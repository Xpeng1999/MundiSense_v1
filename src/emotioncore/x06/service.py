"""Resident frozen Q_mix, feature encoders, D1_fmhx heads and decision parameters."""
import math
import time
from pathlib import Path
import numpy as np
from emotioncore.eval.metrics import LABELS
from emotioncore.x05.calibration import base_scores, predictions
from emotioncore.x06.common import R2, R4B, R5, read, sha256
from emotioncore.x06.gate import route, verify_only_gate_changed


class Service:
    def __init__(self):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel, CLIPImageProcessor, CLIPModel, Qwen2_5OmniProcessor
        from emotioncore.x03.q_smoke import MODEL, model_and_optimizer
        from emotioncore.x04.heads import build_model
        torch.set_num_threads(4)
        self.torch = torch
        self.bundle = read('configs/calibration_bundle_v1.1.json')
        verify_only_gate_changed(Path('configs/calibration_bundle_v1.json').read_bytes(),
                                 Path('configs/calibration_bundle_v1.1.json').read_bytes())
        self.processor = Qwen2_5OmniProcessor.from_pretrained(str(MODEL), local_files_only=True)
        self.tokens = [self.processor.tokenizer(c, add_special_tokens=False).input_ids[0] for c in LABELS]
        self.q, optimizer, _ = model_and_optimizer()
        del optimizer
        adapter = R4B / 'Q/final/best_adapter_eval8'
        assert sha256(adapter / 'adapter_model.safetensors') == read(R4B / 'Q/final/summary.json')['best_checkpoint_sha256']
        self.q.load_adapter(str(adapter), adapter_name='best')
        self.q.set_adapter('best')
        self.q.eval().requires_grad_(False)
        self.a_processor = AutoFeatureExtractor.from_pretrained(str(R2 / 'models/A3'), local_files_only=True)
        self.a = AutoModel.from_pretrained(str(R2 / 'models/A3'), local_files_only=True).eval().to('cuda:0')
        self.v2_processor = CLIPImageProcessor.from_pretrained(str(R2 / 'models/V2'), local_files_only=True)
        self.v2 = CLIPModel.from_pretrained(str(R2 / 'models/V2'), local_files_only=True).eval().to('cuda:0')
        self.v1 = torch.load('/home/administrator/emotion-data/open_resources_v1/models/emotiefflib/enet_b0_8_best_afew.pt',
                             map_location='cpu', weights_only=False)
        self.v1_w = self.v1.classifier[0].weight.detach().clone().to('cuda:0')
        self.v1_b = self.v1.classifier[0].bias.detach().clone().to('cuda:0')
        self.v1.classifier = torch.nn.Identity()
        self.v1.eval().to('cuda:0')
        self.heads = {}
        for name, dimension in [('A3', 1024), ('V2', 768), ('V1', 1288)]:
            path = R5 / f'heads/D1_fmhx/{name}/final_head.pt'
            assert sha256(path) == read(path.parent / 'summary.json')['checkpoint']['sha256']
            head = build_model(dimension, name == 'A3', 'cpu')
            head.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['state_dict'])
            self.heads[name] = head.eval()
        self.previous = {}

    def head_probability(self, name, frames):
        torch = self.torch
        count = frames.shape[-2]
        maximum = 128 if name == 'A3' else 8
        indices = np.linspace(0, count - 1, min(count, maximum)).round().astype(int)
        packed = np.zeros((*frames.shape[:-2], maximum, frames.shape[-1]), dtype=np.float16)
        packed[..., :len(indices), :] = frames[..., indices, :]
        tensor = torch.from_numpy(packed.astype(np.float32)[None])
        with torch.inference_mode():
            p = self.heads[name](tensor, torch.tensor([len(indices)])).softmax(1)[0, :7]
        p = p / p.sum()
        return p.numpy().astype(float).tolist()

    @staticmethod
    def audio(path):
        import soundfile as sf
        import scipy.signal
        signal, rate = sf.read(path, dtype='float32', always_2d=True)
        signal = signal.mean(1)
        if rate != 16000:
            divisor = math.gcd(rate, 16000)
            signal = scipy.signal.resample_poly(signal, 16000 // divisor, rate // divisor)
        if not len(signal) or not np.isfinite(signal).all():
            raise ValueError('STOP empty or nonfinite audio')
        return np.asarray(signal, dtype=np.float32)

    def q_probability(self, form, signal, video):
        import cv2
        from emotioncore.x03.q_smoke import PREFIX, PROMPT, assert_modality_receipt
        tic = time.monotonic()
        content, frames, fps = [], None, None
        if form in ('V', 'AV'):
            capture = cv2.VideoCapture(str(video))
            frames = []
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            finally:
                capture.release()
            if len(frames) != 8 or any(f.shape != (336, 336, 3) for f in frames):
                raise ValueError('STOP invalid frozen Q visual input')
            content.append({'type': 'video', 'video': str(video)})
        video_decode_s = time.monotonic() - tic
        tic = time.monotonic()
        if form in ('A', 'AV'):
            content.append({'type': 'audio', 'audio': 'segment.wav'})
        content.append({'type': 'text', 'text': PROMPT})
        conversation = [{'role': 'system', 'content': [{'type': 'text', 'text':
            '你是外显情绪识别模型，只从可听和可见信息判断。'}]}, {'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) + PREFIX
        args = dict(text=text, audio=[signal] if form in ('A', 'AV') else None,
                    videos=[frames] if frames is not None else None, return_tensors='pt',
                    padding=True, use_audio_in_video=False)
        if fps is not None:
            args['fps'] = fps
        values = self.processor(**args)
        assert_modality_receipt(form, set(values))
        values = {k: v.to('cuda:0') for k, v in values.items()}
        self.torch.cuda.synchronize()
        preprocess_s = time.monotonic() - tic
        tic = time.monotonic()
        with self.torch.inference_mode():
            output = self.q(**values, use_cache=False, use_audio_in_video=False)
            p = output.logits[0, -1].float()[self.tokens].softmax(0).cpu().tolist()
        self.last_q_timing = {'video_decode': video_decode_s, 'preprocess': preprocess_s,
                              'forward': time.monotonic() - tic}
        del values, output
        return p

    def components(self, item, system='H3', has_speech=True, compare_both=False):
        from PIL import Image
        from emotioncore.x02.visual_v1 import MEAN, STD, SIZE
        torch = self.torch
        timing = {}; output = {}
        has_face = item.get('q_video') is not None and item['q_video']['status'] == 'ok'
        name = system if has_face and has_speech else 'H_A' if has_speech else 'H_V'
        needed = set(self.bundle['systems'][name]['components'])
        if compare_both:
            needed |= set(self.bundle['systems']['H_ref']['components'])
        signal = None
        tic = time.monotonic()
        if has_speech:
            signal = self.audio(item['wav'])
        timing['audio_decode'] = time.monotonic() - tic
        with torch.inference_mode():
            if 'A3' in needed:
                tic = time.monotonic()
                values = self.a_processor(signal, sampling_rate=16000, return_tensors='pt').to('cuda:0')
                states = self.a(**values, output_hidden_states=True).hidden_states[-4:]
                features = torch.stack(states).squeeze(1).float().cpu().numpy().astype(np.float16)
                output['A3'] = self.head_probability('A3', features)
                timing['A3'] = time.monotonic() - tic
                del values, states, features
            images = []
            if has_face:
                for path in item['face']['crop_paths']:
                    with Image.open(path) as image:
                        images.append(image.convert('RGB').copy())
            for name in ('V2', 'V1'):
                if name not in needed:
                    continue
                tic = time.monotonic()
                if not images:
                    output[name] = [1 / 7] * 7
                elif name == 'V2':
                    values = self.v2_processor(images=images, return_tensors='pt').to('cuda:0')
                    features = self.v2.get_image_features(**values).pooler_output.float().cpu().numpy()
                    output[name] = self.head_probability(name, features.astype(np.float16))
                    del values, features
                else:
                    mean = torch.tensor(MEAN).view(3, 1, 1); std = torch.tensor(STD).view(3, 1, 1)
                    batch = torch.stack([(torch.from_numpy(np.asarray(im.resize((SIZE, SIZE)), dtype=np.float32) / 255.)
                              .permute(2, 0, 1) - mean) / std for im in images]).to('cuda:0')
                    features = self.v1(batch)
                    logits = torch.nn.functional.linear(features, self.v1_w, self.v1_b)
                    combined = np.concatenate((features.float().cpu().numpy(), logits.float().cpu().numpy()), axis=-1)
                    output[name] = self.head_probability(name, combined.astype(np.float16))
                    del batch, features, logits, combined
                timing[name] = time.monotonic() - tic
            for name in ('Q_A', 'Q_AV', 'Q_V'):
                if name not in needed:
                    continue
                tic = time.monotonic()
                form = name[2:]
                if not has_face and form == 'V':
                    output[name] = [1 / 7] * 7
                else:
                    effective = 'A' if form == 'AV' and not has_face else form
                    if effective == 'A' and 'Q_A' in output:
                        output[name] = output['Q_A']
                    else:
                        output[name] = self.q_probability(effective, signal, item['q_video']['output'] if has_face else None)
                        timing.update({name + '_' + key: value for key, value in self.last_q_timing.items()})
                timing[name] = time.monotonic() - tic
        return output, has_face, timing

    def decide(self, component_probs, system, *, has_face=True, has_speech=True,
               scores=(), alpha=0., stream_id='', use_gate=True):
        routing, reason = route(has_face, has_speech, scores) if use_gate else ('full', 'equivalence_without_gate')
        if routing == 'reject':
            return {'label': 'reject', 'confidence': 0., 'route': routing, 'reason': reason}
        name = system if routing == 'full' else routing
        params = self.bundle['systems'][name]
        logp = np.log(np.clip(np.asarray([component_probs[c] for c in params['components']])[None], 1e-12, 1))
        scores = base_scores(logp, params)
        p, _, _, _ = predictions(scores, params['neutral_bias'])
        p = p[0]
        if has_speech and alpha:
            key = (stream_id, name)
            if key in self.previous:
                p = (1 - alpha) * p + alpha * self.previous[key]
            self.previous[key] = p.copy()
        decision = np.log(np.clip(p, 1e-12, 1))
        decision[0] += params['neutral_bias']
        label = int(decision.argmax()); confidence = float(p[label])
        point = '90' if has_speech else '80'
        accepted = confidence >= params['thresholds'][point]
        return {'label': LABELS[label] if accepted else 'reject', 'unrejected_label': LABELS[label],
                'confidence': confidence, 'route': name, 'reason': reason,
                'calibrated_probabilities': p.tolist(), 'rejection_point': point}
