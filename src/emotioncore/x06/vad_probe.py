"""CPU-only API verification on the model provider's bundled example."""
from emotioncore.x06.common import *
from emotioncore.x06.vad import Posterior, segments
from emotioncore.x06.service import Service
from emotioncore.x06.online_vad import posterior_chunks


def run():
    output = ROOT / 'vad_online_probe'
    meta = start(output)
    path = Path(read(ROOT / 'vad_models/summary.json')['fsmn']) / 'example/vad_example.wav'
    audio = Service.audio(path)
    result = {}
    for name in ('silero', 'fsmn'):
        model = Posterior(name)
        probability = model.score(audio)
        import numpy as np
        incremental = np.concatenate([chunk['probabilities'] for chunk in posterior_chunks(model, audio)])
        assert np.array_equal(probability, incremental)
        result[name] = {'frames': len(probability), 'minimum': float(probability.min()),
                        'maximum': float(probability.max()),
                        'segments': segments(probability, .5, .2), 'incremental_posteriors_identical': True}
    finish(output, meta, {'input': spec(path), 'results': result, 'gpu_seconds': 0,
                          'purpose': 'provider example API validation only; no parameter selection'})


if __name__ == '__main__':
    run()
