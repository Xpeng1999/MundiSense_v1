"""CPU-only API verification on the model provider's bundled example."""
from emotioncore.x06.common import *
from emotioncore.x06.vad import Posterior, segments
from emotioncore.x06.service import Service


def run():
    output = ROOT / 'vad_api_probe'
    meta = start(output)
    path = Path(read(ROOT / 'vad_models/summary.json')['fsmn']) / 'example/vad_example.wav'
    audio = Service.audio(path)
    result = {}
    for name in ('silero', 'fsmn'):
        probability = Posterior(name).score(audio)
        result[name] = {'frames': len(probability), 'minimum': float(probability.min()),
                        'maximum': float(probability.max()),
                        'segments': segments(probability, .5, .2)}
    finish(output, meta, {'input': spec(path), 'results': result, 'gpu_seconds': 0,
                          'purpose': 'provider example API validation only; no parameter selection'})


if __name__ == '__main__':
    run()
