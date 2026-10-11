"""Deterministic speaker timelines with separate original audio/video clocks."""
import random
from emotioncore.x06.common import FMH, TEST, SEED


def build_timelines(records, seed=SEED):
    """MP4 duration defines each truth interval; no source duration is rounded."""
    result = {}
    for speaker in sorted(FMH | TEST):
        selected = sorted((r for r in records if r['speaker_id'] == speaker),
                          key=lambda r: (r['group'], r['dialogue'], r['utt_id']))
        rng = random.Random(seed + int(speaker))
        entries = []
        cursor = 0.
        for index, row in enumerate(selected):
            duration = float(row['mp4']['duration_s'])
            if duration <= 0:
                raise ValueError('STOP invalid source duration')
            entries.append({'kind': 'utterance', 'synthetic': False,
                            'utt_id': row['utt_id'], 'speaker_id': speaker,
                            'start_s': cursor, 'end_s': cursor + duration,
                            'duration_s': duration, 'audio_duration_s': row['wav']['duration_s'],
                            'label': row['labels']['M'], 'video': row['mp4'], 'audio': row['wav']})
            cursor += duration
            if index < len(selected) - 1:
                gap = rng.uniform(.3, 1.5)
                entries.append({'kind': 'gap', 'synthetic': True, 'speaker_id': speaker,
                                'start_s': cursor, 'end_s': cursor + gap,
                                'noise_reference_utt_id': row['utt_id'],
                                'noise_seed': rng.getrandbits(32), 'hold_previous_frame': True})
                cursor += gap
        result[speaker] = {'speaker_id': speaker,
                           'cohort': 'fmh' if speaker in FMH else 'et_test',
                           'duration_s': cursor, 'utterances': len(selected), 'entries': entries}
    return result


def overlap_seconds(start, end, other_start, other_end):
    return max(0., min(end, other_end) - max(start, other_start))
