"""Label-free CSV identity preflight shared by L2 and pre-lock metrics."""
import pandas as pd
from emotioncore.x07.common import checked, ids_sha


def check_identity(source, split, expected_ids):
    path = checked(source)
    identity = pd.read_csv(path, usecols=['video_id', 'clip_id', 'mode'],
                           dtype=str, keep_default_na=False)
    selected = identity[identity['mode'] == split]
    ids = [f'{r.video_id}__{r.clip_id}' for r in selected.itertuples()]
    if (len(ids) != len(expected_ids) or len(set(ids)) != len(ids)
            or set(ids) != set(expected_ids)):
        raise RuntimeError('STOP metric identity mismatch before label access')
    # skiprows uses physical lines; only count lines, never parse their labels.
    with path.open(newline='') as stream:
        line_count = sum(1 for _ in stream)
    if line_count != len(identity) + 1:
        raise RuntimeError('STOP CSV is not one physical line per record')
    checked(source)
    return {'split': split, 'source': source, 'record_count': len(identity),
            'physical_line_count': line_count, 'selected_count': len(ids),
            'ids_sha256': ids_sha(ids), 'ordered_ids': ids,
            'selected_lines': [int(i) + 1 for i in selected.index],
            'id_set_check': 'passed', 'one_line_per_record_check': 'passed'}
