import json
import pytest
from emotioncore.x04.targets import load_sims_rows


def test_sims_target_labels_join_only_frozen_valid_manifest(tmp_path):
    adapter=tmp_path/'adapter.jsonl';manifest=tmp_path/'valid.jsonl'
    rows=[{'utt_id':f'video__{i:04d}','dataset':'CH-SIMS-v2',
           'speaker_id':'sims_valid','official_split':'sims_valid'} for i in range(647)]
    labels=[{'video_id':'video','clip_id':f'{i:04d}','mode':'valid','label':i/646}
            for i in range(647)]
    def write(path, records):path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    write(adapter,rows);write(manifest,labels)
    joined=load_sims_rows(adapter,manifest)
    assert [r['label'] for r in joined]==[r['label'] for r in labels]
    assert 'label' not in rows[0]
    labels[0]['mode']='test';write(manifest,labels)
    with pytest.raises(ValueError):load_sims_rows(adapter,manifest)
    labels[0]['mode']='valid';labels.reverse();write(manifest,labels)
    with pytest.raises(ValueError):load_sims_rows(adapter,manifest)
