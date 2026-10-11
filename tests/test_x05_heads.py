import numpy as np
import pytest
from emotioncore.x05.heads import select_training


def row(uid, speaker, corpus='ET', split='train'):
    return dict(utt_id=uid,speaker_id=speaker,corpus=corpus,official_split=split)


def test_training_membership_and_order():
    cv=[row('a','01'),row('b','06'),row('c','10'),row('d','11'),row('e','19')]
    extra=[row('m','MER','MER'),row('excluded','MER','MER'),row('test','MER','MER','test'),
           row('s','SIMS','CH-SIMS')]
    rows,ids=select_training(cv,extra,{'excluded'})
    assert [rows[i]['utt_id'] for i in ids]==['a','e','m']
    _,all_ids=select_training(cv,extra,{'excluded'},reproduce=True)
    np.testing.assert_array_equal(all_ids,[0,1,2,3,4,5])


def test_training_rejects_et_test():
    with pytest.raises(ValueError,match='leakage'):
        select_training([row('t','02',split='test')],[],set())
