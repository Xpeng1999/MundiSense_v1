import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
from emotioncore.x07.common import committed_protocol,verify_precedes,load_completed,claim_metrics,write_new
from emotioncore.x07.inventory import ids_sha,spec
from emotioncore.x07.leakage import REQUIRED,assert_source,assert_exclusions
from emotioncore.x07.metrics import sentiment,compute
from emotioncore.x07.identity import check_identity
from emotioncore.eval.metrics import LABELS,sims_metrics

def command(repo,*args):
    return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()

def test_protocol_committed_once_before_inference(tmp_path):
    command(tmp_path,'init','-q');command(tmp_path,'config','user.name','Test')
    command(tmp_path,'config','user.email','test@example.invalid')
    (tmp_path/'code.py').write_text('pass\n');command(tmp_path,'add','code.py');command(tmp_path,'commit','-qm','code')
    code=command(tmp_path,'rev-parse','HEAD');p=tmp_path/'protocol.json'
    p.write_text(json.dumps({'code_commit':code}))
    with pytest.raises(subprocess.CalledProcessError):committed_protocol(p,cwd=tmp_path)
    command(tmp_path,'add','protocol.json');command(tmp_path,'commit','-qm','freeze')
    _,receipt=committed_protocol(p,cwd=tmp_path)
    verify_precedes(receipt,'2099-01-01T00:00:00+00:00')
    with pytest.raises(RuntimeError):verify_precedes(receipt,'2000-01-01T00:00:00+00:00')
    p.write_text(json.dumps({'code_commit':code,'modified':True}))
    with pytest.raises(RuntimeError):committed_protocol(p,cwd=tmp_path)
    command(tmp_path,'add','protocol.json');command(tmp_path,'commit','-qm','illegal modification')
    with pytest.raises(RuntimeError):committed_protocol(p,cwd=tmp_path)

@pytest.mark.parametrize('source',REQUIRED)
@pytest.mark.parametrize('alias',['video_0003__0006','sims_video_0003_0006','sims_video_0003__0006'])
def test_each_leakage_assertion(source,alias):
    assert assert_source([{'utt_id':'clean','source':source}],{'video_0003__0006'})['overlap']==0
    with pytest.raises(RuntimeError):assert_source([{'utt_id':alias,'source':source}],{'video_0003__0006'})

def test_leakage_checks_media_paths():
    with pytest.raises(RuntimeError):
        assert_source([{'utt_id':'other','video_path':'/root/Raw/video_0003/0006.mp4'}],{'video_0003__0006'})

def test_three_suspected_duplicates_must_be_excluded():
    excluded=['mer_sample_00002517','mer_sample_00000876','mer_samplenew3_00104997']
    pairs=[{'source_id':x,'target_id':'sims_video_0003_0006'} for x in excluded]
    report={'directions':{'SIMS_test':{'pairs':pairs}},'excluded_ids':excluded}
    ids={'video_0003__0006'}
    assert assert_exclusions(report,[],{},ids)['absent_from_all_training_sources']
    for name in REQUIRED:
        with pytest.raises(RuntimeError):assert_exclusions(report,[],{name:[{'utt_id':excluded[0]}]},ids)
    report['directions']['SIMS_test']['pairs']=pairs[:2]
    with pytest.raises(RuntimeError):assert_exclusions(report,[],{},ids)

def test_metrics_require_complete_exact_population_and_run_once(tmp_path):
    ids=['video_1__01','video_2__01']
    with pytest.raises(RuntimeError):load_completed(tmp_path,ids)
    receipts={}
    for uid in ids:
        path=tmp_path/(uid+'.json');write_new(path,{'id':uid});receipts[uid]=spec(path)
    complete={'ids_sha256':ids_sha(ids),'n':len(ids),'predictions':receipts}
    write_new(tmp_path/'complete.json',complete)
    assert len(load_completed(tmp_path,ids)[1])==2
    claim_metrics(tmp_path)
    with pytest.raises(FileExistsError):claim_metrics(tmp_path)
    with pytest.raises(RuntimeError):load_completed(tmp_path,ids[:1])
    Path(receipts[ids[0]]['path']).write_text('{}')
    with pytest.raises(RuntimeError):load_completed(tmp_path,ids)

def test_sentiment_matches_x01_definition_and_uniform():
    p=np.array([.1,.25,.15,.1,.05,.2,.15])
    expected=p[LABELS.index('joy')]-sum(p[LABELS.index(k)] for k in ('anger','disgust','fear','sadness'))
    assert sentiment(p)==pytest.approx(expected)
    assert sentiment(np.ones(7)/7)==pytest.approx(-3/7)
    assert sentiment(np.eye(7)).tolist()==[0,1,-1,-1,-1,-1,0]

def test_metrics_missing_denominator_and_whole_video_bootstrap():
    rows=[];labels={}; old=[]
    for i,(label,p,available) in enumerate([(1,[0,1,0,0,0,0,0],True),(-1,[0,0,1,0,0,0,0],True),
            (0,[1,0,0,0,0,0,0],True),(-1,[1/7]*7,False)]):
        uid=f'video_{i//2}__{i}';labels[uid]=label
        v={'probabilities':p,'available':available,'accepted_r90':available,'op_n':int(np.argmax(p)) if available else -1}
        rows.append({'id':uid,'systems':{n:v for n in ('H3','H_A','H_V')}})
        old.append({'label':label,'s':float(sentiment(p)) if available else None})
    got=compute(rows,labels,repetitions=40,seed=20261014)['H3']
    assert got['acc2']==sims_metrics(old)['acc_2_nonzero']==pytest.approx(2/3)
    assert got['pearson_available']==pytest.approx(1)
    assert got['video_count']==2 and got['op_r90_coverage']==.75
    assert got['neutral_non_neutral_rate']==0
    assert got==compute(rows,labels,repetitions=40,seed=20261014)['H3']

def test_metric_entrypoint_refuses_early_and_second_label_read(tmp_path,monkeypatch):
    from emotioncore.x07 import metrics
    import pandas as pd
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    meta=tmp_path/'meta.csv';meta.write_text('video_id,clip_id,mode,label\nvideo_1,01,test,1\n')
    uid='video_1__01'; directory=tmp_path/'test';directory.mkdir()
    config={'inventory':{'splits':{'test':{'ids':[uid]}},'meta_csv':spec(meta)},
            'bootstrap':{'repetitions':2,'seed':20261014}}
    write_new(tmp_path/'leakage.json',{'identity_checks':{'test':check_identity(spec(meta),'test',[uid])}})
    monkeypatch.setattr(metrics,'ROOT',tmp_path);monkeypatch.setattr(metrics,'PROTOCOL',protocol)
    monkeypatch.setattr(metrics,'authority',lambda:(config,{'commit':'frozen','commit_unix':0}))
    monkeypatch.setattr(metrics,'stage_gate',lambda split:None)
    calls=[]
    original=pd.read_csv
    def read_once(*args,**kwargs):
        if 'label' in kwargs.get('usecols',[]): calls.append('labels')
        return original(*args,**kwargs)
    monkeypatch.setattr(pd,'read_csv',read_once)
    monkeypatch.setattr(metrics,'compute',lambda *a,**kw:{'H3':{'pearson_available':1.}})
    with pytest.raises(RuntimeError):metrics.run('test')
    assert calls==[] and not (directory/'metrics.started.json').exists()
    pred=directory/'pred.json';write_new(pred,{'id':uid})
    write_new(directory/'complete.json',{'ids_sha256':ids_sha([uid]),'n':1,'predictions':{uid:spec(pred)},
        'protocol_sha256':spec(protocol)['sha256'],'first_inference_at':'2026-10-14T00:00:00+00:00'})
    metrics.run('test');assert calls==['labels']
    with pytest.raises(FileExistsError):metrics.run('test')
    assert calls==['labels']

def test_development_gate_blocks_test_before_prepare(tmp_path,monkeypatch):
    from emotioncore.x07 import common
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    monkeypatch.setattr(common,'PROTOCOL',protocol)
    write_new(tmp_path/'leakage.json',{'status':'passed','protocol_sha256':spec(protocol)['sha256']})
    write_new(tmp_path/'valid/metrics.json',{'protocol_sha256':spec(protocol)['sha256'],
        'systems':{'H3':{'pearson_available':.779}}})
    with pytest.raises(RuntimeError):common.stage_gate('test',tmp_path)

def test_decisions_preserve_frozen_h3_and_secondary_parameters():
    from emotioncore.x07.infer import decisions,unavailable
    from emotioncore.x06.decision import apply
    from emotioncore.x05.calibration import base_scores,predictions
    bundle=json.loads(Path('configs/calibration_bundle_v1.1.json').read_text())
    p={c:[.1,.3,.2,.15,.1,.1,.05] for c in ['Q_AV','Q_A','Q_V','A3','V2','V1']}
    before=json.dumps(bundle,sort_keys=True)
    for face in (True,False):
        got=decisions(bundle,p,face,[1.])
        reference=apply(bundle,p,'H3',has_face=face,speech=True,scores=[1.],alpha=0)
        assert got['H3']['probabilities']==reference['calibrated_probabilities']
        assert got['H3']['accepted_r90']==(reference['label']!='reject')
        assert got['H_A']['available']
        assert got['H_V']['available']==face
    assert json.dumps(bundle,sort_keys=True)==before
    assert not unavailable('decode_failed')['accepted_r90']


@pytest.mark.parametrize('split',['valid','test'])
def test_proposed_loader_never_parses_other_split_label(tmp_path,split):
    from emotioncore.x07.metrics import labels_for_split
    opposite='test' if split=='valid' else 'valid'
    path=tmp_path/'meta.csv'
    path.write_text(f'video_id,clip_id,mode,label\nvideo_1,01,{split},0.6\nvideo_2,02,{opposite},FORBIDDEN_LABEL_MUST_NOT_PARSE\n')
    identity=check_identity(spec(path),split,['video_1__01'])
    assert labels_for_split(path,identity)=={'video_1__01':.6}
    with pytest.raises(RuntimeError):check_identity(spec(path),split,['wrong__id'])

def test_proposed_loader_rejects_multiline_before_labels(tmp_path):
    path=tmp_path/'meta.csv'
    path.write_text('video_id,clip_id,mode,label,text\nvideo_1,01,valid,0.6,"two\nlines"\n')
    with pytest.raises(RuntimeError,match='physical line'):
        check_identity(spec(path),'valid',['video_1__01'])


@pytest.mark.parametrize('failure',['id_set','physical_lines','receipt_mismatch','meta_sha'])
def test_identity_failure_never_claims_metric_lock(tmp_path,monkeypatch,failure):
    from emotioncore.x07 import metrics
    import pandas as pd
    uid='video_1__01'; directory=tmp_path/'test';directory.mkdir()
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    meta=tmp_path/'meta.csv'
    meta.write_text('video_id,clip_id,mode,label,text\nvideo_1,01,test,0.5,single\n')
    identity=check_identity(spec(meta),'test',[uid])
    if failure=='id_set':
        meta.write_text('video_id,clip_id,mode,label,text\nvideo_2,01,test,FORBIDDEN,single\n')
    elif failure=='physical_lines':
        meta.write_text('video_id,clip_id,mode,label,text\nvideo_1,01,test,FORBIDDEN,"two\nlines"\n')
    elif failure=='receipt_mismatch':
        identity['record_count']+=1
    config={'inventory':{'splits':{'test':{'ids':[uid]}},'meta_csv':spec(meta)}}
    if failure=='meta_sha':
        meta.write_text(meta.read_text()+'\n')
    write_new(tmp_path/'leakage.json',{'identity_checks':{'test':identity}})
    pred=directory/'pred.json';write_new(pred,{'id':uid})
    write_new(directory/'complete.json',{'ids_sha256':ids_sha([uid]),'n':1,'predictions':{uid:spec(pred)},
        'protocol_sha256':spec(protocol)['sha256'],'first_inference_at':'2026-10-14T00:00:00+00:00'})
    monkeypatch.setattr(metrics,'ROOT',tmp_path);monkeypatch.setattr(metrics,'PROTOCOL',protocol)
    monkeypatch.setattr(metrics,'authority',lambda:(config,{'commit':'frozen','commit_unix':0}))
    monkeypatch.setattr(metrics,'stage_gate',lambda split:None)
    original=pd.read_csv
    def identities_only(*a,**kw):
        assert kw['usecols']==['video_id','clip_id','mode']
        return original(*a,**kw)
    monkeypatch.setattr(pd,'read_csv',identities_only)
    with pytest.raises(RuntimeError):metrics.run('test')
    assert not (directory/'metrics.started.json').exists()
    assert not (directory/'metrics.json').exists()


def test_v2_freeze_preserves_all_scientific_definitions():
    from emotioncore.x07.freeze_v2 import build,V1,V1_SHA
    assert spec(V1)['sha256']==V1_SHA
    v1=json.loads(V1.read_text());before=json.dumps(v1,sort_keys=True)
    v2=build(v1,'new-code',{'new.py':'new-sha'},'later',{'authorization':'revision-1'})
    changed={k for k in v1 if v1[k]!=v2[k]}
    assert changed=={'version','registered_at','code_commit','code_sha256','execution'}
    assert set(v2)-set(v1)=={'revision'}
    assert {k for k in v1['execution'] if v1['execution'][k]!=v2['execution'][k]}=={'labels'}
    assert set(v2['execution'])-set(v1['execution'])=={'root','protocol'}
    assert json.dumps(v1,sort_keys=True)==before
