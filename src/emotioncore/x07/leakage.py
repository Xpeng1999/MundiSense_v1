"""L2 assertions, independently checked for every requested training source."""
import re
from emotioncore.x07.common import *

REQUIRED = ('Q_epoch0','Q_epoch1','D1_A3','D1_V2','D1_V1','calibration','MER_dedup')

def aliases(uid):
    video,clip = uid.split('__')
    return {uid, f'sims_{video}_{clip}', f'sims_{uid}', f'{video}_{clip}',f'{video}/{clip}.mp4'}

def assert_source(rows, test_ids):
    forbidden = set().union(*(aliases(uid) for uid in test_ids))
    hits=set()
    for row in rows:
        uid=row.get('utt_id',row.get('id'))
        if not isinstance(uid,str) or not uid:
            raise RuntimeError('STOP training row has no auditable ID')
        # Check canonical IDs and historical SIMS aliases, including media paths.
        for token in re.findall(r'[A-Za-z0-9_./-]+',json.dumps(row)):
            suffix=re.search(r'video_\d+/\d+\.mp4$',token)
            if token in forbidden or (suffix and suffix.group() in forbidden):
                hits.add(uid)
        if 'video_id' in row and 'clip_id' in row and f"{row['video_id']}__{row['clip_id']}" in test_ids:
            hits.add(uid)
    if hits:
        raise RuntimeError('STOP SIMS-test leakage: '+','.join(sorted(hits)))
    return {'rows':len(rows),'unique_ids':len({r.get('utt_id',r.get('id')) for r in rows}),'overlap':0}

def assert_exclusions(report, mer_rows, all_sources, test_ids):
    pairs=report['directions']['SIMS_test']['pairs']
    excluded={p['source_id'] for p in pairs}
    expected={'mer_sample_00002517','mer_sample_00000876','mer_samplenew3_00104997'}
    if excluded != expected or len(pairs)!=3 or not excluded <= set(report['excluded_ids']):
        raise RuntimeError('STOP three suspected MER duplicates not established')
    target_aliases=set().union(*(aliases(uid) for uid in test_ids))
    if any(p['target_id'] not in target_aliases for p in pairs):
        raise RuntimeError('STOP duplicate targets not in lockbox')
    for name, rows in {'MER_dedup':mer_rows,**all_sources}.items():
        if excluded & {r.get('utt_id',r.get('id')) for r in rows}:
            raise RuntimeError('STOP suspected duplicate retained in '+name)
    return {'excluded_ids':sorted(excluded),'pairs':pairs,'absent_from_all_training_sources':True}

def run():
    protocol,_=authority()
    inv=protocol['inventory']; sources=inv['leakage_sources']
    try:
        if set(sources)!=set(REQUIRED):
            raise RuntimeError('STOP missing leakage source')
        test=set(inv['splits']['test']['ids']); data={}; checks={}
        for name in REQUIRED:
            path=checked(sources[name])
            data[name]=[json.loads(line) for line in path.read_text().splitlines() if line]
            checks[name]={**assert_source(data[name],test),'source':sources[name]}
        duplicates=assert_exclusions(read(checked(inv['dedup_report'])),data['MER_dedup'],data,test)
        write_new(ROOT/'leakage.json',{'status':'passed','at':now(),'protocol_sha256':sha(PROTOCOL),
            'checks':checks,'suspected_duplicates':duplicates,'test_count':len(test)})
    except Exception as exc:
        stop('L2',exc)
        raise

if __name__=='__main__':
    run()
