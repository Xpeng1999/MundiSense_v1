"""Unchanged S2 overlap scoring with explicit split/merge decomposition."""
import argparse,numpy as np
from emotioncore.x06b.common import *
from emotioncore.x06.aggregate import overlaps,metrics
from emotioncore.x06.decision import apply
from emotioncore.x06.gate import route


def score(bundle,truth,segments):
    matched=overlaps(truth,segments)
    if not matched:return {'label':'reject','reason':'uncovered_by_vad'}
    has_face=any(segments[i]['has_face'] for i,_ in matched)
    ambiguous=any(route(segments[i]['has_face'],True,segments[i]['track_scores'])[1]=='ambiguous_multi_face' for i,_ in matched)
    keys=bundle['systems']['H3' if has_face else 'H_A']['components'];total=sum(w for _,w in matched)
    probabilities={c:(sum(w*np.asarray(segments[i]['component_probs'][c]) for i,w in matched)/total).tolist() for c in keys}
    return apply(bundle,probabilities,'H3',has_face=has_face,speech=True,scores=[1.,.5] if ambiguous else [],alpha=0)


def run(cohort):
    no_stop();out,meta=start('r4_analysis_'+cohort);bundle=read('configs/calibration_bundle_v1.1.json')
    source=read(ROOT/f'r4_vad_{cohort}/summary.json')['predictions']
    correct_source=read(ROOT/f'r4_correct_{cohort}/summary.json')['predictions']
    speech=[r for r in rows(checked(source)) if not r['visual_only']]
    correct={r['utt_id']:r for r in rows(checked(correct_source))};results=[];pred=[];gold=[];baseline=[]
    categories={k:[] for k in ['uncovered','one_to_one','merged_only','split_only','merged_and_split']}
    for speaker in sorted(FMH if cohort=='fmh' else TEST):
        truths=[t for t in read(ROOT/f'preflight/dialogue_{speaker}.json')['entries'] if t['kind']=='utterance']
        segments=[s for s in speech if s['speaker_id']==speaker]
        truth_counts=[len(overlaps(s,truths)) for s in segments]
        for t in truths:
            matched=overlaps(t,segments);split=len(matched)>1;merged=any(truth_counts[i]>1 for i,_ in matched)
            group='uncovered' if not matched else 'merged_and_split' if split and merged else 'merged_only' if merged else 'split_only' if split else 'one_to_one'
            decision=score(bundle,t,segments);old=correct[t['utt_id']]['decision']['label']
            categories[group].append(len(results));gold.append(t['label']);pred.append(decision['label']);baseline.append(old)
            results.append({'utt_id':t['utt_id'],'speaker_id':speaker,'truth':t['label'],'group':group,
                'vad_segments':len(matched),'correct_label':old,**decision})
    a,b=metrics(gold,baseline),metrics(gold,pred);delta=b['macro_f1']-a['macro_f1'];groups={}
    for k,ix in categories.items():groups[k]={'n':len(ix),'fraction':len(ix)/len(gold),'metrics':metrics([gold[i] for i in ix],[pred[i] for i in ix])}
    changed=[r['group'] in ('merged_only','split_only','merged_and_split') for r in results]
    counterfactual=metrics(gold,[baseline[i] if changed[i] else pred[i] for i in range(len(gold))])
    (out/'paired.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    finish(out,meta,{'status':'completed','cohort':cohort,'n':len(gold),'correct_boundaries':a,'vad':b,'vad_minus_correct_macro_f1':delta,
        'acceptance':delta>=-.03 if cohort=='fmh' else None,'groups':groups,'split_merge_replaced_with_correct':counterfactual,
        'paired':spec(out/'paired.jsonl'),'predictions':source,'correct_predictions':correct_source,'gpu_seconds':0,
        'limitation':'Other speakers provide gap duration only; their sound is assumed absent from the target microphone'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cohort',choices=['fmh','et_test']);run(p.parse_args().cohort)
