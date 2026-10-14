"""Formal CPU-only startup inventory and conservative GPU budget."""
import math
from emotioncore.x06b.common import *
from emotioncore.x06b.dialogue import build


def estimate(n,dialogues,rates,old_replay,visual_rate):
    speech_seconds=sum(e['duration_s'] for t in dialogues.values() for e in t['entries'] if e['kind']=='utterance')
    silence_windows=sum(math.floor((e['end_s']-e['start_s'])/2) for t in dialogues.values() for e in t['entries'] if e['kind']=='gap')
    # Same H3 pathways, excluding H_ref-only forwards from the shared old job.
    h3=rates['both']-rates['Q_A']-rates['V1']
    vad_n=max(math.ceil(n*1.2),math.ceil(speech_seconds/15))
    visual_n=silence_windows+math.ceil(n*.1)
    stages={'R2_cpu_only':0.,'R3_correct':n*h3,'R4_correct_on_dialogue':n*h3,
            'R4_vad':vad_n*h3+visual_n*visual_rate,'R5':old_replay}
    future=sum(stages.values()); reserve=.2*future+300
    return {'stage_gpu_seconds':stages,'future_base_s':future,'reserve_s':reserve,
            'estimated_total_gpu_hours':(future+reserve)/3600,'speech_segments_estimate':vad_n,
            'silence_windows_estimate':visual_n,'metadata_gap_windows':silence_windows,'h3_seconds_per_segment':h3,
            'visual_seconds_per_segment':visual_rate,'correct_boundary_n_per_stage':n}


def run():
    no_stop();out,meta=start('preflight'); records=rows(MANIFEST)
    dialogues=build(records);n=sum(t['utterances'] for t in dialogues.values());assert n==5195
    for speaker,timeline in dialogues.items():write_json_atomic(out/f'dialogue_{speaker}.json',timeline)
    old=read('experiments/X06a_stream/results/revision_2_budget_before_s7.json');checked(old['source']);rates=old['data']['rates_s']
    replay=read('experiments/X06a_stream/results/revision_2_gpu_jobs_realtime_H3.json');checked(replay['source'])
    reference=read('experiments/X06a_stream/results/preflight.json');checked(reference['source'])
    visual=reference['data']['q_rates']['V']['seconds_per_sentence']+rates['V2']+rates['V1']
    result=estimate(n,dialogues,rates,replay['data']['charged_gpu_s'],visual)
    status='passed' if result['estimated_total_gpu_hours']<=14 else 'STOP_budget_estimate'
    result.update(status=status,limit_gpu_hours=14,used_gpu_hours=0,gpu_seconds=0,
        sources={'rates':old['source'],'old_replay':replay['source'],'visual':reference['source'],'manifest':spec(MANIFEST)},
        dialogue_timelines={s:spec(out/f'dialogue_{s}.json') for s in dialogues},
        notes=['CPU-only R2; no GPU decoding chosen','R3 and R4 each include their own correct-boundary control',
               'R4 estimates 1.2 voice segments per truth, plus metadata gap windows and 0.1 visual windows per truth; exact frozen VAD counts must replace estimates before R4 GPU launch',
               'R5 reserves measured old H3 full-job wall time rather than assuming realtime optimization success',
               'H3 rate subtracts measured H_ref-only Q-A and V1 costs; preserves remaining measured process overhead',
               '20 percent future work plus 300 seconds initialization reserve'])
    finish(out,meta,result);write_json_atomic(ROOT/'budget_current.json',spec(out/'summary.json'))
    print(json.dumps({'status':status,'hours':result['estimated_total_gpu_hours']}))
    if status!='passed':stop('startup',status,spec(out/'summary.json'))


if __name__=='__main__':run()
