"""Re-estimate all A3 GPU arms after the first complete outer-fold grid."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import math
from pathlib import Path
import numpy as np
from emotioncore.x02 import head_h1
from emotioncore.x02.runtime import git_provenance, write_json_atomic
from emotioncore.x04.heads import FOLDS_SHA
from emotioncore.x04.report import replace_section


def project(grid, counts, charged_hours, remaining_gpu_hours, cpu_remaining_hours=0.):
    if len(grid) != 6 or any(not r.get('train_batches') or not r.get('validation_batches') for r in grid):
        raise ValueError('First D0 outer-fold grid must include all six timed configurations')
    train_step = max(r['train_seconds']/r['train_batches'] for r in grid)
    val_step = max(r['validation_seconds']/r['validation_batches'] for r in grid)
    overhead = max(max(0.,r['elapsed_s']-r['train_seconds']-r['validation_seconds']) for r in grid)
    selection_epochs = min(50, max(r['epochs_executed'] for r in grid)+5)
    refit_epochs = max(r['best_epoch'] for r in grid)
    def arm_cost(spec, epochs, refit):
        selection = sum(spec['grid_count']*(epochs*(math.ceil(n/128)*train_step+math.ceil(v/128)*val_step)+overhead)
                        for n,v in zip(spec['selection_n'],spec['validation_n']))
        if spec['arm']=='D0':
            selection -= spec['grid_count']*(epochs*(math.ceil(spec['selection_n'][0]/128)*train_step+math.ceil(spec['validation_n'][0]/128)*val_step)+overhead)
        refits = sum(refit*math.ceil(n/128)*train_step+overhead for n in spec['refit_n'])
        # Outer prediction has the same shapes as inner validation.
        prediction = sum(math.ceil(n/128)*val_step for n in spec['outer_n'])
        return (selection+refits+prediction)/3600
    arms = {s['arm']:{'remaining_gpu_hours':arm_cost(s,selection_epochs,refit_epochs),
                       'remaining_gpu_hours_if_all_reach_50':arm_cost(s,50,50),**s} for s in counts}
    head_hours=sum(v['remaining_gpu_hours'] for v in arms.values())
    total=charged_hours+1.2*(head_hours+remaining_gpu_hours)+1/6
    serial_remaining=head_hours+remaining_gpu_hours
    return {'status':'measured_first_outer_fold_grid','head_device':'cuda:0','gpu_heads_max_concurrency':1,
        'train_seconds_per_batch':train_step,'validation_seconds_per_batch':val_step,
        'config_overhead_seconds':overhead,'selection_epochs_projection':selection_epochs,'refit_epochs_projection':refit_epochs,
        'assumptions':'Maximum measured per-batch cost; max observed grid epochs plus patience5 capped50; max observed best epoch for refits; 50-epoch bound separately reported. No early-stopping cap or grid changed.',
        'A3_arms':arms,'A3_remaining_gpu_hours':head_hours,'charged_gpu_hours':charged_hours,
        'remaining_other_gpu_hours':remaining_gpu_hours,'reserve_multiplier':1.2,'load_reserve_hours':1/6,
        'total_projection_gpu_hours':total,'limit_gpu_hours':20,'passed':total<=20,
        'remaining_wall_hours_projection':max(serial_remaining,cpu_remaining_hours)*1.2,
        'CPU_visual_remaining_hours_projection':cpu_remaining_hours,
        'total_projection_if_all_reach_50':charged_hours+1.2*(sum(v['remaining_gpu_hours_if_all_reach_50'] for v in arms.values())+remaining_gpu_hours)+1/6,
        'Q_training':False}


def run(args):
    git=git_provenance(Path.cwd())
    if head_h1.sha256(Path('configs/folds_v1.json'))!=FOLDS_SHA:raise ValueError('Frozen fold SHA changed')
    folds=json.loads(Path('configs/folds_v1.json').read_text())
    cv=head_h1.load_rows(Path('experiments/X01_benchmark/results/et_utterances.jsonl'),folds)
    el=np.load(args.r2/'packs/A3/lengths.npy');nl=np.load(args.derived/'packs/A3/lengths.npy')
    extra=[json.loads(s) for s in (args.derived/'manifests/all_features.jsonl').read_text().splitlines()]
    dedup=json.loads(args.dedup.read_text());excluded=set(dedup['excluded_ids'])
    mer=sum(r['corpus']=='MER' and r['official_split']=='train' and r['utt_id'] not in excluded and nl[i]>0 for i,r in enumerate(extra))
    audio_mer=sum(r['corpus']=='MER' and r['official_split']=='train' and r['utt_id'] not in set(dedup['audio_confirmed_ids']) and nl[i]>0 for i,r in enumerate(extra))
    crema=sum(r['corpus']=='CREMA' and nl[i]>0 for i,r in enumerate(extra))
    speakers=np.asarray([r['speaker_id'] for r in cv]);specs=[]
    for arm,n in [('D0',0),('D1',mer),('D2',mer),('D3',mer+crema)]+([('D1_audioonly',audio_mer)] if dedup['fingerprint_available'] else []):
        selection=[];validation=[];refit=[];outer=[]
        for f in folds['folds']:
            selection.append(n+(int(((el>0)&np.isin(speakers,f['inner_train_speakers'])).sum()) if arm!='D2' else 0))
            validation.append(int(((el>0)&np.isin(speakers,f['inner_validation_speakers'])).sum()))
            refit.append(n+(int(((el>0)&~np.isin(speakers,f['outer_holdout_speakers'])).sum()) if arm!='D2' else 0))
            outer.append(int(np.isin(speakers,f['outer_holdout_speakers']).sum()))
        refit.append(n+(int((el>0).sum()) if arm!='D2' else 0))
        specs.append({'arm':arm,'grid_count':6 if arm=='D0' else 18,'selection_n':selection,'validation_n':validation,'refit_n':refit,'outer_n':outer})
    first=json.loads(args.first_grid.read_text());preflight=json.loads(args.budget.read_text())
    visual_cpu=args.cpu_remaining_hours
    cpu_path=Path('experiments/X04a_data/results/cpu_timing_estimate.json')
    if cpu_path.exists():
        # Keep the older measured CPU plan visible; do not silently treat its
        # 3--5-day estimate for CPU A3 as the current GPU schedule.
        cpu_measurement=json.loads(cpu_path.read_text())
        projected=[]
        for encoder in ('V2','V1'):
            measured=cpu_measurement['benchmarks'][encoder]
            epoch=cpu_measurement['X02_history'][encoder]['best_epoch_max']
            for s in specs:
                if s['arm']=='D1_audioonly':continue
                t=measured['steady_train_median_s'];v=measured['steady_forward_median_s']
                seconds=sum(s['grid_count']*min(50,epoch+5)*(math.ceil(n/128)*t+math.ceil(k/128)*v)
                            for n,k in zip(s['selection_n'],s['validation_n']))
                seconds+=sum(epoch*math.ceil(n/128)*t for n in s['refit_n'])
                projected.append(seconds/3600)
        # Eight independently parallel CPU jobs; retain 20% elsewhere for IO.
        visual_cpu=max(visual_cpu,max(projected)-args.cpu_elapsed_hours,0.)
    receipt=project(first['grid'],specs,args.charged_gpu_hours,preflight['Q_mer_test_projection_hours'],visual_cpu)
    receipt.update(git=git,at=dt.datetime.now(dt.timezone.utc).isoformat(),first_grid_sha256=head_h1.sha256(args.first_grid),
                   budget_sha256=head_h1.sha256(args.budget),pipeline_wall_elapsed_hours=args.wall_elapsed_hours,
                   total_wall_hours_projection=args.wall_elapsed_hours+receipt['remaining_wall_hours_projection'])
    write_json_atomic(args.output/'wall_time_estimate.json',receipt)
    body=(f"A3 首个外折完整6配置网格完成，新取数方式与 X02 完全相等。所有 A3 头串行使用 cuda:0，V1/V2 头在 CPU 并行。\n\n"
          f"GPU 已计入 {receipt['charged_gpu_hours']:.3f} 小时；全部 A3 臂剩余投影 {receipt['A3_remaining_gpu_hours']:.3f} 小时，"
          f"Q MER-test 预留 {receipt['remaining_other_gpu_hours']:.3f} 小时。总 GPU 投影 **{receipt['total_projection_gpu_hours']:.3f} 小时**，"
          f"含20%余量及10分钟加载预留，上限20小时；门禁{'通过' if receipt['passed'] else '失败，停止后续模型'}。\n\n"
          f"总墙钟投影 {receipt['total_wall_hours_projection']:.2f} 小时，剩余 {receipt['remaining_wall_hours_projection']:.2f} 小时；"
          f"采用实测最大每批耗时，选参按首折最大实际epoch加5（不超过50），重训按最大最佳epoch。"
          f"若所有配置均达到50epoch，GPU总量为{receipt['total_projection_if_all_reach_50']:.3f}小时；此为单独报告的上界。"
          f"实际早停、共享内存/IO和后续臂收敛不同会改变时长，实际GPU用时仍硬性累计限制20小时。\n\n"
          "首折回执和逐臂预测见 wall_time_estimate.json。根分区最低80,000,000,000字节；所有新增派生媒体在H盘。")
    report=Path('experiments/X04a_data/REPORT.md').read_text()
    (args.output/'REPORT.md').write_text(replace_section(report,12,body))
    if not receipt['passed']:raise RuntimeError('STOP projected X04a GPU time exceeds20h after first A3 grid')


def main():
    p=argparse.ArgumentParser();p.add_argument('--derived',type=Path,required=True);p.add_argument('--dedup',type=Path,required=True)
    p.add_argument('--first-grid',type=Path,required=True);p.add_argument('--budget',type=Path,required=True)
    p.add_argument('--charged-gpu-hours',type=float,required=True);p.add_argument('--wall-elapsed-hours',type=float,required=True)
    p.add_argument('--cpu-remaining-hours',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu-elapsed-hours',type=float,default=0.)
    p.add_argument('--r2',type=Path,default=Path('/home/administrator/emotion-data/emotioncore_runs/X02'))
    run(p.parse_args())


if __name__=='__main__':main()
