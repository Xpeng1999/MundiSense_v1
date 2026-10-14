"""Report-only backlog and memory plots, with raw sources retained on server."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from emotioncore.x06b.common import *
from emotioncore.x06.live import quantiles


def run():
    no_stop();out,meta=start('curves');latest=read(ROOT/'r5/summary.json');previous=read(OLD/'realtime_H3/summary.json')
    old_source=previous['events_file'];new_source=latest['latency_file'];memory_source=latest['memory_file']
    old,new,memory=rows(checked(old_source)),rows(checked(new_source)),rows(checked(memory_source))
    receipts={r['segment_id']:r for r in rows(checked(previous['latency_receipts']))}
    old_starts={};start_sources={}
    for s in sorted(TEST):
        first=next(r for r in old if r['speaker_id']==s)
        path=OLD/'realtime_H3/inputs'/first['segment_id']/'input.json'
        old_starts[s]=read(path)['input_stream_started_monotonic'];start_sources[s]=spec(path)
    old_voice=[r for r in old if not r['visual_only']]
    truth_delays=[];uncovered=[]
    for s in sorted(TEST):
        current=[r for r in old_voice if r['speaker_id']==s]
        for t in source_timeline(s)['entries']:
            if t['kind']!='utterance':continue
            matched=[r for r in current if r['start_s']<t['end_s'] and r['end_s']>t['start_s']]
            if not matched:uncovered.append(t['utt_id']);continue
            written=max(receipts[r['segment_id']]['after_flush_monotonic'] for r in matched)
            truth_delays.append(written-old_starts[s]-t['end_s'])
    old_comparison={'latency_from_vad':quantiles([receipts[r['segment_id']]['latency_s'] for r in old_voice]),
        'latency_from_scheduled_vad':quantiles([receipts[r['segment_id']]['after_flush_monotonic']-old_starts[r['speaker_id']]-r['decision_s'] for r in old_voice]),
        'latency_from_truth_end':quantiles(truth_delays),'truth_delay_n':len(truth_delays),'uncovered_truth_ids':uncovered,
        'maximum_input_lag_s':max(r['input_schedule_lag_s'] for r in old)}
    offsets={};duration=0.
    for s in sorted(TEST):offsets[s]=duration;duration+=source_timeline(s)['duration_s']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    for ax,data,label,color in [(axes[0],old,'X06a H3','#9b5252'),(axes[1],new,'X06b H3','#245da6')]:
        x=[(offsets[r['speaker_id']]+r['decision_s'])/60 for r in data];y=[r['input_schedule_lag_s'] for r in data]
        ax.plot(x,y,color=color,linewidth=.8);ax.set_ylabel('Input backlog (s)');ax.set_title(label,loc='left');ax.grid(alpha=.2)
    axes[1].axhline(2,color='#444444',linestyle='--',linewidth=.8,label='2 s threshold');axes[1].legend(loc='upper left');axes[1].set_xlabel('Cumulative media time (min)')
    fig.savefig(out/'backlog.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    ax.plot([r['elapsed_s']/60 for r in memory],[r['rss_bytes']/2**30 for r in memory],color='#245da6',linewidth=.8,label='Process tree RSS')
    baseline=latest['rss_at_10min']/2**30
    ax.axvline(latest['rss_10min_sample_s']/60,color='#444444',linestyle=':',label='10 min baseline sample')
    ax.axhline(baseline*1.1,color='#9b5252',linestyle='--',label='Baseline + 10%')
    ax.set(xlabel='Elapsed replay time (min)',ylabel='RSS (GiB)',title='X06b H3 memory — same process tree through four streams');ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'memory.png');plt.close(fig)
    finish(out,meta,{'status':'completed','plots':{'backlog':spec(out/'backlog.png'),'memory':spec(out/'memory.png')},
        'old_comparison':old_comparison,'sources':{'old_events':old_source,'old_receipts':previous['latency_receipts'],'old_start_inputs':start_sources,'new_latency':new_source,'memory':memory_source},'gpu_seconds':0,
        'notes':['Backlog panels have separate linear y scales; every event plotted, no smoothing',
                 'Memory includes producer RSS at exit; X06a has no matching 10-minute-baseline curve',
                 'Curves are report-only; no parameter selection or fitting']})

if __name__=='__main__':run()
