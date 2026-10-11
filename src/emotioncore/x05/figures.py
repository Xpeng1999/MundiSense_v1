"""Static, source-backed calibration and selective prediction figures."""
import os
os.environ.setdefault('MPLCONFIGDIR','/mnt/h/emotionAI/emotioncore_runs/X05/matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from emotioncore.eval.metrics import LABELS
from emotioncore.x05.common import ROOT,read,spec,start,finish

NAMES=('H_ref','H3','H_A','H_V')
COLORS=('#0072B2','#D55E00','#009E73','#CC79A7')


def save(fig,output,name):
    files=[]
    for suffix in ('png','svg'):
        path=output/f'{name}.{suffix}';fig.savefig(path,dpi=220,facecolor='white',bbox_inches='tight')
        files.append(spec(path))
    plt.close(fig)
    return files


def plot_cohort(metrics,output,prefix,title):
    files=[]
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for ax,name in zip(axes.flat,NAMES):
        result=metrics[name]
        ax.plot([0,1],[0,1],color='.65',linestyle=':',linewidth=1)
        for key,color,marker,label in [('op_a_top',COLORS[0],'o','OP-A top-label'),
                                     ('op_n_selected',COLORS[1],'s','OP-N selected-label')]:
            bins=[b for b in result['reliability'][key] if b['n']]
            ax.plot([b['confidence'] for b in bins],[b['accuracy'] for b in bins],
                    marker=marker,color=color,label=label,linewidth=1.5,markersize=4)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean calibrated probability',ylabel='Observed correctness',
            title=f"{name} | selected ECE {result['selected_label_ece']:.3f}")
        ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle(f'{title}: reliability (15 equal-width bins)')
    files+=save(fig,output,prefix+'_reliability')
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for ax,name,color in zip(axes.flat,NAMES,COLORS):
        result=metrics[name];curve=result['selective']['curve']
        ax.plot(curve['coverage'],curve['risk'],color=color,label='Selected-label confidence')
        if 'margin_control' in result:
            curve=result['margin_control']['curve']
            ax.plot(curve['coverage'],curve['risk'],color='.4',linestyle='--',label='Top-two margin (control)')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Coverage',ylabel='Error among accepted samples',
               title=f"{name} | AURC {result['selective']['aurc']:.3f}")
        ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle(f'{title}: risk vs coverage')
    files+=save(fig,output,prefix+'_risk_coverage')
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for ax,name,color in zip(axes.flat,NAMES,COLORS):
        points=metrics[name]['selective']['macro_f1_curve']
        for key,style,label in [('accepted_macro_f1','-','Accepted samples only'),
                                ('reject_in_recall_macro_f1','--','Rejects retained in recall denominator')]:
            ax.plot([p['coverage'] for p in points],[p[key] for p in points],color=color,linestyle=style,label=label)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Coverage',ylabel='Macro-F1',title=name)
        ax.grid(alpha=.18);ax.legend(fontsize=7)
    fig.suptitle(f'{title}: macro-F1 vs coverage')
    files+=save(fig,output,prefix+'_f1_coverage')
    return files


def run():
    output=ROOT/'figures';meta=start(output)
    methods=read(ROOT/'methods/summary.json')['selected_metrics']
    for name,curves in read(ROOT/'diagnostics/summary.json')['curves'].items():
        methods[name].update(curves)
    deployment=read(ROOT/'deployment/summary.json')['fmh_loso']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none'})
    files=plot_cohort(methods,output,'q_et_oof','Q_et + original D1, leave-one-fold-out')
    files+=plot_cohort({n:deployment[n]['all'] for n in NAMES},output,'q_mix_fmh','Q_mix + D1_fmhx, FMH leave-one-speaker-out')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,color in zip(NAMES,COLORS):
        result=methods[name]
        points={float(k):v for k,v in result['neutral_curve'].items()}
        points[.15]=result;x=sorted(points)
        for ax,key in zip(axes,('macro_f1','nonneutral_macro_recall')):
            ax.plot(x,[points[t][key] for t in x],marker='o',color=color,label=name)
            ax.set(xlabel='Fitting-set neutral false-positive target',ylabel=key.replace('_',' '),ylim=(0,1))
            ax.grid(alpha=.18);ax.legend()
    fig.suptitle('Q_et OOF: fixed neutral operating-point sensitivity (report only)')
    files+=save(fig,output,'neutral_sensitivity')
    fig,ax=plt.subplots(figsize=(10,3.5),layout='constrained')
    data=np.asarray([[methods[n]['classwise_ece'][label] for label in LABELS] for n in NAMES])
    im=ax.imshow(data,cmap='viridis',vmin=0,vmax=max(.03,float(data.max())))
    ax.set(xticks=range(7),xticklabels=LABELS,yticks=range(4),yticklabels=NAMES,
           title='Q_et OOF: per-class ECE (15 equal-width bins)')
    for i in range(4):
        for j in range(7):ax.text(j,i,f'{data[i,j]:.3f}',ha='center',va='center',
                                   color='white' if data[i,j]<im.norm.vmax*.5 else 'black')
    fig.colorbar(im,ax=ax,label='ECE',shrink=.8)
    files+=save(fig,output,'classwise_ece')
    finish(output,meta,{'files':files,'methods_source':spec(ROOT/'methods/summary.json'),
                        'deployment_source':spec(ROOT/'deployment/summary.json')})


if __name__=='__main__':run()
