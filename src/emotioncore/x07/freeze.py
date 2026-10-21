"""Write the L1 protocol once, only after the complete code has been committed."""
from emotioncore.x07.common import *

def run():
    if git('status','--porcelain','--untracked-files=normal'):
        raise RuntimeError('Commit code and preflight evidence before creating protocol')
    inventory=read('experiments/X07_lockbox/results/asset_inventory.json')
    bundle=read('configs/calibration_bundle_v1.1.json')
    provenance=read('experiments/X07_lockbox/results/training_provenance.json')
    for epoch,rec in enumerate(provenance['Q_run']['data']['epochs']):
        source=inventory['leakage_sources'][f'Q_epoch{epoch}']
        if source['sha256']!=rec['sha256'] or source['path']!=rec['path']:
            raise RuntimeError('STOP Q training manifest provenance mismatch')
    if len(provenance['Q_run']['data']['epochs'])!=2:
        raise RuntimeError('STOP Q must have two final epochs')
    if inventory['leakage_sources']['calibration']['sha256']!=bundle['fitting_set']['sha256']:
        raise RuntimeError('STOP calibration fitting provenance mismatch')
    for name in ('A3','V2','V1'):
        if inventory['assets'][name+'_head'][0]['sha256']!=bundle['components'][name]['sha256']:
            raise RuntimeError('STOP expert SHA does not match accepted bundle')
        if (provenance[name]['data']['training_ids']['sha256']!=inventory['leakage_sources']['D1_'+name]['sha256']
                or provenance[name]['data']['checkpoint']['sha256']!=bundle['components'][name]['sha256']):
            raise RuntimeError('STOP expert training ID provenance mismatch')
    adapter=next(p for p in inventory['assets']['Q_adapter'] if p['path'].endswith('/adapter_model.safetensors'))
    if adapter['sha256']!=bundle['components']['Q_AV']['sha256']:
        raise RuntimeError('STOP Q adapter SHA mismatch')
    configs=['calibration_bundle_v1.json','calibration_bundle_v1.1.json','x05_method_freeze.json',
             'x06_vad_freeze.json','x06_silence_freeze.json','x06_smoothing_freeze.json','x06b_face_freeze.json']
    code=git('rev-parse','HEAD')
    files=[p for p in git('ls-files','src','tests').splitlines() if p.endswith('.py')]
    protocol={
        'version':1,'registered_at':now(),'code_commit':code,
        'step0':{'pr':8,'head':'16d5244447d9c4f07ba4c582dfdd686df9d58c48',
                 'merge_commit':'5e947e9bcc8c1e99eba02e225e37db6cbcb9a4e5','x07_pr_must_remain_unmerged':True},
        'code_sha256':{p:sha(p) for p in files},
        'configs':{p:spec('configs/'+p) for p in configs},
        'inventory':inventory,
        'training_provenance':provenance,
        'system':{'primary':'H3','secondary':['H_A','H_V'],'Q':'Q_mix best_adapter_eval8',
            'experts':'D1_fmhx','bundle':'v1.1','frame_hz':5,'yunet_scale':.5,'detector_instances':8,
            'max_pending_frames':16,'crop':'unchanged X06b original-resolution rule',
            'segmentation':'whole clip, exactly one utterance; VAD is NOT executed','smoothing_alpha':0,
            'vad_registered_only':{'model':'FSMN','threshold':.3,'min_silence_s':.4},
            'q_forwards':'one Q-A, one Q-AV, one Q-V for a usable face clip; no-face reuses Q-A and HV is unavailable',
            'no_training_fitting_or_parameter_selection':True,
            'audio':'ffmpeg pcm_f32le, mono, 16000 Hz',
            'H3_gate':'frozen X06b H3, no-face -> HA, ambiguous multi-face -> reject at R90',
            'HA_gate':'audio only; no face/multi-face filtering',
            'HV_gate':'face only, no-face unavailable, multi-face reject; report existing R90 threshold',
            'HV_R80':'unchanged deployment setting, not used for the requested R90 secondary comparison'},
        'metrics':{
            'score':'calibrated pre-neutral-bias p(joy) - p(anger)-p(disgust)-p(fear)-p(sadness)',
            'labels':['neutral','joy','sadness','anger','fear','disgust','surprise'],
            'primary':'H3 pearson_available',
            'pearson_available':'Pearson on available media, including gated rejection and H3 no-face -> HA fallback',
            'spearman_available':'Pearson of average-tie ranks on same available media',
            'acc2':'all nonzero ground-truth labels as denominator; unavailable or score=0 is incorrect; otherwise compare signs (X01)',
            'pearson_all_uniform':'all 1034 test / 647 valid IDs; unavailable media probabilities uniform, score -3/7',
            'op_r90_coverage':'accepted by fixed per-system R90 threshold and modality/multi-face gates / all IDs',
            'pearson_accepted':'Pearson restricted to R90 accepted IDs',
            'neutral_non_neutral_rate':'available prediction with OP-N label != neutral / all ground-truth label=0 IDs',
            'all_systems':['H3','H_A','H_V'],
            'unavailable':'missing file, missing audio, ffprobe/audio/native-frame decode failure -> all systems uniform/unavailable; no-face -> HV uniform/unavailable only',
            'degenerate':'null when correlation has <2 samples or zero variance; never substitute zero'},
        'bootstrap':{'unit':'video ID before __','repetitions':2000,'seed':20261014,
            'rng':'numpy.random.default_rng','sampling':'draw N videos from N unique videos with replacement; retain every clip and multiplicity; then metric-specific filters',
            'interval':'two-sided 95% percentile [2.5,97.5]','degenerate':'omit undefined replicate for that metric and report valid count','shared_draws':True},
        'gates':{'L2':'all seven source assertions and all three suspected MER exclusions must pass',
            'L3_H3_pearson_available_min':.78,
            'reviewer_acceptance':{'H3_pearson_available_min':.75,'H3_video_ci95_lower_min':.70,'H3_acc2_min':.80},
            'root_free_bytes_min':80_000_000_000,'gpu_seconds_max':14400,'gpu_termination_reserve_seconds':12},
        'execution':{'order':['L1 commit','L2 leakage','L3 valid inference','L3 metrics once','L3 gate','L4 test inference','L4 metrics once'],
            'labels':'meta usecols video_id/clip_id/mode only until the corresponding complete.json; label column opened inside exclusive metrics lock',
            'metric_lock':'O_EXCL before label read; no second attempt, overwrite, rerun or modification after metrics',
            'resume':'only documented infrastructure failure, same code and config, skip every persisted completed clip',
            'budget':'sum of complete GPU child wall times including loading and waiting; CPU preparation excluded',
            'storage_gate_scope':'WSL execution host root /; all new media and outputs on /mnt/h'},
        'comparison':{'valid':'same code and protocol, all same metrics','et_test':'cite X06b R3 existing 7-class results, never rerun',
            'et_source':spec('experiments/X06b_realtime/results/r3_analysis_et_test_retry1.json'),
            'fmh_source':spec('experiments/X06b_realtime/results/r3_analysis_fmh.json')},
        'interpretation':spec('experiments/X07_lockbox/OPEN_QUESTIONS.md'),
        'instructions':spec('experiments/X07_lockbox/protocol/X07_PROMPT.md')}
    write_new(PROTOCOL,protocol)
    print(json.dumps({'protocol':spec(PROTOCOL),'code_commit':code}))

if __name__=='__main__':
    run()
