"""Narrow, evidence-bound authorization for the reviewed reporting failure."""
from pathlib import Path
from emotioncore.x06b import common as c

PLAN = Path('configs/x06b_resume_20261013.json')


def declaration():
    plan=c.read(PLAN)
    c.checked(plan['approval_snapshot'])
    if plan['approval_snapshot']['sha256']!=plan['approval_original_file']['sha256']:
        raise RuntimeError('STOP recovery approval copy differs')
    if plan['gpu_limit_hours']!=14 or not plan['no_gpu_reinference']:
        raise RuntimeError('STOP recovery scope differs')
    if Path(plan['historical_stop']['path'])!=c.ROOT/'STOP.json':
        raise RuntimeError('STOP recovery references another stop')
    for source in plan['preserved']:c.checked(source)
    if c.spec(c.ROOT/'STOP.json')!=plan['historical_stop']:
        raise RuntimeError('STOP original stop changed')
    return plan


def validated_authorization():
    path=c.ROOT/'resume_authorization/summary.json';meta_path=path.parent/'run_meta.json'
    if not path.exists() or not meta_path.exists():
        raise RuntimeError('STOP receipt present without completed recovery authorization')
    plan=declaration();meta=c.read(meta_path)
    if not meta.get('ended_at') or meta['git']['dirty'] or meta['summary']!=c.spec(path):
        raise RuntimeError('STOP incomplete recovery authorization provenance')
    c.checked(meta['summary']);receipt=c.read(path)
    if (receipt['status']!='authorized' or receipt['registration']!=c.spec(PLAN)
            or receipt['historical_stop']!=plan['historical_stop']):
        raise RuntimeError('STOP recovery authorization does not match evidence')
    return receipt


def historical_failure():
    """The sole unfinished run allowed at audit, still explicitly failed."""
    validated_authorization();plan=declaration();failed=c.read(c.checked(plan['failed_run_meta']))
    driver=c.read(c.checked(plan['failed_stage_summary']))
    records=[r for r in driver['stages'] if r['stage']==plan['failed_stage']]
    if (failed.get('ended_at') or failed.get('summary') or failed['git']['dirty']
            or driver['status']!='stopped' or driver['failed_stage']!=plan['failed_stage']
            or len(records)!=1 or records[0]['returncode']==0 or not records[0].get('ended_at')):
        raise RuntimeError('STOP historical failure provenance differs')
    c.checked(records[0]['log'])
    return {'status':'historical_failed','run_meta':plan['failed_run_meta'],
            'driver_summary':plan['failed_stage_summary'],'exit_record':records[0],
            'retry_output':plan['retry_output'],'original_metadata_unchanged':True}


def analysis_task(cohort):
    if cohort=='et_test' and (c.ROOT/'resume_authorization/summary.json').exists():
        validated_authorization()
        return declaration()['retry_output']
    return 'r3_analysis_'+cohort


def authorize():
    from emotioncore.x06b.budget import ledger
    from emotioncore.x06b.resume import active_processes
    c.authority();plan=declaration()
    if active_processes() or list((c.ROOT/'stops').glob('*.json')):
        raise RuntimeError('STOP active process or newer stop prevents recovery')
    if (c.ROOT/plan['retry_output']).exists() or (c.ROOT/'stages_resume1').exists():
        raise RuntimeError('STOP recovery output already exists')
    for source in plan['reused_sources'].values():c.checked(source)
    tests=c.read(c.checked(plan['verified_tests']))
    if tests['returncode']!=0:raise RuntimeError('STOP reporting fix tests failed')
    c.checked(tests['stdout'])
    control_source=c.spec(c.ROOT/'unit_tests_resume_control/summary.json')
    control=c.read(c.checked(control_source));control_meta=c.read(c.ROOT/'unit_tests_resume_control/run_meta.json')
    if (control['returncode']!=0 or not control_meta.get('ended_at') or control_meta['git']['dirty']
            or control_meta['summary']!=control_source
            or control_meta['git']['commit']!=c.git_provenance(Path.cwd())['commit']):
        raise RuntimeError('STOP recovery control requires tests on the same clean commit')
    c.checked(control['stdout'])
    jobs=ledger();used=sum(r['charged_gpu_s'] for r in jobs)/3600
    if (any(r['status']!='completed' for r in jobs) or
            {Path(r['source']['path']).parent.name for r in jobs}!={'r3_fmh','r3_et_test'} or
            abs(used-plan['gpu_used_at_stop'])>1e-12):
        raise RuntimeError('STOP GPU ledger changed since reviewed stop')
    # Validate the incomplete attempt before sealing authorization; never edit it.
    failed=c.read(c.checked(plan['failed_run_meta']))
    if failed.get('ended_at') or failed.get('summary') or failed['git']['dirty']:
        raise RuntimeError('STOP reviewed failure metadata changed')
    out,meta=c.start('resume_authorization')
    c.finish(out,meta,{'status':'authorized','registration':c.spec(PLAN),
        'approval_original_file':plan['approval_original_file'],'approval_snapshot':plan['approval_snapshot'],
        'historical_stop':plan['historical_stop'],'preserved':plan['preserved'],
        'reused_sources':plan['reused_sources'],'retry_output':plan['retry_output'],
        'gpu_hours_before_recovery':used,'gpu_limit_hours':14,'gpu_seconds':0,
        'authorization_scope':plan['scope'],'remaining_order':plan['remaining_order'],
        'recovery_control_tests':control_source,'reporting_fix_tests':plan['verified_tests'],
        'new_stops_remain_blocking':True,'no_gpu_reinference':True})
    historical_failure()


if __name__=='__main__':authorize()
