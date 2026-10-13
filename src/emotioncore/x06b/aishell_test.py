"""CPU-only frozen FSMN/0.3/0.4 resegmentation of verified test posteriors."""
import numpy as np
from emotioncore.x06b.common import *
from emotioncore.x06b.prepare import frozen_face
from emotioncore.x06.aishell import reference
from emotioncore.x06.vad import segments,mask_segments,intervals_from_mask,boundary_counts,prf


def posterior_source(item,root):
    if 'posteriors' in item:return checked(item['posteriors']['fsmn'])
    # The original test runner did not put posterior hashes in sources.json.
    # Use its exact documented filename, record its hash now, and reproduce
    # the previously published test metrics before accepting this cache.
    return root/'posteriors'/(Path(item['rttm']['path']).stem+'_fsmn.npz')


def run():
    frozen_face();out,meta=start('aishell_test_04')
    previous_source=read('experiments/X06a_stream/results/vad_test.json')['source']
    previous=read(checked(previous_source));sources=read(checked(previous['sources']))['files']
    reference_result=previous['results'][0];config=reference_result['config']
    assert config['model']=='fsmn' and config['threshold']==.3
    frame=np.zeros(3,dtype=np.int64);boundary=frame.copy();count=0;duration=0.;evidence=[]
    old_frame=frame.copy();old_boundary=frame.copy();old_count=0
    for item in sources:
        path=posterior_source(item,OLD/'aishell_test');annotation=checked(item['rttm'])
        posterior_receipt=spec(path)
        with np.load(path) as archive:
            truth,valid=reference(annotation,len(archive['probability']))
            assert np.array_equal(truth,archive['truth']) and np.array_equal(valid,archive['valid'])
            found=segments(archive['probability'],.3,.4,valid);mask=mask_segments(found,len(truth))
            historical=segments(archive['probability'],config['threshold'],config['min_silence'],valid)
            old_mask=mask_segments(historical,len(truth))
        frame+=[int((mask&truth&valid).sum()),int((mask&~truth&valid).sum()),int((~mask&truth&valid).sum())]
        boundary+=boundary_counts(found,intervals_from_mask(truth&valid));count+=len(found);duration+=sum(r['end_s']-r['start_s'] for r in found)
        old_frame+=[int((old_mask&truth&valid).sum()),int((old_mask&~truth&valid).sum()),int((~old_mask&truth&valid).sum())]
        old_boundary+=boundary_counts(historical,intervals_from_mask(truth&valid));old_count+=len(historical)
        checked(posterior_receipt);evidence.append({'posterior':posterior_receipt,'rttm':spec(annotation)})
    assert len(evidence)==20
    assert prf(old_frame)==reference_result['frame'] and prf(old_boundary)==reference_result['boundary']
    assert old_count==reference_result['segment_count'], 'STOP historical posterior metric mismatch'
    finish(out,meta,{'status':'completed','split':'test','config':{'model':'fsmn','threshold':.3,'min_silence':.4},
        'frame':prf(frame),'boundary':prf(boundary),'mean_segment_s':duration/max(1,count),'segment_count':count,
        'sources':evidence,'gpu_seconds':0,'parameters_selected':False,'reused_frozen_posteriors':True,
        'historical_metrics_reproduced':True,'historical_summary':previous_source,
        'legacy_cache_hashes_recorded_at_read':True})

if __name__=='__main__':run()
