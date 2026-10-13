"""CPU-only frozen FSMN/0.3/0.4 resegmentation of verified test posteriors."""
import numpy as np
from emotioncore.x06b.common import *
from emotioncore.x06b.prepare import frozen_face
from emotioncore.x06.aishell import reference
from emotioncore.x06.vad import segments,mask_segments,intervals_from_mask,boundary_counts,prf


def run():
    frozen_face();out,meta=start('aishell_test_04')
    sources=read(checked(read(OLD/'aishell_test/summary.json')['sources']))['files']
    frame=np.zeros(3,dtype=np.int64);boundary=frame.copy();count=0;duration=0.;evidence=[]
    for item in sources:
        path=checked(item['posteriors']['fsmn']);annotation=checked(item['rttm'])
        with np.load(path) as archive:
            truth,valid=reference(annotation,len(archive['probability']))
            assert np.array_equal(truth,archive['truth']) and np.array_equal(valid,archive['valid'])
            found=segments(archive['probability'],.3,.4,valid);mask=mask_segments(found,len(truth))
        frame+=[int((mask&truth&valid).sum()),int((mask&~truth&valid).sum()),int((~mask&truth&valid).sum())]
        boundary+=boundary_counts(found,intervals_from_mask(truth&valid));count+=len(found);duration+=sum(r['end_s']-r['start_s'] for r in found)
        evidence.append({'posterior':spec(path),'rttm':spec(annotation)})
    assert len(evidence)==20
    finish(out,meta,{'status':'completed','split':'test','config':{'model':'fsmn','threshold':.3,'min_silence':.4},
        'frame':prf(frame),'boundary':prf(boundary),'mean_segment_s':duration/max(1,count),'segment_count':count,
        'sources':evidence,'gpu_seconds':0,'parameters_selected':False,'reused_frozen_posteriors':True})

if __name__=='__main__':run()
