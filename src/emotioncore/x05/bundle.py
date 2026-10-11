"""Deterministic deployment interface for the frozen X05 parameter bundle."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from emotioncore.eval.metrics import LABELS
from emotioncore.x05.calibration import base_scores,predictions
from emotioncore.x05.gate import route

DEFAULT_PATH=Path(__file__).resolve().parents[3]/'configs/calibration_bundle_v1.json'


def verify_method_freeze(path, expected_sha, require_committed=False):
    path=Path(path);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('STOP method freeze SHA mismatch')
    if require_committed:
        root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
        rel=path.resolve().relative_to(root).as_posix()
        committed=subprocess.check_output(['git','show',f'HEAD:{rel}'])
        if committed!=raw:
            raise ValueError('STOP method freeze must be committed before Q_mix analysis')
    return json.loads(raw)


class Bundle:
    def __init__(self,parameters):
        if parameters['version']!=1 or parameters['labels']!=list(LABELS):
            raise ValueError('Unsupported parameter bundle')
        self.parameters=parameters

    @classmethod
    def load(cls,path=DEFAULT_PATH):
        parameters=json.loads(Path(path).read_text())
        freeze=Path(path).parent/'x05_method_freeze.json'
        verify_method_freeze(freeze,parameters['method_freeze_sha256'])
        return cls(parameters)

    def apply(self,component_probs,availability):
        routing,reason=route(availability['has_face'],availability['face_count'],
                             availability.get('face_quality'),availability['has_speech'])
        if routing=='reject':return 'reject',0.,routing
        full=availability.get('full_system','H_ref')
        if full not in ('H_ref','H3'):raise ValueError('full_system must be H_ref or H3')
        name=full if routing=='full' else routing
        params=self.parameters['systems'][name];inputs=[]
        for component in params['components']:
            p=np.asarray(component_probs[component],dtype=float)
            if p.shape!=(7,) or not np.isfinite(p).all() or (p<0).any() or p.sum()<=0:
                raise ValueError(f'Invalid probabilities: {component}')
            inputs.append(p/p.sum())
        logp=np.log(np.clip(np.asarray(inputs)[None,:,:],1e-12,1))
        _,label,confidence,_=predictions(base_scores(logp,params),params['neutral_bias'])
        mode=availability.get('rejection_point','90')
        if mode not in ('90','80',None):raise ValueError('rejection_point must be 90, 80 or None')
        accepted=mode is None or confidence[0]>=params['thresholds'][mode]
        return LABELS[int(label[0])] if accepted else 'reject',float(confidence[0]),name


def apply(component_probs,availability):
    """Default OP-R90; full system defaults to H_ref. See X06_HANDOFF.md."""
    return Bundle.load().apply(component_probs,availability)
