"""Separate fusion/calibration fits and fixed OP-N/OP-R definitions."""
import math
import numpy as np
from scipy.optimize import minimize
from emotioncore.x03.fusion import fit_fusion, fusion_logits, fit_neutral_bias, softmax, nll

METHODS = ('temperature', 'temperature_bias', 'class_temperature_bias')
SYSTEMS = {'H_ref': ('Q_A','Q_AV','A3','V2','V1'), 'H3': ('Q_AV','A3','V2'),
           'H_A': ('Q_A','A3'), 'H_V': ('Q_V','V2','V1'),
           'Q_A': ('Q_A',), 'Q_V': ('Q_V',), 'A3': ('A3',), 'V2': ('V2',)}


def calibrate(logits, parameters):
    return logits / np.exp(np.asarray(parameters['log_temperature'])) + np.asarray(parameters['bias'])


def fit_calibrator(logits, y, method):
    if method not in METHODS or not len(y):
        raise ValueError('Invalid calibration method or empty fit')
    nt = 7 if method == METHODS[2] else 1
    nb = 0 if method == METHODS[0] else 7
    def objective(theta):
        scaled = logits / np.exp(theta[:nt])
        bias = theta[nt:] - theta[nt:].mean() if nb else np.zeros(7)
        scores = scaled + bias
        p = softmax(scores)
        g = p.copy(); g[np.arange(len(y)), y] -= 1; g /= len(y)
        gt = (-g*scaled).sum(0) if nt==7 else np.array([(-g*scaled).sum()])
        gb = g.sum(0); gb -= gb.mean()
        gradient = np.r_[gt, gb] if nb else gt
        return nll(scores,y)+1e-6*float(theta@theta), gradient+2e-6*theta
    result = minimize(objective,np.zeros(nt+nb),jac=True,method='L-BFGS-B',
                      bounds=[(-4,4)]*nt+[(None,None)]*nb,
                      options={'maxiter':400,'ftol':1e-10})
    if not result.success and np.linalg.norm(result.jac,np.inf)>1e-4:
        raise RuntimeError(f'Calibration optimization failed: {result.message}')
    return {'method':method,'log_temperature':result.x[:nt].tolist(),
            'bias':(result.x[nt:]-result.x[nt:].mean()).tolist() if nb else [0.]*7}


def fit_base(logp, y, method):
    # Single-expert controls follow X03: no fusion weight/bias is fitted.
    fusion = fit_fusion(logp,y).tolist() if logp.shape[1]>1 else None
    scores = fusion_logits(logp,np.asarray(fusion)) if fusion is not None else logp[:,0]
    return {'fusion':fusion,'calibration':fit_calibrator(scores,y,method)}


def base_scores(logp, parameters):
    scores = (fusion_logits(logp,np.asarray(parameters['fusion']))
              if parameters['fusion'] is not None else logp[:,0])
    return calibrate(scores,parameters['calibration'])


def predictions(scores, neutral_bias):
    p = softmax(scores)
    decisions = scores.copy(); decisions[:,0] += neutral_bias
    labels = decisions.argmax(1)
    confidence = p[np.arange(len(p)),labels]
    sorted_p = np.sort(p,axis=1)
    return p, labels, confidence, sorted_p[:,-1]-sorted_p[:,-2]


def coverage_threshold(confidence, coverage):
    if not len(confidence) or coverage not in (.9,.8):
        raise ValueError('Only OP-R90 and OP-R80 are registered')
    # Accept >= threshold. Include all ties; report empirical coverage.
    return float(np.sort(confidence)[::-1][math.ceil(coverage*len(confidence))-1])


def fit_operating(scores, y):
    neutral = fit_neutral_bias(scores,y,.15)
    _,_,confidence,_ = predictions(scores,neutral)
    return {'neutral_bias':neutral,
            'thresholds':{str(int(c*100)):coverage_threshold(confidence,c) for c in (.9,.8)},
            'neutral_curve_biases':{str(t):fit_neutral_bias(scores,y,t) for t in (.05,.10,.20,.25)}}


def select_method(scores):
    best = min(scores.values())
    return next(method for method in METHODS if scores[method] <= best+.002)


def fit_held_fold(logp, labels_for_indices, folds, held, method):
    """The fitting API never receives the held fold's labels."""
    if sorted(set(folds))!=list(range(5)):
        raise ValueError('STOP expected all five OOF folds')
    fitting=np.flatnonzero(folds!=held); evaluation=np.flatnonzero(folds==held)
    if len(set(folds[fitting]))!=4 or not len(evaluation):
        raise ValueError('STOP four fitting folds required')
    y=np.asarray(labels_for_indices(fitting))
    parameters=fit_base(logp[fitting],y,method)
    parameters.update(fit_operating(base_scores(logp[fitting],parameters),y))
    return evaluation, base_scores(logp[evaluation],parameters), parameters, fitting


def fit_held_speaker(oof_logp,oof_y,fmh_logp,labels_for_indices,speakers,held,method):
    """FMH held-speaker labels cannot be accessed through this fitting API."""
    fitting=np.flatnonzero(speakers!=held); evaluation=np.flatnonzero(speakers==held)
    if len(set(speakers[fitting]))!=2 or not len(evaluation):
        raise ValueError('STOP expected two FMH fitting speakers')
    y=np.asarray(labels_for_indices(fitting))
    parameters=fit_base(np.concatenate((oof_logp,fmh_logp[fitting])),
                        np.concatenate((oof_y,y)),method)
    parameters.update(fit_operating(base_scores(fmh_logp[fitting],parameters),y))
    return evaluation,base_scores(fmh_logp[evaluation],parameters),parameters,fitting
