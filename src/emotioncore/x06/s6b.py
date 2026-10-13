"""Paired, utterance-bootstrap robustness; frozen bundle and no modality gates."""
import argparse
import numpy as np
from emotioncore.eval.metrics import LABELS
from emotioncore.x05.calibration import base_scores, predictions
from emotioncore.x05.data import COMPONENTS
from emotioncore.x05.deployment import load_cohort, target_maps, target_inputs
from emotioncore.x06.common import *
from emotioncore.x06.revision2 import REV, checked, s6b_gate
from emotioncore.x06.analysis import truths


def frozen_decisions(probabilities, parameters):
    matrix = np.stack([probabilities[c] for c in parameters['components']], axis=1)
    p, label, confidence, _ = predictions(base_scores(np.log(np.clip(matrix, 1e-12, 1)), parameters),
                                           parameters['neutral_bias'])
    r90 = np.where(confidence >= parameters['thresholds']['90'], label, -1)
    return {'op_n': label, 'r90': r90, 'confidence': confidence}


def macro_f1(truth, prediction):
    cells = np.bincount(truth * 8 + np.where(prediction < 0, 7, prediction), minlength=56).reshape(7, 8)
    tp = np.diag(cells[:, :7]); denominator = cells.sum(1) + cells[:, :7].sum(0)
    return float(np.mean(np.divide(2 * tp, denominator, out=np.zeros(7), where=denominator > 0)))


def paired(truth, old, new, iterations=2000):
    if not len(truth) or len(truth) != len(old['op_n']) or len(truth) != len(new['op_n']):
        raise ValueError('STOP unequal paired populations')
    rng = np.random.default_rng(SEED); draws = {'op_n': [], 'r90': [], 'coverage': []}
    for _ in range(iterations):
        ix = rng.integers(0, len(truth), len(truth))
        for metric in ('op_n', 'r90'):
            draws[metric].append(macro_f1(truth[ix], new[metric][ix]) - macro_f1(truth[ix], old[metric][ix]))
        draws['coverage'].append(float(np.mean(new['r90'][ix] >= 0) - np.mean(old['r90'][ix] >= 0)))
    result = {}
    for metric in ('op_n', 'r90'):
        a, b = macro_f1(truth, old[metric]), macro_f1(truth, new[metric])
        result[metric] = {'offline_macro_f1': a, 'stream_macro_f1': b, 'delta': b - a,
                          'ci95': np.percentile(draws[metric], [2.5, 97.5]).tolist(),
                          'label_agreement': float(np.mean(old[metric] == new[metric]))}
    a, b = float(np.mean(old['r90'] >= 0)), float(np.mean(new['r90'] >= 0))
    result['coverage'] = {'offline': a, 'stream': b, 'delta': b - a,
                          'ci95': np.percentile(draws['coverage'], [2.5, 97.5]).tolist()}
    return result


def run(cohort):
    output = REV / ('s6b_' + cohort); meta = start(output)
    if cohort == 'fmh':
        old = load_cohort('fmh', 'mix'); ids = old['ids']; y = old['y']; sources = old['sources']
        baseline = {c: old['p'][:, i] for i, c in enumerate(COMPONENTS)}
        source = read(ROOT / 'fmh_stream_oracle/summary.json')['predictions']
        speakers = FMH
    else:
        sources = {}; ids, maps = target_maps('et_test', sources)
        logp, bad = target_inputs(ids, maps, COMPONENTS)
        baseline = {c: np.exp(logp[:, i]) for i, c in enumerate(COMPONENTS)}
        y = np.asarray([LABELS.index(maps['Q_A'][uid]['truth_M']) for uid in ids])
        source = read(REV / 'et_remaining/summary.json')['predictions']; speakers = TEST
    fresh = {r['utt_id']: r for r in rows(checked(source))}
    if len(ids) != (3266 if cohort == 'fmh' else 1929) or set(fresh) != set(ids):
        raise ValueError('STOP S6b population mismatch')
    truth_map = {r['utt_id']: (s, r['label']) for s in speakers for r in truths(s)}
    if set(truth_map) != set(ids) or any(fresh[uid]['speaker_id'] != truth_map[uid][0]
            or LABELS[y[i]] != truth_map[uid][1] for i, uid in enumerate(ids)):
        raise ValueError('STOP S6b identity/label pairing mismatch')
    names = set().union(*(r['component_probs'] for r in fresh.values()))
    stream = {c: np.asarray([fresh[uid]['component_probs'][c] for uid in ids]) for c in names}
    bundle = read('configs/calibration_bundle_v1.json'); reports = {}; paired_rows = []
    for system in ('H3', 'H_ref'):
        params = bundle['systems'][system]
        before, after = frozen_decisions(baseline, params), frozen_decisions(stream, params)
        reports[system] = paired(y, before, after)
        reports[system]['one_component_replaced'] = {}
        for component in params['components']:
            replaced = frozen_decisions({**baseline, component: stream[component]}, params)
            reports[system]['one_component_replaced'][component] = {
                metric: {'agreement_with_offline': float(np.mean(replaced[metric] == before[metric])),
                         'macro_f1_delta': macro_f1(y, replaced[metric]) - macro_f1(y, before[metric])}
                for metric in ('op_n', 'r90')}
        for i, uid in enumerate(ids):
            paired_rows.append({'utt_id': uid, 'speaker_id': truth_map[uid][0], 'system': system,
                'truth': int(y[i]), 'offline_op_n': int(before['op_n'][i]), 'stream_op_n': int(after['op_n'][i]),
                'offline_r90': int(before['r90'][i]), 'stream_r90': int(after['r90'][i]),
                'offline_confidence': float(before['confidence'][i]), 'stream_confidence': float(after['confidence'][i])})
    write_rows(output / 'paired.jsonl', paired_rows)
    summary = {'cohort': cohort, 'n': len(ids), 'systems': reports, 'bootstrap': {
        'unit': 'paired utterance', 'iterations': 2000, 'seed': SEED, 'interval': '95% percentile'},
        'multiface_and_modality_gates': False, 'parameters_refitted': False, 'gpu_seconds': 0,
        'predictions': source, 'offline_sources': sources, 'paired': spec(output / 'paired.jsonl'),
        'h3_fmh_ci_acceptance': reports['H3']['op_n']['ci95'][0] >= -.02 if cohort == 'fmh' else None,
        'status': 'STOP_s6b_point' if cohort == 'fmh' and reports['H3']['op_n']['delta'] < -.05 else 'completed'}
    finish(output, meta, summary)
    if cohort == 'fmh':
        s6b_gate(summary)
    print(json.dumps({'cohort': cohort, 'n': len(ids), 'status': summary['status'], 'systems': reports}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('cohort', choices=('fmh', 'et_test'))
    run(parser.parse_args().cohort)
