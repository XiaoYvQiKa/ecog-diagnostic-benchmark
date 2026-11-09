"""
Train-internal hyperparameter selection for the traditional classification methods.

Rationale
---------
The original ablation (run_ablations.py) selected the CSP frequency band, the
window length, and the LightGBM-vs-SVM choice by evaluating each candidate on
the official TEST segment and picking the best mean macro-F1. That constitutes
test-set-informed hyperparameter selection.

This script moves that selection onto a train-internal validation split:
  - the 400 s training segment is split temporally into
    train (first 3/5 = 240 s) and validation (last 2/5 = 160 s),
    matching the Wiener-filter internal split (3/5 + 2/5) used on the
    regression side;
  - each candidate configuration is fit on the train split and scored on the
    validation split (mean macro-F1 across subjects);
  - the winning configuration is then re-fit on the FULL 400 s training segment
    and evaluated once on the official 200 s test segment, which yields the
    numbers reported in the main comparison.

Electrode quality control and per-channel z-score are unchanged from the main
pipeline (load_subject): QC uses raw amplitude/kurtosis and the test/train RMS
ratio for broken-electrode removal, and z-scores are fit on the full training
segment and applied to train/val/test identically. The train/val split is made
on the raw signal BEFORE windowing, so no window straddles the split boundary.

Usage:
    python run_hyperparam_selection_val.py
"""
import dataclasses
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.config import BenchmarkConfig
from common.data_loader import (
    download_dataset,
    generate_labels,
    load_subject,
    window_majority_labels,
)
from common.features import extract_spectral_features, extract_csp_features
from common.models import CSPLDA, LightGBMClassifier, SpectralSVM
from common.evaluation import evaluate_classification

SUBJECTS = ['sub1', 'sub2', 'sub3']
VAL_RATIO = 0.4  # last 40% of training held out for validation (3/5 + 2/5)

CSP_BANDS = {
    'allgamma (8-200)':  (8,   200),
    'widegamma (30-200)': (30, 200),
    'highgamma (60-200)': (60, 200),
    'optimal (65-175)':  (65,  175),
}
WINDOW_LENGTHS_MS = [250, 500, 1000]

SEP = '=' * 74


def split_train_val(train_ecog, train_dg, val_ratio=VAL_RATIO):
    """Temporal split of the 400 s training segment into train / validation.

    train_dg is already downsampled to 25 Hz by load_subject, so both arrays
    are split at the same relative boundary.
    """
    n_eco = train_ecog.shape[1]
    n_dg = train_dg.shape[0]
    n_val_eco = int(n_eco * val_ratio)
    n_val_dg = int(n_dg * val_ratio)
    tr_ecog = train_ecog[:, :-n_val_eco]
    val_ecog = train_ecog[:, -n_val_eco:]
    tr_dg = train_dg[:-n_val_dg]
    val_dg = train_dg[-n_val_dg:]
    return tr_ecog, tr_dg, val_ecog, val_dg


def _fit_eval_spectral(ecog_tr, labels_tr, ecog_ev, labels_ev, feat_cfg, model):
    """Fit + evaluate a spectral-feature classifier (LGB or SVM)."""
    X_tr, _ = extract_spectral_features(ecog_tr, feat_cfg)
    X_ev, _ = extract_spectral_features(ecog_ev, feat_cfg)
    maj_tr = window_majority_labels(labels_tr, feat_cfg.window_samples, feat_cfg.step_samples)
    maj_ev = window_majority_labels(labels_ev, feat_cfg.window_samples, feat_cfg.step_samples)
    y_tr = maj_tr[:len(X_tr)]
    y_ev = maj_ev[:len(X_ev)]
    model.fit(X_tr, y_tr)
    yp = model.predict(X_ev)
    ypb = model.predict_proba(X_ev)
    return evaluate_classification(y_ev, yp, y_proba=ypb)


def _fit_eval_csp(ecog_tr, labels_tr, ecog_ev, labels_ev, csp_cfg, feat_cfg):
    """Fit + evaluate CSP+LDA."""
    csp_win = int(csp_cfg.window_ms * feat_cfg.fs / 1000)
    csp_step = int(csp_cfg.step_ms * feat_cfg.fs / 1000)
    maj_tr = window_majority_labels(labels_tr, csp_win, csp_step)
    maj_ev = window_majority_labels(labels_ev, csp_win, csp_step)
    X_tr, _, csp_f = extract_csp_features(ecog_tr, maj_tr, csp_cfg, feat_cfg, csp_filters=None)
    y_tr = maj_tr[:len(X_tr)]
    X_ev, _, _ = extract_csp_features(ecog_ev, maj_ev, csp_cfg, feat_cfg, csp_filters=csp_f)
    y_ev = maj_ev[:len(X_ev)]
    m = CSPLDA(csp_cfg)
    m.fit(X_tr, y_tr)
    yp = m.predict(X_ev)
    ypb = m.predict_proba(X_ev)
    return evaluate_classification(y_ev, yp, y_proba=ypb)


def _mean_val_f1(results_by_subj, metric='macro_f1'):
    return float(np.mean([results_by_subj[s][metric] for s in SUBJECTS]))


def main():
    download_dataset()
    base = BenchmarkConfig()
    t0 = time.time()

    # ── Pre-load and pre-split all subjects ──
    data = {}
    for s in SUBJECTS:
        d = load_subject(s)
        tr_ecog, tr_dg, val_ecog, val_dg = split_train_val(d['train_ecog'], d['train_dg'])
        tr_labels = generate_labels(tr_dg)
        tr_max = tr_dg.max(axis=0)
        val_labels = generate_labels(val_dg, max_angles=tr_max)
        # test labels use the FULL training maxima (identical to main pipeline)
        test_labels = generate_labels(d['test_dg'], max_angles=d['train_dg'].max(axis=0))
        data[s] = dict(
            tr_ecog=tr_ecog, tr_dg=tr_dg, tr_labels=tr_labels, tr_max=tr_max,
            val_ecog=val_ecog, val_dg=val_dg, val_labels=val_labels,
            full_ecog=d['train_ecog'], full_dg=d['train_dg'],
            test_ecog=d['test_ecog'], test_dg=d['test_dg'], test_labels=test_labels,
        )

    report = {}

    # ── 1. CSP frequency band selection on VALIDATION ──
    print('\n' + SEP)
    print('  CSP FREQUENCY BAND (selected on train-internal validation)')
    print(SEP)
    band_val = {}
    for name, band in CSP_BANDS.items():
        csp_cfg = dataclasses.replace(base.csp, freq_band=band, n_components=3)
        band_val[name] = {}
        for s in SUBJECTS:
            r = _fit_eval_csp(data[s]['tr_ecog'], data[s]['tr_labels'],
                              data[s]['val_ecog'], data[s]['val_labels'],
                              csp_cfg, base.features)
            band_val[name][s] = r
        mf1 = _mean_val_f1(band_val[name])
        print(f'  {name:<22} val F1: ' + '  '.join(
            f"{band_val[name][s]['macro_f1']:.3f}" for s in SUBJECTS) + f'  mean={mf1:.3f}')
    best_band = max(band_val, key=lambda k: _mean_val_f1(band_val[k]))
    print(f'  >>> selected band: {best_band}  (val F1={_mean_val_f1(band_val[best_band]):.3f})')
    report['csp_band'] = {'val_results': {k: {s: v['macro_f1'] for s, v in vv.items()}
                                          for k, vv in band_val.items()},
                          'selected': best_band}
    best_band_hz = CSP_BANDS[best_band]

    # ── 2. CSP window length selection on VALIDATION ──
    print('\n' + SEP)
    print('  CSP WINDOW LENGTH (selected on train-internal validation)')
    print(SEP)
    csp_win_val = {}
    for w in WINDOW_LENGTHS_MS:
        csp_cfg = dataclasses.replace(base.csp, freq_band=best_band_hz, window_ms=w, step_ms=40)
        csp_win_val[w] = {}
        for s in SUBJECTS:
            r = _fit_eval_csp(data[s]['tr_ecog'], data[s]['tr_labels'],
                              data[s]['val_ecog'], data[s]['val_labels'],
                              csp_cfg, base.features)
            csp_win_val[w][s] = r
        mf1 = _mean_val_f1(csp_win_val[w])
        print(f'  {w} ms  val F1: ' + '  '.join(
            f"{csp_win_val[w][s]['macro_f1']:.3f}" for s in SUBJECTS) + f'  mean={mf1:.3f}')
    best_csp_win = max(csp_win_val, key=lambda k: _mean_val_f1(csp_win_val[k]))
    print(f'  >>> selected CSP window: {best_csp_win} ms  (val F1={_mean_val_f1(csp_win_val[best_csp_win]):.3f})')
    report['csp_window'] = {'val_results': {w: {s: v['macro_f1'] for s, v in vv.items()}
                                            for w, vv in csp_win_val.items()},
                            'selected': best_csp_win}

    # ── 3. LGB window length selection on VALIDATION ──
    print('\n' + SEP)
    print('  LGB WINDOW LENGTH (selected on train-internal validation)')
    print(SEP)
    lgb_win_val = {}
    for w in WINDOW_LENGTHS_MS:
        feat_cfg = dataclasses.replace(base.features, window_length_ms=w, step_ms=40)
        lgb_win_val[w] = {}
        for s in SUBJECTS:
            r = _fit_eval_spectral(data[s]['tr_ecog'], data[s]['tr_labels'],
                                   data[s]['val_ecog'], data[s]['val_labels'],
                                   feat_cfg, LightGBMClassifier(base.lgb))
            lgb_win_val[w][s] = r
        mf1 = _mean_val_f1(lgb_win_val[w])
        print(f'  {w} ms  val F1: ' + '  '.join(
            f"{lgb_win_val[w][s]['macro_f1']:.3f}" for s in SUBJECTS) + f'  mean={mf1:.3f}')
    best_lgb_win = max(lgb_win_val, key=lambda k: _mean_val_f1(lgb_win_val[k]))
    print(f'  >>> selected LGB window: {best_lgb_win} ms  (val F1={_mean_val_f1(lgb_win_val[best_lgb_win]):.3f})')
    report['lgb_window'] = {'val_results': {w: {s: v['macro_f1'] for s, v in vv.items()}
                                            for w, vv in lgb_win_val.items()},
                            'selected': best_lgb_win}

    # ── 4. LGB vs SVM on VALIDATION ──
    print('\n' + SEP)
    print('  CLASSIFIER CHOICE: LGB vs SVM (selected on train-internal validation)')
    print(SEP)
    feat_cfg = dataclasses.replace(base.features, window_length_ms=best_lgb_win, step_ms=40)
    clf_val = {'LGB': {}, 'SVM': {}}
    for s in SUBJECTS:
        clf_val['LGB'][s] = _fit_eval_spectral(data[s]['tr_ecog'], data[s]['tr_labels'],
                                               data[s]['val_ecog'], data[s]['val_labels'],
                                               feat_cfg, LightGBMClassifier(base.lgb))
        clf_val['SVM'][s] = _fit_eval_spectral(data[s]['tr_ecog'], data[s]['tr_labels'],
                                               data[s]['val_ecog'], data[s]['val_labels'],
                                               feat_cfg, SpectralSVM(base.svm))
    for name in ['LGB', 'SVM']:
        mf1 = _mean_val_f1(clf_val[name])
        print(f'  {name:<4} val F1: ' + '  '.join(
            f"{clf_val[name][s]['macro_f1']:.3f}" for s in SUBJECTS) + f'  mean={mf1:.3f}')
    best_clf = max(clf_val, key=lambda k: _mean_val_f1(clf_val[k]))
    print(f'  >>> selected classifier: {best_clf}  (val F1={_mean_val_f1(clf_val[best_clf]):.3f})')
    report['classifier'] = {'val_results': {k: {s: v['macro_f1'] for s, v in vv.items()}
                                            for k, vv in clf_val.items()},
                            'selected': best_clf}

    # ── 5. Final evaluation on the official TEST segment ──
    print('\n' + SEP)
    print('  FINAL EVALUATION (fit on full 400 s train, evaluate on 200 s test)')
    print(SEP)
    final = {}

    csp_final_cfg = dataclasses.replace(base.csp, freq_band=best_band_hz, window_ms=best_csp_win, step_ms=40)
    for s in SUBJECTS:
        r = _fit_eval_csp(data[s]['full_ecog'], generate_labels(data[s]['full_dg']),
                          data[s]['test_ecog'], data[s]['test_labels'],
                          csp_final_cfg, base.features)
        final.setdefault('CSP+LDA', {})[s] = r
    for s in SUBJECTS:
        m = final['CSP+LDA'][s]
        print(f'  CSP+LDA  {s}: F1={m["macro_f1"]:.3f}  BalAcc={m["balanced_accuracy"]:.3f}  '
              f'kappa={m["cohen_kappa"]:.3f}')
    csp_mean_f1 = np.mean([final['CSP+LDA'][s]['macro_f1'] for s in SUBJECTS])
    print(f'  >>> CSP+LDA mean macro-F1 = {csp_mean_f1:.3f}')

    lgb_final_cfg = dataclasses.replace(base.features, window_length_ms=best_lgb_win, step_ms=40)
    clf_factory = LightGBMClassifier if best_clf == 'LGB' else SpectralSVM
    clf_cfg = base.lgb if best_clf == 'LGB' else base.svm
    for s in SUBJECTS:
        r = _fit_eval_spectral(data[s]['full_ecog'], generate_labels(data[s]['full_dg']),
                               data[s]['test_ecog'], data[s]['test_labels'],
                               lgb_final_cfg, clf_factory(clf_cfg))
        final.setdefault(f'Spectral+{best_clf}', {})[s] = r
    for s in SUBJECTS:
        m = final[f'Spectral+{best_clf}'][s]
        print(f'  Spectral+{best_clf}  {s}: F1={m["macro_f1"]:.3f}  BalAcc={m["balanced_accuracy"]:.3f}  '
              f'kappa={m["cohen_kappa"]:.3f}')
    spec_mean_f1 = np.mean([final[f'Spectral+{best_clf}'][s]['macro_f1'] for s in SUBJECTS])
    print(f'  >>> Spectral+{best_clf} mean macro-F1 = {spec_mean_f1:.3f}')

    report['final_test'] = {
        'CSP+LDA': {s: final['CSP+LDA'][s]['macro_f1'] for s in SUBJECTS},
        f'Spectral+{best_clf}': {s: final[f'Spectral+{best_clf}'][s]['macro_f1'] for s in SUBJECTS},
    }
    report['final_test']['CSP+LDA']['mean'] = round(float(csp_mean_f1), 4)
    report['final_test'][f'Spectral+{best_clf}']['mean'] = round(float(spec_mean_f1), 4)

    # ── Save ──
    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(out_dir, f'hyperparam_selection_val_{ts}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print('\n' + SEP)
    print(f'  DONE in {(time.time() - t0) / 60:.1f} min.  Saved -> {out_path}')
    print('  Summary:')
    print(f'    CSP band     = {best_band}')
    print(f'    CSP window   = {best_csp_win} ms')
    print(f'    LGB window   = {best_lgb_win} ms')
    print(f'    classifier   = {best_clf}')
    print(f'    CSP+LDA test mean F1 = {csp_mean_f1:.3f}')
    print(f'    Spectral+{best_clf} test mean F1 = {spec_mean_f1:.3f}')
    print(SEP)


if __name__ == '__main__':
    main()
