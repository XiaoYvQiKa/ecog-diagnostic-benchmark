# -*- coding: utf-8 -*-
"""Recompute the paired statistics with the two added Miller subjects.

Note: this is a historical reproduction script. It reads per-method result
files from a specific result snapshot (separate per-method JSONs and
``res_npy/`` predictions) that are not shipped in this repository; the
committed outputs under ``sample_extension/results/`` are the authoritative
statistics (see ``sample_extension/README.md``).

Extends the published per-finger Wilcoxon signed-rank analysis from 15 paired
observations (5 fingers x 3 BCI subjects) to 25 (5 fingers x 5 subjects), by
appending the two added subjects (jc, wc) to the three original subjects.

Two tasks, each with four methods:

  Regression (per-finger Pearson r):
    Wiener filter, Switching linear model (est.), FingerFlex, DTCNet.
  Classification (per-finger Macro F1):
    CSP+LDA, Spectral+LGB, EEGNet, EEG-Conformer.

For each task the six within-task pairwise comparisons are tested with a
two-sided Wilcoxon signed-rank test; Cohen's d is computed as
mean(diff) / std(diff, ddof=1), positive in favour of the first (higher-ranked)
method of each pair. Bonferroni alpha = 0.05 / 6 = 0.00833. The subject-level
Friedman test and Demšar critical difference (CD) are also recomputed for
N = 5 subjects.

The 25-observation vectors are assembled strictly from the source result files
so the values are not hand-copied. Sources:

  - Wiener:        regression/wiener/results/results.json
                   (per-finger test_r_rule for sub1..3) and
                   sample_extension/results/wiener_miller_results.json (jc, wc)
  - Switching LM:  regression/switching_lm/results/results.json
                   (subjects[].estimated for sub1..3) and
                   sample_extension/results/switching_lm_miller_results.json
  - FingerFlex:    regression/fingerflex/results/fingerflex_r2_3seed_agg.csv
                   (sub1..3) and the res_npy predictions for jc/wc (3-seed mean)
  - DTCNet:        regression/dtcnet/results/main/results.json (sub1..3)
                   and main_miller/results.json (jc, wc)
  - Classification: classification/results/benchmark_csp_lgb_20260815_024257.json,
                    trial_dl_eegconformer_eegnet_20260815_024212.json (sub1..3),
                    and sample_extension/results/classification_miller_results.json (jc, wc)

Run with the base conda python (has scipy, numpy, lightgbm, braindecode).

Usage:
    python recompute_stats_25pairs.py [--full]
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
FINGERS_L = ['thumb', 'index', 'middle', 'ring', 'little']
BCI_SUBJECTS = ['sub1', 'sub2', 'sub3']
MILLER_SUBJECTS = ['jc', 'wc']
ALL_SUBJECTS = BCI_SUBJECTS + MILLER_SUBJECTS
ALPHA_BONF = 0.05 / 6.0


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _wiener_per_finger():
    """Return {subject: np.array(5)} of Wiener per-finger r (test_r_rule)."""
    p = os.path.join(ROOT, 'Regression', 'wiener', 'results', 'results.json')
    d = json.load(open(p))
    out = {}
    for subj, name in zip(d, BCI_SUBJECTS):
        out[name] = np.array([f['test_r_rule'] for f in subj['fingers']])
    p2 = os.path.join(ROOT, 'sample_extension', 'results', 'wiener_miller_results.json')
    d2 = json.load(open(p2))
    for r in d2:
        out[r['subject']] = np.array([r['per_finger_r'][f] for f in FINGERS_L])
    return out


def _switching_per_finger():
    """Return {subject: np.array(5)} of Switching-LM per-finger r (estimated)."""
    p = os.path.join(ROOT, 'Regression', 'switching_lm', 'results', 'results.json')
    d = json.load(open(p))
    out = {}
    for s in BCI_SUBJECTS:
        out[s] = np.array(d['subjects'][s]['estimated'])
    p2 = os.path.join(ROOT, 'sample_extension', 'results', 'switching_lm_miller_results.json')
    d2 = json.load(open(p2))
    for r in d2:
        out[r['subject']] = np.array([r['per_finger_estimated'][f] for f in FINGERS_L])
    return out


def _fingerflex_per_finger():
    """Return {subject: np.array(5)} of FingerFlex per-finger r (3-seed mean)."""
    p = os.path.join(ROOT, 'Regression', 'fingerflex', 'results',
                     'fingerflex_r2_3seed_agg.csv')
    out = {}
    with open(p, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            sub = row['subject']
            if sub in BCI_SUBJECTS and row['finger'] in FINGERS:
                out.setdefault(sub, {})[row['finger']] = float(row['pearson_r_mean'])
    for s in BCI_SUBJECTS:
        out[s] = np.array([out[s][f] for f in FINGERS])

    # jc / wc from res_npy (3-seed mean per finger)
    base = os.path.join(ROOT, 'Regression', 'fingerflex', 'res_npy')
    for sub in MILLER_SUBJECTS:
        rs = []
        for seed in (0, 1, 2):
            pred = np.load(os.path.join(base, f'prediction_{sub}_seed{seed}.npy'))
            true = np.load(os.path.join(base, f'true_{sub}_seed{seed}.npy'))
            n = min(len(pred), len(true))
            pred, true = pred[:n], true[:n]
            rs.append([np.corrcoef(pred[:, f], true[:, f])[0, 1] for f in range(5)])
        out[sub] = np.mean(rs, axis=0)
    return out


def _dtcnet_per_finger():
    """Return {subject: np.array(5)} of DTCNet per-finger r (3-seed mean)."""
    p = os.path.join(ROOT, 'Regression', 'dtcnet', 'results', 'main', 'results.json')
    d = json.load(open(p))
    out = {}
    for s in BCI_SUBJECTS:
        out[s] = np.array([d[s]['mean'][f] for f in FINGERS_L])
    p2 = os.path.join(ROOT, 'Regression', 'dtcnet', 'results', 'main_miller', 'results.json')
    d2 = json.load(open(p2))
    for s in MILLER_SUBJECTS:
        seeds = d2[s]['per_seed']
        out[s] = np.array([
            np.mean([seeds[str(k)]['mean'][f] for k in (0, 1, 2)]) for f in FINGERS_L
        ])
    return out


def _classification_per_finger():
    """Return {method: {subject: np.array(5)}} of classification per-finger F1.

    The original three subjects are read from the published benchmark
    (classification/results, four per-method JSONs, 5 seeds each);
    the two Miller subjects come from the sample-extension run.
    """
    cb = os.path.join(ROOT, 'Classification', 'results')
    src = {
        'csp': os.path.join(cb, 'benchmark_csp_20261010_130119.json'),
        'lgb': os.path.join(cb, 'benchmark_lgb_20261010_130033.json'),
        'eegnet': os.path.join(cb, 'trial_dl_eegnet_20261010_132637.json'),
        'eegconformer': os.path.join(cb, 'trial_dl_eegconformer_20261010_133124.json'),
    }
    miller_p = os.path.join(ROOT, 'sample_extension', 'results', 'classification_miller_results.json')
    miller = json.load(open(miller_p))

    def mean_pc(data, subject, method, finger):
        vals = [r['results'][method]['per_class'][finger]['f1']
                for r in data['results'] if r['subject'] == subject]
        return np.mean(vals)

    out = {}
    for method, path in src.items():
        data = json.load(open(path))
        out[method] = {}
        for s in BCI_SUBJECTS:
            out[method][s] = np.array([mean_pc(data, s, method, f) for f in FINGERS])

    # Miller subjects
    for method, data in [('csp', miller), ('lgb', miller)]:
        for s in MILLER_SUBJECTS:
            out[method][s] = np.array([
                np.mean([r['results'][method]['per_class'][f]['f1']
                         for r in data['traditional'] if r['subject'] == s]) for f in FINGERS])
    for method, data in [('eegnet', miller), ('eegconformer', miller)]:
        for s in MILLER_SUBJECTS:
            out[method][s] = np.array([
                np.mean([r['results'][method]['per_class'][f]['f1']
                         for r in data['dl'] if r['subject'] == s]) for f in FINGERS])
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _paired_vector(method_data, subjects):
    """Flatten per-subject (5,) vectors into one (5*N,) vector."""
    return np.concatenate([method_data[s] for s in subjects])


def _cohen_d(v1, v2):
    diff = v1 - v2
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def _pairwise_table(name, methods, subjects, n_repro=None):
    """Print 6 pairwise Wilcoxon + Cohen's d for a task."""
    print(f'\n{"=" * 78}')
    print(f'{name}  ({" + ".join(subjects)}, n={5 * len(subjects)})')
    print(f'{"=" * 78}')
    keys = list(methods.keys())
    vecs = {k: _paired_vector(v, subjects) for k, v in methods.items()}

    # header: per-method mean
    for k in keys:
        print(f'  {k:<18} mean={vecs[k].mean():.4f}')

    print(f'\n  {"Comparison":<28} {"Wilcoxon p":>12} {"Cohen d":>10}  sig')
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            v1, v2 = vecs[keys[i]], vecs[keys[j]]
            stat, p = wilcoxon(v1, v2)
            d = _cohen_d(v1, v2)
            sig = 'yes' if p < ALPHA_BONF else 'no'
            print(f'  {keys[i] + " vs " + keys[j]:<28} {p:>12.5g} {d:>+10.3f}  {sig}')

    # subject-level Friedman (N = len(subjects))
    mat = np.array([[methods[k][s].mean() for s in subjects] for k in keys])
    try:
        st, pf = friedmanchisquare(*[mat[i] for i in range(len(keys))])
        print(f'\n  Friedman (subject-level, N={len(subjects)}): chi2={st:.3f}, p={pf:.4f}')
    except Exception as e:
        print(f'\n  Friedman skipped: {e}')

    # Demšar CD (k methods, N subjects, alpha=0.05)
    k = len(keys)
    N = len(subjects)
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}.get(k, 2.569)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
    print(f'  Demšar CD (k={k}, N={N}, alpha=0.05): q={q}, CD={cd:.3f}')
    # average ranks (per subject, rank 1 = best)
    ranks = np.array([np.argsort(np.argsort(-mat[:, s])) + 1 for s in range(N)])
    avg_rank = ranks.mean(axis=0)
    print(f'  Average ranks (lower = better): ' +
          ', '.join(f'{keys[i]}={avg_rank[i]:.2f}' for i in range(k)))
    return vecs, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='also print 15-pair reproduction')
    a = ap.parse_args()

    # ---- assemble regression methods ----
    reg = {
        'Wiener': _wiener_per_finger(),
        'SwitchingLM': _switching_per_finger(),
        'FingerFlex': _fingerflex_per_finger(),
        'DTCNet': _dtcnet_per_finger(),
    }

    # ---- assemble classification methods ----
    cls_pf = _classification_per_finger()
    cls = {
        'CSP+LDA': cls_pf['csp'],
        'Spectral+LGB': cls_pf['lgb'],
        'EEGNet': cls_pf['eegnet'],
        'EEG-Conformer': cls_pf['eegconformer'],
    }

    if a.full:
        # reproduction check on the original 3 subjects only
        print('#' * 78)
        print('# REPRODUCTION CHECK (original 3 subjects, n=15)')
        print('#' * 78)
        _pairwise_table('REGRESSION reproduction', reg, BCI_SUBJECTS)
        _pairwise_table('CLASSIFICATION reproduction', cls, BCI_SUBJECTS)

    print('\n' + '#' * 78)
    print('# EXTENDED ANALYSIS (5 subjects, n=25)')
    print('#' * 78)
    _pairwise_table('REGRESSION extended', reg, ALL_SUBJECTS)
    _pairwise_table('CLASSIFICATION extended', cls, ALL_SUBJECTS)

    # classification pooled traditional vs deep (n=25)
    print('\n' + '=' * 78)
    print('CLASSIFICATION pooled traditional vs deep (n=25)')
    print('=' * 78)
    trad_vec = (_paired_vector(cls['CSP+LDA'], ALL_SUBJECTS) +
                _paired_vector(cls['Spectral+LGB'], ALL_SUBJECTS)) / 2
    dl_vec = (_paired_vector(cls['EEGNet'], ALL_SUBJECTS) +
              _paired_vector(cls['EEG-Conformer'], ALL_SUBJECTS)) / 2
    stat, p = wilcoxon(trad_vec, dl_vec)
    d = _cohen_d(trad_vec, dl_vec)
    print(f'  trad mean={trad_vec.mean():.4f}, DL mean={dl_vec.mean():.4f}, '
          f'p={p:.5g}, Cohen d={d:+.3f}')

    # dump the 25-vectors for the paper
    dump = {
        'regression': {k: {s: v[s].tolist() for s in ALL_SUBJECTS} for k, v in reg.items()},
        'classification': {k: {s: v[s].tolist() for s in ALL_SUBJECTS} for k, v in cls.items()},
    }
    out_p = os.path.join(ROOT, 'sample_extension', 'results', 'stats_25pairs_vectors.json')
    json.dump(dump, open(out_p, 'w'), indent=2)
    print(f'\n25-pair vectors saved -> {out_p}')


if __name__ == '__main__':
    main()
