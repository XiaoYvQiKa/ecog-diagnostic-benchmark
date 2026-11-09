# -*- coding: utf-8 -*-
"""Full 25-pair statistics summary (regression + classification).

Computes and persists every pairwise Wilcoxon p, Cohen's d, Friedman test, and
Demsar CD for the 5-subject (n=25) extended analysis, and also the original
3-subject (n=15) reproduction for cross-checking. All values are read strictly
from the result files (no hand-copying).

Usage:
    python write_stats_summary.py
"""
import json
import os

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

import recompute_stats_25pairs as R

ALPHA_BONF = 0.05 / 6.0


def pairwise(vecs, order):
    rows = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = order[i], order[j]
            v1, v2 = vecs[a], vecs[b]
            stat, p = wilcoxon(v1, v2)
            d = R._cohen_d(v1, v2)
            rows.append({
                'comparison': f'{a} vs {b}',
                'wilcoxon_p': float(p),
                'cohens_d': float(d),
                'significant_after_bonferroni': bool(p < ALPHA_BONF),
            })
    return rows


def friedman_cd(vecs, order, subjects):
    # per method, per-subject mean (each method's 25-vector is laid out as
    # 5 subjects x 5 fingers in subject order)
    mat = np.array([[(vecs[k][si * 5:(si + 1) * 5]).mean() for si in range(len(subjects))]
                    for k in order])
    st, pf = friedmanchisquare(*[mat[i] for i in range(len(order))])
    k = len(order)
    N = len(subjects)
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}.get(k, 2.569)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
    ranks = np.array([np.argsort(np.argsort(-mat[:, s])) + 1 for s in range(N)])
    avg_rank = ranks.mean(axis=0)
    return {
        'friedman_chi2': float(st),
        'friedman_p': float(pf),
        'demsar_cd': float(cd),
        'average_ranks': {order[i]: float(avg_rank[i]) for i in range(k)},
        'per_method_mean': {order[i]: float(mat[i].mean()) for i in range(k)},
    }


def main():
    reg = {
        'FingerFlex': R._fingerflex_per_finger(),
        'DTCNet': R._dtcnet_per_finger(),
        'Wiener': R._wiener_per_finger(),
        'SwitchingLM': R._switching_per_finger(),
    }
    cls_pf = R._classification_per_finger()
    cls = {
        'CSP+LDA': cls_pf['csp'],
        'Spectral+LGB': cls_pf['lgb'],
        'EEGNet': cls_pf['eegnet'],
        'EEG-Conformer': cls_pf['eegconformer'],
    }

    out = {'alpha_bonferroni': ALPHA_BONF}

    # ---- n=15 reproduction ----
    out['n15'] = {}
    for name, methods, order in [
        ('regression', reg, ['FingerFlex', 'DTCNet', 'Wiener', 'SwitchingLM']),
        ('classification', cls, ['CSP+LDA', 'Spectral+LGB', 'EEGNet', 'EEG-Conformer']),
    ]:
        vecs = {k: R._paired_vector(v, R.BCI_SUBJECTS) for k, v in methods.items()}
        out['n15'][name] = {
            'pairwise': pairwise(vecs, order),
            'friedman_cd': friedman_cd(vecs, order, R.BCI_SUBJECTS),
        }
        if name == 'classification':
            tv = (vecs['CSP+LDA'] + vecs['Spectral+LGB']) / 2
            dv = (vecs['EEGNet'] + vecs['EEG-Conformer']) / 2
            stat, p = wilcoxon(tv, dv)
            out['n15'][name]['pooled_trad_vs_deep'] = {
                'p': float(p), 'cohens_d': R._cohen_d(tv, dv),
                'trad_mean': float(tv.mean()), 'deep_mean': float(dv.mean()),
            }

    # ---- n=25 extended ----
    out['n25'] = {}
    for name, methods, order in [
        ('regression', reg, ['FingerFlex', 'DTCNet', 'Wiener', 'SwitchingLM']),
        ('classification', cls, ['CSP+LDA', 'Spectral+LGB', 'EEGNet', 'EEG-Conformer']),
    ]:
        vecs = {k: R._paired_vector(v, R.ALL_SUBJECTS) for k, v in methods.items()}
        out['n25'][name] = {
            'pairwise': pairwise(vecs, order),
            'friedman_cd': friedman_cd(vecs, order, R.ALL_SUBJECTS),
        }
        if name == 'classification':
            tv = (vecs['CSP+LDA'] + vecs['Spectral+LGB']) / 2
            dv = (vecs['EEGNet'] + vecs['EEG-Conformer']) / 2
            stat, p = wilcoxon(tv, dv)
            out['n25'][name]['pooled_trad_vs_deep'] = {
                'p': float(p), 'cohens_d': R._cohen_d(tv, dv),
                'trad_mean': float(tv.mean()), 'deep_mean': float(dv.mean()),
            }

    out_p = os.path.join(R.ROOT, 'sample_extension', 'results', 'stats_summary_n15_n25.json')
    json.dump(out, open(out_p, 'w'), indent=2)
    print(f'saved -> {out_p}')

    # concise console report
    for scope in ('n15', 'n25'):
        print(f'\n===== {scope} =====')
        for task in ('regression', 'classification'):
            b = out[scope][task]
            print(f'\n-- {task} --')
            fc = b['friedman_cd']
            print(f'  mean: ' + ', '.join(f'{k}={v:.4f}' for k, v in fc['per_method_mean'].items()))
            print(f'  Friedman chi2={fc["friedman_chi2"]:.3f} p={fc["friedman_p"]:.4g}; CD={fc["demsar_cd"]:.3f}; ranks=' +
                  ', '.join(f'{k}={v:.2f}' for k, v in fc['average_ranks'].items()))
            for row in b['pairwise']:
                sig = 'yes' if row['significant_after_bonferroni'] else 'no'
                print(f'  {row["comparison"]:<34} p={row["wilcoxon_p"]:.5g} d={row["cohens_d"]:+.2f} {sig}')
            if 'pooled_trad_vs_deep' in b:
                pd_ = b['pooled_trad_vs_deep']
                print(f'  [pooled] trad vs deep: p={pd_["p"]:.5g} d={pd_["cohens_d"]:+.2f}')


if __name__ == '__main__':
    main()
