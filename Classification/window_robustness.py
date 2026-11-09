# -*- coding: utf-8 -*-
"""Window-robustness control for the cross-task reversal.

Sweeps the classification-side analysis window and checks that the
"traditional > deep" direction of the reversal stays stable, i.e. that the
reversal is not an artifact of one particular window choice.

  CSP+LDA      : window_ms in {250, 500, 1000}
  Spectral+LGB : window_length_ms in {500, 1000, 1500}

The deep baselines (EEGNet / EEG-Conformer) are left at their fixed 1000 ms
trial (their locked benchmark setting); this script only needs to show the
traditional methods remain above the deep methods across windows.

Run from classification/:
    python window_robustness.py                 # full (3 subjects, 1 seed each)
    python window_robustness.py --seeds 42 142 242   # multi-seed
"""
import argparse
import json
from dataclasses import replace

from common.config import BenchmarkConfig
from common.run_traditional import run_traditional_subject

# Locked deep-method macro-F1 (5-seed, from the published results) -- used only
# as the fixed reference line for the "traditional > deep" direction check.
DEEP_F1 = {"eegnet": 0.238, "conformer": 0.228}


def run_window_sweep(csp_windows, lgb_windows, subjects, seeds):
    out = {}
    for w in csp_windows:
        cfg = BenchmarkConfig()
        cfg = replace(cfg, csp=replace(cfg.csp, window_ms=w))
        key = f"csp_{w}ms"
        out[key] = {}
        for seed in seeds:
            for subj in subjects:
                r = run_traditional_subject(subj, seed, ["csp"], cfg, verbose=False)
                m = r["results"]["csp"]
                out[key].setdefault(subj, []).append(m["macro_f1"])
                print(f"  [csp {w}ms] {subj} seed={seed}: F1={m['macro_f1']:.3f}  "
                      f"BalAcc={m['balanced_accuracy']:.3f}")

    for w in lgb_windows:
        cfg = BenchmarkConfig()
        cfg = replace(cfg, features=replace(cfg.features, window_length_ms=w))
        key = f"lgb_{w}ms"
        out[key] = {}
        for seed in seeds:
            for subj in subjects:
                r = run_traditional_subject(subj, seed, ["lgb"], cfg, verbose=False)
                m = r["results"]["lgb"]
                out[key].setdefault(subj, []).append(m["macro_f1"])
                print(f"  [lgb {w}ms] {subj} seed={seed}: F1={m['macro_f1']:.3f}  "
                      f"BalAcc={m['balanced_accuracy']:.3f}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["sub1", "sub2", "sub3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = ap.parse_args()

    out = run_window_sweep([250, 500, 1000], [500, 1000, 1500],
                           args.subjects, args.seeds)

    with open("results/window_robustness.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nsaved -> results/window_robustness.json")

    # summary: per-window mean F1 vs deep reference
    print("\n" + "=" * 70)
    print("SUMMARY: macro-F1 mean across subjects (traditional vs deep reference)")
    print("=" * 70)
    import numpy as np
    for key, d in out.items():
        subj_means = [np.mean(v) for v in d.values()]
        print(f"  {key:<12} mean F1 = {np.mean(subj_means):.3f}   "
              f"(per-subject {[round(x,3) for x in subj_means]})")
    print(f"  {'EEGNet':<12} mean F1 = {DEEP_F1['eegnet']:.3f}   (fixed)")
    print(f"  {'Conformer':<12} mean F1 = {DEEP_F1['conformer']:.3f}   (fixed)")
    print("=" * 70)


if __name__ == "__main__":
    main()
