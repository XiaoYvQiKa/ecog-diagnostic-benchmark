# -*- coding: utf-8 -*-
"""Electrode-QC ablation for the Wiener filter.

Quantifies the performance perturbation of the 11 Wiener relative-quantile
extra channels by comparing four channel-set configurations per subject:

  none      : no electrode QC (all channels kept)
  cross4    : only the 4 cross-verified channels removed (classification track
              absolute-threshold criteria, 0-based physical indices)
  all15     : the full 15 channels removed (4 cross-verified + 11 Wiener extras)
  auto95    : the paper's default train-only relative-quantile screen (pct=95),
              kept here to VERIFY that it flags exactly the same 15 channels
              as the explicit all15 list (channel-list correctness check)

All indices are 0-based physical (regression-track convention). Output is a
JSON + a printed comparison table with per-finger r, official_r (ring excluded)
and avg_r (5-finger mean).

Run from the wiener/code directory:
    python qc_ablation.py
"""
import json
import os

import numpy as np

from run_reproduction import run_subject

# 0-based physical channel indices.
CROSS_VERIFIED = {1: [54], 2: [20, 37], 3: [49]}          # classification-track 4
ALL_15 = {
    1: [4, 25, 28, 50, 54],
    2: [20, 22, 30, 31, 37],
    3: [14, 22, 43, 44, 49],
}                                                          # 4 cross-verified + 11 extras

CONFIGS = [
    ("none",    "no QC",      None,                 None),
    ("cross4",  "4 cross-verified", CROSS_VERIFIED, None),
    ("all15",   "4 + 11 (15)",     ALL_15,          None),
    ("auto95",  "auto pct=95",     None,             True),
]

FINGERS = ["thumb", "index", "middle", "ring", "little"]


def run_config(name, bad_dict, auto_qc):
    rows = {}
    for s in (1, 2, 3):
        kw = dict(subject=s, verbose=False)
        if bad_dict is not None:
            kw["explicit_bad_channels"] = bad_dict[s]
        elif auto_qc:
            kw["electrode_qc"] = True
            kw["qc_percentile"] = 95.0
        else:
            kw["electrode_qc"] = False
        res = run_subject(**kw)
        rows[s] = res
    return rows


def main():
    out = {}
    for key, label, bad_dict, auto_qc in CONFIGS:
        print(f"running config [{key}] {label} ...")
        res = run_config(key, bad_dict, auto_qc)
        out[key] = {str(s): res[s] for s in (1, 2, 3)}
        for s in (1, 2, 3):
            r = res[s]
            fs = [f["test_r_rule"] for f in r["fingers"]]
            official = r["official_test_r"]
            avg = r["avg_test_r"]
            print(f"  S{s}: official_r={official:+.3f}  avg_r={avg:+.3f}  "
                  f"per-finger={[round(v, 3) for v in fs]}")

    # persist
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(here, "results", "qc_ablation_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved -> {out_path}")

    # summary comparison table
    print("\n" + "=" * 90)
    print("SUMMARY: official_r (ring excluded) per config per subject")
    print("=" * 90)
    hdr = f"{'config':<10}" + "".join(f"{'S'+str(s):>10}" for s in (1, 2, 3)) + f"{'mean':>10}"
    print(hdr)
    print("-" * 90)
    for key, label, _, _ in CONFIGS:
        vals = [out[key][str(s)]["official_test_r"] for s in (1, 2, 3)]
        print(f"{key:<10}" + "".join(f"{v:>10.3f}" for v in vals)
              + f"{np.mean(vals):>10.3f}")
    print("-" * 90)
    base = [out["none"][str(s)]["official_test_r"] for s in (1, 2, 3)]
    c4 = [out["cross4"][str(s)]["official_test_r"] for s in (1, 2, 3)]
    a15 = [out["all15"][str(s)]["official_test_r"] for s in (1, 2, 3)]
    auto = [out["auto95"][str(s)]["official_test_r"] for s in (1, 2, 3)]
    print(f"{'delta(4-15)':<10}" + "".join(f"{(c-a):>10.3f}" for c, a in zip(c4, a15))
          + f"{np.mean(c4)-np.mean(a15):>10.3f}")
    print(f"{'auto==all15?':<10}"
          + "".join(f"{'OK' if abs(x-y) < 1e-9 else 'DIFF':>10}" for x, y in zip(auto, a15)))
    print("=" * 90)


if __name__ == "__main__":
    main()
