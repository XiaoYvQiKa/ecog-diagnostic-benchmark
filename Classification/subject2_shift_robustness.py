# -*- coding: utf-8 -*-
"""Label-distribution-shift robustness on Subject S2.

S2 shows a large train/test label shift (rest fraction collapses from train to
test), which the reviewers flagged as interfering with several models. This
script does two things:

  (1) quantifies the S2 shift (train vs test class distribution) and, for
      reference, the S1/S3 distributions;
  (2) measures how much the built-in class re-weighting mitigates it, by
      comparing LightGBM on S2 with class_weight = None vs "balanced".

The finding feeds directly into the reviewer response: if "balanced" recovers
little, we state that the shift is partly mitigated but S2 remains an
anomalous subject, and we flag the affected observations in the Results.

Run from classification/:
    python subject2_shift_robustness.py
"""
import json
from dataclasses import replace

import numpy as np

from common.config import BenchmarkConfig
from common.run_traditional import run_traditional_subject, load_data_for_subject
from common.data_loader import get_label_distribution


def quantify_shift():
    print("=" * 70)
    print("(1) TRAIN/TEST LABEL DISTRIBUTION (per subject)")
    print("=" * 70)
    shifts = {}
    for subj in ["sub1", "sub2", "sub3"]:
        d = load_data_for_subject(subj, verbose=False)
        tr = get_label_distribution(d["train_labels"])
        te = get_label_distribution(d["test_labels"])
        shifts[subj] = {"train": tr, "test": te}
        print(f"\n  {subj}:")
        print(f"    train: " + ", ".join(f"{k}:{v[1]:.0f}%" for k, v in tr.items()))
        print(f"    test : " + ", ".join(f"{k}:{v[1]:.0f}%" for k, v in te.items()))
    return shifts


def class_weight_ablation(subjects, seeds):
    print("\n" + "=" * 70)
    print("(2) LightGBM class_weight = None vs balanced (macro-F1)")
    print("=" * 70)
    out = {}
    for cw in (None, "balanced"):
        label = "none" if cw is None else "balanced"
        cfg = BenchmarkConfig()
        cfg = replace(cfg, lgb=replace(cfg.lgb, class_weight=cw))
        out[label] = {}
        print(f"\n  --- class_weight = {label} ---")
        for subj in subjects:
            f1s = []
            for seed in seeds:
                r = run_traditional_subject(subj, seed, ["lgb"], cfg, verbose=False)
                f1s.append(r["results"]["lgb"]["macro_f1"])
            out[label][subj] = f1s
            mu = float(np.mean(f1s))
            print(f"    {subj}: F1 = {mu:.3f}  (per-seed {[round(x,3) for x in f1s]})")
    return out


def main():
    shifts = quantify_shift()
    abl = class_weight_ablation(["sub1", "sub2", "sub3"], [42, 142, 242])

    with open("results/subject2_shift_robustness.json", "w") as fh:
        json.dump({"shifts": {k: {kk: {str(a): b for a, b in vv.items()}
                                  for kk, vv in v.items()} for k, v in shifts.items()},
                   "class_weight_ablation": abl}, fh, indent=2, default=str)
    print("\nsaved -> results/subject2_shift_robustness.json")


if __name__ == "__main__":
    main()
