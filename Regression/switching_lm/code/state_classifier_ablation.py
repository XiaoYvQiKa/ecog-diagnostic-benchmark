# -*- coding: utf-8 -*-
"""Ablation for the Switching LM state classifier.

Two controls, both targeting the root cause of the state-classifier failure:

  (A) features control  -- AR coefficients (paper) vs gamma-band power.
      If balanced accuracy stays near chance on gamma power, the bottleneck
      is the task / data, not the AR representation.
  (B) thresholds control -- sweep STATE_ON_THRESHOLD_FRAC (0.10 / 0.20 / 0.30)
      to measure how sensitive the state definition (and downstream decode)
      is to the activation threshold.

Reports, per subject, the test-set state-classifier balanced accuracy and the
estimated-decoder official_r (ring excluded) for every condition.

Run from switching_lm/code:
    python state_classifier_ablation.py            # full sweep (~30-40 min)
    python state_classifier_ablation.py --quick    # 1 subject, 1 config smoke
"""
import argparse
import json
import os

import numpy as np

import config as C
from data_io import load_subject_clean
from state_labels import make_state_labels
from evaluate_decoder import fit_decoder, evaluate

TUNED = {1: dict(tau_samples=75, M=30), 2: dict(tau_samples=150, M=30),
         3: dict(tau_samples=500, M=30)}
N_SELECT_CHANNELS = 15
LAMBDA_S = 0.0
LAMBDA_K = 1.0


def balanced_accuracy(true_state, pred_state, n_states):
    recalls = []
    for k in range(1, n_states + 1):
        mask = true_state == k
        if mask.sum() == 0:
            continue
        recalls.append(float(np.mean(pred_state[mask] == true_state[mask])))
    return float(np.mean(recalls))


def run_one(n, feature_type, frac):
    C.STATE_ON_THRESHOLD_FRAC = frac          # monkey-patch the threshold
    sd = load_subject_clean(n)
    sl = make_state_labels(sd.train_glove, sd.test_glove, subject=n)
    tau, M = TUNED[n]["tau_samples"], TUNED[n]["M"]
    bundle = fit_decoder(
        sd.train_ecog, sd.train_glove, sl.train_state,
        n_select_channels=N_SELECT_CHANNELS, lambda_s=LAMBDA_S,
        tau_samples=tau, lambda_k=LAMBDA_K, M=M, feature_type=feature_type,
    )
    res = evaluate(bundle, sd.test_ecog, sd.test_glove, sl.test_state)
    pred = res["pred_state_aligned"]
    true = res["state_true_aligned"]
    ba = balanced_accuracy(true, pred, C.N_STATES)
    official = float(np.nanmean([res["estimated"][i] for i in (0, 1, 2, 4)]))
    avg = float(np.nanmean(res["estimated"]))
    return ba, official, avg, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        ba, official, avg, _ = run_one(1, "gamma", 0.20)
        print(f"SMOKE S1 gamma frac=0.20: BA={ba:.3f} official_r={official:+.3f} avg_r={avg:+.3f}")
        return

    out = {"features": {}, "thresholds": {}}

    # (A) features control, threshold fixed at the paper's 0.20
    print("=" * 70)
    print("(A) FEATURES CONTROL  (frac = 0.20)")
    print("=" * 70)
    for ft in ("ar", "gamma"):
        print(f"\n--- feature = {ft} ---")
        bas, offs, avgs = [], [], []
        for n in (1, 2, 3):
            ba, official, avg, _ = run_one(n, ft, 0.20)
            bas.append(ba); offs.append(official); avgs.append(avg)
            print(f"  S{n}: BA={ba:.3f}  official_r={official:+.3f}  avg_r={avg:+.3f}")
        out["features"][ft] = {"ba": bas, "official_r": offs, "avg_r": avgs}
        print(f"  mean: BA={np.mean(bas):.3f}  official_r={np.mean(offs):+.3f}")

    # (B) thresholds control, feature fixed at AR (the paper's)
    print("\n" + "=" * 70)
    print("(B) THRESHOLDS CONTROL  (feature = ar)")
    print("=" * 70)
    for frac in (0.10, 0.20, 0.30):
        print(f"\n--- frac = {frac} ---")
        bas, offs, avgs = [], [], []
        for n in (1, 2, 3):
            ba, official, avg, _ = run_one(n, "ar", frac)
            bas.append(ba); offs.append(official); avgs.append(avg)
            print(f"  S{n}: BA={ba:.3f}  official_r={official:+.3f}  avg_r={avg:+.3f}")
        out["thresholds"][str(frac)] = {"ba": bas, "official_r": offs, "avg_r": avgs}
        print(f"  mean: BA={np.mean(bas):.3f}  official_r={np.mean(offs):+.3f}")

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "results", "state_classifier_ablation.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved -> state_classifier_ablation.json")


if __name__ == "__main__":
    main()
