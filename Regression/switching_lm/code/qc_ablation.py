# -*- coding: utf-8 -*-
"""Electrode-QC ablation for the Switching Linear Model.

Compares two channel-set configurations per subject by monkey-patching
config.BAD_CHANNELS (0-based physical indices) before load_subject_clean:

  cross4 : only the 4 cross-verified channels removed (classification track
           absolute-threshold criteria)
  all15  : the full 15 channels removed (4 cross-verified + 11 Wiener extras),
           i.e. the paper's current result

The only variable that changes is the excluded channel set; every other
hyperparameter (tau/M/K/lambda_s/lambda_k) stays at final_run.py's locked
values. Reports official_r (ring excluded), avg_r (5-finger mean) and the
test-set state-classification accuracy per subject.

Run from switching_lm/code:
    python qc_ablation.py
"""
import json
import os

import numpy as np

import config as C
from data_io import load_subject_clean
from state_labels import make_state_labels
from evaluate_decoder import fit_decoder, evaluate

CROSS_VERIFIED = {1: [54], 2: [20, 37], 3: [49]}
ALL_15 = {
    1: [4, 25, 28, 50, 54],
    2: [20, 22, 30, 31, 37],
    3: [14, 22, 43, 44, 49],
}

TUNED = {1: dict(tau_samples=75, M=30), 2: dict(tau_samples=150, M=30),
         3: dict(tau_samples=500, M=30)}
N_SELECT_CHANNELS = 15
LAMBDA_S = 0.0
LAMBDA_K = 1.0


def official_r(per_finger):
    return float(np.nanmean([per_finger[i] for i in (0, 1, 2, 4)]))


def run_config(bad_dict):
    per_subject = {}
    for n in (1, 2, 3):
        C.BAD_CHANNELS = dict(bad_dict)      # monkey-patch for this config
        sd = load_subject_clean(n)
        sl = make_state_labels(sd.train_glove, sd.test_glove, subject=n)
        tau, M = TUNED[n]["tau_samples"], TUNED[n]["M"]
        bundle = fit_decoder(
            sd.train_ecog, sd.train_glove, sl.train_state,
            n_select_channels=N_SELECT_CHANNELS, lambda_s=LAMBDA_S,
            tau_samples=tau, lambda_k=LAMBDA_K, M=M,
        )
        res = evaluate(bundle, sd.test_ecog, sd.test_glove, sl.test_state)
        est = res["estimated"]
        state_acc = float(np.mean(res["pred_state_aligned"] == res["state_true_aligned"]))
        per_subject[n] = {
            "estimated": [float(v) for v in est],
            "official_r": official_r(est),
            "avg_r": float(np.nanmean(est)),
            "state_accuracy": state_acc,
        }
    return per_subject


def main():
    out = {}
    for key, bad_dict in (("cross4", CROSS_VERIFIED), ("all15", ALL_15)):
        print(f"running config [{key}] ...")
        out[key] = run_config(bad_dict)
        for n in (1, 2, 3):
            d = out[key][n]
            print(f"  S{n}: official_r={d['official_r']:+.3f}  avg_r={d['avg_r']:+.3f}  "
                  f"state_acc={d['state_accuracy']:.3f}  "
                  f"per-finger={[round(v, 3) for v in d['estimated']]}")

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(here, "results", "qc_ablation_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved -> {out_path}")

    print("\n" + "=" * 80)
    print("SUMMARY: official_r (ring excluded), cross4 vs all15")
    print("=" * 80)
    print(f"{'config':<10}" + "".join(f"{'S'+str(s):>10}" for s in (1, 2, 3)) + f"{'mean':>10}")
    print("-" * 80)
    for key in ("cross4", "all15"):
        vals = [out[key][n]["official_r"] for n in (1, 2, 3)]
        print(f"{key:<10}" + "".join(f"{v:>10.3f}" for v in vals) + f"{np.mean(vals):>10.3f}")
    print("-" * 80)
    d = [out["cross4"][n]["official_r"] - out["all15"][n]["official_r"]
         for n in (1, 2, 3)]
    print(f"{'delta':<10}" + "".join(f"{v:>10.3f}" for v in d) + f"{np.mean(d):>10.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
