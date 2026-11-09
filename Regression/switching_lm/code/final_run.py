"""
FINAL test-set evaluation, using the tuned hyperparameters found by
tune_regression.py (per-subject tau/M for the H_k/baseline regression
branch). The classifier branch (K, lambda_s) is NOT re-tuned further --
diagnose_confusion.py / diagnose_lag.py established that its ceiling is a
feature-representation limit (can't resolve finger identity), not a
hyperparameter one, so K=15/lambda_s=0 (no degenerate all-zero collapse)
is used consistently across subjects.

Run in the same folder as the other files, with data/ populated:

    python final_run.py

Prints the paper's Table 3 layout (baseline / oracle / estimated) on the
held-out TEST split for all 3 subjects, plus test-set state classification
accuracy (expected to stay near chance for S1/S2, per the documented
feature-limitation finding).
"""
import json
import os
import time

import numpy as np

import config as C
from data_io import load_subject_clean
from state_labels import make_state_labels
from evaluate_decoder import fit_decoder, evaluate

# per-subject regression hyperparameters, from tune_regression.py's BEST rows
TUNED = {
    1: dict(tau_samples=75, M=30),
    2: dict(tau_samples=150, M=30),
    3: dict(tau_samples=500, M=30),
}
LAMBDA_K = 1.0          # tune_regression.py found lambda_k doesn't matter once M is set
N_SELECT_CHANNELS = 15  # classifier branch -- not re-tuned, see module docstring
LAMBDA_S = 0.0          # avoids the degenerate all-zero-rows collapse seen at high lambda_s


def main():
    all_results = {}
    fit_times = {}

    for n in [1, 2, 3]:
        print(f"{'='*60}")
        print(f"SUBJECT {n}")
        print(f"{'='*60}")

        sd = load_subject_clean(n)
        sl = make_state_labels(sd.train_glove, sd.test_glove, subject=n)

        tau_samples = TUNED[n]["tau_samples"]
        M = TUNED[n]["M"]
        print(f"  using tuned tau_samples={tau_samples} ({tau_samples/C.FS_ECOG*1000:.0f}ms), M={M}")

        t0 = time.time()
        bundle = fit_decoder(
            sd.train_ecog, sd.train_glove, sl.train_state,
            n_select_channels=N_SELECT_CHANNELS, lambda_s=LAMBDA_S,
            tau_samples=tau_samples, lambda_k=LAMBDA_K, M=M,
        )
        fit_time = time.time() - t0
        print(f"  fit time: {fit_time:.1f}s")
        fit_times[n] = fit_time

        res = evaluate(bundle, sd.test_ecog, sd.test_glove, sl.test_state)
        all_results[n] = res

        print()
        print(f'  {"finger":8s} {"(a) baseline":>14s} {"(b) oracle":>12s} {"(c) estimated":>14s}')
        for j, name in enumerate(C.FINGER_NAMES):
            print(f'  {name:8s} {res["baseline"][j]:14.3f} {res["oracle"][j]:12.3f} {res["estimated"][j]:14.3f}')
        avg_b, avg_o, avg_e = (np.nanmean(res["baseline"]), np.nanmean(res["oracle"]),
                               np.nanmean(res["estimated"]))
        print(f'  {"AVERAGE":8s} {avg_b:14.3f} {avg_o:12.3f} {avg_e:14.3f}')

        state_acc = np.mean(res["pred_state_aligned"] == res["state_true_aligned"])
        print(f"  test-set state classification accuracy: {state_acc:.3f}")
        print()

    print(f"{'='*60}")
    print("SUMMARY (mean +/- SD across 3 subjects)")
    print(f"{'='*60}")
    for label in ["baseline", "oracle", "estimated"]:
        subj_avgs = [np.nanmean(all_results[n][label]) for n in [1, 2, 3]]
        print(f"  {label:10s}: {np.mean(subj_avgs):.3f} +/- {np.std(subj_avgs, ddof=1):.3f}  "
              f"(per-subject: {[round(v,3) for v in subj_avgs]})")
    print()
    print(f"  paper reference (Table 3 avg, real data): baseline={C.PAPER_LINEAR_BASELINE}  "
          f"oracle={C.PAPER_ORACLE_STATE}  estimated={C.PAPER_ESTIMATED_STATE}")

    # --- Persist results for reproducibility (added 2026-10-10) ---
    # Save the full per-finger r (5 fingers) for all three variants plus the
    # test-set state accuracy, so both the 5-finger mean (avg_r) and the
    # official 4-finger mean (ring excluded) can be recomputed from this file.
    out = {
        "method": "switching_linear_model",
        "hyperparams": {
            "tuned": {str(k): v for k, v in TUNED.items()},
            "lambda_k": LAMBDA_K,
            "n_select_channels": N_SELECT_CHANNELS,
            "lambda_s": LAMBDA_S,
        },
        "subjects": {},
        "runtime": {
            "fit_time_s": {str(n): round(float(fit_times[n]), 2) for n in [1, 2, 3]},
            "note": "fit_time_s = wall-clock fit_decoder time (group-lasso BCD state classifier + per-state ridge regression) on CPU; run-to-run variation is dominated by machine load, not algorithmic instability. Inference latency is measured separately by diagnostics/measure_inference_ms.py and reported in the paper appendix (Table D3 / G1).",
        },
    }
    for n in [1, 2, 3]:
        res = all_results[n]
        out["subjects"][f"sub{n}"] = {
            "baseline": [float(v) for v in res["baseline"]],
            "oracle": [float(v) for v in res["oracle"]],
            "estimated": [float(v) for v in res["estimated"]],
            "state_accuracy": float(np.mean(res["pred_state_aligned"] == res["state_true_aligned"])),
            "fit_time_s": round(float(fit_times[n]), 2),
        }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
