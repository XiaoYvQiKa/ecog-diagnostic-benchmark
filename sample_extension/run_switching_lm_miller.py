# -*- coding: utf-8 -*-
"""Switching Linear Model run on the added Miller subjects (jc, wc).

Reuses make_state_labels + fit_decoder + evaluate from switching_lm/code/,
replacing only the data source:
  - ECoG : P.PREP_DIR/{subj}/ecog_{train,test}.npy  (time, ch) @1000 Hz
  - flex : P.PREP_DIR/{subj}/flex_{train,test}.npy   (time, 5) @25 Hz de-based

The prep output is 25 Hz de-based flexion; the Switching Linear Model expects a
1000 Hz zero-order-held glove trace, so it is repeated back to 1000 Hz first.
make_state_labels uses a robust percentile threshold (translation-invariant), so
after de-baselining rest is near 0 and flexion positive, matching the original
BCI train_dg semantics. official_r is the four-finger mean of the estimated
decoder excluding the ring finger.

Usage:
    python run_switching_lm_miller.py jc wc
"""
import json
import sys
import time

import numpy as np

import paths as P

sys.path.insert(0, str(P.SW_DIR))
import config as C  # noqa: E402  (switching_lm method config)
from state_labels import make_state_labels  # noqa: E402
from evaluate_decoder import fit_decoder, evaluate  # noqa: E402


def flex_25_to_1k(flex_25, n_samples):
    """De-based flex (T25, 5) -> 1000 Hz zero-order-held, truncated to n_samples."""
    flex_1k = np.repeat(flex_25, 40, axis=0)              # (T25*40, 5)
    if flex_1k.shape[0] > n_samples:
        flex_1k = flex_1k[:n_samples]
    return flex_1k


def official_r(arr):
    """Four-finger mean excluding the ring finger."""
    ring = C.FINGER_NAMES.index("ring")
    return float(np.mean([arr[j] for j in range(5) if j != ring]))


def tier_summary(r, r2, cr2):
    """Per-tier aggregates for the appendix Table D4: avg_r (5 fingers),
    official_r (4 fingers excl. ring), mean R2 (4 excl. ring), and mean
    calibration gap (4 excl. ring), each a plain arithmetic mean."""
    ring = C.FINGER_NAMES.index("ring")
    off = [j for j in range(5) if j != ring]
    return {
        "avg_r": round(float(np.nanmean(r)), 4),
        "official_r": round(float(np.nanmean([r[j] for j in off])), 4),
        "mean_r2": round(float(np.nanmean([r2[j] for j in off])), 4),
        "mean_gap": round(float(np.nanmean([cr2[j] - r2[j] for j in off])), 4),
    }


def run_subject(subject: str) -> dict:
    t_start = time.time()
    ecog_train = np.load(P.PREP_DIR / subject / "ecog_train.npy").astype(np.float64)  # (T, ch)
    ecog_test = np.load(P.PREP_DIR / subject / "ecog_test.npy").astype(np.float64)
    flex_train_25 = np.load(P.PREP_DIR / subject / "flex_train.npy").astype(np.float64)
    flex_test_25 = np.load(P.PREP_DIR / subject / "flex_test.npy").astype(np.float64)

    # Zero-order-hold flex back to 1000 Hz and align to the 25 Hz grid.
    flex_train = flex_25_to_1k(flex_train_25, ecog_train.shape[0])
    flex_test = flex_25_to_1k(flex_test_25, ecog_test.shape[0])
    n_tr = min(ecog_train.shape[0], flex_train.shape[0])
    n_te = min(ecog_test.shape[0], flex_test.shape[0])
    ecog_train, flex_train = ecog_train[:n_tr], flex_train[:n_tr]
    ecog_test, flex_test = ecog_test[:n_te], flex_test[:n_te]

    print(f"\n{'='*70}")
    print(f"SwitchingLM on {subject}: ecog train {ecog_train.shape}, flex train {flex_train.shape}")

    sl = make_state_labels(flex_train, flex_test, subject=0)
    bundle = fit_decoder(ecog_train, flex_train, sl.train_state,
                         n_select_channels=15, lambda_s=0.0, tau_samples=200, lambda_k=1.0)
    res = evaluate(bundle, ecog_test, flex_test, sl.test_state)

    est = np.array(res["estimated"])
    base = np.array(res["baseline"])
    ora = np.array(res["oracle"])
    est_r2 = np.array(res["estimated_r2"])
    est_cr2 = np.array(res["estimated_calibrated_r2"])
    base_r2 = np.array(res["baseline_r2"])
    ora_r2 = np.array(res["oracle_r2"])
    base_cr2 = np.array(res["baseline_calibrated_r2"])
    ora_cr2 = np.array(res["oracle_calibrated_r2"])
    state_acc = float(np.mean(res["pred_state_aligned"] == res["state_true_aligned"]))

    print(f"  {'finger':8s} {'baseline':>10s} {'oracle':>10s} {'estimated':>10s}")
    for j, name in enumerate(C.FINGER_NAMES):
        print(f"  {name:8s} {base[j]:10.3f} {ora[j]:10.3f} {est[j]:10.3f}")
    print(f"  state classification accuracy: {state_acc:.3f}")
    print(f"  official_r(estimated, 4 fingers) = {official_r(est):.3f}  "
          f"avg_r(5 fingers) = {float(np.nanmean(est)):.3f}")

    result = {
        "method": "switching_lm",
        "subject": subject,
        "per_finger_estimated": {C.FINGER_NAMES[j]: float(est[j]) for j in range(5)},
        "per_finger_baseline": {C.FINGER_NAMES[j]: float(base[j]) for j in range(5)},
        "per_finger_oracle": {C.FINGER_NAMES[j]: float(ora[j]) for j in range(5)},
        "per_finger_estimated_r2": {C.FINGER_NAMES[j]: float(est_r2[j]) for j in range(5)},
        "per_finger_estimated_calibrated_r2": {C.FINGER_NAMES[j]: float(est_cr2[j]) for j in range(5)},
        "avg_r_estimated": round(float(np.nanmean(est)), 3),
        "official_r_estimated": round(official_r(est), 3),
        "official_r2_estimated": official_r(est_r2),
        "official_gap_estimated": official_r(est_cr2 - est_r2),
        "tier_baseline": tier_summary(base, base_r2, base_cr2),
        "tier_oracle": tier_summary(ora, ora_r2, ora_cr2),
        "tier_estimated": tier_summary(est, est_r2, est_cr2),
        "state_classification_accuracy": round(state_acc, 3),
        "seconds": round(time.time() - t_start, 1),
    }
    return result


if __name__ == "__main__":
    subs = sys.argv[1:] if len(sys.argv) > 1 else ["jc", "wc"]
    all_res = [run_subject(s) for s in subs]
    out_path = P.OUT_DIR / "switching_lm_miller_results.json"
    out_path.write_text(json.dumps(all_res, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_path}")
