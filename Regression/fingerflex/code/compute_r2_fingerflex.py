"""
Compute R2 / calibrated_r2 / Pearson r on the test set for the three FingerFlex
subjects. Data source: res_npy/prediction_subX.npy, res_npy/true_subX.npy
(generated automatically after the test run; no retraining or re-inference).

The r2_score / calibrated_r2 helpers are reused verbatim from the Wiener
reproduction's regressor.py.
"""

import numpy as np
import pandas as pd

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
SUBJECTS = ["sub1", "sub2", "sub3"]
RES_DIR = "res_npy"  # relative path; run from the FingerFlex project root


def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    """Standard Pearson correlation coefficient"""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    n = min(len(pred), len(target))
    pred, target = pred[:n], target[:n]
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    denom = np.sqrt((pred_c ** 2).sum() * (target_c ** 2).sum())
    return float("nan") if denom == 0 else float((pred_c * target_c).sum() / denom)


def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Coefficient of determination, 1 - SSE/SST.
    Reported alongside Pearson r because they answer different questions and
    can disagree sharply:
      * r  asks whether the prediction has the right SHAPE...
      * R2 asks whether the prediction has the right shape AND the right
        scale and offset...
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    n = min(len(pred), len(target))
    pred, target = pred[:n], target[:n]
    sse = float(((target - pred) ** 2).sum())
    sst = float(((target - target.mean()) ** 2).sum())
    return float("nan") if sst == 0 else 1.0 - sse / sst


def calibrated_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """
    R2 after the single best affine rescaling of the prediction, which equals
    r^2 exactly. The gap between this and the raw R2 is precisely the amount
    of performance lost to miscalibration rather than to bad decoding.
    """
    r = pearson_r(pred, target)
    return float("nan") if not np.isfinite(r) else r * r


def main():
    rows = []

    for subject in SUBJECTS:
        pred_path = f"{RES_DIR}/prediction_{subject}.npy"
        true_path = f"{RES_DIR}/true_{subject}.npy"

        try:
            pred = np.load(pred_path)
            true = np.load(true_path)
        except FileNotFoundError as e:
            print(f"[skip] {subject}: file not found {e.filename}")
            continue

        # expected shape (5, T): 5 fingers x time; if (T, 5), transpose
        if pred.shape[0] != 5 and pred.shape[-1] == 5:
            pred = pred.T
        if true.shape[0] != 5 and true.shape[-1] == 5:
            true = true.T

        print(f"{subject}: pred shape={pred.shape}, true shape={true.shape}")

        finger_r2, finger_cal_r2, finger_r = [], [], []

        for i, finger in enumerate(FINGER_NAMES):
            p, t = pred[i], true[i]
            r = pearson_r(p, t)
            r2 = r2_score(p, t)
            cal_r2 = calibrated_r2(p, t)

            finger_r.append(r)
            finger_r2.append(r2)
            finger_cal_r2.append(cal_r2)

            rows.append({
                "subject": subject,
                "finger": finger,
                "pearson_r": round(r, 4),
                "r2": round(r2, 4),
                "calibrated_r2": round(cal_r2, 4),
                "calibration_gap": round(cal_r2 - r2, 4),  # larger means scale/offset misaligned while the shape itself is correct
            })

        # subject-level summary (5-finger mean = avg_r)
        rows.append({
            "subject": subject,
            "finger": "AVG(5 fingers, avg_r)",
            "pearson_r": round(np.mean(finger_r), 4),
            "r2": round(np.mean(finger_r2), 4),
            "calibrated_r2": round(np.mean(finger_cal_r2), 4),
            "calibration_gap": round(np.mean(finger_cal_r2) - np.mean(finger_r2), 4),
        })

        # official_r: 4-finger mean excluding Ring, matching the Wiener/Switching LM official competition convention
        ring_idx = FINGER_NAMES.index("Ring")
        official_r_list = [r for i, r in enumerate(finger_r) if i != ring_idx]
        official_r2_list = [r2 for i, r2 in enumerate(finger_r2) if i != ring_idx]
        official_cal_r2_list = [c for i, c in enumerate(finger_cal_r2) if i != ring_idx]

        rows.append({
            "subject": subject,
            "finger": "OFFICIAL(4 fingers, Ring excluded)",
            "pearson_r": round(np.mean(official_r_list), 4),
            "r2": round(np.mean(official_r2_list), 4),
            "calibrated_r2": round(np.mean(official_cal_r2_list), 4),
            "calibration_gap": round(np.mean(official_cal_r2_list) - np.mean(official_r2_list), 4),
        })
        rows.append({"subject": "", "finger": "", "pearson_r": "", "r2": "", "calibrated_r2": "", "calibration_gap": ""})

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    # concise summary: keep only the avg_r and official_r rows for the benchmark comparison table
    summary_rows = df[df["finger"].isin(["AVG(5 fingers, avg_r)", "OFFICIAL(4 fingers, Ring excluded)"])]
    print("\n=== avg_r / official_r summary (aligned with benchmark spec section 5) ===")
    print(summary_rows.to_string(index=False))

    df.to_csv("fingerflex_r2_results.csv", index=False, encoding="utf-8-sig")
    print("\nSaved: fingerflex_r2_results.csv")


if __name__ == "__main__":
    main()
