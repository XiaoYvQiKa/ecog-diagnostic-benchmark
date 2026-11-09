"""
Compute Pearson r / R2 / calibrated_r2 / calibration_gap on the test set for
three subjects x three seeds (9 runs total), per finger, and aggregate the
three seeds per subject as mean +/- std.

Data source: res_npy/prediction_{subject}_seed{seed}.npy,
             res_npy/true_{subject}_seed{seed}.npy
(generated automatically after the 3-seed test run; no retraining or
re-inference).

The r2_score / calibrated_r2 helpers are identical to the single-seed
compute_r2_fingerflex.py. The only changes here are the seed-suffixed filename
parsing and the multi-seed aggregation (mean +/- std).
"""

import numpy as np
import pandas as pd

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
SUBJECTS = ["sub1", "sub2", "sub3"]
SEEDS = [0, 1, 2]
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
    """Coefficient of determination, 1 - SSE/SST."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    n = min(len(pred), len(target))
    pred, target = pred[:n], target[:n]
    sse = float(((target - pred) ** 2).sum())
    sst = float(((target - target.mean()) ** 2).sum())
    return float("nan") if sst == 0 else 1.0 - sse / sst


def calibrated_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """R2 after the single best affine rescaling of the prediction (= r^2)."""
    r = pearson_r(pred, target)
    return float("nan") if not np.isfinite(r) else r * r


def load_one(subject: str, seed: int):
    """Load the prediction/true arrays for one (subject, seed) pair, normalize shape, and return a per-finger dict"""
    pred_path = f"{RES_DIR}/prediction_{subject}_seed{seed}.npy"
    true_path = f"{RES_DIR}/true_{subject}_seed{seed}.npy"

    try:
        pred = np.load(pred_path)
        true = np.load(true_path)
    except FileNotFoundError as e:
        print(f"[skip] {subject} seed{seed}: file not found {e.filename}")
        return None

    # expected shape (5, T): 5 fingers x time; if (T, 5), transpose
    if pred.shape[0] != 5 and pred.shape[-1] == 5:
        pred = pred.T
    if true.shape[0] != 5 and true.shape[-1] == 5:
        true = true.T

    print(f"{subject} seed{seed}: pred shape={pred.shape}, true shape={true.shape}")

    finger_r, finger_r2, finger_cal_r2 = [], [], []
    for i in range(5):
        p, t = pred[i], true[i]
        finger_r.append(pearson_r(p, t))
        finger_r2.append(r2_score(p, t))
        finger_cal_r2.append(calibrated_r2(p, t))

    return {"r": finger_r, "r2": finger_r2, "cal_r2": finger_cal_r2}


def main():
    raw_rows = []       # one row per (subject, seed, finger): raw detail
    agg_rows = []       # one row per (subject, finger): 3-seed mean +/- std

    for subject in SUBJECTS:
        # per_seed[seed] = {"r": [...5 fingers], "r2": [...], "cal_r2": [...]}
        per_seed = {}
        for seed in SEEDS:
            result = load_one(subject, seed)
            if result is not None:
                per_seed[seed] = result

        if len(per_seed) == 0:
            print(f"[warning] {subject} has no usable seed data; skipping")
            continue

        # ---- raw detail: one row per seed per finger ----
        for seed, result in per_seed.items():
            for i, finger in enumerate(FINGER_NAMES):
                r, r2, cal_r2 = result["r"][i], result["r2"][i], result["cal_r2"][i]
                raw_rows.append({
                    "subject": subject, "seed": seed, "finger": finger,
                    "pearson_r": round(r, 4),
                    "r2": round(r2, 4),
                    "calibrated_r2": round(cal_r2, 4),
                    "calibration_gap": round(cal_r2 - r2, 4),
                })

        # ---- per-finger 3-seed aggregation: mean +/- std ----
        n_seeds = len(per_seed)
        for i, finger in enumerate(FINGER_NAMES):
            r_vals = [per_seed[s]["r"][i] for s in per_seed]
            r2_vals = [per_seed[s]["r2"][i] for s in per_seed]
            cal_vals = [per_seed[s]["cal_r2"][i] for s in per_seed]
            gap_vals = [c - r2 for c, r2 in zip(cal_vals, r2_vals)]

            agg_rows.append({
                "subject": subject, "finger": finger, "n_seeds": n_seeds,
                "pearson_r_mean": round(np.mean(r_vals), 4),
                "pearson_r_std": round(np.std(r_vals, ddof=1), 4) if n_seeds > 1 else "",
                "r2_mean": round(np.mean(r2_vals), 4),
                "r2_std": round(np.std(r2_vals, ddof=1), 4) if n_seeds > 1 else "",
                "calibrated_r2_mean": round(np.mean(cal_vals), 4),
                "calibrated_r2_std": round(np.std(cal_vals, ddof=1), 4) if n_seeds > 1 else "",
                "calibration_gap_mean": round(np.mean(gap_vals), 4),
                "calibration_gap_std": round(np.std(gap_vals, ddof=1), 4) if n_seeds > 1 else "",
            })

        # ---- subject level: 3-seed mean +/- std of avg_r (5 fingers) and official_r (4 fingers, Ring excluded) ----
        ring_idx = FINGER_NAMES.index("Ring")
        avg_r_per_seed, off_r_per_seed = [], []
        avg_r2_per_seed, off_r2_per_seed = [], []
        avg_cal_per_seed, off_cal_per_seed = [], []

        for seed, result in per_seed.items():
            avg_r_per_seed.append(np.mean(result["r"]))
            avg_r2_per_seed.append(np.mean(result["r2"]))
            avg_cal_per_seed.append(np.mean(result["cal_r2"]))

            off_r = [v for i, v in enumerate(result["r"]) if i != ring_idx]
            off_r2 = [v for i, v in enumerate(result["r2"]) if i != ring_idx]
            off_cal = [v for i, v in enumerate(result["cal_r2"]) if i != ring_idx]
            off_r_per_seed.append(np.mean(off_r))
            off_r2_per_seed.append(np.mean(off_r2))
            off_cal_per_seed.append(np.mean(off_cal))

        def mean_std(vals):
            m = round(float(np.mean(vals)), 4)
            s = round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else ""
            return m, s

        for label, r_list, r2_list, cal_list in [
            ("AVG(5 fingers, avg_r)", avg_r_per_seed, avg_r2_per_seed, avg_cal_per_seed),
            ("OFFICIAL(4 fingers, Ring excluded)", off_r_per_seed, off_r2_per_seed, off_cal_per_seed),
        ]:
            r_m, r_s = mean_std(r_list)
            r2_m, r2_s = mean_std(r2_list)
            cal_m, cal_s = mean_std(cal_list)
            gap_list = [c - r2 for c, r2 in zip(cal_list, r2_list)]
            gap_m, gap_s = mean_std(gap_list)
            agg_rows.append({
                "subject": subject, "finger": label, "n_seeds": n_seeds,
                "pearson_r_mean": r_m, "pearson_r_std": r_s,
                "r2_mean": r2_m, "r2_std": r2_s,
                "calibrated_r2_mean": cal_m, "calibrated_r2_std": cal_s,
                "calibration_gap_mean": gap_m, "calibration_gap_std": gap_s,
            })

    raw_df = pd.DataFrame(raw_rows)
    agg_df = pd.DataFrame(agg_rows)

    print("\n===== raw detail (one row per seed per finger) =====")
    print(raw_df.to_string(index=False))

    print("\n===== per-finger + subject-level 3-seed mean +/- std summary =====")
    print(agg_df.to_string(index=False))

    raw_df.to_csv("fingerflex_r2_3seed_raw.csv", index=False, encoding="utf-8-sig")
    agg_df.to_csv("fingerflex_r2_3seed_agg.csv", index=False, encoding="utf-8-sig")
    print("\nSaved: fingerflex_r2_3seed_raw.csv (raw detail, 27 rows = 3 subjects x 3 seeds x 5 fingers + summary rows)")
    print("Saved: fingerflex_r2_3seed_agg.csv (aggregated table, mean +/- std, aligned with the benchmark spec)")


if __name__ == "__main__":
    main()
