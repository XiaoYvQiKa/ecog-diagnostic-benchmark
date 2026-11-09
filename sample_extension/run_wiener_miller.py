# -*- coding: utf-8 -*-
"""Wiener filter run on the added Miller subjects (jc, wc).

Reuses the AM feature extraction, forward selection, and ridge regression from
wiener/code/, replacing only the data source:
  - ECoG : C.PREP_DIR/{subj}/ecog_{train,test}.npy  (time, ch) @1000 Hz
  - target : C.PREP_DIR/{subj}/flex_{train,test}.npy  (time, 5) @25 Hz de-based

Protocol identical to the original BCI IV-4 Wiener run: three AM bands
(sub 1-60 / gamma 60-100 / fastgamma 100-200 Hz, 40 ms squared-sum), a
train-only relative-quantile electrode screen (95th percentile), forward
selection (inner 3/5), full-train refit, and test Pearson r. official_r is the
four-finger mean excluding the ring finger.

Usage:
    python run_wiener_miller.py jc wc
"""
import json
import sys
import time

import numpy as np

import paths as C

sys.path.insert(0, str(C.WIENER_DIR))
from features import extract_am_features, align  # noqa: E402
from memory_stack import build_xy  # noqa: E402
from regressor import fit_wiener, pearson_r, r2_score, calibrated_r2  # noqa: E402
from selection import forward_select  # noqa: E402


def electrode_qc_mask(Xtr, names, pct=95.0):
    """Train-only relative-quantile electrode screen (identical to the original)."""
    Xl = np.log1p(np.maximum(Xtr, 0.0))
    by_ch = {}
    for j in range(Xtr.shape[1]):
        col = Xl[:, j]
        mean, std = col.mean(), col.std()
        kurt = 0.0 if std == 0 else float(((col - mean) ** 4).mean() / std ** 4 - 3.0)
        p50, p99 = np.percentile(col, [50, 99])
        max_ratio = float((col.max() - p50) / max(p99 - p50, 1e-9))
        _, ch = names[j]
        entry = by_ch.setdefault(ch, {"kurt": -np.inf, "ratio": -np.inf})
        entry["kurt"] = max(entry["kurt"], kurt)
        entry["ratio"] = max(entry["ratio"], max_ratio)
    kurt_thr = np.percentile([e["kurt"] for e in by_ch.values()], pct)
    ratio_thr = np.percentile([e["ratio"] for e in by_ch.values()], pct)
    flagged = {ch for ch, e in by_ch.items()
               if e["kurt"] >= kurt_thr or e["ratio"] >= ratio_thr}
    mask = np.array([ch not in flagged for _, ch in names])
    return mask, sorted(flagged)


def run_subject(subject: str) -> dict:
    t_start = time.time()
    ecog_train = np.load(C.PREP_DIR / subject / "ecog_train.npy").astype(np.float64)  # (T, ch)
    ecog_test = np.load(C.PREP_DIR / subject / "ecog_test.npy").astype(np.float64)
    flex_train = np.load(C.PREP_DIR / subject / "flex_train.npy").astype(np.float64)  # (T25, 5)
    flex_test = np.load(C.PREP_DIR / subject / "flex_test.npy").astype(np.float64)

    print(f"\n{'='*70}")
    print(f"Wiener on {subject}: ecog train {ecog_train.shape}, flex train {flex_train.shape}")

    t0 = time.time()
    Xtr, names = extract_am_features(ecog_train, standardize=False)
    ytr = flex_train
    Xtr, ytr = align(Xtr, ytr)
    Xte, _ = extract_am_features(ecog_test, standardize=False)
    yte = flex_test
    Xte, yte = align(Xte, yte)
    print(f"  AM features: train {Xtr.shape}, test {Xte.shape}  ({time.time()-t0:.1f}s)")

    mask, flagged = electrode_qc_mask(Xtr, names, pct=95.0)
    Xtr, Xte = Xtr[:, mask], Xte[:, mask]
    names = [n for n, k in zip(names, mask) if k]
    print(f"  electrode QC (train-only, 95th pct): flagged {len(flagged)} channels")

    rows = []
    for f in range(5):
        t1 = time.time()
        sel = forward_select(Xtr, ytr[:, f], feature_names=names)
        Dtr, dtr = build_xy(Xtr, ytr[:, f:f + 1], columns=sel.selected)
        fit = fit_wiener(Dtr, dtr)
        Dte, dte = build_xy(Xte, yte[:, f:f + 1], columns=sel.selected)
        pred = fit.predict(Dte)
        r = float(pearson_r(pred, dte))
        r2 = float(r2_score(pred, dte))
        cr2 = float(calibrated_r2(pred, dte))
        rows.append({"finger": ["thumb", "index", "middle", "ring", "little"][f], "r": r,
                     "r2": r2, "calibrated_r2": cr2,
                     "n_selected": sel.n_selected, "seconds": time.time() - t1})
        print(f"  {rows[-1]['finger']:<7s} r={r:+.3f} r2={r2:+.3f}  ({sel.n_selected} feat, {rows[-1]['seconds']:.0f}s)")

    rs = [r["r"] for r in rows]
    official_r = float(np.mean([rs[i] for i in (0, 1, 2, 4)]))
    avg_r = float(np.mean(rs))
    official_r2 = float(np.mean([rows[i]["r2"] for i in (0, 1, 2, 4)]))
    official_gap = float(np.mean([rows[i]["calibrated_r2"] - rows[i]["r2"] for i in (0, 1, 2, 4)]))
    result = {
        "method": "wiener",
        "subject": subject,
        "per_finger_r": {r["finger"]: r["r"] for r in rows},
        "per_finger_r2": {r["finger"]: r["r2"] for r in rows},
        "per_finger_calibrated_r2": {r["finger"]: r["calibrated_r2"] for r in rows},
        "avg_r": round(avg_r, 3),
        "official_r": round(official_r, 3),
        "official_r2": official_r2,
        "official_gap": official_gap,
        "flagged_channels_1based": [c + 1 for c in flagged],
        "seconds": round(time.time() - t_start, 1),
    }
    print(f"  avg_r(5)={avg_r:.3f}  official_r(4)={official_r:.3f}  total={result['seconds']:.1f}s")
    return result


if __name__ == "__main__":
    subs = sys.argv[1:] if len(sys.argv) > 1 else ["jc", "wc"]
    all_res = [run_subject(s) for s in subs]
    out_path = C.OUT_DIR / "wiener_miller_results.json"
    out_path.write_text(json.dumps(all_res, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_path}")
