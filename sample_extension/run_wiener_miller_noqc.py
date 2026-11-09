# -*- coding: utf-8 -*-
"""Wiener no-QC ablation on the Miller subjects (jc, wc).

Identical to run_wiener_miller.py except that the train-only relative-quantile
electrode screen is skipped, so all channels enter forward selection. This
quantifies the performance impact of the channels that the relative-quantile
screen excludes, mirroring the 11-channel ablation in Appendix F.1.

Usage:
    python run_wiener_miller_noqc.py jc wc
"""
import json
import sys
import time

import numpy as np

import paths as C

sys.path.insert(0, str(C.WIENER_DIR))
from features import extract_am_features, align  # noqa: E402
from memory_stack import build_xy  # noqa: E402
from regressor import fit_wiener, pearson_r  # noqa: E402
from selection import forward_select  # noqa: E402


def run_subject_noqc(subject: str) -> dict:
    t_start = time.time()
    ecog_train = np.load(C.PREP_DIR / subject / "ecog_train.npy").astype(np.float64)
    ecog_test = np.load(C.PREP_DIR / subject / "ecog_test.npy").astype(np.float64)
    flex_train = np.load(C.PREP_DIR / subject / "flex_train.npy").astype(np.float64)
    flex_test = np.load(C.PREP_DIR / subject / "flex_test.npy").astype(np.float64)

    print(f"\n{'='*70}")
    print(f"Wiener (no-QC) on {subject}: ecog train {ecog_train.shape}")

    Xtr, names = extract_am_features(ecog_train, standardize=False)
    ytr = flex_train
    Xtr, ytr = align(Xtr, ytr)
    Xte, _ = extract_am_features(ecog_test, standardize=False)
    yte = flex_test
    Xte, yte = align(Xte, yte)
    print(f"  AM features: train {Xtr.shape}, test {Xte.shape}  (no channel screen)")

    rows = []
    for f in range(5):
        sel = forward_select(Xtr, ytr[:, f], feature_names=names)
        Dtr, dtr = build_xy(Xtr, ytr[:, f:f + 1], columns=sel.selected)
        fit = fit_wiener(Dtr, dtr)
        Dte, dte = build_xy(Xte, yte[:, f:f + 1], columns=sel.selected)
        pred = fit.predict(Dte)
        r = float(pearson_r(pred, dte))
        rows.append({"finger": ["thumb", "index", "middle", "ring", "little"][f], "r": r})
        print(f"  {rows[-1]['finger']:<7s} r={r:+.3f}")

    rs = [r["r"] for r in rows]
    official_r = float(np.mean([rs[i] for i in (0, 1, 2, 4)]))
    avg_r = float(np.mean(rs))
    result = {
        "method": "wiener_noqc",
        "subject": subject,
        "per_finger_r": {r["finger"]: r["r"] for r in rows},
        "avg_r": round(avg_r, 3),
        "official_r": round(official_r, 3),
        "seconds": round(time.time() - t_start, 1),
    }
    print(f"  avg_r(5)={avg_r:.3f}  official_r(4)={official_r:.3f}  total={result['seconds']:.1f}s")
    return result


if __name__ == "__main__":
    subs = sys.argv[1:] if len(sys.argv) > 1 else ["jc", "wc"]
    all_res = [run_subject_noqc(s) for s in subs]
    out_path = C.OUT_DIR / "wiener_miller_noqc_results.json"
    out_path.write_text(json.dumps(all_res, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_path}")
