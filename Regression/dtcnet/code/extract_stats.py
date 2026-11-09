"""Extract the 15 per-finger r values for the regression-side significance test.

The reported regression statistics use two layers:
  Layer 1: subject-level direction consistency (official_r, sign of per-subject difference)
  Layer 2: per-finger Wilcoxon signed-rank test (5 fingers × 3 subjects = 15 paired raw r values)

This script reads results.json and prints:
  1. the 15 raw r values (order: Thumb/Index/Middle/Ring/Little × sub1/sub2/sub3)
  2. per-subject official_r (4-finger mean excluding Ring): Layer 1
  3. per-subject avg_r (5-finger mean): auxiliary

Usage:
  python extract_stats.py --input ../results/main/results.json
"""
import json, argparse
import numpy as np

FINGER_ORDER = ['thumb', 'index', 'middle', 'ring', 'little']
SUBJECT_ORDER = ['sub1', 'sub2', 'sub3']
FINGER_LABELS = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../results/main/results.json")
    a = ap.parse_args()

    with open(a.input) as f:
        results = json.load(f)

    # 1. 15 raw r values (strictly aligned pairing order)
    r15 = []
    for sub in SUBJECT_ORDER:
        if sub not in results:
            print(f"[WARN] {sub} missing from results file")
            continue
        m = results[sub]['mean']
        for f in FINGER_ORDER:
            r15.append(m[f])

    print("=" * 60)
    print("DTCNet 15 raw r values (5 fingers × 3 subjects)")
    print("Order: Thumb/Index/Middle/Ring/Little × sub1/sub2/sub3")
    print("=" * 60)
    if len(r15) == 15:
        print("r15 = [")
        for sub in SUBJECT_ORDER:
            m = results[sub]['mean']
            vals = [m[f] for f in FINGER_ORDER]
            print(f"  # {sub}: " + ", ".join(f"{v:.4f}" for v in vals))
        print("]")
        print(f"\n# paste-ready 15-vector for the paired Wilcoxon test:")
        print("r15 = [" + ", ".join(f"{v:.4f}" for v in r15) + "]")
    else:
        print(f"expected 15 values, found {len(r15)}")

    # 2. Layer 1: official_r + avg_r per subject
    print("\n" + "=" * 60)
    print("Layer 1: subject-level official_r / avg_r")
    print("=" * 60)
    for sub in SUBJECT_ORDER:
        if sub not in results:
            continue
        m = results[sub]['mean']
        print(f"{sub}: official_r={m['official_r']:.4f}  avg_r={m['avg_r']:.4f}")

    # 3. mean
    if len(r15) == 15:
        off = [results[s]['mean']['official_r'] for s in SUBJECT_ORDER if s in results]
        avg = [results[s]['mean']['avg_r'] for s in SUBJECT_ORDER if s in results]
        print(f"\nMean: official_r={np.mean(off):.4f}  avg_r={np.mean(avg):.4f}")


if __name__ == "__main__":
    main()
