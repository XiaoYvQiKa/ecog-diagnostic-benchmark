# -*- coding: utf-8 -*-
"""Cross-validation of the 11 Wiener relative-quantile extra channels
against the CLASSIFICATION-track absolute thresholds.

For each subject, load the raw ECoG and compute, per channel:
  - train excess kurtosis (Fisher)
  - train RMS as a fraction of the median train RMS (dead-channel ratio)
  - test/train RMS ratio
then apply the classification track's three absolute rules
  (kurtosis > 50, RMS < median/10, test/train RMS ratio > 10)
to every channel. Reports, for the 4 cross-verified channels and the
11 extra channels separately, which absolute rules (if any) each channel
violates, so we can state whether the 11 extras are physically broken
(absolute) or merely heavy-tailed relative outliers (relative-quantile).

Indices are 0-based physical; prints 1-based for the paper.
"""
import os

import numpy as np
from scipy.io import loadmat
from scipy import stats

DATA_DIR = os.environ.get("BCI_DATA_DIR", os.path.expanduser("~/BCI_data"))

CROSS_VERIFIED = {1: [54], 2: [20, 37], 3: [49]}
EXTRA_11 = {
    1: [4, 25, 28, 50],
    2: [22, 30, 31],
    3: [14, 22, 43, 44],
}

KURT_ABS = 50.0
RMS_RATIO = 10.0


def analyze(subject):
    mat = loadmat(os.path.join(DATA_DIR, f"sub{subject}_comp.mat"))
    train = np.asarray(mat["train_data"], dtype=np.float64).T   # (n_ch, T_train)
    test = np.asarray(mat["test_data"], dtype=np.float64).T     # (n_ch, T_test)
    n_ch = train.shape[0]

    tr_kurt = stats.kurtosis(train, axis=1, fisher=True)
    tr_rms = np.sqrt((train ** 2).mean(axis=1))
    te_rms = np.sqrt((test ** 2).mean(axis=1))
    median_rms = np.median(tr_rms)
    dead_ratio = tr_rms / (median_rms + 1e-12)   # < 0.1 -> dead
    ratio = te_rms / (tr_rms + 1e-12)

    def rules(c):
        flags = []
        if tr_kurt[c] > KURT_ABS:
            flags.append("KURT")
        if dead_ratio[c] < 0.1:
            flags.append("DEAD")
        if ratio[c] > RMS_RATIO:
            flags.append("RATIO")
        return flags

    print(f"\n===== Subject {subject}  (n_ch={n_ch}) =====")
    print(f"{'ch(1b)':>7} {'group':>7} {'tr_kurt':>9} {'RMS/med':>9} {'te/tr':>9} {'classif-flags':>14}")
    for grp, chans in (("cross4", CROSS_VERIFIED[subject]), ("extra", EXTRA_11[subject])):
        for c in chans:
            fl = ",".join(rules(c)) if rules(c) else "none"
            print(f"{c+1:>7} {grp:>7} {tr_kurt[c]:>9.2f} {dead_ratio[c]:>9.3f} "
                  f"{ratio[c]:>9.2f} {fl:>14}")

    # summary: how many of the 11 extras violate ANY absolute rule
    extras = EXTRA_11[subject]
    flagged_extras = [c for c in extras if rules(c)]
    cross = CROSS_VERIFIED[subject]
    flagged_cross = [c for c in cross if rules(c)]
    print(f"  cross-verified channels violating absolute rules: "
          f"{[c+1 for c in flagged_cross]}")
    print(f"  EXTRA channels violating absolute rules:         "
          f"{[c+1 for c in flagged_extras]}  "
          f"({len(flagged_extras)}/{len(extras)})")


for s in (1, 2, 3):
    analyze(s)
