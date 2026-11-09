# -*- coding: utf-8 -*-
"""Unified data preparation for the two added Miller subjects (jc, wc).

Produces per-subject train/test intermediates (raw-unit ECoG, de-based flex,
glove-threshold labels) shared by the four regression methods and the four
classification methods.

Outputs (C.OUT_DIR/{subject}/):
    ecog_train.npy    (n_keep, T_train)  @1000 Hz, amplifier units (not z-scored)
    ecog_test.npy     (n_keep, T_test)   @1000 Hz
    flex_train.npy    (5, T_train_25)    @25 Hz, de-based relative flexion
    flex_test.npy     (5, T_test_25)
    labels_train.npy  (T_train_25,)      0=rest, 1-5=finger (glove threshold)
    labels_test.npy   (T_test_25,)
    meta.json         trial bounds, bad channels, baseline, max angles

Protocol (aligned with the original BCI IV-4 study):
    1. Split: cue 0 -> 1-5 rising edges mark trial onsets; the first
       ceil(2/3*N) trials form train and the rest form test, a 2:1 trial
       ratio matching the original 400 s / 200 s 2:1 ratio.
    2. Bad channels: three absolute thresholds (train kurtosis, train RMS,
       test/train RMS ratio). All three quantities are scale-invariant.
    3. De-baselining: flex is a 25 Hz zero-order-held uint16 absolute position
       with a non-zero rest offset. Subtract the per-finger median over the
       train rest segments (cue == 0) to obtain relative flexion comparable to
       the BCI train_dg (rest near 0, flexion positive). Train and test share
       the same baseline and the same max angles.
    4. Labels: generate_labels (Yao & Shoaran 2019; 10% of max plus a 5-tap
       median filter at 25 Hz); max angles come from the train relative flexion.
    5. Units: ECoG stays in amplifier units (same as the BCI train_data); each
       method z-scores internally, absorbing session gain differences.

Usage:
    python prep_miller.py jc wc
"""
import json
import sys

import numpy as np
import scipy.io as sio
from scipy import stats
from scipy.signal import medfilt

import paths as C

FS_ECOG = 1000
FS_GLOVE = 25
DOWNSAMPLE = FS_ECOG // FS_GLOVE          # 40
KURT_ABS = 50.0
DEAD_RATIO = 0.1
RMS_RATIO = 10.0
ANGLE_THRESHOLD_RATIO = 0.10
MEDIAN_FILTER_KERNEL = 5
FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]


def load_raw(subject: str):
    """Read {subject}_fingerflex.mat and {subject}_stim.mat."""
    m = sio.loadmat(str(C.MILLER_DIR / f"{subject}_fingerflex.mat"))
    data = np.asarray(m["data"], dtype=np.float64)          # (T, ch) @1000Hz amplifier units
    flex = np.asarray(m["flex"], dtype=np.float64)          # (T, 5)  @25Hz ZOH uint16
    cue = np.asarray(m["cue"], dtype=np.float64).ravel()    # (T,)
    stim = None
    stim_path = C.MILLER_DIR / f"{subject}_stim.mat"
    if stim_path.exists():
        sm = sio.loadmat(str(stim_path))
        stim = np.asarray(sm["stim"], dtype=np.float64).ravel()
    regions = None
    if "elec_regions" in m:
        regions = np.asarray(m["elec_regions"]).ravel().astype(int)
    return data, flex, cue, stim, regions


def extract_trial_starts(cue: np.ndarray) -> np.ndarray:
    """Rising edges of cue (0 -> 1-5) as trial onset indices at 1000 Hz."""
    cue_int = np.round(cue).astype(int)
    starts = np.where((cue_int[1:] >= 1) & (cue_int[:-1] == 0))[0] + 1
    return starts


def generate_labels(dg, max_angles=None, threshold_ratio=ANGLE_THRESHOLD_RATIO,
                    median_kernel=MEDIAN_FILTER_KERNEL):
    """Glove-threshold discrete labels (Yao & Shoaran 2019). dg: (n_time, 5)."""
    n_time = len(dg)
    if max_angles is None:
        max_angles = dg.max(axis=0)
    thresholds = max_angles * threshold_ratio          # (5,)
    active = dg > thresholds[np.newaxis, :]            # (n_time, 5)
    n_active = active.sum(axis=1)
    labels = np.zeros(n_time, dtype=np.int32)
    single = n_active == 1
    labels[single] = np.argmax(active[single], axis=1) + 1
    multi = n_active > 1
    if multi.any():
        labels[multi] = np.argmax(dg[multi], axis=1) + 1
    labels = medfilt(labels, kernel_size=median_kernel).astype(np.int32)
    return labels


def label_distribution(labels):
    """Return {label: fraction} including rest (0)."""
    vals, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    return {int(v): round(float(c / total), 4) for v, c in zip(vals, counts)}


def prepare_subject(subject: str) -> dict:
    data, flex, cue, stim, regions = load_raw(subject)
    T, n_ch = data.shape

    # 1. Trial bounds and 2:1 split.
    starts = extract_trial_starts(cue)
    n_trials = len(starts)
    n_train_trials = int(np.ceil(n_trials * 2 / 3))
    train_start = int(starts[0])
    train_end = int(starts[n_train_trials]) if n_train_trials < n_trials else T

    # 2. Bad-channel QC (train kurtosis/RMS, test/train RMS ratio).
    tr = data[train_start:train_end]
    te = data[train_end:]
    tr_kurt = stats.kurtosis(tr, axis=0, fisher=True)
    tr_rms = np.sqrt((tr ** 2).mean(axis=0))
    te_rms = np.sqrt((te ** 2).mean(axis=0))
    med_rms = np.median(tr_rms)
    bad = set()
    bad |= set(np.where(tr_kurt > KURT_ABS)[0].tolist())
    bad |= set(np.where(tr_rms < med_rms * DEAD_RATIO)[0].tolist())
    bad |= set(np.where(te_rms / (tr_rms + 1e-12) > RMS_RATIO)[0].tolist())
    bad = sorted(bad)
    keep = [c for c in range(n_ch) if c not in bad]

    # 3. Split at 1000 Hz.
    ecog_train = data[train_start:train_end][:, keep]     # (T_tr, n_keep)
    ecog_test = data[train_end:][:, keep]                 # (T_te, n_keep)
    flex_train_1k = flex[train_start:train_end]           # (T_tr, 5)
    flex_test_1k = flex[train_end:]                       # (T_te, 5)
    cue_train_1k = cue[train_start:train_end]
    cue_test_1k = cue[train_end:]

    # 4. Downsample flex to 25 Hz and de-baseline with the train rest median.
    flex_train_25 = flex_train_1k[::DOWNSAMPLE]           # (T_tr_25, 5)
    flex_test_25 = flex_test_1k[::DOWNSAMPLE]
    cue_train_25 = cue_train_1k[::DOWNSAMPLE]
    cue_test_25 = cue_test_1k[::DOWNSAMPLE]

    rest_train = cue_train_25 == 0
    baseline = np.median(flex_train_25[rest_train], axis=0)   # (5,)
    flex_train_rel = flex_train_25 - baseline
    flex_test_rel = flex_test_25 - baseline

    # 5. Glove-threshold labels (train max sets the threshold, shared by test).
    max_angles = flex_train_rel.max(axis=0)
    labels_train = generate_labels(flex_train_rel, max_angles)
    labels_test = generate_labels(flex_test_rel, max_angles)

    # 6. Stim cross-check (diagnostic, non-blocking).
    stim_agree = None
    if stim is not None:
        stim_train_25 = stim[train_start:train_end][::DOWNSAMPLE]
        stim_test_25 = stim[train_end:][::DOWNSAMPLE]
        for name, lab, st in (("train", labels_train, stim_train_25),
                              ("test", labels_test, stim_test_25)):
            valid = st >= 0                                   # exclude -1/-2
            agree = float(np.mean(lab[valid] == np.round(st[valid]).astype(int)))
            if name == "train":
                stim_agree = {"train": agree}
            else:
                stim_agree["test"] = agree

    # 7. Save.
    subj_dir = C.OUT_DIR / subject
    subj_dir.mkdir(parents=True, exist_ok=True)
    np.save(subj_dir / "ecog_train.npy", ecog_train)          # float64, faithful to the Wiener path
    np.save(subj_dir / "ecog_test.npy", ecog_test)
    np.save(subj_dir / "flex_train.npy", flex_train_rel)      # float64
    np.save(subj_dir / "flex_test.npy", flex_test_rel)
    np.save(subj_dir / "labels_train.npy", labels_train.astype(np.int32))
    np.save(subj_dir / "labels_test.npy", labels_test.astype(np.int32))

    meta = {
        "subject": subject,
        "T_seconds": round(T / FS_ECOG, 3),
        "n_ch_raw": n_ch,
        "n_ch_keep": len(keep),
        "bad_channels_1based": [c + 1 for c in bad],
        "n_trials": n_trials,
        "n_train_trials": n_train_trials,
        "n_test_trials": n_trials - n_train_trials,
        "train_seconds": round((train_end - train_start) / FS_ECOG, 3),
        "test_seconds": round((T - train_end) / FS_ECOG, 3),
        "fs_ecog": FS_ECOG,
        "fs_glove": FS_GLOVE,
        "baseline_flex": [round(float(b), 3) for b in baseline],
        "max_angles_relative": [round(float(m), 3) for m in max_angles],
        "label_threshold_ratio": ANGLE_THRESHOLD_RATIO,
        "median_filter_kernel": MEDIAN_FILTER_KERNEL,
        "stim_agreement": stim_agree,
        "ecog_unit": "amplifier units (1 unit = 0.0298 uV; downstream z-score absorbs gain)",
        "label_source": "glove threshold (Yao & Shoaran 2019), cue used for trial boundary only",
    }
    (subj_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # 8. Print verification.
    print(f"\n{'='*72}")
    print(f"SUBJECT {subject}: T={T/FS_ECOG:.2f}s, n_ch={n_ch}->{len(keep)} keep, "
          f"bad(1-based)={[c+1 for c in bad]}")
    print(f"  trials={n_trials}, train={n_train_trials} [{train_start/FS_ECOG:.2f}s,"
          f"{train_end/FS_ECOG:.2f}s]={meta['train_seconds']}s, "
          f"test={n_trials-n_train_trials} [{train_end/FS_ECOG:.2f}s,{T/FS_ECOG:.2f}s]"
          f"={meta['test_seconds']}s")
    print(f"  baseline (rest position): {[round(b,1) for b in baseline]}")
    print(f"  max_angles (relative): {[round(m,1) for m in max_angles]}")
    print(f"  flex_train range: [{flex_train_rel.min():.1f}, {flex_train_rel.max():.1f}]  "
          f"flex_test: [{flex_test_rel.min():.1f}, {flex_test_rel.max():.1f}]")
    print(f"  labels_train dist: {label_distribution(labels_train)}")
    print(f"  labels_test  dist: {label_distribution(labels_test)}")
    if stim_agree is not None:
        print(f"  stim agreement (train/test): {stim_agree['train']:.3f} / {stim_agree['test']:.3f}")
    print(f"  saved to {subj_dir}")
    return meta


if __name__ == "__main__":
    subs = sys.argv[1:] if len(sys.argv) > 1 else ["jc", "wc"]
    for s in subs:
        prepare_subject(s)
