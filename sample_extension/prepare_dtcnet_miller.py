# -*- coding: utf-8 -*-
"""DTCNet data preparation for the added Miller subjects (jc, wc).

Reuses the DTCNet preprocessing functions (per-channel z-score + median removal
before band-pass, 40-300 Hz band-pass, 60 Hz notch, complex Morlet with
n_cycles=7, ::10 downsampling to 100 Hz), reading from the prep outputs that are
already trial-split, bad-channel-cleaned, and de-based. Unlike FingerFlex, DTCNet
uses median removal (not common-average), normalization before filtering, a
hand-written complex Morlet, and ::10 decimation of the finger trace (no
interpolation).

Outputs (P.DTC_DATA_MILLER/sub{sid}_{train,test}_{spec,finger}.npy):
    spec   (ch, 40, T@100Hz)
    finger (5, T@100Hz)  de-based flexion repeated 4x (equivalent to ::10)

Usage:
    python prepare_dtcnet_miller.py jc wc
"""
import pathlib
import sys

import numpy as np

import paths as P

sys.path.insert(0, str(P.DTC_CODE_DIR))
from preprocess_raw import bandpass_filter, notch_filter, per_channel_normalize, morlet_spec  # noqa: E402


def process_one(subject: str):
    ecog_train = np.load(P.PREP_DIR / subject / "ecog_train.npy").astype(np.float64)  # (T, ch)
    ecog_test = np.load(P.PREP_DIR / subject / "ecog_test.npy").astype(np.float64)
    flex_train = np.load(P.PREP_DIR / subject / "flex_train.npy").astype(np.float64)  # (T_25, 5)
    flex_test = np.load(P.PREP_DIR / subject / "flex_test.npy").astype(np.float64)

    tr_e = ecog_train.T   # (ch, T)
    te_e = ecog_test.T
    print(f"[{subject}] n_ch={tr_e.shape[0]}, train={tr_e.shape[1]/1000:.2f}s, "
          f"test={te_e.shape[1]/1000:.2f}s")

    # Per-channel normalization on raw ECoG (train statistics), before filtering.
    tr_e, te_e = per_channel_normalize(tr_e, te_e)
    print(f"[{subject}] per-channel normalization done")

    for name, eco in (("train", tr_e), ("test", te_e)):
        eco = bandpass_filter(eco)
        eco = notch_filter(eco)
        print(f"[{subject}] {name}: Morlet (n_cycles=7) ...")
        eco = morlet_spec(eco)
        eco_ds = eco[:, :, ::10]                       # (ch, 40, T/10) @100Hz
        np.save(P.DTC_DATA_MILLER / f"sub{subject}_{name}_spec.npy", eco_ds)
        print(f"[{subject}] {name} spec {eco_ds.shape}")

    # Finger: de-based flex (T_25,5) -> (5, T_25) -> repeat 4 -> (5, T_100).
    finger_train = np.repeat(flex_train.T, 4, axis=1).astype(np.float32)   # (5, T_100)
    finger_test = np.repeat(flex_test.T, 4, axis=1).astype(np.float32)
    np.save(P.DTC_DATA_MILLER / f"sub{subject}_train_finger.npy", finger_train)
    np.save(P.DTC_DATA_MILLER / f"sub{subject}_test_finger.npy", finger_test)
    print(f"[{subject}] finger train {finger_train.shape}, test {finger_test.shape}")
    print(f"[{subject}] DONE")


if __name__ == "__main__":
    P.DTC_DATA_MILLER.mkdir(parents=True, exist_ok=True)
    subs = sys.argv[1:] if len(sys.argv) > 1 else ["jc", "wc"]
    for s in subs:
        process_one(s)
