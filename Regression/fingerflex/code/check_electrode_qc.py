"""
Electrode quality-control diagnostic script (parameterized; run for any of S1/S2/S3).

Logic: for each channel, compute the amplitude statistic (std) of the train
segment (first 400 s) and the test segment (last 200 s) separately. If one
channel's train/test amplitude ratio is far higher than the rest, that channel
likely suffers from a physical artifact (loose/broken electrode contact, wire
failure) rather than a genuine neural change, because its signal properties
jump sharply between train and test.

Reads only the raw .mat files; does not run prepare_data.ipynb's full
preprocessing, so it runs quickly.

Usage: change SUBJECT below and rerun to diagnose S1/S2/S3 independently.
"""

import numpy as np
import scipy.io
import pathlib

# ---- Change this line to switch subject ----
SUBJECT = "sub1"    # sub1 / sub2 / sub3
# -----------------------------

PATH = f"{pathlib.Path().resolve()}/data/pure_data"

RATIO_FLAG_THRESHOLD = 8   # Lowered threshold: S3's ch49 is an extreme case at 252x, but the
                            # ch54 flagged on S1 is only ~11x, which a 50x threshold would miss.
                            # A moderate anomaly (clearly outside the normal range of the other
                            # channels) is still worth noting, not only the one-off extremes.

# Known bad-channel records (for cross-checking after running a new subject)
# sub3: independently cross-verified by FingerFlex + Wiener + Switching LM, ch49, ratio 252.63x
# sub1: reported as ch55 (1-indexed) by the external FBTTR paper, i.e. ch54 0-indexed.
#       Whether the Wiener / Switching LM reproductions excluded this channel themselves
#       has not been directly confirmed.
KNOWN_BAD_CHANNELS = {
    "sub1": [54],  # 0-indexed, from the external FBTTR paper; used here for cross-validation
    "sub3": [49],  # 0-indexed, confirmed, train/test amplitude ratio 252.63x
}


def main():
    print(f"Diagnosing raw ECoG data for {SUBJECT} ...")

    train_data = scipy.io.loadmat(f"{PATH}/{SUBJECT}_comp.mat")["train_data"].astype("float64")  # (time, channels)

    comp_data = scipy.io.loadmat(f"{PATH}/{SUBJECT}_comp.mat")
    if "test_data" in comp_data:
        test_ecog = comp_data["test_data"].astype("float64")
        source = "test_data field of sub_comp.mat"
    else:
        print("[note] no test_data field in sub_comp.mat; falling back to a last-1/3 vs first-2/3 "
              "split of train_data (approximate diagnosis, not a strict train/test amplitude ratio)")
        n = train_data.shape[0]
        test_ecog = train_data[int(n * 2 / 3):]
        train_data = train_data[:int(n * 2 / 3)]
        source = "within-train_data front/back split (approximate)"

    n_channels = train_data.shape[1]
    print(f"Data source: {source}, {n_channels} channels\n")

    train_std = train_data.std(axis=0)
    test_std = test_ecog.std(axis=0)

    # Amplitude ratio: max/min so the ratio is always >= 1.
    ratio = np.maximum(train_std, test_std) / np.minimum(train_std, test_std)

    order = np.argsort(-ratio)  # descending

    print(f"{'channel (0-idx)':<16}{'train_std':<14}{'test_std':<14}{'ratio':<10}")
    print("-" * 54)
    for idx in order[:15]:  # show the top-15 channels by ratio
        flag = " <-- suspicious" if ratio[idx] > RATIO_FLAG_THRESHOLD else ""
        print(f"{idx:<16}{train_std[idx]:<14.4f}{test_std[idx]:<14.4f}{ratio[idx]:<10.2f}{flag}")

    flagged = np.where(ratio > RATIO_FLAG_THRESHOLD)[0]
    print(f"\nChannels above threshold ({RATIO_FLAG_THRESHOLD}x), 0-indexed: {flagged.tolist()}")

    if SUBJECT in KNOWN_BAD_CHANNELS:
        known = KNOWN_BAD_CHANNELS[SUBJECT]
        overlap = set(flagged.tolist()) & set(known)
        print(f"Known bad channels: {known}, overlap with this run: "
              f"{sorted(overlap) if overlap else '(none; check threshold or data consistency)'}")
    else:
        if len(flagged) == 0:
            print(f"{SUBJECT}: no channel exceeds the threshold; electrode quality looks normal "
                  f"(still worth inspecting the top-15 ratio distribution for gradual anomalies)")
        else:
            print(f"{SUBJECT}: {len(flagged)} suspicious channel(s) found. This is the first QC pass "
                  f"for {SUBJECT}; confirm whether they reflect real physical problems (e.g. plot the "
                  f"raw waveform for abnormal spikes/drift)")


if __name__ == "__main__":
    main()
