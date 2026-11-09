"""
S3 electrode QC cross-validation: check whether the raw S3 ECoG data used by
FingerFlex contains the same bad channel(s) already identified by the Wiener
filter and the Switching Linear Model (in particular ch49, train/test
amplitude ratio > 1100x).

Logic: for each channel, compute the amplitude statistic (here std) of the
train segment (first 400 s) and the test segment (last 200 s) separately. If
the train_std / test_std ratio (or its inverse) is far higher than the other
channels, that channel's signal properties jump sharply between train and
test, which likely reflects a physical artifact (loose/broken electrode
contact, wire failure) rather than a genuine neural change.

Reads only the raw .mat files; does not run prepare_data.ipynb's full
preprocessing, so it runs quickly.
"""

import numpy as np
import scipy.io
import pathlib

PATH = f"{pathlib.Path().resolve()}/data/pure_data"
SUBJECT = "sub3"

RATIO_FLAG_THRESHOLD = 50  # Flag channels whose amplitude ratio exceeds this multiple. The Wiener
                            # reproduction reports ch49 at >1100x; the threshold is set lower here so
                            # less extreme but still anomalous channels are not missed.


def main():
    print(f"Diagnosing raw ECoG data for {SUBJECT} ...")

    train_data = scipy.io.loadmat(f"{PATH}/{SUBJECT}_comp.mat")["train_data"].astype("float64")  # (time, channels)
    test_data = scipy.io.loadmat(f"{PATH}/{SUBJECT}_testlabels.mat")

    # sub_testlabels.mat contains only labels (test_dg); the raw test ECoG is in the test_data field
    # of sub_comp.mat (if present). If absent, fall back to a front/back split of train_data (with a note).
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
    print("If ch49 from the Wiener notes is also in this list (note the possible 0-indexed vs "
          "1-indexed offset of 1, i.e. Wiener's channel 49 may correspond to index 48 here), then the "
          "raw data used by FingerFlex is contaminated by the same physical bad channel, because "
          "FingerFlex currently performs no channel selection and feeds all 62/64 channels into the model.")


if __name__ == "__main__":
    main()
