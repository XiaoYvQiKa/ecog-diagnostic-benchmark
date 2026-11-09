"""
Check for electrode artifacts, mirroring the Wiener filter's method (its
documentation, section 4): a test/train max-abs-value ratio check per
channel. A ratio >> 1 (Wiener saw ~700,000x on their own S3 case) means
the test segment has an amplitude excursion training never showed -- a
strong artifact signature.

EXTENDED to all three subjects (previously only run on S3, which is how
channel 49 was originally found). S1/S2's currently-excluded channels
(config.BAD_CHANNELS) were reused directly from Wiener's list WITHOUT
independent verification on this pipeline -- this run checks whether S1/S2
have any ADDITIONAL bad channels Wiener's own detection method (tuned for
their AM/band-power features) might have missed, that show up on this
pipeline's own features instead.

Uses load_subject_clean(), so already-known bad channels (Wiener's list)
are excluded before this check runs -- this specifically hunts for
channels NOT already on that list.

Run in the same folder as the other files, with data/ populated:

    python check_electrode_artifacts.py
"""
import numpy as np

import config as C
from data_io import load_subject_clean
from regression_features import savgol_smooth_all_channels

TOP_N = 15  # show the N channels with the largest test/train ratio

for SUBJECT in [1, 2, 3]:
    print(f"{'='*70}")
    print(f"SUBJECT {SUBJECT}")
    print(f"{'='*70}")

    sd = load_subject_clean(SUBJECT)
    n_ch = sd.n_channels
    print(f"{n_ch} channels remain after excluding config.BAD_CHANNELS[{SUBJECT}] "
          f"= {C.BAD_CHANNELS.get(SUBJECT, [])}")
    print()

    print("-- RAW ECoG: test/train max-abs-value ratio per channel --")
    train_max = np.max(np.abs(sd.train_ecog), axis=0)
    test_max = np.max(np.abs(sd.test_ecog), axis=0)
    ratio = test_max / np.maximum(train_max, 1e-12)
    order = np.argsort(ratio)[::-1]
    print(f'{"channel":>8s} {"train_max":>12s} {"test_max":>12s} {"ratio":>12s}')
    for ch in order[:TOP_N]:
        flag = "  <-- SUSPECT" if ratio[ch] > 10 else ""
        print(f'{ch:8d} {train_max[ch]:12.2f} {test_max[ch]:12.2f} {ratio[ch]:12.1f}{flag}')

    print()
    print("-- SMOOTHED regression-feature signal: same check (what baseline actually uses) --")
    smoothed_train = savgol_smooth_all_channels(sd.train_ecog)
    smoothed_test = savgol_smooth_all_channels(sd.test_ecog)
    train_max_s = np.max(np.abs(smoothed_train), axis=0)
    test_max_s = np.max(np.abs(smoothed_test), axis=0)
    ratio_s = test_max_s / np.maximum(train_max_s, 1e-12)
    order_s = np.argsort(ratio_s)[::-1]
    print(f'{"channel":>8s} {"train_max":>12s} {"test_max":>12s} {"ratio":>12s}')
    for ch in order_s[:TOP_N]:
        flag = "  <-- SUSPECT" if ratio_s[ch] > 10 else ""
        print(f'{ch:8d} {train_max_s[ch]:12.2f} {test_max_s[ch]:12.2f} {ratio_s[ch]:12.1f}{flag}')

    print()
    n_suspect = np.sum(ratio_s > 10)
    print(f"Channels with smoothed test/train ratio > 10: {n_suspect}")
    if n_suspect > 0:
        print(f"Suspect channel indices (post-exclusion numbering, NOT original "
              f"channel indices): {sorted(order_s[:n_suspect].tolist())}")
        print("NEXT STEP: map these back to original channel indices (account for the")
        print("already-excluded ones shifting the numbering), refit and re-evaluate to")
        print("confirm removing them actually improves r2_score before trusting this.")
    else:
        print("No obvious single-channel amplitude artifact found beyond what's already excluded.")
    print()

print("Done. For S1/S2, 'no suspects' means Wiener's list, reused as-is, appears")
print("sufficient on this pipeline's own features too -- independent confirmation,")
print("not just an assumption. Any suspects found need the refit-and-confirm step")
print("above before being added to config.BAD_CHANNELS.")
