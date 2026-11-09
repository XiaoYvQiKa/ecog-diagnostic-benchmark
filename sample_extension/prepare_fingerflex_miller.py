# -*- coding: utf-8 -*-
"""FingerFlex data preparation for the added Miller subjects (jc, wc).

Reuses the original FingerFlex preprocessing pipeline (40-300 Hz band-pass +
50 Hz harmonic notch + per-channel z-score + common-average -> 40 Morlet bands
-> ::10 downsampling to 100 Hz), but reads from the prep outputs, which are
already trial-split (2:1), bad-channel-cleaned, and de-based. The flex input is
therefore used directly (25 Hz de-based relative flexion) and cubic-interpolated
to 100 Hz, instead of the deprecated time-based 2/3 split.

Outputs (P.FINGERFLEX_PROCESSED/{subject}/train|val/):
    ecog_data.npy       (ch, 40, time@100Hz) after RobustScaler
    fingerflex_data.npy (5, time@100Hz)      after MinMaxScaler

MinMaxScaler(flex) and RobustScaler(ECoG) are both fit on train and shared with
test. MinMaxScaler is per-finger translation-invariant, so de-baselining does not
change the normalized flex; it is kept for semantic consistency with the other
methods.

Usage:
    python prepare_fingerflex_miller.py jc wc
"""
import pathlib
import sys

import numpy as np
import scipy.interpolate
import mne

import paths as P

L_FREQ, H_FREQ = 40, 300
WAVELET_NUM = 40
DOWNSAMPLE_FS = 100
TIME_DELAY_SEC = 0.2


def filter_ecog_data(sig, fs=1000, powerline_freq=50):
    harmonics = np.array([i * powerline_freq for i in range(1, (fs // 2) // powerline_freq)])
    signal_filtered = mne.filter.filter_data(sig, fs, l_freq=L_FREQ, h_freq=H_FREQ)
    signal_removed = mne.filter.notch_filter(signal_filtered, fs, freqs=harmonics)
    return signal_removed


def normalize(sig):
    means = np.mean(sig, axis=1, keepdims=True)
    stds = np.std(sig, axis=1, keepdims=True)
    transformed = (sig - means) / stds
    common_average = np.median(transformed, axis=0, keepdims=True)
    return transformed - common_average


def compute_spectrogramms(sig, fs=1000,
                          freqs=np.logspace(np.log10(L_FREQ), np.log10(H_FREQ), WAVELET_NUM)):
    n_ch = len(sig)
    spec = mne.time_frequency.tfr_array_morlet(
        sig.reshape(1, n_ch, -1), sfreq=fs, freqs=freqs,
        output="power", verbose=False, n_jobs=6)[0]
    return spec  # (ch, 40, time)


def downsample_spectrogramms(spec, cur_fs=1000, new_fs=DOWNSAMPLE_FS):
    coef = cur_fs // new_fs
    return spec[:, :, ::coef]


def interpolate_flex_25_to_100(flex_25):
    """flex_25: (5, T_25) 25 Hz de-based -> (5, T_100) 100 Hz cubic."""
    flex = np.c_[flex_25, flex_25.T[-1]]              # (5, T_25+1) pad the last value
    upscaling_ratio = DOWNSAMPLE_FS // 25             # 4
    ts = np.asarray(range(flex.shape[1])) * upscaling_ratio
    funcs = [scipy.interpolate.interp1d(ts, flex_ch, kind="cubic") for flex_ch in flex]
    ts_needed = np.asarray(range(flex.shape[1] * upscaling_ratio)[:-upscaling_ratio])
    return np.array([[f(t) for t in ts_needed] for f in funcs])


def crop_for_time_delay(flex, spec, delay_sec, fs):
    d = int(delay_sec * fs)
    flex_cropped = flex[..., d:]
    spec_cropped = spec[..., : spec.shape[2] - d]
    return flex_cropped, spec_cropped


def ecog_to_spec(ecog):
    """ecog (ch, T) 1000 Hz -> spec (ch, 40, T/10) 100 Hz."""
    filtered = filter_ecog_data(ecog)
    normed = normalize(filtered)
    spec = compute_spectrogramms(normed)
    return downsample_spectrogramms(spec)


def process_one(subject):
    ecog_train = np.load(P.PREP_DIR / subject / "ecog_train.npy").astype(np.float64)  # (T, ch)
    ecog_test = np.load(P.PREP_DIR / subject / "ecog_test.npy").astype(np.float64)
    flex_train_25 = np.load(P.PREP_DIR / subject / "flex_train.npy").astype(np.float64)  # (T_25, 5)
    flex_test_25 = np.load(P.PREP_DIR / subject / "flex_test.npy").astype(np.float64)

    train_ecog_raw = ecog_train.T   # (ch, T)
    test_ecog_raw = ecog_test.T
    print(f"[{subject}] n_ch={train_ecog_raw.shape[0]}, train={train_ecog_raw.shape[1]/1000:.2f}s, "
          f"test={test_ecog_raw.shape[1]/1000:.2f}s")

    # Flex 25 -> 100 Hz interpolation.
    train_flex_i = interpolate_flex_25_to_100(flex_train_25.T)   # (5, T_100)
    test_flex_i = interpolate_flex_25_to_100(flex_test_25.T)

    # ECoG spectrogram.
    print(f"[{subject}] computing train spectrogram ...")
    spec_train = ecog_to_spec(train_ecog_raw)
    print(f"[{subject}] computing test spectrogram ...")
    spec_test = ecog_to_spec(test_ecog_raw)

    # Crop the 0.2 s delay.
    train_flex_c, spec_train_c = crop_for_time_delay(train_flex_i, spec_train, TIME_DELAY_SEC, DOWNSAMPLE_FS)
    test_flex_c, spec_test_c = crop_for_time_delay(test_flex_i, spec_test, TIME_DELAY_SEC, DOWNSAMPLE_FS)
    print(f"[{subject}] train spec {spec_train_c.shape}, flex {train_flex_c.shape}")

    # MinMaxScaler(flex) + RobustScaler(ECoG), both fit on train.
    from sklearn.preprocessing import MinMaxScaler, RobustScaler
    scaler = MinMaxScaler()
    scaler.fit(train_flex_c.T)
    train_flex_s = scaler.transform(train_flex_c.T).T
    test_flex_s = scaler.transform(test_flex_c.T).T

    transformer = RobustScaler(unit_variance=True, quantile_range=(0.1, 0.9))
    transformer.fit(spec_train_c.T.reshape(-1, WAVELET_NUM * spec_train_c.shape[0]))
    train_spec_s = transformer.transform(
        spec_train_c.T.reshape(-1, WAVELET_NUM * spec_train_c.shape[0])).reshape(-1, WAVELET_NUM, spec_train_c.shape[0]).T
    test_spec_s = transformer.transform(
        spec_test_c.T.reshape(-1, WAVELET_NUM * spec_test_c.shape[0])).reshape(-1, WAVELET_NUM, spec_test_c.shape[0]).T

    save_dir = P.FINGERFLEX_PROCESSED / subject
    for split in ["train", "val"]:
        (save_dir / split).mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "train" / "ecog_data.npy", train_spec_s)
    np.save(save_dir / "train" / "fingerflex_data.npy", train_flex_s)
    np.save(save_dir / "val" / "ecog_data.npy", test_spec_s)
    np.save(save_dir / "val" / "fingerflex_data.npy", test_flex_s)
    print(f"[{subject}] saved scaled data to {save_dir}")
    print(f"[{subject}] DONE")


if __name__ == "__main__":
    subs = sys.argv[1:] if len(sys.argv) > 1 else ["jc", "wc"]
    for s in subs:
        process_one(s)
