# -*- coding: utf-8 -*-
"""Ablation -- gamma-band power features for the moving-finger state
classifier, replacing the paper's AR-coefficient features (the features
control).

High-gamma band power is the classic discriminative signal for ECoG motor
decoding (indexes local cortical activation), so it is the natural "more
discriminative" alternative to the AR coefficients. If the state classifier's
balanced accuracy stays near chance on these features, the bottleneck is the
task / data, not the AR feature representation.

Feature construction (per channel, per band):
  1. zero-phase band-pass filter (Butterworth, order 4)
  2. short-window band power (mean squared amplitude over a non-overlapping
     window of `window_samples`, default 300 = 300 ms, matching the AR
     feature's AR_WINDOW_SAMPLES so the two feature families share the same
     temporal grain)
  3. natural-cubic-spline interpolation from window centers back to the full
     T-sample trace, exactly mirroring ar_features' spline step, so both
     feature families output (T, ...) at the native 1000 Hz resolution.

For n_channels channels and 2 bands (gamma 60-100 Hz, fast-gamma 100-200 Hz),
this yields n_channels * 2 features -- the same dimensionality as the AR
features (n_channels * 2 coefficients at shift 0), keeping the comparison
dimension-matched.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import CubicSpline

GAMMA_BANDS = {"gamma": (60.0, 100.0), "fastgamma": (100.0, 200.0)}


def _bandpass_sos(fs: float, low: float, high: float, order: int = 4):
    nyq = fs / 2.0
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


@dataclass
class GammaFeatureSet:
    """features: (T, n_channels * n_bands), plus the (channel, band) label for
    every column (mirrors ar_features.ARFeatureSet's shape contract)."""
    features: np.ndarray
    channel_of_col: np.ndarray   # (n_cols,) int
    band_of_col: np.ndarray      # (n_cols,) int


def _band_power_trace(x: np.ndarray, fs: float, low: float, high: float,
                      window_samples: int) -> np.ndarray:
    sos = _bandpass_sos(fs, low, high)
    filt = sosfiltfilt(sos, x)
    T = len(filt)
    n_windows = T // window_samples
    usable = n_windows * window_samples
    power = (filt[:usable] ** 2).reshape(n_windows, window_samples).mean(axis=1)
    centers = (np.arange(n_windows) + 0.5) * window_samples
    spline = CubicSpline(centers, power, bc_type="natural")
    return spline(np.arange(T))


def extract_gamma_features(ecog: np.ndarray, channels=None, fs: float = None,
                           window_samples: int = None) -> GammaFeatureSet:
    fs = 1000.0 if fs is None else fs
    window_samples = 300 if window_samples is None else window_samples
    channels = range(ecog.shape[1]) if channels is None else list(channels)

    n_bands = len(GAMMA_BANDS)
    T = ecog.shape[0]
    n_cols = len(channels) * n_bands
    feats = np.empty((T, n_cols), dtype=np.float64)
    ch_col = np.empty(n_cols, dtype=int)
    band_col = np.empty(n_cols, dtype=int)

    col = 0
    for ch in channels:
        x = ecog[:, ch]
        for bi, (_, (lo, hi)) in enumerate(GAMMA_BANDS.items()):
            feats[:, col] = _band_power_trace(x, fs, lo, hi, window_samples)
            ch_col[col] = ch
            band_col[col] = bi
            col += 1

    return GammaFeatureSet(features=feats, channel_of_col=ch_col,
                           band_of_col=band_col)
