# ECoG Finger-Movement Classification Benchmark

A diagnostic benchmark for ECoG-based finger-movement classification on BCI
Competition IV, Dataset 4 (five subjects: S1–S3 plus the Miller-library
subjects jc and wc, 400 s train / 200 s test per subject). Four methods —
CSP+LDA, Spectral+LightGBM, EEGNet, and EEG-Conformer — are evaluated under a
single unified protocol. Each method operationalizes one assumption about the
structure of ECoG signals, so the comparison reads out, per assumption, where
it holds and where it fails (rather than ranking methods).

## Structure

| Path | Contents |
|---|---|
| `common/` | shared pipeline (config, data loading, features, models, evaluation, DL utilities, training loops) |
| `csp_lda/` | CSP+LDA runner |
| `spectral_lgb/` | Spectral+LightGBM runner |
| `eegnet/` | EEGNet runner |
| `eeg_conformer/` | EEG-Conformer runner |
| `run_ablations.py` | frequency-band / window-length / classifier ablations (traditional) |
| `run_dl_ablations.py` | Mixup and SimCLR-pretraining ablations (deep learning) |
| `run_hyperparam_selection_val.py` | train-internal validation hyperparameter selection |
| `run_label_alignment_ablation.py` | label-alignment convention ablation |
| `subject2_shift_robustness.py`, `window_robustness.py` | robustness analyses |
| `compute_combined_stats.py` | Friedman, per-finger Wilcoxon, Cohen's d |
| `figures_final.py` | publication figures (PNG + PDF) |
| `preprocess_raw.py`, `measure_cost.py`, `diagnose_channels.py` | preprocessing / cost / QC diagnostics |

## Environment

- Python 3.13
- numpy, scipy, scikit-learn, lightgbm, mne, matplotlib
- PyTorch 2.x (EEGNet / EEG-Conformer)
- braindecode 1.3+ (DL utilities)

Install with `pip install -r requirements.txt`.

## Data

The competition data are downloaded automatically on first run. Set
`BCI_DATA_DIR` to control the cache location (default: `./data/`). The
official test labels (`true_labels.zip`, sub1/sub2/sub3_testlabels.mat) must be
placed under `BCI_DATA_DIR/test_labels/`. The two added Miller subjects (jc,
wc) are prepared by `../sample_extension/`.

## Quick start

```bash
cd classification
python csp_lda/run.py --subjects sub1 sub2 sub3       # CSP+LDA (5 seeds)
python spectral_lgb/run.py --subjects sub1 sub2 sub3  # Spectral+LightGBM (5 seeds)
python eegnet/run.py --subjects sub1 sub2 sub3        # EEGNet (3 seeds, GPU)
python eeg_conformer/run.py --subjects sub1 sub2 sub3 # EEG-Conformer (3 seeds, GPU)
```

Ablations and statistics:

```bash
python run_ablations.py --ablation all
python run_dl_ablations.py --ablation all
python run_hyperparam_selection_val.py
python compute_combined_stats.py
python figures_final.py
```

## Electrode quality control

Three absolute thresholds are applied on training data only (no test leakage):
spiky channels (kurtosis > 50), dead channels (RMS < median/10), and
test-time breakage (test/train RMS ratio > 10×). This removes sub1 ch55,
sub2 ch21+ch38, and sub3 ch50 (61/46/63 channels remain), matching the
regression task.

## Results (five-subject means)

| Method | Mean F1 | BalAcc | Cohen's κ |
|---|---|---|---|
| CSP+LDA | 0.416 | 0.414 | 0.315 |
| Spectral+LGB | 0.381 | 0.388 | 0.300 |
| EEGNet | 0.232 | 0.253 | 0.125 |
| EEG-Conformer | 0.228 | 0.266 | 0.129 |

Per-subject macro F1:

| Method | S1 | S2 | S3 | jc | wc |
|---|---|---|---|---|---|
| CSP+LDA | 0.432 | 0.271 | 0.601 | 0.407 | 0.370 |
| Spectral+LGB | 0.450 | 0.210 | 0.565 | 0.352 | 0.326 |
| EEGNet | 0.223 | 0.256 | 0.236 | 0.262 | 0.184 |
| EEG-Conformer | 0.260 | 0.184 | 0.239 | 0.258 | 0.198 |

Traditional methods outperform deep learning: pooled per-finger paired
Wilcoxon (5 fingers × 5 subjects, n = 25), p = 2.7e-5, Cohen's d = 0.98.
Subject-level Friedman test: p = 0.014 (n = 5). Full per-finger, per-method
tables and the Demšar critical-difference analysis are in the paper (Results,
and Appendix).
