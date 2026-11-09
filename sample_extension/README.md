# Sample extension: adding two Miller-library subjects (jc, wc)

This directory extends the benchmark from the three BCI Competition IV
Dataset 4 subjects (S1–S3) to five subjects by adding two subjects (jc, wc)
from the nine-subject open ECoG finger-flexion library of Miller (2019),
selected by an objective motor-region criterion (Methods, Section 2.1).

It reuses each method's own feature/model/evaluation code and replaces only
the data source, so the added subjects are processed identically to the three
original subjects. The paired statistics are then recomputed over 25 paired
observations (5 fingers × 5 subjects).

## Pipeline

1. **Prepare** the Miller subjects into the shared BCI-compatible layout
   (`prep_miller.py`): raw amplifier ECoG (1000 Hz), de-based relative flexion
   (25 Hz), and glove-threshold labels, plus per-channel z-score on train
   statistics.

2. **Run** each method on jc/wc (regression and classification):

   ```bash
   python run_wiener_miller.py jc wc
   python run_switching_lm_miller.py jc wc
   python run_classification_miller.py jc wc
   python run_dtcnet_miller.py jc wc
   python prepare_fingerflex_miller.py jc wc
   ```

   (FingerFlex runs through its notebook; `prepare_fingerflex_miller.py`
   produces its spectrogram inputs.)

3. **Recompute** the five-subject statistics (`recompute_stats_25pairs.py`,
   `write_stats_summary.py`): per-finger paired Wilcoxon signed-rank tests,
   Cohen's d, Friedman test and Demšar critical-difference diagram over
   N = 5 subjects, plus the 15-vs-25 observation comparison.

## Path configuration

`paths.py` defines all machine-specific roots once. Set `BCI_DATA_DIR` to the
location of the downloaded datasets (default `~/BCI_data`); the Miller data
live under `BCI_DATA_DIR/Miller_fingerflex`.

## Results

The committed five-subject summaries are in `results/`:

- `classification_5subj_summary.json` — per-method, per-subject F1 / BalAcc / κ
  and the five-subject means.
- `regression_5subj_summary.json` — per-method, per-subject r / R² /
  calibration gap and the five-subject means.
- `stats_summary_n15_n25.json` — the full n=15 vs n=25 statistical comparison
  (Wilcoxon p, Cohen's d, Friedman, Demšar CD).
- `stats_25pairs_vectors.json` — the raw 25-observation vectors.

## Data

Miller, K. J. (2019). A library of human electrocorticographic data for
individual finger movements (open data).
