# Switching Linear Model: ECoG Finger-Flexion Regression

Switching linear decoder for continuous finger-flexion regression, following
Flamary & Rakotomamonjy (2011/2012). One of four regression methods in the
diagnostic benchmark on BCI Competition IV, Dataset 4.

The method operationalizes a hypothesis that a single global linear map is
insufficient: the decoder switches among per-state ridge regressions H_k, one
per movement state (five fingers + rest), with the active state predicted by a
joint sparse (group-lasso) classifier f(x) on AR features.

## Pipeline

1. **State labels** (`state_labels.py`): a moving-finger state sequence is
   derived from the glove trajectory (no explicit trial-cue channel exists in
   ds4), as in the paper's oracle analysis.
2. **AR features** (`ar_features.py`): the first two AR coefficients over
   300-sample windows, with time-shifts ±ts.
3. **Channel selection** (`channel_selection.py`): K channels maximizing
   validation correlation.
4. **State classifier** (`state_classifier.py`): joint sparse (group-lasso)
   moving-finger classifier f(x).
5. **Regression features** (`regression_features.py`): Savitzky-Golay smoothed
   (order 3, 0.4 s) signal at time lags [t−τ, t, t+τ].
6. **Per-state ridge regression** (`finger_regressor.py`): H_k with M
   largest-|h| features retained.
7. **Assembly + evaluation** (`evaluate_decoder.py`).

## Data

Place the BCI Competition IV Dataset 4 files under `code/data/` (same layout as
the Wiener filter; see the top-level README). `config.py` points `DATA_DIR` at
`code/data` by default.

## Usage

```bash
cd code
python tune_regression.py      # per-subject tau/M for the regression branch
python tune_classifier.py      # K / lambda_s for the classifier branch
python final_run.py            # final test-set evaluation (3 subjects)
```

`final_run.py` prints the paper's Table 3 layout (baseline / oracle / estimated)
and writes per-finger r plus test-set state accuracy to `results/results.json`
(reproducibility). A full end-to-end run is available via `run_real_data.py`;
smoke tests use `synthetic_blocks.py`. Diagnostic and verification scripts live
under `diagnostics/`.

## Results

| Variant | official_r (mean over 5 subjects) |
|---|---|
| estimated state (competition result) | 0.146 |

Per-subject estimated-state r: S1 0.117 / S2 0.187 / S3 0.375 / jc 0.038 / wc 0.014. Baseline and
oracle variants (single global linear map, and switching with true state) are
reported in `results/results.json` alongside the estimated variant. The
Switching Linear Model reports a negative official_R² with positive official_r
(a calibration shortfall discussed in the paper).

## Repository layout

```
switching_lm/
├── code/          # core pipeline + tuning + synthetic smoke test
├── diagnostics/   # diagnose_* / check_* / inspect_* / measure_* / verify_*
├── results/       # results.json (written by final_run.py)
├── requirements.txt
└── README.md
```

## Citation

Flamary, R. & Rakotomamonjy, A. (2012). Decoding finger movements from ECoG
signals using switching linear models. *Frontiers in Neuroscience*, 6, 29.

Miller, K. J. & Schalk, G. (2008). BCI Competition IV, Dataset 4.
