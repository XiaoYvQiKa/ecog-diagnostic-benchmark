# Wiener Filter: ECoG Finger-Flexion Regression

Closed-form Wiener filter decoder for continuous finger-flexion regression from
ECoG, following Liang & Bougrain (2012). One of four regression methods in the
diagnostic benchmark (classical linear → deep learning) on BCI Competition IV,
Dataset 4.

The method operationalizes a specific hypothesis: whether a linear,
time-invariant mapping suffices to decode continuous finger trajectories from
ECoG. It is not presented as a standalone "best score" candidate.

## Pipeline

Band-specific amplitude-modulation (AM) features are extracted from three FIR
bands (sub 1–60 Hz, gamma 60–100 Hz, fast-gamma 100–200 Hz) over 40 ms
non-overlapping windows, aligned to the 25 Hz dataglove rate. A k=25-lag memory
stack adds ~1 s of temporal context; greedy forward selection (≤10
channel×band pairs, internal 3/5–2/5 split) prunes the features; and a
pseudo-inverse regression (`pinv(XᵀX)`) maps features to flexion trajectories.
The model has ~261 parameters.

## Data

Place the BCI Competition IV Dataset 4 files under `code/data/`:

```
code/data/sub{n}_comp.mat        # train_data, train_dg, test_data
code/data/sub{n}_testlabels.mat  # test_dg
```

`config.py` points `DATA_DIR` at `code/data` by default. See the top-level README
for download instructions.

## Usage

```bash
cd code
python run_reproduction.py               # all three subjects (~6 min)
python run_reproduction.py --subject 1   # single subject
python run_reproduction.py --solver inv  # solver ablation
```

The competition split is fixed temporal: first 400 s train, last 200 s test.
The test set is touched exactly once, at the end. Electrode quality control is
on by default (`--no-electrode-qc` to disable); it uses a train-only
kurtosis/extreme-value screen and excludes flagged electrodes before feature
selection (see `detect_bad_channels.py`, `diagnose.py`).

## Results

| Metric | Value (mean over 5 subjects) |
|---|---|
| official_r (4 fingers, ring excluded) | 0.409 |
| official_R² | −0.005 |
| calibration gap | 0.22 |

Per-subject official_r: S1 0.447 / S2 0.323 / S3 0.611 / jc 0.405 / wc 0.257. Full per-finger
r / R² / calibrated-R² / NRMSE / MAE are in `results/results.json`
(reproducible via `run_reproduction.py --save results/results.json`).

## Repository layout

```
wiener/
├── code/          # source: config, data_io, filters, features, memory_stack,
│                  #        selection, regressor, run_reproduction, diagnostics, tests
├── results/       # results.json
├── requirements.txt
└── README.md
```

## Citation

Liang, N. & Bougrain, L. (2012). Decoding finger flexion from band-specific
ECoG signals. *Frontiers in Neuroscience*, 6, 91.

Miller, K. J. & Schalk, G. (2008). BCI Competition IV, Dataset 4.
