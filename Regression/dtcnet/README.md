# DTCNet Regression: Reproduction of Wang et al. (2025)

Dilated-Transposed Convolution Network (DTCNet) for ECoG finger-flexion
regression on BCI Competition IV, Dataset 4.

Wang et al. (2025), *Frontiers in Computational Neuroscience*.
DOI: [10.3389/fncom.2025.1627819](https://doi.org/10.3389/fncom.2025.1627819)

## Environment

```bash
pip install -r requirements.txt
# GPU (CUDA 12.4):
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

- Python 3.13, PyTorch 2.6+cu124, numpy, scipy.

## Data preparation

1. Download BCI Competition IV, Dataset 4 from
   [bbci.de](https://www.bbci.de/competition/iv/#dataset4).
2. Obtain the official test labels (released after the competition; see
   `true_labels.zip`).
3. Preprocess:

```bash
python code/preprocess_raw.py
```

The raw `.mat` files are read from `BCI_RAW_DIR` (default `../data_raw`) and the
preprocessed spectrograms are written to `BCI_DATA_DIR` (default
`../data_dtcnet`) as `sub{1,2,3}_{train,test}_{spec,finger}.npy`.

Environment variables (all optional, with relative defaults under the package
root):

| Variable | Default | Purpose |
|:--|:--|:--|
| `BCI_RAW_DIR` | `../data_raw` | raw `.mat` files |
| `BCI_DATA_DIR` | `../data_dtcnet` | preprocessed spectrograms + finger data |
| `BCI_ABLATION_DIR` | `../data_ablation` | frequency-resolution ablation spectrograms |
| `BCI_KEEPBAD_DIR` | `../data_dtcnet_keepbad` | keep-bad-channel control data |

## Training

```bash
python code/run_direct.py --subject all
```

- GPU: ~5 h per subject (RTX 4060).
- Outputs to `results/main/` (per-seed metrics, mean ± std, loss curves).

Multi-seed configuration (seeds 0, 1, 2) reports mean ± std over seeds to
quantify training randomness.

## Preprocessing (Wang et al. 2025, Section 2.2)

| Step | Operation |
|:--|:--|
| 1 | Per-channel z-score normalization + median removal (on raw ECoG, train statistics only) |
| 2 | Bandpass 40–300 Hz |
| 3 | Notch 60 Hz (line noise) |
| 4 | Complex Morlet wavelet → spectrogram (40 bands, n_cycles = 7) |
| 5 | Temporal downsampling 1000 → 100 Hz |

Bad electrodes (amplitude anomaly > 10× median, independently confirmed across
methods): sub1 ch55, sub2 ch21 + ch38, sub3 ch50.

Finger data: data-glove angles at 25 Hz, `::10` decimation to 100 Hz (raw
readings preserved, no interpolation).

## Model architecture

| Component | Details |
|:--|:--|
| Input | Morlet spectrogram (ch × 40 freq × 256 time @ 100 Hz) |
| Feature reduction | 1×1 conv + 3×1 conv (ch×freq → 48) |
| Encoder | 5 dilated conv blocks (48→64→96→128→128; dilation 1,2,3,1,2) + average pooling |
| Decoder | 5 transposed-conv blocks + skip connections |
| Output | 1×1 conv → 5 fingers, per-time-step trajectory (batch, 5, 256) |
| Parameters | 675 K–708 K (varies with channel count) |
| Training | Adam (lr = 8.42e-5, wd = 1e-6), MSE + cosine loss, dropout 0.1 |

> **Difference from the original paper**: the encoder widths here are
> [48, 64, 96, 128, 128] rather than [64, 128, 256, 512, 512]. The paper reports
> 550–790 K parameters, but [64, ..., 512] widths yield ~4 M parameters, an
> internal contradiction. We therefore match the reported parameter count rather
> than the stated widths; this is documented in the manuscript.

## Evaluation metrics

| Metric | Definition |
|:--|:--|
| **r** | Pearson correlation (per finger, across time) |
| **official_r** | 4-finger mean excluding the ring finger (primary) |
| **avg_r** | 5-finger mean (auxiliary) |
| **R²** | coefficient of determination, 1 − SSE/SST |
| **calibrated R²** | per-finger r² (upper bound of R² under affine rescaling) |
| **calibration gap** | r² − R² (recoverable scale/bias error) |
| **MAE** | mean absolute error (data-glove units) |

Evaluation protocol: per-time-step trajectory output; the last time step of each
sliding window (stride = 1) is taken for scoring, identical to the single-point
protocol.

## Ablations and analysis

```bash
python code/gen_ablation_data.py   # frequency-resolution spectrograms (1/10/20 band)
bash code/run_ablation.sh          # frequency-resolution + output-layer ablations
python code/export_results.py      # benchmark tables (Markdown + CSV)
python code/extract_stats.py       # per-finger r values for the paired Wilcoxon test (15 for the three BCI subjects; 25 with the two added Miller subjects)
python code/analyze_supplementary.py  # per-finger correlation + training curves
python code/draw_figures.py        # architecture / correlation / ablation figures
python code/measure_dtcnet_latency.py  # inference latency + parameter count (Table G1)
python code/verify_25hz.py         # 100 Hz → 25 Hz block-average equivalence check
```

> `verify_25hz.py` loads the seed-0 weights from `results/main/sub{sid}_seed0_model.pt`.
> Run `run_direct.py` first to produce them; the `.pt` files are not shipped in this
> repository (weights excluded). `measure_dtcnet_latency.py` needs no data or weights.

## Statistical protocol (regression)

- Layer 1: subject-level direction consistency of `official_r`.
- Layer 2: per-finger paired Wilcoxon signed-rank test (5 fingers × 5 subjects
  = 25 pairs for the five-subject benchmark; 5 fingers × 3 BCI subjects = 15
  pairs within this package), Bonferroni-corrected α = 0.0083 across the
  6 method pairs.

## Repository layout

```
code/       source modules and runners
results/    main + ablation + latency result JSONs and loss curves (weights excluded)
benchmark/  exported benchmark tables (Markdown + CSV)
figures/    generated figures and the finger-correlation CSV
docs/       this README
```

## Code lock

All code files are checksummed per-file in `.code_lock.sha256`; verify with
`sha256sum -c .code_lock.sha256` (run from the package root).

## Citation

```bibtex
@article{wang2025dtcnet,
  title={DTCNet: finger flexion decoding with three-dimensional ECoG data},
  author={Wang, Y. and others},
  journal={Frontiers in Computational Neuroscience},
  year={2025},
  doi={10.3389/fncom.2025.1627819}
}
```
