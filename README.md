# ECoG Finger-Movement Decoding Benchmark

**Eight ECoG finger-movement decoders — four classification, four regression — compared across five subjects as hypothesis tests, not a leaderboard.**

A unified diagnostic benchmark for ECoG-based individual finger-movement
decoding, spanning two tasks (five-finger classification and finger-flexion
regression) and eight methods, evaluated on five subjects. The benchmark is a
set of hypothesis tests rather than a leaderboard: each method operationalizes
one assumption about the structure of ECoG signals, and the comparison reads
out, per assumption, under which data conditions it holds and where it fails.

## Scope

- **Subjects**: five subjects. The three BCI Competition IV Dataset 4 subjects
  (S1, S2, S3) plus two added subjects (jc, wc) drawn from the nine-subject
  open ECoG finger-flexion library of Miller (2019), selected by an objective
  motor-region criterion (Methods, Section 2.1).
- **Tasks**: five-finger classification (six classes incl. rest) and
  continuous finger-flexion regression.
- **Protocol**: within-subject, a fixed temporal train/test split, a single
  unified metric convention per task, and a three-tier statistical framework
  (per-finger Wilcoxon signed-rank tests, Cohen's *d*, and Demšar critical
  difference diagrams).

## Structure

| Directory | Contents |
|---|---|
| `Regression/` | four regression methods (Wiener filter, Switching Linear Model, FingerFlex, DTCNet), each in its own subdirectory |
| `Classification/` | four classification methods (CSP+LDA, Spectral+LightGBM, EEGNet, EEG-Conformer), each in its own subdirectory plus a shared `common/` pipeline |
| `sample_extension/` | scripts that add the two Miller-library subjects (jc, wc) and recompute the five-subject statistics |
| `requirements.txt` | top-level environment (union of both tasks; per-method files remain in each directory) |

## Methods and hypotheses

| Task | Method | Operationalized hypothesis |
|---|---|---|
| Regression | Wiener filter | a linear, time-invariant map suffices |
| Regression | Switching Linear Model | per-state linear maps are needed |
| Regression | FingerFlex | a learned Morlet time-frequency representation suffices |
| Regression | DTCNet | per-time-step supervision plus a retained channel-frequency 2D structure helps |
| Classification | CSP+LDA | spatial filtering is sufficient |
| Classification | Spectral+LightGBM | frequency-band decomposition plus trees competes |
| Classification | EEGNet | a compact CNN survives limited trials |
| Classification | EEG-Conformer | self-attention adds value on small samples |

## Reproduction

Install the shared environment with `pip install -r requirements.txt` (see
each method's `README.md` for its own requirements). Each method directory is
self-contained. The BCI Competition IV Dataset 4 classification data are
downloaded automatically on first run; the regression and Miller-library data
are downloaded manually, following each directory's `README.md`. Set
`BCI_DATA_DIR` to control the cache location. The official
classification test labels must be placed under the test-label directory. See
each directory's `README.md` for the environment and the exact commands. The
five-subject summary numbers are recomputed by `sample_extension/` from each
method's per-subject outputs.

## Key results (five-subject means)

Regression (official *r*, four-finger convention, ring excluded):

| Method | official_r | official_R² | calibration gap |
|---|---|---|---|
| Wiener filter | 0.409 | −0.005 | 0.22 |
| Switching Linear Model | 0.146 | −0.305 | 0.35 |
| FingerFlex | 0.583 | 0.195 | 0.16 |
| DTCNet | 0.426 | −0.309 | 0.53 |

Classification (macro-F1):

| Method | macro-F1 |
|---|---|
| CSP+LDA | 0.416 |
| Spectral+LightGBM | 0.381 |
| EEGNet | 0.232 |
| EEG-Conformer | 0.228 |

The two tasks reverse the deep-learning advantage: deep learning wins in
regression (FingerFlex 0.583 versus Wiener 0.409) whereas traditional methods
win in classification (0.40 versus 0.23 pooled). Full per-finger and
per-subject tables, and the statistical analysis (25 paired observations,
5 fingers × 5 subjects), are in the paper (Results, Sections 3.1–3.6, and the
Appendix).

## Citation

If you use this benchmark, cite the original method papers (per-method
`README.md`) and the datasets: BCI Competition IV Dataset 4 (Miller & Schalk,
2008) and the Miller (2019) open ECoG finger-flexion library.
