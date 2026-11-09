# Spectral+LightGBM

Six-band spectral power features (Welch PSD) classified by gradient-boosted
trees (LightGBM), following Yao & Shoaran (2019). One of four classification
methods in the benchmark; it operationalizes the hypothesis that an explicit
frequency-band decomposition competes with (and, here, trails) explicit
spatial filtering.

## Run

```bash
cd classification
python spectral_lgb/run.py --subjects sub1 sub2 sub3   # 5 seeds
```

The runner delegates to the shared pipeline in `../common/` (subject-specific
400 s / 200 s split, electrode QC, per-channel z-score, 10%-threshold label
generation, full-window majority vote).

## Result (five-subject mean)

| Metric | Value |
|---|---|
| Macro F1 | 0.381 |
| Balanced accuracy | 0.388 |
| Cohen's κ | 0.300 |

Per-subject macro F1: S1 0.450 / S2 0.210 / S3 0.565 / jc 0.352 / wc 0.326.

## Citation

Yao, Y. & Shoaran, M. (2019). Enhanced classification of invasive
electrocorticography data using a combined approach of band power features and
gradient boosting. *IEEE Transactions on Neural Systems and Rehabilitation
Engineering*, 27(5), 825–834.
