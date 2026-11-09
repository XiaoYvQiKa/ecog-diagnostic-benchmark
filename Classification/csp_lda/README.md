# CSP+LDA

Common Spatial Patterns spatial filtering followed by linear discriminant
analysis, following Blankertz et al. (2008). One of four classification
methods in the benchmark; it operationalizes the hypothesis that a
hand-crafted spatial filter (high-gamma band, one-vs-rest CSP) is sufficient
to separate the six movement classes.

## Run

```bash
cd classification
python csp_lda/run.py --subjects sub1 sub2 sub3   # 5 seeds, deterministic
```

The runner delegates to the shared pipeline in `../common/` (subject-specific
400 s / 200 s split, electrode QC, per-channel z-score, 10%-threshold label
generation, full-window majority vote).

## Result (five-subject mean)

| Metric | Value |
|---|---|
| Macro F1 | 0.416 |
| Balanced accuracy | 0.414 |
| Cohen's κ | 0.315 |

Per-subject macro F1: S1 0.432 / S2 0.271 / S3 0.601 / jc 0.407 / wc 0.370.

## Citation

Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Müller, K.-R. (2008).
Optimizing spatial filters for robust EEG single-trial analysis. *IEEE Signal
Processing Magazine*, 25(1), 41–56.
