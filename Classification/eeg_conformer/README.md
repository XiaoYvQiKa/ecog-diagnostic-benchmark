# EEG-Conformer

A convolutional transformer for EEG/ECoG decoding, following Song et al.
(2023). One of four classification methods in the benchmark; it
operationalizes the hypothesis that self-attention adds value over convolution
on small ECoG finger-movement datasets.

## Run

```bash
cd classification
python eeg_conformer/run.py --subjects sub1 sub2 sub3   # 3 seeds, GPU
```

The runner delegates to the shared deep-learning pipeline in `../common/`
(trial extraction from continuous ECoG, 1000 -> 500 Hz downsampling, Mixup
regularization, early stopping).

## Result (five-subject mean)

| Metric | Value |
|---|---|
| Macro F1 | 0.228 |
| Balanced accuracy | 0.266 |
| Cohen's κ | 0.129 |

Per-subject macro F1: S1 0.260 / S2 0.184 / S3 0.239 / jc 0.258 / wc 0.198.

## Citation

Song, Y., Zheng, Q., Liu, B., & Gao, X. (2023). EEG-Conformer: convolutional
transformer for EEG decoding. *IEEE Transactions on Neural Systems and
Rehabilitation Engineering*, 31, 265–274.
