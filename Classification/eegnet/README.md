# EEGNet

A compact temporal-convolutional network for EEG/ECoG decoding, following
Lawhern et al. (2018). One of four classification methods in the benchmark;
it operationalizes the hypothesis that a compact CNN can be trained on the
limited trial counts of ECoG finger-movement data.

## Run

```bash
cd classification
python eegnet/run.py --subjects sub1 sub2 sub3   # 3 seeds, GPU
```

The runner delegates to the shared deep-learning pipeline in `../common/`
(trial extraction from continuous ECoG, 1000 -> 500 Hz downsampling, Mixup
regularization, early stopping).

## Result (five-subject mean)

| Metric | Value |
|---|---|
| Macro F1 | 0.232 |
| Balanced accuracy | 0.253 |
| Cohen's κ | 0.125 |

Per-subject macro F1: S1 0.223 / S2 0.256 / S3 0.236 / jc 0.262 / wc 0.184.

## Citation

Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., &
Lance, B. J. (2018). EEGNet: a compact convolutional neural network for
EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5),
056013.
