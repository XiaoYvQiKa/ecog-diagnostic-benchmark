"""
Measure exact training time and per-trial inference latency for EEGNet / EEG-Conformer.

Standalone measurement script (not part of the benchmark core) that reports
cross-method computational cost (train time, per-trial inference latency, and
parameter count) for the two deep-learning methods.

Usage:
  python measure_cost.py --subject sub1 --seed 42
  BCI_DATA_DIR=/path/to/data python measure_cost.py --subject sub1 --seed 42
"""
import argparse, os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.data_loader import download_dataset, load_subject
from common.dl_utils import (downsample_ecog, extract_trials,
                                   extract_test_trials, TRIAL_SAMPLES)
from common.run_dl import train_dl


def measure_latency_ms(model, Xte_t, device, n_warmup=10, n_measure=200):
    """Per-trial forward latency (ms), single sample, eval mode, GPU-synchronized."""
    model.eval()
    x = Xte_t[:1].to(device)  # single trial (1, n_ch, 500)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_measure):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_measure
    return dt * 1000.0  # ms per trial


def run(subject_id, seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

    d = load_subject(subject_id)
    ecog_train_500 = downsample_ecog(d['train_ecog'])
    ecog_test_500 = downsample_ecog(d['test_ecog'])
    train_max = d['train_dg'].max(axis=0)
    Xtr, ytr = extract_trials(ecog_train_500, d['train_dg'], seed=seed)
    Xte, yte = extract_test_trials(ecog_test_500, d['test_dg'], train_max=train_max)

    Xtr_t = torch.FloatTensor(Xtr)
    ytr_t = torch.LongTensor(ytr)
    Xte_t = torch.FloatTensor(Xte)
    yte_t = torch.LongTensor(yte)
    n_ch = Xtr.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out = {'subject': subject_id, 'seed': seed, 'n_ch': n_ch,
           'n_train': int(len(ytr)), 'n_test': int(len(yte)), 'device': str(device)}

    # ---- EEGNet ----
    from braindecode.models import EEGNet
    m = EEGNet(n_outputs=6, n_chans=n_ch, n_times=TRIAL_SAMPLES,
               F1=8, D=2, F2=16, kernel_length=250,
               pool_mode='mean', drop_prob=0.5, final_conv_length='auto')
    yp, ypb, ttrain, tinfer, npar = train_dl(
        m, Xtr_t, ytr_t, Xte_t, yte_t,
        lr=1e-3, betas=(0.9, 0.999), batch_size=64,
        n_epochs=100, patience=15, mixup_alpha=0.0)
    lat = measure_latency_ms(m, Xte_t, device)
    out['eegnet'] = {'train_s': round(ttrain, 2), 'train_min': round(ttrain / 60, 2),
                     'infer_ms': round(lat, 3), 'n_params': npar}
    print(f"EEGNet     train={ttrain:.1f}s ({ttrain/60:.2f} min)  latency={lat:.3f} ms/trial  params={npar}", flush=True)

    # ---- EEG-Conformer ----
    from braindecode.models import EEGConformer
    m = EEGConformer(n_outputs=6, n_chans=n_ch,
                     n_filters_time=40, filter_time_length=50,
                     pool_time_length=150, pool_time_stride=30,
                     drop_prob=0.5, num_layers=6, num_heads=10,
                     att_drop_prob=0.5, n_times=TRIAL_SAMPLES, final_fc_length='auto')
    yp, ypb, ttrain, tinfer, npar = train_dl(
        m, Xtr_t, ytr_t, Xte_t, yte_t,
        lr=2e-4, betas=(0.5, 0.999), batch_size=72,
        n_epochs=100, patience=15, mixup_alpha=0.0)
    lat = measure_latency_ms(m, Xte_t, device)
    out['eegconformer'] = {'train_s': round(ttrain, 2), 'train_min': round(ttrain / 60, 2),
                           'infer_ms': round(lat, 3), 'n_params': npar}
    print(f"Conformer   train={ttrain:.1f}s ({ttrain/60:.2f} min)  latency={lat:.3f} ms/trial  params={npar}", flush=True)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', default='sub1')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='results/cost_results.json')
    a = ap.parse_args()

    download_dataset()
    r = run(a.subject, a.seed)

    allr = {}
    if os.path.exists(a.out):
        with open(a.out) as f:
            allr = json.load(f)
    allr[a.subject] = r
    with open(a.out, 'w') as f:
        json.dump(allr, f, indent=2)
    print("SAVED ->", a.out)


if __name__ == '__main__':
    main()
