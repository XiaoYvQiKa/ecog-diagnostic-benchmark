# -*- coding: utf-8 -*-
"""Diagnose the scale / bias offset of a DTCNet prediction on one Miller subject.

Loads the seed-0 model and the test spectrogram + finger trace, reproduces the
sliding-window inference from train.py evaluate, and reports, per finger:
  - pred_std / target_std (whether the prediction scale is systematically small)
  - pred_mean vs target_mean (bias)
  - the optimal affine rescaling a*pred + b (scale coefficient a, offset b)
  - r, R2, and the R2 after optimal rescaling (the recoverable part of the gap)

This separates "scale too small", "bias offset", and "weak signal (overfitting)"
as competing explanations for a low R2.

Usage:
    python verify_dtcnet_scale.py wc
"""
import os
import sys

import numpy as np
import torch

import paths as P

sys.path.insert(0, str(P.DTC_CODE_DIR))
from train import C, load_subject  # noqa: E402
from model import DTCNet  # noqa: E402

FINGER = ["thumb", "index", "middle", "ring", "little"]


def main():
    sid = sys.argv[1] if len(sys.argv) > 1 else "wc"
    seed = 0
    cfg = C()
    cfg.data_root = str(P.DTC_DATA_MILLER)
    cfg.output_dir = str(P.DTC_DIR / "results" / "main_miller")
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = os.path.join(cfg.output_dir, f"sub{sid}_seed{seed}_model.pt")
    print(f"loading {model_path}")
    spec_path = os.path.join(cfg.data_root, f"sub{sid}_test_spec.npy")
    _probe = np.load(spec_path, mmap_mode='r')
    n_ch, n_fr = _probe.shape[0], _probe.shape[1]
    del _probe
    m = DTCNet(n_channels=n_ch, n_freqs=n_fr, dropout=0.1, output_mode="trajectory").to(cfg.device)
    m.load_state_dict(torch.load(model_path, map_location=cfg.device, weights_only=False))
    m.eval()

    _, _, ds_test, _, _ = load_subject(sid, cfg)
    s_data = ds_test.s   # (ch, freq, T)
    f_data = ds_test.f   # (5, T)
    T = s_data.shape[2]
    w = cfg.window
    n_tot = (T - w) // 1 + 1
    print(f"test: T={T}, windows={n_tot}")

    bs_eval = 16
    P_, Tp = [], []
    with torch.no_grad():
        for start in range(0, n_tot, bs_eval):
            end = min(start + bs_eval, n_tot)
            xb = np.zeros((end - start, s_data.shape[0], s_data.shape[1], w), dtype=np.float32)
            yb = np.zeros((end - start, 5), dtype=np.float32)
            for j, idx in enumerate(range(start, end)):
                st = idx * 1
                xb[j] = s_data[..., st:st + w]
                yb[j] = f_data[:, st + w - 1]
            pred = m(torch.from_numpy(xb).to(cfg.device))
            pred = pred[:, :, -1]
            P_.append(pred.cpu().numpy())
            Tp.append(yb)

    p = np.concatenate(P_).astype(np.float64)
    t = np.concatenate(Tp).astype(np.float64)
    print(f"pred shape {p.shape}, target shape {t.shape}")

    print(f"\n{'finger':8s} {'t_std':>8s} {'p_std':>8s} {'p/t_std':>8s} "
          f"{'t_mean':>9s} {'p_mean':>9s} {'a':>7s} {'b':>9s} {'r':>7s} {'R2':>8s} {'R2_resc':>8s}")
    for i, name in enumerate(FINGER):
        tp, pp = t[:, i], p[:, i]
        a, b = np.polyfit(pp, tp, 1)          # optimal affine rescaling tp ~ a*pp + b
        pp_resc = a * pp + b
        r = float(np.corrcoef(pp, tp)[0, 1])
        sst = np.sum((tp - tp.mean()) ** 2)
        r2 = 1 - np.sum((pp - tp) ** 2) / (sst + 1e-12)
        r2_resc = 1 - np.sum((pp_resc - tp) ** 2) / (sst + 1e-12)
        print(f"{name:8s} {tp.std():8.1f} {pp.std():8.1f} {pp.std()/(tp.std()+1e-12):8.3f} "
              f"{tp.mean():9.1f} {pp.mean():9.1f} {a:7.3f} {b:9.1f} {r:7.3f} {r2:8.3f} {r2_resc:8.3f}")

    print(f"\n=== summary ===")
    print(f"pred_std/target_std per finger = "
          f"{[round(p[:, i].std()/(t[:, i].std()+1e-12), 3) for i in range(5)]}")
    print(f"optimal rescaling a (scale) = "
          f"{[round(float(np.polyfit(p[:, i], t[:, i], 1)[0]), 3) for i in range(5)]}")
    print(f"optimal rescaling b (bias) = "
          f"{[round(float(np.polyfit(p[:, i], t[:, i], 1)[1]), 1) for i in range(5)]}")


if __name__ == "__main__":
    main()
