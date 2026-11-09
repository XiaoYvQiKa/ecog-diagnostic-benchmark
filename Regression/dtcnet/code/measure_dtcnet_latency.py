"""Measure DTCNet per-sample inference latency and parameter count.

Reports the single-window forward latency (GPU if available) and the number of
trainable parameters for the three channel counts used in the manuscript
(61 / 46 / 63, after bad-electrode exclusion). The numbers back Table G1.

Usage:
    python code/measure_dtcnet_latency.py [--out ../results/dtcnet_latency.json]
"""
import sys, os, time, json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DTCNet

# Parse optional --out path (default: ../results/dtcnet_latency.json)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "results", "dtcnet_latency.json")
if "--out" in sys.argv:
    out_path = sys.argv[sys.argv.index("--out") + 1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device={device}  torch={torch.__version__}")

results = {}
for ch in [61, 46, 63]:   # sub1 / sub2 / sub3 channel counts (after bad-electrode exclusion)
    m = DTCNet(n_channels=ch, n_freqs=40, dropout=0.1, output_mode='trajectory').to(device)
    m.eval()
    x = torch.randn(1, ch, 40, 256, device=device)
    with torch.no_grad():
        for _ in range(10):       # warmup
            m(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            m(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 200
    lat_ms = dt * 1000.0
    npar = m.get_param_count()
    results[ch] = {'params': npar, 'latency_ms': round(lat_ms, 3)}
    print(f"ch={ch:2d}: params={npar:>8,}  latency={lat_ms:.3f} ms/sample", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"SAVED -> {out_path}")
