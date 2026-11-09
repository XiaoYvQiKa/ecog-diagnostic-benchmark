"""DTCNet training launcher (multi-seed, resume-safe).

Usage:
  python run_direct.py              # all 3 subjects, trajectory output (main experiment)
  python run_direct.py --subject 1  # single subject
  python run_direct.py --output single  # single-point output (output-layer ablation)
  python run_direct.py --data-root /path/to/data_ablation/freq_20band  # frequency-resolution ablation
  python run_direct.py --seed 42    # single-seed run (ablations use a fixed seed = 42)
"""
import sys, os, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from train import C, train_subject, set_seed

cfg = C()
cfg.output_dir = os.path.join('..', 'results', 'main')
os.makedirs(cfg.output_dir, exist_ok=True)

# parse --output (trajectory / single)
if '--output' in sys.argv:
    idx = sys.argv.index('--output')
    cfg.output_mode = sys.argv[idx + 1]

# parse --data-root (for frequency-resolution ablation)
if '--data-root' in sys.argv:
    idx = sys.argv.index('--data-root')
    cfg.data_root = sys.argv[idx + 1]

# parse --out (results directory)
if '--out' in sys.argv:
    idx = sys.argv.index('--out')
    cfg.output_dir = sys.argv[idx + 1]
    os.makedirs(cfg.output_dir, exist_ok=True)

# redirect stdout to log file (line-buffered, UTF-8)
log_path = os.path.join(cfg.output_dir, 'train_log.txt')
log_fh = open(log_path, 'a', encoding='utf-8', buffering=1)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, log_fh)

# parse --subject
subs = [1, 2, 3]
if '--subject' in sys.argv:
    idx = sys.argv.index('--subject')
    arg = sys.argv[idx + 1]
    subs = [1, 2, 3] if arg == 'all' else [int(arg)]

# 3-seed configuration (assesses training randomness; per-seed results + mean ± std)
SEEDS = [0, 1, 2]

# parse --seed <int>  → single-seed run (used by the ablations, which are fixed at seed 42)
if '--seed' in sys.argv:
    idx = sys.argv.index('--seed')
    SEEDS = [int(sys.argv[idx + 1])]

print(f"Config: output_mode={cfg.output_mode}  data_root={cfg.data_root}  out={cfg.output_dir}  seeds={SEEDS}")

# load existing results (resume-safe)
rf = os.path.join(cfg.output_dir, 'results.json')
all_r = {}
if os.path.exists(rf):
    with open(rf) as f:
        all_r = json.load(f)

for s in subs:
    entry = all_r.get(f"sub{s}", {})
    per_seed = entry.get("per_seed", {})
    if len(per_seed) >= len(SEEDS):
        print(f"Subject {s} already has {len(SEEDS)} seeds, skipping.")
        continue

    print(f"\n{'='*60}\nSubject {s}\n{'='*60}")
    for seed in SEEDS:
        sk = str(seed)
        if sk in per_seed:
            print(f"  seed {seed} already done, skipping.")
            continue
        print(f"  --- seed {seed} ---")
        try:
            r, best_train_loss = train_subject(s, cfg, seed=seed)
            per_seed[sk] = {"mean": r, "best_train_loss": best_train_loss}
            all_r[f"sub{s}"] = {"seeds": SEEDS, "per_seed": per_seed, "n": len(per_seed)}
            with open(rf, "w") as f:
                json.dump(all_r, f, indent=2)
            print(f"  [SAVED] sub{s} seed{seed}")
        except Exception as e:
            print(f"  [ERROR] sub{s} seed{seed}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()  # GPU cleanup between runs

    # once all seeds complete, compute mean ± std over seeds
    if len(per_seed) == len(SEEDS):
        mean, std = {}, {}
        first = per_seed[str(SEEDS[0])]["mean"]
        for key in first.keys():
            if key == "infer_ms":
                continue
            vals = [per_seed[str(sd)]["mean"][key] for sd in SEEDS]
            mean[key] = float(np.mean(vals))
            std[key] = float(np.std(vals))
        all_r[f"sub{s}"] = {"seeds": SEEDS, "per_seed": per_seed,
                            "mean": mean, "std": std, "n": len(per_seed)}
        with open(rf, "w") as f:
            json.dump(all_r, f, indent=2)
        print(f"  [FINAL] sub{s}: official_r = {mean['official_r']:.4f} ± {std['official_r']:.4f}")

print(f"\nDone. Subjects completed: {[k for k in all_r]}")
sys.stdout.files = [f for f in sys.stdout.files if f != log_fh]
log_fh.close()
