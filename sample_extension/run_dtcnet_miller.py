# -*- coding: utf-8 -*-
"""DTCNet training on the added Miller subjects (jc, wc), three seeds each.

Reuses train_subject + set_seed from dtcnet/code/train.py, replacing only the
data root and the subject identifier (jc/wc strings instead of 1/2/3). All other
hyperparameters (window 256 / stride 1 / lr 8.42e-5 / wd 1e-6 / dropout 0.1 /
MSE + cosine / trajectory output) inherit the original configuration. Three
seeds ([0, 1, 2]) report mean +- std, matching the original protocol.

Training is resume-safe: train.py checkpoints every 5 epochs, and results.json
is read to skip completed seeds. stdout/stderr are also tee-ed to train_log.txt
so progress survives a killed task.

Usage:
    python run_dtcnet_miller.py jc wc
"""
import os
import sys
import json

import numpy as np
import torch

import paths as P

sys.path.insert(0, str(P.DTC_CODE_DIR))
from train import C, train_subject, set_seed  # noqa: E402

cfg = C()
cfg.data_root = str(P.DTC_DATA_MILLER)
cfg.output_dir = str(P.DTC_DIR / "results" / "main_miller")
os.makedirs(cfg.output_dir, exist_ok=True)

# Tee stdout/stderr to a log file so progress survives a killed task.
_log_path = os.path.join(cfg.output_dir, "train_log.txt")
_log_fh = open(_log_path, "a", encoding="utf-8", buffering=1)


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


sys.stdout = _Tee(sys.stdout, _log_fh)
sys.stderr = _Tee(sys.stderr, _log_fh)

SEEDS = [0, 1, 2]
OUT_JSON = os.path.join(cfg.output_dir, "results.json")


def main():
    subs = sys.argv[1:] if len(sys.argv) > 1 else ["jc", "wc"]
    all_r = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            all_r = json.load(f)

    for s in subs:
        entry = all_r.get(s, {})
        per_seed = entry.get("per_seed", {})
        for seed in SEEDS:
            sk = str(seed)
            if sk in per_seed:
                print(f"{s} seed {seed} already done, skip.")
                continue
            print(f"\n=== {s} seed {seed} ===", flush=True)
            try:
                r, best_train_loss = train_subject(s, cfg, seed=seed)
                per_seed[sk] = {"mean": r, "best_train_loss": best_train_loss}
                all_r[s] = {"seeds": SEEDS, "per_seed": per_seed, "n": len(per_seed)}
                with open(OUT_JSON, "w") as f:
                    json.dump(all_r, f, indent=2)
                print(f"[SAVED] {s} seed {seed}")
            except Exception as e:
                print(f"[ERROR] {s} seed {seed}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                torch.cuda.empty_cache()

        if len(per_seed) == len(SEEDS):
            first = per_seed[str(SEEDS[0])]["mean"]
            mean, std = {}, {}
            for key in first.keys():
                if key == "infer_ms":
                    continue
                vals = [per_seed[str(sd)]["mean"][key] for sd in SEEDS]
                mean[key] = float(np.mean(vals))
                std[key] = float(np.std(vals))
            all_r[s] = {"seeds": SEEDS, "per_seed": per_seed, "mean": mean, "std": std, "n": len(per_seed)}
            with open(OUT_JSON, "w") as f:
                json.dump(all_r, f, indent=2)
            print(f"[FINAL] {s}: official_r = {mean['official_r']:.4f} +- {std['official_r']:.4f}")

    print(f"\nDone. Subjects completed: {[k for k in all_r]}")


if __name__ == "__main__":
    main()
