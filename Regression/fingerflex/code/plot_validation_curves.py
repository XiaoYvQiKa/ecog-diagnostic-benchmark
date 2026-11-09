"""
FingerFlex validation-curve plotting script.

Reads the per-epoch corr_mean_val recorded in lightning_logs/version_X/metrics.csv
and plots the full validation curve for the three subjects (not a few scattered
checkpoint points, but the true per-epoch data that PyTorch Lightning logs
automatically).

Usage:
    python plot_validation_curves.py

Before running, confirm that the version numbers in VERSION_MAP below match the
actual subjects (e.g. spot-check with `cat lightning_logs\\version_X\\metrics.csv`).
"""

import pandas as pd
import matplotlib.pyplot as plt

# ---- Mapping inferred from timestamps; spot-check before running ----
VERSION_MAP = {
    "S1 (62ch)": "lightning_logs/version_1/metrics.csv",
    "S2 (48ch)": "lightning_logs/version_3/metrics.csv",
    "S3 (63ch, ch49 excluded)": "lightning_logs/version_5/metrics.csv",
}

COLORS = {
    "S1 (62ch)": "#1F3864",
    "S2 (48ch)": "#ED7D31",
    "S3 (63ch, ch49 excluded)": "#70AD47",
}

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

for label, path in VERSION_MAP.items():
    df = pd.read_csv(path)

    # metrics.csv can hold multiple rows per epoch (train/val recorded separately);
    # keep only rows with a corr_mean_val value, and aggregate by epoch (last non-null).
    if "corr_mean_val" not in df.columns:
        print(f"[warning] {path} has no corr_mean_val column; actual columns: {list(df.columns)}")
        continue

    val_df = df.dropna(subset=["corr_mean_val"])[["epoch", "corr_mean_val"]]
    val_df = val_df.groupby("epoch", as_index=False).last()  # last record per epoch
    val_df = val_df.sort_values("epoch")

    print(f"{label}: {len(val_df)} epoch points, range epoch {val_df['epoch'].min()}~{val_df['epoch'].max()}, "
          f"max corr_mean_val={val_df['corr_mean_val'].max():.4f} (epoch {val_df.loc[val_df['corr_mean_val'].idxmax(), 'epoch']:.0f})")

    ax.plot(val_df["epoch"], val_df["corr_mean_val"], marker='o', markersize=4,
            linewidth=2, label=label, color=COLORS[label])

    # Mark the best point
    best_idx = val_df["corr_mean_val"].idxmax()
    best_epoch = val_df.loc[best_idx, "epoch"]
    best_val = val_df.loc[best_idx, "corr_mean_val"]
    ax.annotate(f'{best_val:.3f}', xy=(best_epoch, best_val),
                xytext=(0, 8), textcoords="offset points",
                ha='center', fontsize=8, color=COLORS[label], fontweight='bold')

ax.axhline(0.74, color='#C00000', linestyle='--', linewidth=1, label="Paper's target (0.74)", zorder=0)

ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Validation Pearson r (corr_mean_val)', fontsize=11)
ax.set_title('FingerFlex: Validation Curve by Subject', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#E7E6E6', linewidth=0.6, zorder=0)

plt.tight_layout()
plt.savefig('fingerflex_validation_curves.png', dpi=300, bbox_inches='tight')
print("\nSaved: fingerflex_validation_curves.png")
plt.show()
