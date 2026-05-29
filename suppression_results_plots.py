import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv(
    r"C:/Users/20231229/OneDrive - TU Eindhoven/Documents/Thesis Code/results/suppression_results/suppression_results.csv"
)

# Compute mean and std across seeds
grouped = df.groupby(['suppression_type', 'strength']).agg(
    baseline_rel_mean=('baseline_relative', 'mean'),
    baseline_rel_std=('baseline_relative', 'std'),
    inbiased_rel_mean=('inbiased_relative', 'mean'),
    inbiased_rel_std=('inbiased_relative', 'std'),
).reset_index()

# ============================================================
# PLOT SETTINGS
# ============================================================

BASELINE_COLOR = '#2196F3'   # blue
INBIASED_COLOR = '#FF5722'   # orange
LINEWIDTH = 2.0
MARKERSIZE = 7
ALPHA_FILL = 0.15

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150
})

# ============================================================
# FIGURE 1 — SHAPE SUPPRESSION (Main thesis figure)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    'Figure 1: Shape Suppression — Baseline vs InBiaseD ResNet-18',
    fontsize=14, fontweight='bold', y=1.02
)

# --- Patch Shuffle ---
ax = axes[0]
data = grouped[grouped['suppression_type'] == 'patch_shuffle'].sort_values('strength')

ax.plot(
    data['strength'], data['baseline_rel_mean'],
    color=BASELINE_COLOR, linewidth=LINEWIDTH,
    marker='o', markersize=MARKERSIZE, label='Baseline'
)
ax.fill_between(
    data['strength'],
    data['baseline_rel_mean'] - data['baseline_rel_std'],
    data['baseline_rel_mean'] + data['baseline_rel_std'],
    alpha=ALPHA_FILL, color=BASELINE_COLOR
)

ax.plot(
    data['strength'], data['inbiased_rel_mean'],
    color=INBIASED_COLOR, linewidth=LINEWIDTH,
    marker='s', markersize=MARKERSIZE, label='InBiaseD'
)
ax.fill_between(
    data['strength'],
    data['inbiased_rel_mean'] - data['inbiased_rel_std'],
    data['inbiased_rel_mean'] + data['inbiased_rel_std'],
    alpha=ALPHA_FILL, color=INBIASED_COLOR
)

ax.set_xlabel('Grid Size (small=global, large=local)')
ax.set_ylabel('Relative Accuracy')
ax.set_title('Patch Shuffle\n(Global → Local Shape Suppression)')
ax.set_ylim(0, 1.05)
ax.set_xticks(data['strength'])
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

# --- Patch Rotation ---
ax = axes[1]
data = grouped[grouped['suppression_type'] == 'patch_rotation'].sort_values('strength')

ax.plot(
    data['strength'], data['baseline_rel_mean'],
    color=BASELINE_COLOR, linewidth=LINEWIDTH,
    marker='o', markersize=MARKERSIZE, label='Baseline'
)
ax.fill_between(
    data['strength'],
    data['baseline_rel_mean'] - data['baseline_rel_std'],
    data['baseline_rel_mean'] + data['baseline_rel_std'],
    alpha=ALPHA_FILL, color=BASELINE_COLOR
)

ax.plot(
    data['strength'], data['inbiased_rel_mean'],
    color=INBIASED_COLOR, linewidth=LINEWIDTH,
    marker='s', markersize=MARKERSIZE, label='InBiaseD'
)
ax.fill_between(
    data['strength'],
    data['inbiased_rel_mean'] - data['inbiased_rel_std'],
    data['inbiased_rel_mean'] + data['inbiased_rel_std'],
    alpha=ALPHA_FILL, color=INBIASED_COLOR
)

ax.set_xlabel('Grid Size (small=global, large=local)')
ax.set_ylabel('Relative Accuracy')
ax.set_title('Patch Rotation\n(Local Shape Suppression)')
ax.set_ylim(0, 1.05)
ax.set_xticks(data['strength'])
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    r"C:/Users/20231229/OneDrive - TU Eindhoven/Documents/Thesis Code/results/figure1_shape_suppression.png",
    bbox_inches='tight', dpi=150
)
plt.show()
print("Figure 1 saved!")

# ============================================================
# FIGURE 2 — TEXTURE AND COLOR SUPPRESSION (Control figure)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    'Figure 2: Control Suppressions — Texture and Color',
    fontsize=14, fontweight='bold', y=1.02
)

# --- Gaussian Blur ---
ax = axes[0]
data = grouped[grouped['suppression_type'] == 'gaussian_blur'].sort_values('strength')

ax.plot(
    data['strength'], data['baseline_rel_mean'],
    color=BASELINE_COLOR, linewidth=LINEWIDTH,
    marker='o', markersize=MARKERSIZE, label='Baseline'
)
ax.fill_between(
    data['strength'],
    data['baseline_rel_mean'] - data['baseline_rel_std'],
    data['baseline_rel_mean'] + data['baseline_rel_std'],
    alpha=ALPHA_FILL, color=BASELINE_COLOR
)

ax.plot(
    data['strength'], data['inbiased_rel_mean'],
    color=INBIASED_COLOR, linewidth=LINEWIDTH,
    marker='s', markersize=MARKERSIZE, label='InBiaseD'
)
ax.fill_between(
    data['strength'],
    data['inbiased_rel_mean'] - data['inbiased_rel_std'],
    data['inbiased_rel_mean'] + data['inbiased_rel_std'],
    alpha=ALPHA_FILL, color=INBIASED_COLOR
)

ax.set_xlabel('Kernel Size')
ax.set_ylabel('Relative Accuracy')
ax.set_title('Gaussian Blur\n(Texture Suppression)')
ax.set_ylim(0, 1.05)
ax.set_xticks(data['strength'])
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

# --- Grayscale ---
ax = axes[1]
data = grouped[grouped['suppression_type'] == 'grayscale'].sort_values('strength')

ax.plot(
    data['strength'], data['baseline_rel_mean'],
    color=BASELINE_COLOR, linewidth=LINEWIDTH,
    marker='o', markersize=MARKERSIZE, label='Baseline'
)
ax.fill_between(
    data['strength'],
    data['baseline_rel_mean'] - data['baseline_rel_std'],
    data['baseline_rel_mean'] + data['baseline_rel_std'],
    alpha=ALPHA_FILL, color=BASELINE_COLOR
)

ax.plot(
    data['strength'], data['inbiased_rel_mean'],
    color=INBIASED_COLOR, linewidth=LINEWIDTH,
    marker='s', markersize=MARKERSIZE, label='InBiaseD'
)
ax.fill_between(
    data['strength'],
    data['inbiased_rel_mean'] - data['inbiased_rel_std'],
    data['inbiased_rel_mean'] + data['inbiased_rel_std'],
    alpha=ALPHA_FILL, color=INBIASED_COLOR
)

ax.set_xlabel('Grayscale Strength (alpha)')
ax.set_ylabel('Relative Accuracy')
ax.set_title('Grayscale\n(Color Suppression)')
ax.set_ylim(0, 1.05)
ax.set_xticks(data['strength'])
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    r"C:/Users/20231229/OneDrive - TU Eindhoven/Documents/Thesis Code/results/figure2_control_suppression.png",
    bbox_inches='tight', dpi=150
)
plt.show()
print("Figure 2 saved!")

# ============================================================
# FIGURE 3 — COMBINED OVERVIEW (For thesis presentation)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Figure 3: Feature Suppression Analysis — Baseline vs InBiaseD ResNet-18 on TinyImageNet',
    fontsize=14, fontweight='bold'
)

suppression_configs = [
    ('patch_shuffle', 'Grid Size (small=global, large=local)',
     'Patch Shuffle (Shape)', axes[0, 0]),
    ('patch_rotation', 'Grid Size (small=global, large=local)',
     'Patch Rotation (Shape)', axes[0, 1]),
    ('gaussian_blur', 'Kernel Size',
     'Gaussian Blur (Texture)', axes[1, 0]),
    ('grayscale', 'Grayscale Strength (alpha)',
     'Grayscale (Color)', axes[1, 1]),
]

for sup_type, xlabel, title, ax in suppression_configs:
    data = grouped[
        grouped['suppression_type'] == sup_type
    ].sort_values('strength')

    ax.plot(
        data['strength'], data['baseline_rel_mean'],
        color=BASELINE_COLOR, linewidth=LINEWIDTH,
        marker='o', markersize=MARKERSIZE, label='Baseline'
    )
    ax.fill_between(
        data['strength'],
        data['baseline_rel_mean'] - data['baseline_rel_std'],
        data['baseline_rel_mean'] + data['baseline_rel_std'],
        alpha=ALPHA_FILL, color=BASELINE_COLOR
    )

    ax.plot(
        data['strength'], data['inbiased_rel_mean'],
        color=INBIASED_COLOR, linewidth=LINEWIDTH,
        marker='s', markersize=MARKERSIZE, label='InBiaseD'
    )
    ax.fill_between(
        data['strength'],
        data['inbiased_rel_mean'] - data['inbiased_rel_std'],
        data['inbiased_rel_mean'] + data['inbiased_rel_std'],
        alpha=ALPHA_FILL, color=INBIASED_COLOR
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Relative Accuracy')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(data['strength'])
    ax.axhline(
        y=1.0, color='gray',
        linestyle='--', linewidth=0.8, alpha=0.5
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    r"C:/Users/20231229/OneDrive - TU Eindhoven/Documents/Thesis Code/results/figure3_combined_suppression.png",
    bbox_inches='tight', dpi=150
)
plt.show()
print("Figure 3 saved!")