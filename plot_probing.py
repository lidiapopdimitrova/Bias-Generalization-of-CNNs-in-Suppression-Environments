import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv(
    r"C:\Users\20231229\OneDrive - TU Eindhoven\Documents\Thesis Code\results\probing_results.csv"
)

# Compute mean and std across seeds
grouped = df.groupby(['model', 'layer', 'suppression']).agg(
    mean_change=('rep_change', 'mean'),
    std_change=('rep_change', 'std'),
).reset_index()

# Layer ordering
layer_order = ['layer1', 'layer2', 'layer3', 'layer4']
layer_labels = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']

# ============================================================
# PLOT SETTINGS
# ============================================================

BASELINE_COLOR = '#2196F3'   # blue
INBIASED_COLOR = '#FF5722'   # orange
LINEWIDTH = 2.0
MARKERSIZE = 8
ALPHA_FILL = 0.15

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150
})

def get_model_data(grouped, suppression, model):
    data = grouped[
        (grouped['suppression'] == suppression) &
        (grouped['model'] == model)
    ].copy()
    data['layer_idx'] = data['layer'].map(
        {l: i for i, l in enumerate(layer_order)}
    )
    return data.sort_values('layer_idx')


def plot_suppression(ax, suppression, title, ylabel=True):
    """Plot representational change for one suppression type."""
    for model, color, marker, label in [
        ('baseline', BASELINE_COLOR, 'o', 'Baseline'),
        ('inbiased', INBIASED_COLOR, 's', 'InBiaseD'),
    ]:
        data = get_model_data(grouped, suppression, model)
        x = range(len(layer_order))

        ax.plot(
            x, data['mean_change'],
            color=color, linewidth=LINEWIDTH,
            marker=marker, markersize=MARKERSIZE,
            label=label
        )
        ax.fill_between(
            x,
            data['mean_change'] - data['std_change'],
            data['mean_change'] + data['std_change'],
            alpha=ALPHA_FILL, color=color
        )

    ax.set_xticks(range(len(layer_order)))
    ax.set_xticklabels(layer_labels)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel('Representational Change\n(1 − Cosine Similarity)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)


# ============================================================
# FIGURE 1 — SHAPE SUPPRESSION (Main finding)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    'Figure 4: Layer-wise Representational Change Under Shape Suppression',
    fontsize=14, fontweight='bold', y=1.02
)

plot_suppression(axes[0], 'global_shape',
                 'Global Shape Suppression\n(Patch Shuffle grid=2)')
plot_suppression(axes[1], 'local_shape',
                 'Local Shape Suppression\n(Patch Shuffle grid=12)',
                 ylabel=False)

plt.tight_layout()
plt.savefig(
    r"C:\Users\20231229\OneDrive - TU Eindhoven\Documents\Thesis Code\results\figure4_probing_shape.png",
    bbox_inches='tight', dpi=150
)
plt.show()
print("Figure 4 saved!")


# ============================================================
# FIGURE 2 — TEXTURE AND COLOR (Controls)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    'Figure 5: Layer-wise Representational Change Under Texture and Color Suppression',
    fontsize=14, fontweight='bold', y=1.02
)

plot_suppression(axes[0], 'texture',
                 'Texture Suppression\n(Gaussian Blur k=11)')
plot_suppression(axes[1], 'color',
                 'Color Suppression\n(Full Grayscale)',
                 ylabel=False)

plt.tight_layout()
plt.savefig(
    r"C:\Users\20231229\OneDrive - TU Eindhoven\Documents\Thesis Code\results\figure5_probing_controls.png",
    bbox_inches='tight', dpi=150
)
plt.show()
print("Figure 5 saved!")


# ============================================================
# FIGURE 3 — COMBINED OVERVIEW (All four suppressions)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Figure 6: Layer-wise Representational Change — Baseline vs InBiaseD ResNet-18',
    fontsize=14, fontweight='bold'
)

configs = [
    ('global_shape', 'Global Shape Suppression\n(Patch Shuffle grid=2)', axes[0, 0], True),
    ('local_shape',  'Local Shape Suppression\n(Patch Shuffle grid=12)', axes[0, 1], False),
    ('texture',      'Texture Suppression\n(Gaussian Blur k=11)',         axes[1, 0], True),
    ('color',        'Color Suppression\n(Full Grayscale)',                axes[1, 1], False),
]

for suppression, title, ax, ylabel in configs:
    plot_suppression(ax, suppression, title, ylabel=ylabel)

plt.tight_layout()
plt.savefig(
    r"C:\Users\20231229\OneDrive - TU Eindhoven\Documents\Thesis Code\results\figure6_probing_combined.png",
    bbox_inches='tight', dpi=150
)
plt.show()
print("Figure 6 saved!")


# ============================================================
# FIGURE 4 — DIFFERENCE PLOT (Baseline minus InBiaseD)
# Shows where InBiaseD differs most from baseline
# ============================================================

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle(
    'Figure 7: Difference in Representational Change (InBiaseD − Baseline)\n'
    'Positive = InBiaseD changes more; Negative = InBiaseD changes less',
    fontsize=13, fontweight='bold', y=1.02
)

suppression_configs = [
    ('global_shape', 'Global Shape'),
    ('local_shape',  'Local Shape'),
    ('texture',      'Texture'),
    ('color',        'Color'),
]

for ax, (suppression, title) in zip(axes, suppression_configs):
    baseline_data = get_model_data(grouped, suppression, 'baseline')
    inbiased_data = get_model_data(grouped, suppression, 'inbiased')

    diff = inbiased_data['mean_change'].values - baseline_data['mean_change'].values
    colors = ['#FF5722' if d > 0 else '#2196F3' for d in diff]

    bars = ax.bar(
        range(len(layer_order)), diff,
        color=colors, alpha=0.8, edgecolor='black', linewidth=0.5
    )

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(layer_order)))
    ax.set_xticklabels(layer_labels, rotation=15, ha='right')
    ax.set_title(title)
    if ax == axes[0]:
        ax.set_ylabel('Difference in Representational Change')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, diff):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.002 if val >= 0 else -0.008),
            f'{val:+.3f}',
            ha='center', va='bottom' if val >= 0 else 'top',
            fontsize=9
        )

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF5722', alpha=0.8, label='InBiaseD changes more'),
    Patch(facecolor='#2196F3', alpha=0.8, label='InBiaseD changes less'),
]
fig.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=2,
    bbox_to_anchor=(0.5, -0.05),
    fontsize=11
)

plt.tight_layout()
plt.savefig(
    r"C:\Users\20231229\OneDrive - TU Eindhoven\Documents\Thesis Code\results\figure7_probing_difference.png",
    bbox_inches='tight', dpi=150
)
plt.show()
print("Figure 7 saved!")

print("\nAll probing figures saved successfully!")
