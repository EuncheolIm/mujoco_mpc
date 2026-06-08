#!/usr/bin/env python3
"""Pareto scatter: 4 FlowMPPI modes (subplot per mode).
x: xy [mm], y: contact%. Marker shape = H, color shade = K.
"""
import csv, math, os, statistics as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

F_THRESH = 1.0
WIPE_T   = 5.0
FM_DIR   = "sweep_flowmppi_4modes_3seeds"

MODES   = ["wta1", "wta2", "wta3", "cost"]
MODE_TITLES = {"wta1":"Warm start #1 (fm_frac=1.0)",
               "wta2":"Warm start #2 (shared softmax, half-half)",
               "wta3":"Warm start #3 (per-group, half-half)",
               "cost":"FM as cost"}
K_LIST  = [8, 16, 32, 64, 128]
H_LIST  = ["0.05", "0.10", "0.20", "0.30"]
SEEDS   = [1, 2, 3]

# Per-mode colormap (mode → cmap)
MODE_CMAP = {"wta1":"Reds", "wta2":"Oranges", "wta3":"Greens", "cost":"Blues"}
H_MARKERS = {"0.05":"v", "0.10":"o", "0.20":"s", "0.30":"D"}
# Normalize K to [0.3, 1.0] in the colormap so even smallest K is visible
K_NORM = {K: 0.3 + 0.7 * i / (len(K_LIST)-1) for i, K in enumerate(K_LIST)}

def stats(path):
    rs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try: rs.append({k: float(v) for k, v in r.items() if v != ''})
            except ValueError: pass
    if not rs: return None
    wipe = [r for r in rs if r['time'] > WIPE_T and r['hybrid'] >= 0.5]
    if not wipe: return None
    n = len(wipe)
    contact = 100.0 * sum(1 for r in wipe if r['Fz'] > F_THRESH) / n
    xy = 1000 * math.sqrt(sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2
                              for r in wipe) / n)
    return contact, xy

def cell_mean(paths):
    out = []
    for p in paths:
        if not os.path.exists(p): continue
        s = stats(p)
        if s is not None: out.append(s)
    if not out: return None
    return st.mean(s[0] for s in out), st.mean(s[1] for s in out), len(out)

pts = []
for mode in MODES:
    for K in K_LIST:
        for H in H_LIST:
            paths = [f"{FM_DIR}/{mode}_T{K}_H{H}_s{s}.csv" for s in SEEDS]
            m = cell_mean(paths)
            if m: pts.append((mode, K, H, *m))

# 2x2 grid, one subplot per mode. Zoom to clean-wipe region (xy<5mm).
fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True)
axes_flat = axes.flatten()

XLIM = (1.3, 5.0)
YLIM = (70, 90)

for ax, mode in zip(axes_flat, MODES):
    cmap = cm.get_cmap(MODE_CMAP[mode])
    for m, K, H, c, xy, n in pts:
        if m != mode: continue
        if xy > XLIM[1]: continue   # collision region excluded from zoom
        ax.scatter(xy, c,
                   s=110,
                   c=[cmap(K_NORM[K])],
                   marker=H_MARKERS[H],
                   edgecolors='k', linewidths=0.6, alpha=0.9)
    ax.set_title(MODE_TITLES[mode], fontsize=11)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.grid(True, alpha=0.3)

# Shared axis labels
for ax in axes[1]:
    ax.set_xlabel("xy tracking error [mm]  (lower = better)", fontsize=11)
for ax in axes[:, 0]:
    ax.set_ylabel("wipe contact %  (higher = better)", fontsize=11)

# Annotate 'better' arrow on first subplot
axes[0,0].annotate("better", xy=(0.03, 0.97), xytext=(0.18, 0.85),
                   xycoords='axes fraction', fontsize=10, color='gray',
                   arrowprops=dict(arrowstyle='->', color='gray'))

# Legend: H markers + K colors (using neutral gray cmap for K legend)
h_handles = [Line2D([0],[0], marker=H_MARKERS[h], color='w',
                    markerfacecolor='gray', markeredgecolor='k',
                    markersize=10, label=f"H={h}") for h in H_LIST]
k_gray = cm.get_cmap("Greys")
k_handles = [Line2D([0],[0], marker='o', color='w',
                    markerfacecolor=k_gray(K_NORM[K]), markeredgecolor='k',
                    markersize=10, label=f"K={K}") for K in K_LIST]

fig.legend(handles=h_handles, title="horizon", loc='upper center',
           bbox_to_anchor=(0.30, 0.02), ncol=4, fontsize=10, title_fontsize=11)
fig.legend(handles=k_handles, title="rollouts (K)  ← light: small, dark: large",
           loc='upper center', bbox_to_anchor=(0.72, 0.02), ncol=5,
           fontsize=10, title_fontsize=11)

fig.suptitle("4 FlowMPPI modes — Pareto scatter in clean-wipe region (xy<5mm)\n"
             "mean over 3 seeds; shade ∝ K (rollouts), marker = H (horizon)",
             fontsize=12, y=1.00)
plt.tight_layout(rect=[0, 0.05, 1, 1.0])
out = "pareto_scatter_4modes.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"saved: {out}  ({len(pts)} cells total)")
