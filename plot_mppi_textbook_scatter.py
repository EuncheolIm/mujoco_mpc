#!/usr/bin/env python3
"""Textbook MPPI baseline — xy vs contact% across (K, H) — same scatter style
as pareto_scatter_4modes.png. Single panel, K×H grid (3-seed mean).
"""
import csv, math, os, statistics as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

DIR = "sweep_mppi_textbook_3seeds"
K_LIST = [8, 16, 32, 64, 128]
H_LIST = ["0.05", "0.10", "0.20", "0.30"]
SEEDS  = [1, 2, 3]
F_THRESH = 1.0
WIPE_T   = 5.0

H_MARKERS = {"0.05":"v", "0.10":"o", "0.20":"s", "0.30":"D"}
K_NORM = {K: 0.3 + 0.7 * i / (len(K_LIST)-1) for i, K in enumerate(K_LIST)}

def stats(path):
    n = 0; cn = 0; sq = 0.0
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                t = float(r["time"])
                if t <= WIPE_T: continue
                if float(r["hybrid"]) < 0.5: continue
                fz = float(r["Fz"])
                dx = float(r["ee_x"]) - float(r["tgt_x"])
                dy = float(r["ee_y"]) - float(r["tgt_y"])
            except (ValueError, KeyError):
                continue
            n += 1
            if fz > F_THRESH: cn += 1
            sq += dx*dx + dy*dy
    if n == 0: return None
    return 100.0 * cn / n, 1000 * math.sqrt(sq / n)

def cell_mean(K, H):
    out = []
    for s in SEEDS:
        p = f"{DIR}/T{K}_H{H}_s{s}.csv"
        if not os.path.exists(p): continue
        r = stats(p)
        if r: out.append(r)
    if not out: return None
    return st.mean(s[0] for s in out), st.mean(s[1] for s in out), len(out)

pts = []
for K in K_LIST:
    for H in H_LIST:
        m = cell_mean(K, H)
        if m: pts.append((K, H, *m))

fig, ax = plt.subplots(1, 1, figsize=(11, 7))

cmap = cm.get_cmap("Greys")

for K, H, c, xy, n in pts:
    ax.scatter(xy, c, s=150, c=[cmap(K_NORM[K])], marker=H_MARKERS[H],
               edgecolors='k', linewidths=0.6, alpha=0.9)
ax.set_xlabel("xy tracking error [mm]", fontsize=14)
ax.set_ylabel("wipe contact %", fontsize=14)
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3)
ax.annotate("better", xy=(0.03, 0.97), xytext=(0.18, 0.85),
            xycoords='axes fraction', fontsize=12, color='gray',
            arrowprops=dict(arrowstyle='->', color='gray'))

# Legends
h_handles = [Line2D([0],[0], marker=H_MARKERS[h], color='w',
                    markerfacecolor='gray', markeredgecolor='k',
                    markersize=14, label=f"H={h}") for h in H_LIST]
k_handles = [Line2D([0],[0], marker='o', color='w',
                    markerfacecolor=cmap(K_NORM[K]), markeredgecolor='k',
                    markersize=14, label=f"K={K}") for K in K_LIST]

fig.legend(handles=h_handles, title="horizon",
           loc='upper center', bbox_to_anchor=(0.23, 0.07),
           ncol=4, fontsize=11, title_fontsize=12)
fig.legend(handles=k_handles, title="rollouts (K, light→dark = small→large)",
           loc='upper center', bbox_to_anchor=(0.72, 0.07),
           ncol=5, fontsize=11, title_fontsize=12)

fig.suptitle("Textbook MPPI baseline — xy tracking error vs wipe contact ratio across (K, H)",
             fontsize=18, y=1.015)
fig.text(0.5, 0.96, "xy: lower = better,   contact %: higher = better",
         ha='center', fontsize=14, color='black', style='italic')
plt.tight_layout(rect=[0, 0.10, 1, 1.0])
out = "mppi_textbook_scatter.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"saved: {out}  ({len(pts)} cells)")
