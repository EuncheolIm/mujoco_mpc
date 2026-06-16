#!/usr/bin/env python3
"""MPPI baseline vs FlowMPPI cost — plan_ms vs (K, H).
Shows how planning time scales with K and H for both modes.
2 panels (one per mode); x=K (log), lines per H.
"""
import csv, os, statistics as st
import matplotlib.pyplot as plt

MPPI_DIR = "sweep_mppi_hinge_3seeds"
COST_DIR = "sweep_flowmppi_4modes_3seeds"
K_LIST = [8, 16, 32, 64, 128]
H_LIST = ["0.05", "0.10", "0.20", "0.30"]
SEEDS  = [1, 2, 3]

def plan_ms(path):
    if not os.path.exists(path): return None
    vals = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                pm = float(r.get("plan_ms", "0"))
                if pm > 0: vals.append(pm)
            except: pass
    if not vals: return None
    return st.mean(vals)

def cell(paths):
    out = [plan_ms(p) for p in paths]
    out = [v for v in out if v is not None]
    if not out: return None
    return st.mean(out), (st.stdev(out) if len(out) > 1 else 0)

def gather(prefix_fn):
    """Return dict: data[H] = [(K, mean, std), ...]"""
    data = {h: [] for h in H_LIST}
    for K in K_LIST:
        for H in H_LIST:
            m = cell(prefix_fn(K, H))
            if m: data[H].append((K, m[0], m[1]))
    return data

mppi_data = gather(lambda K,H: [f"{MPPI_DIR}/T{K}_H{H}_s{s}.csv" for s in SEEDS])
cost_data = gather(lambda K,H: [f"{COST_DIR}/cost_T{K}_H{H}_s{s}.csv" for s in SEEDS])

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
H_COLORS = {"0.05":"#2ca02c", "0.10":"#1f77b4", "0.20":"#ff7f0e", "0.30":"#d62728"}

for ax, (mode_data, title) in zip(axes, [(mppi_data, "MPPI baseline"),
                                          (cost_data, "FM as cost")]):
    for H in H_LIST:
        pts = mode_data[H]
        if not pts: continue
        Ks   = [p[0] for p in pts]
        mean = [p[1] for p in pts]
        std  = [p[2] for p in pts]
        lo   = [m - s for m, s in zip(mean, std)]
        hi   = [m + s for m, s in zip(mean, std)]
        ax.plot(Ks, mean, marker="o", lw=2, ms=8,
                color=H_COLORS[H], label=f"H = {H} s")
        ax.fill_between(Ks, lo, hi, color=H_COLORS[H], alpha=0.25, linewidth=0)
        # Also annotate std values next to each point
        for K, m, s in zip(Ks, mean, std):
            ax.annotate(f"±{s:.2f}", (K, m), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=H_COLORS[H], alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_LIST)
    ax.set_xticklabels([str(K) for K in K_LIST])
    ax.set_xlabel("rollouts K  (log scale)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(title="horizon", fontsize=11, title_fontsize=12)

axes[0].set_ylabel("plan_ms  (per planning iteration)", fontsize=13)

fig.suptitle("Per-iteration planning time vs (K, H)  —  3-seed mean ± std",
             fontsize=15, y=1.02)
plt.tight_layout()
out = "plan_ms_vs_KH.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
print("MPPI plan_ms table:")
for H in H_LIST:
    print(f"  H={H}: {[f'{p[0]}→{p[1]:.2f}±{p[2]:.2f}ms' for p in mppi_data[H]]}")
print("FM-cost plan_ms table:")
for H in H_LIST:
    print(f"  H={H}: {[f'{p[0]}→{p[1]:.2f}±{p[2]:.2f}ms' for p in cost_data[H]]}")
