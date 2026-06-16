#!/usr/bin/env python3
"""MPPI baseline vs FM-as-cost — K sweep at H=0.10.
3 metrics × 2 modes line plot (mean ± std over 3 seeds).

Shows:
- Claim A: same K (K=128) comparison
- Claim B: cost K↓ achieves MPPI K=128 performance
"""
import csv, math, statistics as st
import matplotlib.pyplot as plt

MPPI_DIR = "sweep_mppi_3seeds"
COST_DIR = "sweep_flowmppi_4modes_3seeds"
K_LIST   = [8, 16, 32, 64, 128]
H        = "0.10"
SEEDS    = [1, 2, 3]
F_THRESH = 1.0
WIPE_T   = 5.0

def stats(path):
    """contact%, xy[mm], peak_Fz (phase2 wipe portion after transition + 1s)."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(r[k]) for k in
                             ("time","Fz","ee_x","ee_y","tgt_x","tgt_y","hybrid")})
            except (ValueError, KeyError):
                pass
    if not rows: return None
    rows.sort(key=lambda r: r["time"])
    # phase 2 transition (first hybrid>=0.5)
    t_trans = next((r["time"] for r in rows if r["hybrid"] >= 0.5), None)
    if t_trans is None: return None
    # peak Fz in first 1s after phase transition (P2 transient)
    p2_init = [r for r in rows if r["hybrid"] >= 0.5 and r["time"] - t_trans < 1.0]
    peak = max((r["Fz"] for r in p2_init), default=0.0)
    # wipe metrics (after stabilize)
    wipe = [r for r in rows if r["time"] > WIPE_T and r["hybrid"] >= 0.5]
    if not wipe: return None
    n = len(wipe)
    contact = 100 * sum(1 for r in wipe if r["Fz"] > F_THRESH) / n
    sq = sum((r["ee_x"]-r["tgt_x"])**2 + (r["ee_y"]-r["tgt_y"])**2 for r in wipe) / n
    xy = 1000 * math.sqrt(sq)
    return dict(contact=contact, xy=xy, peak=peak)

def cell_stats(paths):
    res = [stats(p) for p in paths]
    res = [r for r in res if r is not None]
    if not res: return None
    return {
        k: (st.mean(r[k] for r in res),
            st.stdev(r[k] for r in res) if len(res) > 1 else 0.0)
        for k in ("contact","xy","peak")
    }

mppi = {}
cost = {}
for K in K_LIST:
    mppi[K] = cell_stats([f"{MPPI_DIR}/T{K}_H{H}_s{s}.csv" for s in SEEDS])
    cost[K] = cell_stats([f"{COST_DIR}/cost_T{K}_H{H}_s{s}.csv" for s in SEEDS])

def get_series(data, key):
    Ks   = [K for K in K_LIST if data[K] is not None]
    mean = [data[K][key][0] for K in Ks]
    std  = [data[K][key][1] for K in Ks]
    return Ks, mean, std

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
METRICS = [
    ("contact", "wipe contact % (Fz > 1 N)", "higher = better"),
    ("xy",      "xy tracking error [mm]",   "lower = better"),
    ("peak",    "Phase-2 peak Fz [N]",      "lower = better"),
]
C_MPPI = "#7f7f7f"
C_COST = "#1f77b4"

for ax, (key, ylabel, hint) in zip(axes, METRICS):
    Km, mm, ms = get_series(mppi, key)
    Kc, mc, sc = get_series(cost, key)
    ax.errorbar(Km, mm, yerr=ms, fmt="o-", color=C_MPPI, lw=2, ms=8,
                capsize=4, label="MPPI baseline")
    ax.errorbar(Kc, mc, yerr=sc, fmt="s-", color=C_COST, lw=2, ms=8,
                capsize=4, label="FM as cost")
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_LIST)
    ax.set_xticklabels([str(K) for K in K_LIST])
    ax.set_xlabel("rollouts K  (log scale)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel}\n({hint})", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=11)

fig.suptitle(f"MPPI baseline vs FM-as-cost — K sweep at H = {H} s (mean ± std, 3 seeds)",
             fontsize=14, y=1.02)
plt.tight_layout()
out = "mppi_vs_cost_ksweep.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
print("\nMetric table (mean only):")
print(f"{'K':>4} | {'MPPI contact':>12} {'MPPI xy':>10} {'MPPI peak':>10} | "
      f"{'cost contact':>12} {'cost xy':>10} {'cost peak':>10}")
for K in K_LIST:
    m = mppi[K] or {}
    c = cost[K] or {}
    def fmt(d, k): return f"{d[k][0]:.2f}" if d else "—"
    print(f"{K:>4} | {fmt(m,'contact'):>12} {fmt(m,'xy'):>10} {fmt(m,'peak'):>10} | "
          f"{fmt(c,'contact'):>12} {fmt(c,'xy'):>10} {fmt(c,'peak'):>10}")
