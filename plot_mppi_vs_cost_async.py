#!/usr/bin/env python3
"""MPPI baseline vs FM-as-cost (async, existing data).
Two figures:
  1. Performance: contact% / xy / peak Fz  vs K (H=0.10)  — mppi_vs_cost_perf.png
  2. Timing: plan_ms vs K (H=0.10)                        — mppi_vs_cost_time.png
Data: sweep_mppi_ftask_3seeds (MPPI, F_task) vs sweep_flowmppi_4modes_3seeds/cost.
"""
import csv, math, os, statistics as st
import matplotlib.pyplot as plt

MPPI_DIR = "sweep_mppi_ftask_3seeds"
COST_DIR = "sweep_flowmppi_4modes_3seeds"
K_LIST = [8, 16, 32, 64, 128]
H = "0.10"
SEEDS = [1, 2, 3]
F_THRESH = 1.0
WIPE_T = 5.0

def stats(p):
    if not os.path.exists(p): return None
    n=0; cn=0; sq=0; t_trans=None; peak=0; pms=[]
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                t=float(r['time']); fz=float(r['Fz']); h=float(r['hybrid'])
                pm=float(r.get('plan_ms','0'))
                if pm>0: pms.append(pm)
                if t_trans is None and h>=0.5: t_trans=t
                if t_trans is not None and t-t_trans<1.0 and h>=0.5 and fz>peak: peak=fz
                if t>WIPE_T and h>=0.5:
                    n+=1
                    if fz>F_THRESH: cn+=1
                    dx=float(r['ee_x'])-float(r['tgt_x']); dy=float(r['ee_y'])-float(r['tgt_y'])
                    sq+=dx*dx+dy*dy
            except: pass
    if n==0: return None
    return (100*cn/n, 1000*math.sqrt(sq/n), peak, st.mean(pms) if pms else 0)

def series(path_fn):
    out = {'contact':[], 'xy':[], 'peak':[], 'plan':[],
           'contact_sd':[], 'xy_sd':[], 'peak_sd':[], 'plan_sd':[]}
    for K in K_LIST:
        vals = [stats(path_fn(K, s)) for s in SEEDS]
        vals = [v for v in vals if v]
        for idx, key in enumerate(['contact','xy','peak','plan']):
            arr = [v[idx] for v in vals]
            out[key].append(st.mean(arr) if arr else 0)
            out[key+'_sd'].append(st.stdev(arr) if len(arr)>1 else 0)
    return out

mppi = series(lambda K, s: f"{MPPI_DIR}/T{K}_H{H}_s{s}.csv")
cost = series(lambda K, s: f"{COST_DIR}/cost_T{K}_H{H}_s{s}.csv")

C_MPPI = "#7f7f7f"; C_COST = "#1f77b4"

# ---------- Figure 1: performance ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
specs = [("contact", "wipe contact %", "higher = better"),
         ("xy", "xy tracking error [mm]", "lower = better"),
         ("peak", "Phase-2 peak Fz [N]", "lower = better")]
for ax, (key, ylabel, hint) in zip(axes, specs):
    ax.errorbar(K_LIST, mppi[key], yerr=mppi[key+'_sd'], fmt='o-', color=C_MPPI,
                lw=2, ms=8, capsize=4, label="MPPI baseline (F_task)")
    ax.errorbar(K_LIST, cost[key], yerr=cost[key+'_sd'], fmt='s-', color=C_COST,
                lw=2, ms=8, capsize=4, label="FM as cost (hinge)")
    ax.set_xscale('log', base=2); ax.set_xticks(K_LIST)
    ax.set_xticklabels([str(k) for k in K_LIST])
    ax.set_xlabel("rollouts K (log)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{ylabel}\n({hint})", fontsize=12)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
fig.suptitle(f"MPPI baseline vs FM-as-cost — performance, H={H} (async, 3-seed mean±std)",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("mppi_vs_cost_perf.png", dpi=150, bbox_inches='tight')
print("saved: mppi_vs_cost_perf.png")

# ---------- Figure 2: timing ----------
fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.errorbar(K_LIST, mppi['plan'], yerr=mppi['plan_sd'], fmt='o-', color=C_MPPI,
            lw=2, ms=9, capsize=4, label="MPPI baseline plan_ms")
ax.errorbar(K_LIST, cost['plan'], yerr=cost['plan_sd'], fmt='s-', color=C_COST,
            lw=2, ms=9, capsize=4, label="FM as cost plan_ms")
ax.set_xscale('log', base=2); ax.set_xticks(K_LIST)
ax.set_xticklabels([str(k) for k in K_LIST])
ax.set_xlabel("rollouts K (log)", fontsize=13)
ax.set_ylabel("plan_ms (per planning iteration)", fontsize=13)
ax.set_title(f"Per-iteration planning time, H={H} (async, 3-seed mean)\n"
             "(FM inference runs async in separate thread — not in plan_ms)",
             fontsize=12)
ax.grid(True, alpha=0.3); ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("mppi_vs_cost_time.png", dpi=150, bbox_inches='tight')
print("saved: mppi_vs_cost_time.png")

# print table
print(f"\nH={H} (3-seed mean):")
print(f"{'K':>4} | {'MPPI c% xy peak plan':>28} | {'cost c% xy peak plan':>28}")
for i,K in enumerate(K_LIST):
    print(f"{K:>4} | {mppi['contact'][i]:>6.1f} {mppi['xy'][i]:>5.2f} {mppi['peak'][i]:>5.1f} {mppi['plan'][i]:>5.2f}      "
          f"| {cost['contact'][i]:>6.1f} {cost['xy'][i]:>5.2f} {cost['peak'][i]:>5.1f} {cost['plan'][i]:>5.2f}")
