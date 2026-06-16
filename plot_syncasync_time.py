#!/usr/bin/env python3
"""Timing analysis for sync vs async sweep.
Two figures:
  1. plan_ms per iteration vs K, per H — same layout as perf plot.
  2. plan iterations per control step (10ms) vs K — shows async refinement budget.
"""
import csv, os, statistics as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DIR="sweep_syncasync"
K_LIST=[8,16,32,64,128]
H_LIST=["0.05","0.10","0.20","0.30"]
MODES=[("mppi","MPPI baseline","#7f7f7f","o"),
       ("cost","FM as cost","#1f77b4","s"),
       ("wta3","WTA3 (FM warmstart)","#d62728","^")]

def plan_ms_stats(p):
    if not os.path.exists(p): return None
    pms=[]; times=[]
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                pm=float(r.get('plan_ms','0'))
                if pm>0: pms.append(pm)
                times.append(float(r['time']))
            except: pass
    if not pms: return None
    return st.mean(pms)
def cell(mode, plan, K, H):
    vs=[plan_ms_stats(f'{DIR}/{mode}_{plan}_K{K}_H{H}_s{s}.csv') for s in [1,2,3]]
    vs=[v for v in vs if v]
    if not vs: return None
    return st.mean(vs)

fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey='row')
for r, plan in enumerate(["async","sync"]):
    for c, H in enumerate(H_LIST):
        ax=axes[r,c]
        for mode_id, label, color, marker in MODES:
            Ks=[]; ys=[]
            for K in K_LIST:
                v=cell(mode_id, plan, K, H)
                if v is None: continue
                Ks.append(K); ys.append(v)
            ax.plot(Ks, ys, marker=marker, color=color, lw=2, ms=10, label=label)
        ax.set_xscale('log', base=2)
        ax.set_xticks(K_LIST); ax.set_xticklabels([str(k) for k in K_LIST])
        ax.set_xlabel("rollouts K", fontsize=11)
        if c==0: ax.set_ylabel(f"{plan.upper()}\n\nplan_ms (per iter)", fontsize=13)
        ax.set_title(f"H={H}", fontsize=12)
        ax.grid(True, alpha=0.3)

# Legend
handles=[Line2D([0],[0], marker=m, color=c, lw=2, ms=12, label=lbl)
         for _, lbl, c, m in MODES]
fig.legend(handles=handles, loc='upper center',
           bbox_to_anchor=(0.5, 0.04), ncol=3, fontsize=13)

fig.suptitle("Per-iteration planning time (ms) — ASYNC vs SYNC across (K, H, mode).  "
             "FM async thread (~21 ms / chunk) not included.", fontsize=14, y=1.00)
plt.tight_layout(rect=[0, 0.06, 1, 1.0])
plt.savefig("syncasync_time.png", dpi=150, bbox_inches='tight')
print("saved: syncasync_time.png")

# Print table
print()
print("plan_ms per iter (3-seed mean) — H=0.10")
print(f'{"K":>4} | {"mppi-A":>7} {"mppi-S":>7} | {"cost-A":>7} {"cost-S":>7} | {"wta3-A":>7} {"wta3-S":>7}')
for K in K_LIST:
    cs=[]
    for m,_,_,_ in MODES:
        for p in ["async","sync"]:
            v=cell(m,p,K,"0.10")
            cs.append(f"{v:.2f}" if v else "  -  ")
    print(f'{K:>4} | {cs[0]:>7} {cs[1]:>7} | {cs[2]:>7} {cs[3]:>7} | {cs[4]:>7} {cs[5]:>7}')
