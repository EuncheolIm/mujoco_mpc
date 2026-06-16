#!/usr/bin/env python3
"""Sync vs async, MPPI / cost / WTA3, full K×H grid.
2 rows (async/sync) × 4 columns (H values) × 3 lines (modes) on contact%.
FAIL / drift (xy>5) marked as no point.
"""
import csv, math, os, statistics as st
import matplotlib.pyplot as plt

DIR="sweep_syncasync"
K_LIST=[8,16,32,64,128]
H_LIST=["0.05","0.10","0.20","0.30"]
MODES=[("mppi","MPPI baseline","#7f7f7f","o"),
       ("cost","FM as cost","#1f77b4","s"),
       ("wta3","WTA3 (FM warmstart)","#d62728","^")]
F=1.0; WIPE=5.0; DRIFT_TH=5.0

def stats(p):
    if not os.path.exists(p): return None
    n=0; cn=0; sq=0
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                t=float(r['time']); fz=float(r['Fz']); h=float(r['hybrid'])
                if t>WIPE and h>=0.5:
                    n+=1
                    if fz>F: cn+=1
                    dx=float(r['ee_x'])-float(r['tgt_x']); dy=float(r['ee_y'])-float(r['tgt_y'])
                    sq+=dx*dx+dy*dy
            except: pass
    if n==0: return None
    return 100*cn/n, 1000*math.sqrt(sq/n)
def cell(mode, plan, K, H):
    vs=[stats(f'{DIR}/{mode}_{plan}_K{K}_H{H}_s{s}.csv') for s in [1,2,3]]
    vs=[v for v in vs if v]
    if not vs: return None
    c=st.mean(v[0] for v in vs); xy=st.mean(v[1] for v in vs)
    return c, xy

fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey='row')
plans=["async","sync"]
for r, plan in enumerate(plans):
    for c, H in enumerate(H_LIST):
        ax=axes[r,c]
        for mode_id, label, color, marker in MODES:
            Ks=[]; ys=[]
            for K in K_LIST:
                t=cell(mode_id, plan, K, H)
                if t is None: continue  # FAIL
                contact, xy = t
                if xy>DRIFT_TH: continue  # drift = task failure
                Ks.append(K); ys.append(contact)
            ax.plot(Ks, ys, marker=marker, color=color, lw=2, ms=10, label=label)
        ax.set_xscale('log', base=2)
        ax.set_xticks(K_LIST); ax.set_xticklabels([str(k) for k in K_LIST])
        ax.set_xlabel("rollouts K", fontsize=11)
        if c==0: ax.set_ylabel(f"{plan.upper()}\n\nwipe contact %", fontsize=13)
        ax.set_title(f"H={H}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(55, 92)

# Global legend (mode markers/colors) below the figure
from matplotlib.lines import Line2D
legend_handles = [Line2D([0],[0], marker=m, color=c, lw=2, ms=12, label=lbl)
                  for _, lbl, c, m in MODES]
fig.legend(handles=legend_handles, loc='upper center',
           bbox_to_anchor=(0.5, 0.04), ncol=3, fontsize=13,
           frameon=True)

fig.suptitle("ASYNC vs SYNC across (K, H, mode) — 3-seed mean.  "
             "Missing points = FAIL (no phase-2) or DRIFT (xy>5mm).",
             fontsize=15, y=1.00)
plt.tight_layout(rect=[0, 0.06, 1, 1.0])
plt.savefig("syncasync_KH.png", dpi=150, bbox_inches='tight')
print("saved: syncasync_KH.png")
