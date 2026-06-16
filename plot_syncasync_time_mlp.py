#!/usr/bin/env python3
"""Per-iteration planning time (plan_ms) — same layout as plot_syncasync_time.py
but adds a 4th line for the MLP student guide.
"""
import csv, os, statistics as st
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DIR = "sweep_syncasync"
K_LIST = [8, 16, 32, 64, 128]
H_LIST = ["0.05", "0.10", "0.20", "0.30"]
MODES = [
    ("mppi", "MPPI baseline",          "#7f7f7f", "o"),
    ("cost", "FM-DiT (FlowMPPI cost)", "#1f77b4", "s"),
    ("wta3", "FM-DiT (WTA warmstart)", "#d62728", "^"),
    ("mlp",  "MLP student (cost)",     "#2ca02c", "D"),
]

def plan_ms_stats(p):
    if not os.path.exists(p): return None
    pms = []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                pm = float(r.get('plan_ms', '0'))
                if pm > 0:
                    pms.append(pm)
            except (ValueError, KeyError):
                pass
    if not pms: return None
    return st.mean(pms)

def cell(mode, plan, K, H):
    vs = [plan_ms_stats(f'{DIR}/{mode}_{plan}_K{K}_H{H}_s{s}.csv')
          for s in (1, 2, 3)]
    vs = [v for v in vs if v]
    if not vs: return None
    return st.mean(vs)

fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey='row')
for r, plan in enumerate(["async", "sync"]):
    for c, H in enumerate(H_LIST):
        ax = axes[r, c]
        for mode_id, label, color, marker in MODES:
            Ks, ys = [], []
            for K in K_LIST:
                v = cell(mode_id, plan, K, H)
                if v is None: continue
                Ks.append(K); ys.append(v)
            if Ks:
                ax.plot(Ks, ys, marker=marker, color=color, lw=2, ms=10,
                        label=label)
        ax.set_xscale('log', base=2)
        ax.set_xticks(K_LIST)
        ax.set_xticklabels([str(k) for k in K_LIST])
        ax.set_xlabel("rollouts K", fontsize=11)
        if c == 0:
            ax.set_ylabel(f"{plan.upper()}\n\nplan_ms (per iter)", fontsize=13)
        ax.set_title(f"H={H}", fontsize=12)
        ax.grid(True, alpha=0.3)

handles = [Line2D([0], [0], marker=m, color=c, lw=2, ms=12, label=lbl)
           for _, lbl, c, m in MODES]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.04),
           ncol=4, fontsize=12)

fig.suptitle("Per-iteration planning time (ms) — MPPI vs FM-DiT vs MLP student "
             "(3-seed mean). Guide inference NOT included in plan_ms.",
             fontsize=14, y=1.00)
plt.tight_layout(rect=[0, 0.06, 1, 1.0])
plt.savefig("syncasync_time_mlp.png", dpi=150, bbox_inches='tight')
print("saved: syncasync_time_mlp.png")

# Table dump at H=0.10
print()
print("plan_ms per iter (3-seed mean) — H=0.10")
header = f'{"K":>4} | '
header += " ".join(f"{m:>10}" for m, *_ in MODES) + "  (async)"
print(header)
for K in K_LIST:
    cs = []
    for m, *_ in MODES:
        v = cell(m, "async", K, "0.10")
        cs.append(f"{v:7.2f}" if v else "   -   ")
    print(f"{K:>4} | " + " ".join(f"{c:>10}" for c in cs))
print()
header = f'{"K":>4} | '
header += " ".join(f"{m:>10}" for m, *_ in MODES) + "  (sync)"
print(header)
for K in K_LIST:
    cs = []
    for m, *_ in MODES:
        v = cell(m, "sync", K, "0.10")
        cs.append(f"{v:7.2f}" if v else "   -   ")
    print(f"{K:>4} | " + " ".join(f"{c:>10}" for c in cs))

# fm_ms (guide inference time) — only meaningful for guided modes.
print()
print("fm_ms per call (3-seed mean) — guide inference cost — H=0.10, sync")
for K in K_LIST:
    line = f"K={K:>3}: "
    for m, lbl, _, _ in MODES:
        if m == "mppi":
            line += f"{lbl}=N/A (no guide)  "
            continue
        vs = []
        for s in (1, 2, 3):
            p = f"{DIR}/{m}_sync_K{K}_H0.10_s{s}.csv"
            if not os.path.exists(p): continue
            ms = []
            with open(p) as f:
                for r in csv.DictReader(f):
                    try:
                        v = float(r.get('fm_ms', '0'))
                        if v > 0: ms.append(v)
                    except (ValueError, KeyError):
                        pass
            if ms: vs.append(st.mean(ms))
        if vs:
            line += f"{lbl}={st.mean(vs):.3f}ms  "
        else:
            line += f"{lbl}=- "
    print(line)
