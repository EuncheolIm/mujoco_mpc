#!/usr/bin/env python3
"""SYNC mode (1 plan/control step): MPPI baseline vs FM-as-cost.
Performance (contact/xy/peak) + timing (plan_ms) vs K, H=0.10, single run.
Data: /tmp/sync_K*.csv (MPPI) and /tmp/synccost_K*.csv (cost).
FAIL cells (no phase-2) shown as gaps.
"""
import csv, math, os
import matplotlib.pyplot as plt

K_LIST = [8, 16, 32, 64, 128]
F_THRESH = 1.0; WIPE_T = 5.0

def stats(p):
    if not os.path.exists(p): return None
    import statistics as st
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
    plan = st.mean(pms) if pms else 0
    if n==0: return dict(contact=None, xy=None, peak=None, plan=plan, fail=True)
    return dict(contact=100*cn/n, xy=1000*math.sqrt(sq/n), peak=peak, plan=plan, fail=False)

mppi = [stats(f"/tmp/sync_K{K}.csv") for K in K_LIST]
cost = [stats(f"/tmp/synccost_K{K}.csv") for K in K_LIST]
# treat drift (xy>5mm) as fail for clarity
for arr in (mppi, cost):
    for s in arr:
        if s and not s['fail'] and s['xy'] and s['xy'] > 5:
            s['fail'] = True  # drift = task fail

C_MPPI="#7f7f7f"; C_COST="#1f77b4"

def xy_series(arr, key):
    xs=[]; ys=[]
    for K, s in zip(K_LIST, arr):
        if s and not s['fail'] and s[key] is not None:
            xs.append(K); ys.append(s[key])
    return xs, ys

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
specs=[("contact","wipe contact %","higher=better"),
       ("xy","xy error [mm]","lower=better"),
       ("peak","Phase-2 peak Fz [N]","lower=better"),
       ("plan","plan_ms","lower=better")]
for ax,(key,ylabel,hint) in zip(axes, specs):
    xm,ym = xy_series(mppi, key)
    xc,yc = xy_series(cost, key)
    ax.plot(xm,ym,'o-',color=C_MPPI,lw=2,ms=9,label="MPPI baseline")
    ax.plot(xc,yc,'s-',color=C_COST,lw=2,ms=9,label="FM as cost")
    # mark FAIL region
    for K,s in zip(K_LIST, mppi):
        if s and s['fail'] and key!='plan':
            ax.axvline(K, color=C_MPPI, ls=':', alpha=0.3)
    ax.set_xscale('log',base=2); ax.set_xticks(K_LIST)
    ax.set_xticklabels([str(k) for k in K_LIST])
    ax.set_xlabel("rollouts K (log)",fontsize=12)
    ax.set_ylabel(ylabel,fontsize=12)
    ax.set_title(f"{ylabel}\n({hint})",fontsize=12)
    ax.grid(True,alpha=0.3); ax.legend(fontsize=10)

fig.suptitle("SYNC mode (1 plan / control step) — MPPI vs FM-as-cost, H=0.10 (single run)\n"
             "FAIL (no phase-2 reach or xy>5mm drift) = missing point. K=8 fails for both.",
             fontsize=13, y=1.04)
plt.tight_layout()
plt.savefig("mppi_vs_cost_sync.png", dpi=150, bbox_inches='tight')
print("saved: mppi_vs_cost_sync.png")
print()
print(f"{'K':>4} | {'MPPI':>26} | {'cost':>26}")
for K, m, c in zip(K_LIST, mppi, cost):
    def f(s):
        if not s: return 'none'
        if s['fail']:
            return f"FAIL (xy={s['xy']:.1f})" if s['xy'] else 'FAIL'
        return f"{s['contact']:.1f}/{s['xy']:.2f}/{s['peak']:.0f}/{s['plan']:.1f}"
    print(f"{K:>4} | {f(m):>26} | {f(c):>26}")
