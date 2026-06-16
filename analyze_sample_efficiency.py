#!/usr/bin/env python3
"""Sample efficiency analysis — the right question.
Find MPPI baseline's best (K, H) by task performance, then ask whether
FM-DiT cost / MLP cost can hit that bar at smaller (K, H).
"""
import csv, math, os, statistics as st

DIR = "sweep_syncasync"
K_LIST = [8, 16, 32, 64, 128]
H_LIST = ["0.05", "0.10", "0.20", "0.30"]
SEEDS = [1, 2, 3]
PLANS = ["async", "sync"]
MODES = ["mppi", "cost", "wta3", "mlp"]
LABEL = {"mppi":"MPPI baseline", "cost":"FM-DiT cost",
         "wta3":"FM-DiT WTA",   "mlp":"MLP cost"}

FZ_THR = 1.0      # Fz>1N counts as contact (matches existing plot)
WIPE_T = 5.0      # phase 2 starts ~t=5s
DRIFT_LIMIT = 5.0  # mm — above this = task failure

def stats(p):
    """Return (contact%, xy_mm, plan_ms_mean, fm_ms_mean) or None."""
    if not os.path.exists(p): return None
    n=cn=0; sq=0.0; pms=[]; fms=[]
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                t=float(r['time']); h=float(r['hybrid'])
                pm=float(r.get('plan_ms','0'));
                fm=float(r.get('fm_ms','0'))
                if pm>0: pms.append(pm)
                if fm>0: fms.append(fm)
                if t>WIPE_T and h>=0.5:
                    n+=1
                    fz=float(r['Fz'])
                    if fz>FZ_THR: cn+=1
                    dx=float(r['ee_x'])-float(r['tgt_x'])
                    dy=float(r['ee_y'])-float(r['tgt_y'])
                    sq+=dx*dx+dy*dy
            except (ValueError, KeyError):
                continue
    if n==0: return None
    return (100*cn/n, 1000*math.sqrt(sq/n),
            st.mean(pms) if pms else 0.0,
            st.mean(fms) if fms else 0.0)

def cell(mode, plan, K, H):
    rs=[stats(f'{DIR}/{mode}_{plan}_K{K}_H{H}_s{s}.csv') for s in SEEDS]
    rs=[r for r in rs if r is not None]
    if not rs: return None
    return tuple(st.mean(r[i] for r in rs) for i in range(4))

# === step 1: dump full grid per mode and plan ===
print("="*84)
print("Full grid (3-seed mean): contact% | xy_mm | plan_ms | fm_ms")
print("="*84)
for plan in PLANS:
    print(f"\n[{plan.upper()}]")
    for mode in MODES:
        print(f"  {LABEL[mode]:>22}  " +
              "        ".join(f"H={H}" for H in H_LIST))
        for K in K_LIST:
            row = f"    K={K:>3}  "
            for H in H_LIST:
                v = cell(mode, plan, K, H)
                if v is None:
                    row += f"{'-':>13}"
                else:
                    contact, xy, pm, fm = v
                    if xy > DRIFT_LIMIT:
                        row += f"  DRIFT(xy={xy:4.1f})"
                    else:
                        row += f"  {contact:5.1f}%|{xy:4.1f}mm"
            print(row)

# === step 2: find MPPI baseline's best ===
print()
print("="*84)
print("MPPI baseline best (K, H) per plan mode — by contact% (xy<5mm only)")
print("="*84)
for plan in PLANS:
    best = []
    for K in K_LIST:
        for H in H_LIST:
            v = cell("mppi", plan, K, H)
            if v is None: continue
            contact, xy, pm, fm = v
            if xy > DRIFT_LIMIT: continue
            best.append((contact, K, H, xy, pm))
    if not best:
        print(f"  {plan}: no valid baseline")
        continue
    best.sort(reverse=True)
    contact, K, H, xy, pm = best[0]
    print(f"  {plan:>6}: K={K:>3} H={H}  →  contact={contact:5.2f}%  "
          f"xy={xy:4.2f}mm  plan_ms={pm:5.2f}")
    print(f"           runner-up: " + ", ".join(
        f"K={k}H={h} {c:.1f}%" for c,k,h,_,_ in best[1:4]))

# === step 3: for each non-MPPI mode, find smallest (K,H) that matches/beats baseline ===
print()
print("="*84)
print("Iso-performance check — smallest (K, H) for each guided mode that")
print("matches MPPI baseline's best contact% (xy<5mm)")
print("="*84)
for plan in PLANS:
    # baseline bar
    base_pts=[]
    for K in K_LIST:
        for H in H_LIST:
            v = cell("mppi", plan, K, H)
            if v is None or v[1] > DRIFT_LIMIT: continue
            base_pts.append((v[0], K, H))
    if not base_pts: continue
    base_pts.sort(reverse=True)
    bar_contact, bar_K, bar_H = base_pts[0][:3]
    base_pm = cell("mppi", plan, bar_K, bar_H)[2]
    print(f"\n[{plan.upper()}] MPPI bar = {bar_contact:.2f}% "
          f"at K={bar_K} H={bar_H} (plan_ms={base_pm:.2f})")
    for mode in ["cost", "wta3", "mlp"]:
        matches=[]
        for K in K_LIST:
            for H in H_LIST:
                v = cell(mode, plan, K, H)
                if v is None or v[1] > DRIFT_LIMIT: continue
                if v[0] >= bar_contact - 0.5:  # allow 0.5pp slack
                    matches.append((K, H, v[0], v[1], v[2], v[3]))
        if not matches:
            print(f"  {LABEL[mode]:>22}: NO config matches the bar")
            continue
        # rank by (K, H) lexicographic — smaller is better
        h2f = {h: float(h) for h in H_LIST}
        matches.sort(key=lambda r: (r[0], h2f[r[1]]))
        K,H,c,xy,pm,fm = matches[0]
        # compute saving estimate
        saving_K = (1 - K / bar_K) * 100
        saving_H = (1 - h2f[H] / h2f[bar_H]) * 100
        print(f"  {LABEL[mode]:>22}: K={K:>3} H={H}  contact={c:5.2f}%  "
              f"xy={xy:4.2f}mm  plan_ms={pm:5.2f}  fm_ms={fm:5.2f}  "
              f"(K↓{saving_K:+.0f}%  H↓{saving_H:+.0f}%)")
