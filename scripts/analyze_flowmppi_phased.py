#!/usr/bin/env python3
"""FlowMPPI 4-mode analyzer (phase 1/2 + timing)."""
import csv, math, os, statistics as st, sys

ROOT = "sweep_flowmppi_4modes_3seeds"
F_THRESH = 1.0
WIPE = 5.0

def stats_one(path):
    rs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try: rs.append({k: float(v) for k, v in r.items()})
            except (ValueError, TypeError): pass
    if not rs: return None
    phase2 = [r for r in rs if r['hybrid'] >= 0.5]
    if not phase2: return None
    t_transition = phase2[0]['time']
    phase1 = [r for r in rs if r['hybrid'] < 0.5]
    p1_any = any(r['Fz'] > F_THRESH for r in phase1)
    p1_peak = max((r['Fz'] for r in phase1), default=0.0)
    t2_init = [r for r in rs if r['hybrid'] >= 0.5 and r['time'] - t_transition < 1.0]
    t2_peak = max((r['Fz'] for r in t2_init), default=0.0)
    wipe = [r for r in rs if r['time'] > WIPE and r['hybrid'] >= 0.5]
    if not wipe: return None
    n = len(wipe)
    fz = [r['Fz'] for r in wipe]
    contact = 100.0 * sum(1 for v in fz if v > F_THRESH) / n
    fz_mean = st.mean(fz)
    xy = 1000 * math.sqrt(sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in wipe)/n)
    plan_ms = [r.get('plan_ms', 0) for r in wipe if r.get('plan_ms', 0) > 0]
    fm_ms = [r.get('fm_ms', 0) for r in wipe if r.get('fm_ms', 0) > 0]
    plan_mean = st.mean(plan_ms) if plan_ms else 0
    fm_mean = st.mean(fm_ms) if fm_ms else 0
    return dict(
        t_trans=t_transition, p1_any=p1_any, p1_peak=p1_peak,
        t2_peak=t2_peak, contact=contact, fz=fz_mean, xy=xy,
        plan_ms=plan_mean, fm_ms=fm_mean,
    )

def fmt(vals, d=1):
    if not vals: return "-"
    if len(vals) < 2: return f"{vals[0]:.{d}f}"
    return f"{st.mean(vals):.{d}f}±{st.stdev(vals):.{d}f}"

MODES = ["wta1", "wta2", "wta3", "cost"]
T_LIST = [8, 16, 32, 64, 128]
H_LIST = ["0.05", "0.10", "0.20", "0.30"]

for mode in MODES:
    print(f"\n=== {mode.upper()} ===")
    print(f"{'K':>4} {'H':>5}  {'P1 any':>7}  {'P2 peak':>10}  {'wipe c%':>10}  {'wipe xy':>10}  {'plan_ms':>10}  {'fm_ms':>10}")
    print("-"*90)
    for K in T_LIST:
        for H in H_LIST:
            res = []
            for s in [1,2,3]:
                p = f"{ROOT}/{mode}_T{K}_H{H}_s{s}.csv"
                if not os.path.exists(p): continue
                r = stats_one(p)
                if r: res.append(r)
            if not res: continue
            p1 = sum(1 for r in res if r['p1_any'])
            p2 = [r['t2_peak'] for r in res]
            cs = [r['contact'] for r in res]
            xys = [r['xy'] for r in res]
            pms = [r['plan_ms'] for r in res if r['plan_ms']>0]
            fms = [r['fm_ms'] for r in res if r['fm_ms']>0]
            print(f"{K:>4} {H:>5}  {p1}/{len(res):>3}    {fmt(p2):>10}  {fmt(cs):>10}  {fmt(xys, 2):>10}  {fmt(pms, 1):>10}  {fmt(fms, 1):>10}")
