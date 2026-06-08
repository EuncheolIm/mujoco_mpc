#!/usr/bin/env python3
"""MPPI baseline 3-seed analyzer with mean ± std."""
import csv, math, os, statistics as st

ROOT = "sweep_mppi_3seeds"
WARMUP = 5.0; F_THRESH = 1.0

def stats_one(path):
    rs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try: rs.append({k: float(v) for k, v in r.items()})
            except (ValueError, TypeError): pass
    rs = [r for r in rs if r['time'] >= WARMUP]
    if not rs: return None
    n = len(rs)
    fz = [r['Fz'] for r in rs]
    contact = 100.0 * sum(1 for v in fz if v > F_THRESH) / n
    fz_mean = st.mean(fz)
    xy = 1000 * math.sqrt(sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in rs)/n)
    return contact, fz_mean, xy

def fmt(vals):
    if len(vals) < 2: return f"{vals[0]:.1f}" if vals else "-"
    return f"{st.mean(vals):.1f}±{st.stdev(vals):.1f}"

print(f"{'T':>4} {'H':>5}  {'contact %':>14}  {'Fz [N]':>14}  {'xy [mm]':>14}")
print("-"*70)
for T in [8, 16, 32, 64, 128]:
    for H in ["0.05", "0.10", "0.20", "0.30"]:
        cs, fzs, xys = [], [], []
        for s in [1, 2, 3]:
            p = f"{ROOT}/T{T}_H{H}_s{s}.csv"
            if not os.path.exists(p): continue
            r = stats_one(p)
            if r: cs.append(r[0]); fzs.append(r[1]); xys.append(r[2])
        if cs:
            print(f"{T:>4} {H:>5}  {fmt(cs):>14}  {fmt(fzs):>14}  {fmt(xys):>14}")
