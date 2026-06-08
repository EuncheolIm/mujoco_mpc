#!/usr/bin/env python3
"""MPPI 3-seed: report post-5s metrics + initial 5s Fz peak (impact)."""
import csv, math, os, statistics as st

ROOT = "sweep_mppi_3seeds"
WARMUP = 5.0; F_THRESH = 1.0

def stats_one(path):
    rs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try: rs.append({k: float(v) for k, v in r.items()})
            except (ValueError, TypeError): pass
    if not rs: return None
    init = [r for r in rs if r['time'] <= WARMUP]
    post = [r for r in rs if r['time'] > WARMUP]
    if not post: return None
    peak_init = max((r['Fz'] for r in init), default=0)
    n = len(post)
    fz = [r['Fz'] for r in post]
    contact = 100.0 * sum(1 for v in fz if v > F_THRESH) / n
    xy = 1000 * math.sqrt(sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in post)/n)
    return contact, xy, peak_init

def fmt(vals):
    if len(vals) < 2: return f"{vals[0]:.1f}" if vals else "-"
    return f"{st.mean(vals):.1f}±{st.stdev(vals):.1f}"

print(f"{'T':>4} {'H':>5}  {'contact %':>12}  {'xy [mm]':>10}  {'init Fz peak [N]':>18}")
print("-"*65)
for T in [8, 16, 32, 64, 128]:
    for H in ["0.05", "0.10", "0.20", "0.30"]:
        cs, xys, peaks = [], [], []
        for s in [1, 2, 3]:
            p = f"{ROOT}/T{T}_H{H}_s{s}.csv"
            if not os.path.exists(p): continue
            r = stats_one(p)
            if r: cs.append(r[0]); xys.append(r[1]); peaks.append(r[2])
        if cs:
            print(f"{T:>4} {H:>5}  {fmt(cs):>12}  {fmt(xys):>10}  {fmt(peaks):>18}")
