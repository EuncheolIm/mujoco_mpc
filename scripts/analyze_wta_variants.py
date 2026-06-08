#!/usr/bin/env python3
"""Analyze WTA variants comparison."""
import csv, glob, math, os, statistics as st

ROOT = "sweep_wta_variants_grid"
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

VARIANTS = [
    ("WTA #1 (full FM warmstart)", "wta1"),
    ("WTA #2 (shared softmax)",    "wta2"),
    ("WTA #3 (half-half WTA)",     "wta3"),
    ("Cost (option E)",             "cost"),
]
HS = ["0.10", "0.20", "0.30"]

print(f"{'variant':<32}  {'ftrack':>7}  " + "  ".join(f"H={h:>4}" for h in HS))
print("-"*90)
for vlabel, vkey in VARIANTS:
    for ft, ftlabel in [("0","0.0"), ("15","1.5")]:
        line = f"{vlabel:<32}  {ftlabel:>7}  "
        for H in HS:
            p = f"{ROOT}/{vkey}_ftrack{ft}_T16_H{H}.csv"
            r = stats_one(p) if os.path.exists(p) else None
            if r:
                c, fm, xy = r
                line += f"  c={c:5.1f}  xy={xy:5.2f}"
            else:
                line += f"  {'--':>14}"
        print(line)
