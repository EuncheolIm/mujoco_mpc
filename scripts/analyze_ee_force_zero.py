#!/usr/bin/env python3
"""Compare EE_Force_SCALE = 0 vs 1 (default) on 4 cells."""
import csv, glob, math, statistics as st

ROOT = "out/sweep_ee_force_zero"
WARMUP = 5.0
EE_W = 7.46
F_THRESH = 1.0

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
    xy = 1000 * math.sqrt(
        sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in rs) / n)
    return contact, fz_mean, xy

def agg(files):
    rs = [stats_one(f) for f in files]; rs = [r for r in rs if r]
    if not rs: return None
    a = list(zip(*rs))
    def ms(x): return st.mean(x), (st.stdev(x) if len(x)>1 else 0.0)
    return ms(a[0]) + ms(a[1]) + ms(a[2])

CELLS = [
    ("yaml_default", "T=16 H=0.20 FM=1.5"),
    ("yaml_H03",     "T=16 H=0.30 FM=1.5"),
    ("phase3_bal",   "T=32 H=0.30 FM=3.0"),
    ("newbest",      "T=16 H=0.20 FM=30 "),
]

print(f"{'cell':30s}  {'EE_FS':5s}  {'contact %':14s}  {'Fz [N]':12s}  {'xy [mm]':10s}")
print("-"*90)
for lbl, desc in CELLS:
    for FS in ["0", "1"]:
        files = sorted(glob.glob(f"{ROOT}/force_{lbl}_FS{FS}_s*.csv"))
        r = agg(files)
        if not r: continue
        cm,cs,fm,fs,xm,xs = r
        tag = f"FS={FS}"
        print(f"{lbl} ({desc})  {tag:5s}  {cm:5.1f}±{cs:4.1f}    {fm:+6.2f}±{fs:4.2f}  {xm:5.2f}±{xs:4.2f}")
    print()
