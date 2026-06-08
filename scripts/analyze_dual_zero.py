#!/usr/bin/env python3
import csv, glob, math, statistics as st

ROOT = "out/sweep_dual_zero"
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
COMBOS = [("1","1","default"), ("0","1","Force=0"), ("0","0","both=0")]

print(f"{'cell':35s}  {'combo':10s}  {'contact %':14s}  {'Fz [N]':14s}  {'xy [mm]':12s}")
print("-"*100)
for lbl, desc in CELLS:
    for FS, ZV, name in COMBOS:
        files = sorted(glob.glob(f"{ROOT}/force_{lbl}_FS{FS}_Z{ZV}_s*.csv"))
        r = agg(files)
        if not r: continue
        cm,cs,fm,fs,xm,xs = r
        print(f"{lbl} ({desc})  {name:10s}  {cm:5.1f}±{cs:4.1f}    {fm:+7.2f}±{fs:5.2f}  {xm:6.2f}±{xs:5.2f}")
    print()
