#!/usr/bin/env python3
"""Analyze F_max sweep (upper-bound hinge)."""
import csv, glob, math, statistics as st

ROOT = "out/sweep_fmax"
WARMUP = 5.0
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
    fz_p95 = sorted(fz)[int(0.95*n)]
    xy = 1000 * math.sqrt(
        sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in rs) / n)
    return contact, fz_mean, fz_p95, xy

def agg(files):
    rs = [stats_one(f) for f in files]; rs = [r for r in rs if r]
    if not rs: return None
    a = list(zip(*rs))
    def ms(x): return st.mean(x), (st.stdev(x) if len(x)>1 else 0.0)
    return ms(a[0]) + ms(a[1]) + ms(a[2]) + ms(a[3])

print(f"{'FM':>4}  {'F_MAX':>6}  {'contact %':14s}  {'Fz mean':14s}  {'Fz p95':14s}  {'xy [mm]':12s}")
print("-"*100)
for FM in ["1.0", "1.5"]:
    for FMAX in [10, 15, 25, 40]:
        files = sorted(glob.glob(f"{ROOT}/force_FM{FM}_FMAX{FMAX}_s*.csv"))
        r = agg(files)
        if not r: continue
        cm,cs,fm,fs,fp95m,fp95s,xm,xs = r
        print(f"{FM:>4}  {FMAX:>6}  {cm:5.1f}±{cs:4.1f}    {fm:+6.2f}±{fs:5.2f}   {fp95m:+6.2f}±{fp95s:5.2f}   {xm:5.2f}±{xs:4.2f}")
