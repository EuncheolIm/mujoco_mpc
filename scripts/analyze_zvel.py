#!/usr/bin/env python3
"""Analyze zvel sweep + per-angle contact for -Y asymmetry."""
import csv, glob, math, statistics as st

ROOT = "out/sweep_zvel"
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
    fz_p95 = sorted(fz)[int(0.95*n)]
    xy = 1000 * math.sqrt(
        sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in rs) / n)
    # per-angle contact
    cx = sum(r['tgt_x'] for r in rs) / n
    cy = sum(r['tgt_y'] for r in rs) / n
    bins = {i: [] for i in range(8)}
    for r in rs:
        th = math.atan2(r['tgt_y']-cy, r['tgt_x']-cx)
        idx = int(((th + math.pi) / (2*math.pi)) * 8) % 8
        bins[idx].append(r['Fz'])
    angle_c = [100.0*sum(1 for v in bins[i] if v>1.0)/max(1,len(bins[i])) for i in range(8)]
    return contact, fz_p95, xy, angle_c

def agg(files):
    rs = [stats_one(f) for f in files]; rs = [r for r in rs if r]
    if not rs: return None
    cs = [r[0] for r in rs]
    p95s = [r[1] for r in rs]
    xys = [r[2] for r in rs]
    angles_each = [r[3] for r in rs]
    angles_mean = [st.mean([a[i] for a in angles_each]) for i in range(8)]
    def ms(x): return st.mean(x), (st.stdev(x) if len(x)>1 else 0.0)
    return ms(cs) + ms(p95s) + ms(xys) + (angles_mean,)

DIRS = ["-X","-X-Y","-Y","+X-Y","+X","+X+Y","+Y","-X+Y"]

print(f"{'ZS':>3}  {'contact %':14s}  {'Fz p95':14s}  {'xy [mm]':12s}")
print("-"*60)
results = {}
for ZS in [0, 1, 3, 10]:
    files = sorted(glob.glob(f"{ROOT}/force_ZS{ZS}_s*.csv"))
    r = agg(files)
    if not r: continue
    cm,cs,pm,ps,xm,xs,angles = r
    results[ZS] = (cm, pm, xm, angles)
    print(f"{ZS:>3}  {cm:5.1f}±{cs:4.1f}    {pm:+6.2f}±{ps:5.2f}   {xm:5.2f}±{xs:4.2f}")

print()
print(f"{'angle':8s} " + " ".join(f"  ZS={zs:<3}" for zs in [0,1,3,10]))
for i in range(8):
    line = f"{DIRS[i]:8s} "
    for zs in [0,1,3,10]:
        if zs in results:
            line += f"  {results[zs][3][i]:5.1f}"
    print(line)
