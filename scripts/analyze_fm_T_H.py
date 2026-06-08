#!/usr/bin/env python3
"""Analyze FM x T x H sweep. Per cell aggregate over 3 seeds."""
import csv, glob, math, statistics as st

ROOT = "out/sweep_fm_T_H"
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
    xy_mm = 1000 * math.sqrt(
        sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in rs) / n)
    return contact, fz_mean, xy_mm

rows = []
for FM in ["3.0", "10.0", "30.0"]:
    for T in [8, 16, 32]:
        for H in ["0.10", "0.20", "0.30"]:
            seeds = sorted(glob.glob(f"{ROOT}/force_FM{FM}_T{T}_H{H}_s*.csv"))
            res = [stats_one(p) for p in seeds]
            res = [r for r in res if r]
            if not res: continue
            arrs = list(zip(*res))
            def m(a): return st.mean(a)
            def s(a): return st.stdev(a) if len(a)>1 else 0.0
            c, fz, xy = m(arrs[0]), m(arrs[1]), m(arrs[2])
            cs, fzs, xys = s(arrs[0]), s(arrs[1]), s(arrs[2])
            rows.append((FM, T, H, c, cs, fz, fzs, xy, xys))

print(f"{'FM':>4}  {'T':>3}  {'H':>5}  {'contact%':>14}  {'Fz [N]':>12}  {'xy [mm]':>14}")
print("-"*80)
for r in rows:
    FM,T,H,c,cs,fz,fzs,xy,xys = r
    print(f"{FM:>4}  {T:>3}  {H:>5}  {c:6.1f}±{cs:4.1f}    "
          f"{fz:+6.2f}±{fzs:4.2f}  {xy:6.2f}±{xys:4.2f}")

# Best by criterion
print("\n=== Top 3 by contact ===")
for r in sorted(rows, key=lambda x: -x[3])[:3]:
    FM,T,H,c,cs,fz,fzs,xy,xys = r
    print(f"FM={FM} T={T} H={H}: contact {c:.1f}%, Fz {fz:+.2f}, xy {xy:.2f}mm")
print("\n=== Top 3 by xy (lowest) ===")
for r in sorted(rows, key=lambda x: x[7])[:3]:
    FM,T,H,c,cs,fz,fzs,xy,xys = r
    print(f"FM={FM} T={T} H={H}: contact {c:.1f}%, Fz {fz:+.2f}, xy {xy:.2f}mm")
print("\n=== Top 3 by contact at xy<5mm ===")
for r in sorted([x for x in rows if x[7]<5.0], key=lambda x: -x[3])[:3]:
    FM,T,H,c,cs,fz,fzs,xy,xys = r
    print(f"FM={FM} T={T} H={H}: contact {c:.1f}%, Fz {fz:+.2f}, xy {xy:.2f}mm")
