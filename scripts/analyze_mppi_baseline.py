#!/usr/bin/env python3
"""MPPI baseline analyzer (T<T>_H<H>.csv format, no seed suffix)."""
import csv, math, os, statistics as st

ROOT = "sweep_mppi_baseline"
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
    fz_p95 = sorted(fz)[int(0.95*n)]
    xy = 1000 * math.sqrt(sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in rs)/n)
    return contact, fz_mean, fz_p95, xy

print(f"{'T':>4}  {'H':>5}  {'contact %':>9}  {'Fz mean':>8}  {'Fz p95':>8}  {'xy [mm]':>9}")
print("-"*60)
rows = []
for T in [8, 16, 32, 64, 128]:
    for H in ["0.05", "0.10", "0.20", "0.30"]:
        p = f"{ROOT}/T{T}_H{H}.csv"
        if not os.path.exists(p): continue
        r = stats_one(p)
        if not r: continue
        c, fm, fp, xy = r
        rows.append((T, float(H), c, fm, fp, xy))
        print(f"{T:>4}  {H:>5}  {c:7.1f}    {fm:+6.2f}    {fp:+6.2f}    {xy:6.2f}")

print("\n=== Top 3 by contact (xy<6mm) ===")
for r in sorted([x for x in rows if x[5]<6.0], key=lambda x: -x[2])[:3]:
    print(f"  T={r[0]} H={r[1]}: c={r[2]:.1f}% Fz {r[3]:+.2f} xy={r[5]:.2f}")
print("\n=== Top 3 by xy ===")
for r in sorted(rows, key=lambda x: x[5])[:3]:
    print(f"  T={r[0]} H={r[1]}: c={r[2]:.1f}% Fz {r[3]:+.2f} xy={r[5]:.2f}")
