#!/usr/bin/env python3
"""Generic T x H sweep analyzer. Usage: analyze_TH.py <sweep_dir> [label]"""
import csv, sys, math, statistics as st, os

ROOT = sys.argv[1] if len(sys.argv) > 1 else "sweep_TH"
LABEL = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(ROOT)
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
    xy = 1000 * math.sqrt(
        sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in rs) / n)
    return contact, fz_mean, fz_p95, xy

print(f"=== {LABEL} ===")
print(f"{'T':>4}  {'H':>5}  {'contact %':>9}  {'Fz mean':>9}  {'Fz p95':>9}  {'xy [mm]':>9}")
print("-"*60)
rows = []
for T in [8, 16, 32, 64, 128]:
    for H in ["0.05", "0.10", "0.20", "0.30"]:
        path = f"{ROOT}/T{T}_H{H}_s1.csv"
        if not os.path.exists(path): continue
        r = stats_one(path)
        if not r: continue
        c, fm, fp, xy = r
        rows.append((T, float(H), c, fm, fp, xy))
        print(f"{T:>4}  {H:>5}  {c:7.1f}    {fm:+6.2f}    {fp:+6.2f}    {xy:6.2f}")

print("\n=== Top 3 by contact (xy<6mm) ===")
for r in sorted([x for x in rows if x[5]<6.0], key=lambda x: -x[2])[:3]:
    T, H, c, fm, fp, xy = r
    print(f"  T={T} H={H}: contact {c:.1f}%, Fz {fm:+.1f} (p95 {fp:+.1f}), xy {xy:.2f}mm")
print("\n=== Top 3 by xy (lowest) ===")
for r in sorted(rows, key=lambda x: x[5])[:3]:
    T, H, c, fm, fp, xy = r
    print(f"  T={T} H={H}: contact {c:.1f}%, Fz {fm:+.1f}, xy {xy:.2f}mm")
