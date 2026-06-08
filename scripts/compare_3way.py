#!/usr/bin/env python3
"""3-way comparison: Cost / WTA TE off / WTA TE on."""
import csv, math, statistics as st, os

SWEEPS = [
    ("Cost / TE on / FM_track=1.5", "sweep_TH"),
    ("WTA  / TE off / FM_track=0",  "sweep_TH_wta"),
    ("WTA  / TE on  / FM_track=0",  "sweep_TH_wta_te_on"),
]
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

# Per-cell comparison
print(f"\n=== contact% (T, H) ===")
print(f"{'T':>4}  {'H':>5}  {'Cost':>8}  {'WTA-off':>8}  {'WTA-on':>8}  {'Δ(WTA-Cost)':>12}")
print("-"*60)
for T in [8, 16, 32, 64, 128]:
    for H in ["0.05", "0.10", "0.20", "0.30"]:
        vals = []
        for _, d in SWEEPS:
            p = f"{d}/T{T}_H{H}_s1.csv"
            r = stats_one(p) if os.path.exists(p) else None
            vals.append(r[0] if r else None)
        if all(v is not None for v in vals):
            d = vals[1] - vals[0]
            print(f"{T:>4}  {H:>5}  {vals[0]:7.1f}   {vals[1]:7.1f}   {vals[2]:7.1f}    {d:+7.1f}")

print(f"\n=== xy [mm] (T, H) ===")
print(f"{'T':>4}  {'H':>5}  {'Cost':>8}  {'WTA-off':>8}  {'WTA-on':>8}  {'Δ(WTA-Cost)':>12}")
print("-"*60)
for T in [8, 16, 32, 64, 128]:
    for H in ["0.05", "0.10", "0.20", "0.30"]:
        vals = []
        for _, d in SWEEPS:
            p = f"{d}/T{T}_H{H}_s1.csv"
            r = stats_one(p) if os.path.exists(p) else None
            vals.append(r[2] if r else None)
        if all(v is not None for v in vals):
            d = vals[1] - vals[0]
            print(f"{T:>4}  {H:>5}  {vals[0]:7.2f}   {vals[1]:7.2f}   {vals[2]:7.2f}    {d:+7.2f}")

# H-dependence of contact (mean across T)
print(f"\n=== contact% vs H (mean over T) ===")
print(f"{'H':>5}  {'Cost':>8}  {'WTA-off':>8}  {'WTA-on':>8}")
print("-"*40)
for H in ["0.05", "0.10", "0.20", "0.30"]:
    avgs = []
    for _, d in SWEEPS:
        cs = []
        for T in [8, 16, 32, 64, 128]:
            p = f"{d}/T{T}_H{H}_s1.csv"
            r = stats_one(p) if os.path.exists(p) else None
            if r: cs.append(r[0])
        avgs.append(st.mean(cs) if cs else None)
    print(f"{H:>5}  {avgs[0]:7.1f}   {avgs[1]:7.1f}   {avgs[2]:7.1f}")
