#!/usr/bin/env python3
"""Analyze 4 modes × T×H grid."""
import csv, math, os, statistics as st

ROOT = "sweep_4modes_grid"
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

MODES = [
    ("WTA #1 (full FM)",     "wta1"),
    ("WTA #2 (shared)",      "wta2"),
    ("WTA #3 (half WTA)",    "wta3"),
    ("Cost (option E)",       "cost"),
]
T_LIST = [8, 16, 32, 64, 128]
H_LIST = ["0.05", "0.10", "0.20", "0.30"]

for name, key in MODES:
    print(f"\n=== {name} ===")
    print(f"{'T':>4}  {'H':>5}  {'contact':>8}  {'Fz_mean':>8}  {'Fz_p95':>8}  {'xy':>7}")
    print("-"*55)
    rows = []
    for T in T_LIST:
        for H in H_LIST:
            p = f"{ROOT}/{key}_T{T}_H{H}.csv"
            r = stats_one(p) if os.path.exists(p) else None
            if r:
                c, fm, fp, xy = r
                rows.append((T, H, c, fm, fp, xy))
                print(f"{T:>4}  {H:>5}  {c:7.1f}    {fm:+7.2f}  {fp:+7.2f}  {xy:6.2f}")
    if rows:
        best_c = max([r for r in rows if r[5]<6.0], key=lambda x: x[2], default=None)
        best_xy = min(rows, key=lambda x: x[5])
        if best_c:
            print(f"  best contact (xy<6): T={best_c[0]} H={best_c[1]} → {best_c[2]:.1f}% / xy={best_c[5]:.2f}")
        print(f"  best xy:              T={best_xy[0]} H={best_xy[1]} → xy={best_xy[5]:.2f} / contact={best_xy[2]:.1f}%")

print("\n=== 4-mode best summary ===")
print(f"{'mode':<22}  {'best xy cell':28}  {'best contact cell (xy<6)':28}")
for name, key in MODES:
    rows = []
    for T in T_LIST:
        for H in H_LIST:
            p = f"{ROOT}/{key}_T{T}_H{H}.csv"
            r = stats_one(p) if os.path.exists(p) else None
            if r: rows.append((T, H, *r))
    if rows:
        best_xy = min(rows, key=lambda x: x[5])
        best_c = max([r for r in rows if r[5]<6.0], key=lambda x: x[2], default=None)
        xy_str = f"T={best_xy[0]} H={best_xy[1]}: {best_xy[5]:.2f}mm c={best_xy[2]:.1f}%"
        c_str = f"T={best_c[0]} H={best_c[1]}: {best_c[2]:.1f}% xy={best_c[5]:.2f}" if best_c else "-"
        print(f"{name:<22}  {xy_str:<28}  {c_str}")
