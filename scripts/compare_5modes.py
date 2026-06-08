#!/usr/bin/env python3
"""5-way: MPPI baseline vs 4 FlowMPPI modes."""
import csv, math, os, statistics as st

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
    xy = 1000 * math.sqrt(sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in rs)/n)
    return contact, xy

# (label, root, file_format)
SWEEPS = [
    ("MPPI",   "sweep_mppi_baseline", "T{T}_H{H}.csv"),
    ("WTA #1", "sweep_4modes_grid",    "wta1_T{T}_H{H}.csv"),
    ("WTA #2", "sweep_4modes_grid",    "wta2_T{T}_H{H}.csv"),
    ("WTA #3", "sweep_4modes_grid",    "wta3_T{T}_H{H}.csv"),
    ("Cost",   "sweep_4modes_grid",    "cost_T{T}_H{H}.csv"),
]

T_LIST = [8, 16, 32, 64, 128]
H_LIST = ["0.05", "0.10", "0.20", "0.30"]

# Per (T, H) — contact / xy
for what, idx in [("contact %", 0), ("xy [mm]", 1)]:
    print(f"\n=== {what} ===")
    print(f"{'T':>4} {'H':>5}  " + "  ".join(f"{l:>9}" for l,_,_ in SWEEPS))
    print("-"*70)
    for T in T_LIST:
        for H in H_LIST:
            line = f"{T:>4} {H:>5}  "
            for _, root, fmt in SWEEPS:
                p = f"{root}/{fmt.format(T=T, H=H)}"
                r = stats_one(p) if os.path.exists(p) else None
                if r:
                    v = r[idx]
                    line += f"  {v:>7.2f}  "
                else:
                    line += f"  {'--':>9}"
            print(line)

# Low-compute win analysis: each mode's best at T<=16
print("\n=== Low-compute (T ≤ 16): each mode's best ===")
print(f"{'mode':<10}  best @ T≤16          best @ T=128")
for lbl, root, fmt in SWEEPS:
    low_rows = []
    high_rows = []
    for T in T_LIST:
        for H in H_LIST:
            p = f"{root}/{fmt.format(T=T, H=H)}"
            r = stats_one(p) if os.path.exists(p) else None
            if not r: continue
            c, xy = r
            if xy < 6.0:  # exclude xy-blown
                if T <= 16: low_rows.append((T, H, c, xy))
                if T == 128: high_rows.append((T, H, c, xy))
    if not low_rows or not high_rows: continue
    low_best = max(low_rows, key=lambda x: x[2])
    high_best = max(high_rows, key=lambda x: x[2])
    print(f"{lbl:<10}  T={low_best[0]} H={low_best[1]}: c={low_best[2]:.1f}% xy={low_best[3]:.2f}  |  T={high_best[0]} H={high_best[1]}: c={high_best[2]:.1f}% xy={high_best[3]:.2f}")
