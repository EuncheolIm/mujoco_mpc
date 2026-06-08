#!/usr/bin/env python3
"""5-way comparison: NEW vs OLD struct for Cost / WTA modes."""
import csv, math, statistics as st, os

SWEEPS = [
    ("Cost-NEW", "sweep_TH"),
    ("Cost-OLD", "sweep_TH_cost_oldstruct"),
    ("WTA-NEW (TEoff)",  "sweep_TH_wta"),
    ("WTA-NEW (TEon)",   "sweep_TH_wta_te_on"),
    ("WTA-OLD",  "sweep_TH_wta_oldstruct"),
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
    xy = 1000 * math.sqrt(sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in rs)/n)
    return contact, st.mean(fz), xy

# Contact% mean over T, by H
print("=== Contact % vs H (mean over T) ===")
print(f"{'H':>5}  " + "  ".join(f"{l:>16}" for l,_ in SWEEPS))
print("-"*100)
for H in ["0.05", "0.10", "0.20", "0.30"]:
    line = f"{H:>5}  "
    for _, d in SWEEPS:
        cs = []
        for T in [8, 16, 32, 64, 128]:
            p = f"{d}/T{T}_H{H}_s1.csv"
            r = stats_one(p) if os.path.exists(p) else None
            if r: cs.append(r[0])
        avg = st.mean(cs) if cs else 0
        line += f"  {avg:>14.1f}  "
    print(line)

print("\n=== Best cell per sweep (xy<6mm) ===")
print(f"{'sweep':>18}  {'T':>3}  {'H':>4}  {'contact':>8}  {'Fz':>7}  {'xy':>6}")
for lbl, d in SWEEPS:
    rows = []
    for T in [8, 16, 32, 64, 128]:
        for H in ["0.05", "0.10", "0.20", "0.30"]:
            p = f"{d}/T{T}_H{H}_s1.csv"
            r = stats_one(p) if os.path.exists(p) else None
            if r and r[2] < 6.0:
                rows.append((T, H, r[0], r[1], r[2]))
    best = sorted(rows, key=lambda x: -x[2])[0] if rows else None
    if best:
        T, H, c, fm, xy = best
        print(f"{lbl:>18}  {T:>3}  {H:>4}  {c:>7.1f}%  {fm:>+7.2f}  {xy:>6.2f}")
