#!/usr/bin/env python3
"""Analyze FlowMPPI WTA diag log."""
import csv, sys, statistics as st

p = sys.argv[1] if len(sys.argv) > 1 else "sweep_compare/wta_verify/diag.csv"
WARMUP = 5.0

rs = []
with open(p) as f:
    for r in csv.DictReader(f):
        try: rs.append({k: float(v) for k, v in r.items()})
        except (ValueError, TypeError): pass
rs = [r for r in rs if r['time'] >= WARMUP]
if not rs: print("no data"); sys.exit(1)

n = len(rs)
fm_wins = sum(1 for r in rs if r['winner'] == 0)
mppi_wins = n - fm_wins
print(f"=== WTA winner stats (n={n} planner iters after warmup) ===")
print(f"FM wins:   {fm_wins} ({100*fm_wins/n:.1f}%)")
print(f"MPPI wins: {mppi_wins} ({100*mppi_wins/n:.1f}%)")

min_fm = [r['min_fm'] for r in rs]
min_mppi = [r['min_mppi'] for r in rs]
print(f"\n=== Best rollout cost (after warmup) ===")
print(f"min_fm:   mean {st.mean(min_fm):.1f}, median {sorted(min_fm)[n//2]:.1f}")
print(f"min_mppi: mean {st.mean(min_mppi):.1f}, median {sorted(min_mppi)[n//2]:.1f}")
gap = [a - b for a, b in zip(min_fm, min_mppi)]
print(f"Δ(fm - mppi) mean: {st.mean(gap):+.1f}  (negative = FM cheaper)")

d_mppi_fm = [r['d_mppi_fm'] for r in rs]
d_mppi_dt = [r['d_mppi_dt'] for r in rs]
print(f"\n=== Plan dynamics ===")
print(f"d_mppi_fm: mean {st.mean(d_mppi_fm):.2f}  (L2 between MPPI nominal and FM nominal)")
print(f"d_mppi_dt: mean {st.mean(d_mppi_dt):.2f}  (MPPI accumulator smoothness)")
