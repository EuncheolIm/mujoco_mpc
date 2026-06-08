#!/usr/bin/env python3
"""Quick metrics for a single CSV."""
import csv, sys, math, statistics as st

CSV = sys.argv[1] if len(sys.argv) > 1 else None
if not CSV: print("usage: analyze_one.py <csv>"); sys.exit(1)
WARMUP = 5.0; F_THRESH = 1.0
rs = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        try: rs.append({k: float(v) for k, v in r.items()})
        except (ValueError, TypeError): pass
rs = [r for r in rs if r['time'] >= WARMUP]
if not rs: print("no data"); sys.exit(1)
n = len(rs)
fz = [r['Fz'] for r in rs]
contact = 100.0 * sum(1 for v in fz if v > F_THRESH) / n
fz_mean = st.mean(fz); fz_p95 = sorted(fz)[int(0.95*n)]
xy = 1000 * math.sqrt(sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in rs) / n)
print(f"{CSV}")
print(f"  contact = {contact:.1f}%  Fz mean={fz_mean:+.2f} p95={fz_p95:+.2f}  xy={xy:.2f}mm")
