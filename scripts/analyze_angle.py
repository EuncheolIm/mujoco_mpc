#!/usr/bin/env python3
"""Bin Fz by circle angle to check direction-dependent contact loss."""
import csv, sys, math
import statistics as st

WARMUP = 5.0

if len(sys.argv) < 2:
    print("usage: analyze_angle.py <csv>"); sys.exit(1)

rs = []
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        try: rs.append({k: float(v) for k, v in r.items()})
        except (ValueError, TypeError): pass
rs = [r for r in rs if r['time'] >= WARMUP]
if not rs: print("no data"); sys.exit(1)

# center of circle ≈ mean of target xy
cx = sum(r['tgt_x'] for r in rs) / len(rs)
cy = sum(r['tgt_y'] for r in rs) / len(rs)

# bin by angle (8 bins of 45°)
bins = {i: [] for i in range(8)}
for r in rs:
    th = math.atan2(r['tgt_y']-cy, r['tgt_x']-cx)
    idx = int(((th + math.pi) / (2*math.pi)) * 8) % 8
    bins[idx].append(r['Fz'])

DIRS = ["-X", "-X,-Y","-Y","+X,-Y","+X","+X,+Y","+Y","-X,+Y"]
print(f"{'angle':14s}  {'n':>4}  {'Fz mean':>9}  {'Fz p50':>8}  {'contact %':>10}")
for i in range(8):
    fz = bins[i]
    if not fz: continue
    contact = 100.0 * sum(1 for v in fz if v > 1.0) / len(fz)
    p50 = sorted(fz)[len(fz)//2]
    print(f"{DIRS[i]:14s}  {len(fz):>4}  {st.mean(fz):+8.2f}   {p50:+7.2f}   {contact:7.1f}")
