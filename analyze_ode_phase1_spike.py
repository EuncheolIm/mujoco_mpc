#!/usr/bin/env python3
"""Find when Fz spike happens, what robot is doing."""
import csv

def find_spike(p, label):
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    't': float(r['time']), 'fz': float(r['Fz']),
                    'z': float(r['ee_z']), 'h': float(r['hybrid']),
                })
            except (ValueError, KeyError):
                continue
    # phase 1 only
    p1 = [r for r in rows if r['h'] < 0.5]
    if not p1: return
    maxr = max(p1, key=lambda r: abs(r['fz']))
    # show 20 samples around it
    idx = p1.index(maxr)
    lo = max(0, idx-5); hi = min(len(p1), idx+15)
    print(f"--- {label}: max|Fz|={abs(maxr['fz']):.1f}N at t={maxr['t']:.3f}s ---")
    for r in p1[lo:hi]:
        print(f"  t={r['t']:.3f}  Fz={r['fz']:+8.2f}  ee_z={r['z']*1000:+7.2f}mm  h={r['h']:.2f}")

# Compare worst-case ODE=1 vs ODE=12
find_spike("sweep_ode_steps/wta3_sync_ode1_K8_s1.csv",   "ode=1 K=8 s=1")
print()
find_spike("sweep_ode_steps/wta3_sync_ode1_K16_s3.csv",  "ode=1 K=16 s=3")
print()
find_spike("sweep_ode_steps/wta3_sync_ode12_K16_s3.csv", "ode=12 K=16 s=3")
print()
find_spike("sweep_ode_steps/wta3_sync_ode12_K32_s1.csv", "ode=12 K=32 s=1")
