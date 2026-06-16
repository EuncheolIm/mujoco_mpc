#!/usr/bin/env python3
"""Detailed Phase 1 trajectory analysis — when does collision happen?"""
import csv, os

DIR = "sweep_ode_steps"

def trace(p, label):
    if not os.path.exists(p):
        return
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    't': float(r['time']),
                    'fz': float(r['Fz']),
                    'h': float(r['hybrid']),
                    'z': float(r['ee_z']),
                    'tgt_z': float(r['tgt_z']),
                })
            except (ValueError, KeyError):
                continue
    # find max Fz during phase 1
    p1 = [r for r in rows if r['h'] < 0.5]
    if not p1:
        print(f"{label}: no phase1")
        return
    maxfz = max(p1, key=lambda r: abs(r['fz']))
    # contact transition
    contact_t = None
    for r in rows:
        if r['h'] >= 0.5:
            contact_t = r['t']; break

    # ee_z just before contact (last 10 phase1 samples)
    last_p1 = p1[-10:]
    print(f"--- {label} ---")
    print(f"  Phase 1 max|Fz|: {abs(maxfz['fz']):.2f} N at t={maxfz['t']:.3f}s "
          f"(ee_z={maxfz['z']*1000:.1f}mm)")
    print(f"  Contact (hybrid>=0.5) at t={contact_t}")
    print(f"  Last 10 Phase 1 samples:")
    for r in last_p1:
        print(f"    t={r['t']:.3f}  Fz={r['fz']:+7.2f}  ee_z={r['z']*1000:+7.2f}mm  hybrid={r['h']:.2f}")

# focus: ODE=12 K=16 seed1 (was best) vs ODE=1 K=16 seed3 (worst)
for ode in [12, 1]:
    for K in [8, 16, 32]:
        for s in [1, 2, 3]:
            trace(f'{DIR}/wta3_sync_ode{ode}_K{K}_s{s}.csv', f"ode={ode} K={K} s={s}")
