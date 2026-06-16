#!/usr/bin/env python3
"""Phase 1 stability analysis for ODE sweep.
Phase 1: hybrid<0.5 (approach to table). Look for collision/oscillation.
Metrics:
  - phase1 max |Fz|: impact force (collision indicator)
  - phase1 time to contact (first hybrid>=0.5)
  - phase1 ee_z min (lowest z reached before contact)
  - phase1 z-velocity at contact (impact velocity)
"""
import csv, os, statistics as st

DIR = "sweep_ode_steps"
ODES = [12, 8, 5, 3, 1]
KS = [8, 16, 32]
SEEDS = [1, 2, 3]

def phase1_stats(p):
    if not os.path.exists(p):
        return None
    t_prev = None; z_prev = None
    max_fz_p1 = 0.0
    impact_fz = 0.0       # Fz at hybrid transition
    impact_vz = 0.0       # estimated dz/dt at hybrid transition
    contact_time = None   # time of first hybrid>=0.5
    p1_count = 0
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                t = float(r['time'])
                h = float(r['hybrid'])
                fz = float(r['Fz'])
                z = float(r['ee_z'])
            except (ValueError, KeyError):
                continue
            if h < 0.5:
                p1_count += 1
                if abs(fz) > max_fz_p1:
                    max_fz_p1 = abs(fz)
                t_prev = t; z_prev = z
            else:
                if contact_time is None:
                    contact_time = t
                    impact_fz = abs(fz)
                    if t_prev is not None and z_prev is not None and (t - t_prev) > 0:
                        impact_vz = (z - z_prev) / (t - t_prev)
    if p1_count == 0:
        return None
    return {
        'max_fz_p1': max_fz_p1,
        'impact_fz': impact_fz,
        'impact_vz_mm_s': impact_vz * 1000.0,
        'contact_time': contact_time if contact_time else float('nan'),
    }

print("Phase 1 (hybrid<0.5) stability — ODE sweep, sync, WTA3")
print()
print(f"{'ODE':>4} {'K':>4} | {'max_Fz_P1':>10} {'impact_Fz':>10} {'impact_vz':>11} {'contact_t':>10}")
print(f"{'':>4} {'':>4} | {'(N)':>10} {'(N)':>10} {'(mm/s)':>11} {'(s)':>10}")
print('-' * 70)

results = {}
for ode in ODES:
    for K in KS:
        rows = []
        for s in SEEDS:
            r = phase1_stats(f'{DIR}/wta3_sync_ode{ode}_K{K}_s{s}.csv')
            if r:
                rows.append(r)
        if not rows:
            print(f"{ode:>4} {K:>4} | (no data)")
            continue
        mean_max_fz = st.mean(r['max_fz_p1'] for r in rows)
        mean_impact_fz = st.mean(r['impact_fz'] for r in rows)
        mean_impact_vz = st.mean(r['impact_vz_mm_s'] for r in rows)
        mean_contact_t = st.mean(r['contact_time'] for r in rows if r['contact_time'])
        results[(ode, K)] = {
            'max_fz_p1': mean_max_fz, 'impact_fz': mean_impact_fz,
            'impact_vz': mean_impact_vz, 'contact_t': mean_contact_t,
        }
        print(f"{ode:>4} {K:>4} | {mean_max_fz:>10.2f} {mean_impact_fz:>10.2f} {mean_impact_vz:>11.1f} {mean_contact_t:>10.3f}")

print()
print("Per-seed max_Fz_P1 (collision force) — K=16")
print(f"{'ODE':>4} | {'s1':>8} {'s2':>8} {'s3':>8}")
for ode in ODES:
    vs = []
    for s in SEEDS:
        r = phase1_stats(f'{DIR}/wta3_sync_ode{ode}_K16_s{s}.csv')
        vs.append(r['max_fz_p1'] if r else None)
    cs = [f"{v:.2f}" if v is not None else "  -  " for v in vs]
    print(f"{ode:>4} | {cs[0]:>8} {cs[1]:>8} {cs[2]:>8}")
