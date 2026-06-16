#!/usr/bin/env python3
"""Check Phase 1 collision for the cells identified as iso-performance best
in analyze_sample_efficiency.py (sync mode).
Same metric as analyze_ode_phase1.py — max |Fz| during hybrid<0.5 descent.
"""
import csv, math, os, statistics as st

DIR = "sweep_syncasync"

# Cells to check (mode, plan, K, H, label).
CELLS = [
    ("mppi", "sync", 128, "0.30", "MPPI baseline best"),
    ("cost", "sync",  32, "0.10", "FM-DiT cost  best"),
    ("wta3", "sync",  16, "0.05", "FM-DiT WTA   best"),
    ("mlp",  "sync",  32, "0.10", "MLP cost     best"),
]
SEEDS = [1, 2, 3]

def phase1(p):
    """Phase 1 (hybrid<0.5) collision stats.
    Returns dict or None if file missing.
    """
    if not os.path.exists(p): return None
    max_fz=0.0; t_max=0.0; z_max=0.0
    impact_fz=0.0; impact_t=None
    contact_time=None
    n_p1=0
    last_z=None; last_t=None
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                t=float(r['time']); fz=float(r['Fz']);
                h=float(r['hybrid']); z=float(r['ee_z'])
            except (ValueError, KeyError):
                continue
            if h < 0.5:
                n_p1 += 1
                if abs(fz) > max_fz:
                    max_fz = abs(fz); t_max = t; z_max = z
                last_z = z; last_t = t
            else:
                if contact_time is None:
                    contact_time = t
                    impact_fz = abs(fz); impact_t = t
    if n_p1 == 0:
        return None
    return {
        'max_fz': max_fz, 't_max': t_max, 'z_max_mm': z_max*1000.0,
        'contact_t': contact_time, 'impact_fz': impact_fz,
        'n_p1': n_p1,
    }

print("="*80)
print("Phase 1 (hybrid<0.5) collision check — best (K, H) per mode, sync")
print("="*80)
print(f"{'label':>22}  K   H     | seed | max_Fz_P1(N)  @t(s)  @ee_z(mm) | contact_t(s)")
print("-"*100)
for mode, plan, K, H, label in CELLS:
    rows=[]
    for s in SEEDS:
        p = f"{DIR}/{mode}_{plan}_K{K}_H{H}_s{s}.csv"
        r = phase1(p)
        if r is None:
            print(f"  {label:>22}  K={K:>3} H={H}  s={s}  (no data)")
            continue
        rows.append(r)
        print(f"  {label:>22}  K={K:>3} H={H}  s={s} |"
              f"  {r['max_fz']:>10.2f}    {r['t_max']:>5.3f}   {r['z_max_mm']:>7.2f}   |"
              f"  {r['contact_t']}")
    if rows:
        mean_max = st.mean(r['max_fz'] for r in rows)
        max_max = max(r['max_fz'] for r in rows)
        print(f"  {'':>22}  {'':>4} {'':>5}        | mean={mean_max:6.2f}  max={max_max:6.2f}")
    print()

# Reference threshold from ODE study: ODE=12 K=32 sync had Phase 1 max ~97N
# (clean approach). ODE=1 K=8 had 631N (hard impact). >300N is "hard collision".
print("Reference thresholds (from sweep_ode_steps analysis):")
print("  <150 N  = soft / acceptable approach")
print("  150-300 N = moderate impact")
print("  >300 N  = hard collision (Phase 1 stability concern)")
