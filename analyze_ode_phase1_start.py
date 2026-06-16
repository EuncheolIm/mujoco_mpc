#!/usr/bin/env python3
"""Look at very start of Phase 1 — when does Fz spike happen, what's ee_z?"""
import csv

def show_start(p, label, n=50):
    with open(p) as f:
        rows = list(csv.DictReader(f))
    print(f"--- {label} (first {n} samples) ---")
    print(f"{'t':>7} {'Fz':>8} {'ee_z':>8} {'tgt_z':>8} {'hyb':>5}")
    for r in rows[:n]:
        t = float(r['time']); fz = float(r['Fz']); z = float(r['ee_z'])
        tz = float(r['tgt_z']); h = float(r['hybrid'])
        print(f"{t:>7.3f} {fz:>+8.2f} {z*1000:>+8.2f} {tz*1000:>+8.2f} {h:>5.2f}")

show_start("sweep_ode_steps/wta3_sync_ode1_K8_s1.csv", "ode=1 K=8 s=1", 30)
print()
show_start("sweep_ode_steps/wta3_sync_ode12_K16_s1.csv", "ode=12 K=16 s=1", 30)
