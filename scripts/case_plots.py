#!/usr/bin/env python3
"""Generate per-case analysis plots for slide deck.

Usage:
  python3 scripts/case_plots.py <force_csv> <label> [output_dir]

Outputs (per case):
  <out>/<label>_xy.png       : 2D xy trajectory (ee vs target)
  <out>/<label>_force.png    : F_press_z time series
  <out>/<label>_metrics.txt  : numeric summary (xy_RMS, F_mean, sustain%, ez_std)
"""
import csv, math, statistics as st, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WARMUP = 5.0
EE_W = 7.46
CONTACT_Z_MAX = 0.312

if len(sys.argv) < 3:
    print(__doc__); sys.exit(1)

csv_path = sys.argv[1]
label    = sys.argv[2]
out_dir  = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(csv_path)
os.makedirs(out_dir, exist_ok=True)

# Load
rs = []
with open(csv_path) as f:
    for r in csv.DictReader(f):
        try: rs.append({k: float(v) for k, v in r.items()})
        except (ValueError, TypeError): pass
rs = [r for r in rs if r['time'] >= WARMUP]
if not rs:
    print(f"No data in {csv_path}"); sys.exit(1)

t   = [r['time'] - WARMUP for r in rs]
eex = [r['ee_x']  for r in rs]
eey = [r['ee_y']  for r in rs]
eez = [r['ee_z']  for r in rs]
tx  = [r['tgt_x'] for r in rs]
ty  = [r['tgt_y'] for r in rs]
fz  = [r['Fz']    for r in rs]      # raw site-frame Fz
fpz = [v - EE_W for v in fz]        # gravity-comp press force

# Metrics
n = len(rs)
xy_rms_mm = 1000 * math.sqrt(sum((a-b)**2 + (c-d)**2 for a,b,c,d in zip(eex,tx,eey,ty)) / n)
ez_std_mm = 1000 * (st.stdev(eez) if n > 1 else 0)
ez_mean_mm = 1000 * st.mean(eez)
fz_mean = st.mean(fz)
fpz_mean = st.mean(fpz)
fpz_p50 = sorted(fpz)[n//2]
fpz_p95 = sorted(fpz)[int(0.95*n)]
sustain = 100 * sum(1 for z in eez if z < CONTACT_Z_MAX) / n

# === Plot 1: xy 2D trajectory ===
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(tx,  ty,  '--', color='gray', linewidth=1.5, label='target', alpha=0.7)
ax.plot(eex, eey, '-',  color='C0',   linewidth=1.2, label='end-effector')
ax.plot(eex[0], eey[0], 'go', markersize=8, label='start')
ax.plot(eex[-1], eey[-1], 'rs', markersize=8, label='end')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title(f'{label}: XY trajectory  (xy_RMS = {xy_rms_mm:.2f} mm)')
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{out_dir}/{label}_xy.png", dpi=150)
plt.close(fig)

# === Plot 2: F_press_z time series ===
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot(t, fpz, '-', color='C1', linewidth=0.8)
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('time [s] (after 5s warmup)')
ax.set_ylabel('F_press_z = Fz_raw − mg [N]')
ax.set_title(f'{label}: Press force  '
             f'(p50={fpz_p50:+.1f}, p95={fpz_p95:+.1f}, mean={fpz_mean:+.2f} N)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{out_dir}/{label}_force.png", dpi=150)
plt.close(fig)

# === Metrics summary ===
summary = (
    f"=== {label} ===\n"
    f"  n_samples         : {n}\n"
    f"  xy_RMS            : {xy_rms_mm:.3f} mm\n"
    f"  ee_z mean / std   : {ez_mean_mm:.2f} / {ez_std_mm:.3f} mm\n"
    f"  sustain (ee_z<312): {sustain:.1f}%\n"
    f"  Fz_raw mean       : {fz_mean:+.3f} N\n"
    f"  F_press_z mean    : {fpz_mean:+.3f} N\n"
    f"  F_press_z p50     : {fpz_p50:+.3f} N\n"
    f"  F_press_z p95     : {fpz_p95:+.3f} N\n"
)
with open(f"{out_dir}/{label}_metrics.txt", 'w') as f:
    f.write(summary)
print(summary)
print(f"Wrote: {label}_xy.png, {label}_force.png, {label}_metrics.txt in {out_dir}")
