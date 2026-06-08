#!/usr/bin/env python3
"""xy 2D + Fz time-series plot for a single cell CSV.
Usage: plot_cell.py <csv> <label> [outdir]"""
import csv, math, sys, os, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print(__doc__); sys.exit(1)
csv_path = sys.argv[1]
label = sys.argv[2]
outdir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(csv_path) or "."
os.makedirs(outdir, exist_ok=True)

rs = []
with open(csv_path) as f:
    for r in csv.DictReader(f):
        try: rs.append({k: float(v) for k, v in r.items()})
        except (ValueError, TypeError): pass
if not rs: print("no data"); sys.exit(1)

# Plot 범위: wipe (원궤적) 시작 시점부터
WIPE = 5.0
rs_plot = [r for r in rs if r['time'] >= WIPE]
if not rs_plot: print("no data after wipe start"); sys.exit(1)

t = [r['time'] for r in rs_plot]
fz = [r['Fz'] for r in rs_plot]
eex = [r['ee_x'] for r in rs_plot]; eey = [r['ee_y'] for r in rs_plot]
tgx = [r['tgt_x'] for r in rs_plot]; tgy = [r['tgt_y'] for r in rs_plot]

# phase transition time (for context only — wipe phase 는 항상 hybrid)
t_switch = next((r['time'] for r in rs if r['hybrid'] >= 0.5), None)

# metrics (wipe phase: t>=5s and hybrid==1)
wipe = [r for r in rs_plot if r['hybrid'] >= 0.5]
if wipe:
    n = len(wipe)
    wipe_c = 100.0 * sum(1 for r in wipe if r['Fz'] > 1.0) / n
    wipe_xy = 1000 * math.sqrt(sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in wipe)/n)
    wipe_fz = st.mean(r['Fz'] for r in wipe)
else:
    wipe_c = wipe_xy = wipe_fz = float('nan')

# === Plot 1: xy 2D ===
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(tgx, tgy, '--', color='gray', linewidth=1.5, label='target', alpha=0.7)
ax.plot(eex, eey, '-', color='C0', linewidth=1.2, label='ee')
ax.plot(eex[0], eey[0], 'go', markersize=8, label='start')
ax.plot(eex[-1], eey[-1], 'rs', markersize=8, label='end')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title(f'{label}: xy trajectory  (wipe xy_RMS = {wipe_xy:.2f} mm)')
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='best'); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{outdir}/{label}_xy.png", dpi=150)
plt.close(fig)

# === Plot 2: Fz time series ===
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot(t, fz, '-', color='C1', linewidth=0.7, label='Fz (site +z = press)')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axhline(10, color='C2', linewidth=0.7, linestyle=':', label='F_des = 10 N')
ax.set_xlabel('time [s]')
ax.set_ylabel('Fz [N]')
ax.set_title(f'{label}: contact force  '
             f'(wipe contact={wipe_c:.1f}%, mean Fz={wipe_fz:+.1f} N)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f"{outdir}/{label}_force.png", dpi=150)
plt.close(fig)

print(f"Wrote: {outdir}/{label}_xy.png, {label}_force.png")
print(f"  t_switch={t_switch:.2f}s, wipe_c={wipe_c:.1f}%, wipe_xy={wipe_xy:.2f}mm, wipe_Fz_mean={wipe_fz:+.2f}N")
