#!/usr/bin/env python3
"""Analyze contact-toggle wobble: ee_z, Fz, hybrid frequency.
Plot a short windowed segment + spectral analysis."""
import csv, math, sys, statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "out/wobble_check/run_500hz.csv"
WARMUP = 8.0   # past stabilization + first half circle

rs = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        try: rs.append({k: float(v) for k, v in r.items()})
        except (ValueError, TypeError): pass
rs = [r for r in rs if r['time'] >= WARMUP]
if not rs:
    print("no data"); sys.exit(1)

t = np.array([r['time'] for r in rs])
fz = np.array([r['Fz'] for r in rs])
eez = np.array([r['ee_z'] for r in rs]) * 1000  # mm
eex = np.array([r['ee_x'] for r in rs])
eey = np.array([r['ee_y'] for r in rs])

# detect contact toggles
in_contact = fz > 1.0
toggles = np.sum(np.abs(np.diff(in_contact.astype(int))))
print(f"samples: {len(rs)}")
print(f"sim time: {t[-1]-t[0]:.2f} s")
sample_dt = (t[-1]-t[0])/max(1,len(rs)-1)
print(f"sample dt: {sample_dt*1000:.2f} ms ({1/sample_dt:.0f} Hz)")
print(f"contact toggles: {toggles}  (~{toggles/(t[-1]-t[0]):.1f}/s)")
print(f"contact %: {100*np.mean(in_contact):.1f}")
print(f"ee_z: mean {np.mean(eez):.2f} mm, std {np.std(eez):.3f} mm, range [{np.min(eez):.2f}, {np.max(eez):.2f}]")
print(f"Fz: mean {np.mean(fz):+.2f} N, std {np.std(fz):.2f} N")

# spectral: ee_z and Fz FFT on a 4s window
win_t = 4.0
n_win = int(win_t / sample_dt)
if len(eez) >= n_win:
    # center window
    mid = len(eez) // 2
    s = eez[mid-n_win//2:mid+n_win//2]
    s = s - np.mean(s)
    f = np.fft.rfft(s * np.hanning(len(s)))
    freq = np.fft.rfftfreq(len(s), sample_dt)
    mag = np.abs(f) / len(s)
    # top 5 freq peaks
    idx = np.argsort(mag[1:])[-5:][::-1] + 1
    print(f"\nee_z spectrum (top peaks):")
    for i in idx:
        print(f"  {freq[i]:6.1f} Hz  mag {mag[i]*1000:.3f} (mm)")

    # same for Fz
    sf = fz[mid-n_win//2:mid+n_win//2]
    sf = sf - np.mean(sf)
    ff = np.fft.rfft(sf * np.hanning(len(sf)))
    magf = np.abs(ff) / len(sf)
    idx = np.argsort(magf[1:])[-5:][::-1] + 1
    print(f"\nFz spectrum (top peaks):")
    for i in idx:
        print(f"  {freq[i]:6.1f} Hz  mag {magf[i]:.2f} (N)")

# time-domain plots, 2s window around a toggle event
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
# zoom to 6s..8s past warmup
t0 = t[0]
m = (t - t0 < 6.0)
axes[0].plot(t[m]-t0, fz[m], '-', linewidth=0.6, color='C1')
axes[0].axhline(1.0, color='gray', ls='--', lw=0.5, label='F=1N (contact thr)')
axes[0].set_ylabel('Fz [N]'); axes[0].legend(loc='upper right'); axes[0].grid(alpha=0.3)
axes[1].plot(t[m]-t0, eez[m], '-', linewidth=0.6, color='C0')
axes[1].set_ylabel('ee_z [mm]'); axes[1].grid(alpha=0.3)
axes[2].plot(t[m]-t0, eex[m], '-', linewidth=0.6, label='ee_x', color='C2')
axes[2].plot(t[m]-t0, eey[m], '-', linewidth=0.6, label='ee_y', color='C3')
axes[2].set_ylabel('ee_x, ee_y [m]'); axes[2].set_xlabel('time [s] (post warmup)')
axes[2].legend(); axes[2].grid(alpha=0.3)
plt.tight_layout()
out_png = CSV.replace('.csv','_wobble.png')
fig.savefig(out_png, dpi=150)
print(f"\nplot: {out_png}")
