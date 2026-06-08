#!/usr/bin/env python3
"""Analyze scale-verify sweep. Per-condition stats:
  - contact %     : fraction of samples with Fz_raw > 1N (after 5s warmup)
  - Fz_mean       : mean raw Fz (N)
  - F_press_mean  : mean (Fz - 7.46)
  - q_track_RMS   : RMS over time of ||qpos - q_fm_target|| (rad)
  - xy_RMS        : RMS xy tracking (mm)
Aggregates 3 seeds per condition -> mean ± std.
"""
import csv, glob, math, os, statistics as st, sys

ROOT = "out/sweep_scale_verify"
WARMUP = 5.0
EE_W = 7.46
F_THRESH = 1.0

def stats_one(path):
    rs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try: rs.append({k: float(v) for k, v in r.items()})
            except (ValueError, TypeError): pass
    rs = [r for r in rs if r['time'] >= WARMUP]
    if not rs: return None
    n = len(rs)
    fz = [r['Fz'] for r in rs]
    contact = 100.0 * sum(1 for v in fz if v > F_THRESH) / n
    fz_mean = st.mean(fz)
    fp_mean = fz_mean - EE_W
    # q-tracking RMS: sum (q_i - qd_i)^2 across joints, then sqrt and time-RMS
    q_err_sq_sum = 0.0
    for r in rs:
        s = 0.0
        for i in range(1, 8):
            s += (r[f'q{i}'] - r[f'qd{i}'])**2
        q_err_sq_sum += s  # already squared norm
    q_rms = math.sqrt(q_err_sq_sum / n)
    xy_mm = 1000 * math.sqrt(
        sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in rs) / n)
    return contact, fz_mean, fp_mean, q_rms, xy_mm

def agg(label, files):
    rows = [stats_one(f) for f in files]
    rows = [r for r in rows if r]
    if not rows: return
    arrs = list(zip(*rows))
    def ms(a): return st.mean(a), (st.stdev(a) if len(a)>1 else 0.0)
    cm, cs = ms(arrs[0]); fzm, fzs = ms(arrs[1])
    fpm, fps = ms(arrs[2]); qm, qs = ms(arrs[3]); xm, xs = ms(arrs[4])
    print(f"{label:14s}  contact {cm:5.1f}±{cs:4.1f}  Fz {fzm:+6.2f}±{fzs:4.2f}  "
          f"F_press {fpm:+6.2f}  qRMS {qm:.4f}±{qs:.4f}  xyRMS {xm:5.2f}±{xs:4.2f}")

print(f"{'label':14s}  {'contact %':14s}  {'Fz [N]':14s}  {'F_press':9s}  {'qRMS [rad]':16s}  {'xyRMS [mm]':14s}")
print("-"*110)
for FM in ["1.5", "3.0", "5.0", "10.0", "30.0"]:
    agg(f"cost_FM={FM}", sorted(glob.glob(f"{ROOT}/force_cost_FM{FM}_s*.csv")))
agg("FM-only", sorted(glob.glob(f"{ROOT}/force_fmonly_s*.csv")))
