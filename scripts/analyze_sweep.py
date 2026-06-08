#!/usr/bin/env python3
"""Analyze horizon-sweep outputs from sweep_horizon.sh.

For each (planner, horizon) run, compute:
  - xy_RMS  : sqrt(mean((ee_x - tgt_x)^2 + (ee_y - tgt_y)^2)) over t >= warmup
  - z_err   : mean(ee_z - tgt_z)                              (m)
  - F_press : mean(F_press_z) where F_press_z = F[2] - ee_weight_N (N)
  - cost_min: median(min_cost) from planner diag CSV
  - cpu_ms  : median(rollouts_ms) from planner diag CSV

Reads $OUTDIR (default /tmp/fmppi_sweep).
"""
import csv, glob, math, os, re, statistics as st, sys

OUTDIR = os.environ.get("OUTDIR", "/tmp/fmppi_sweep")
WARMUP = float(os.environ.get("WARMUP", "5.0"))  # seconds to skip at start
EE_WEIGHT_N = float(os.environ.get("EE_WEIGHT_N", "7.46"))

FORCE_PATTERN = re.compile(r"force_(?P<plan>[a-z]+)_H(?P<h>[0-9.]+)_T(?P<t>\d+)_K(?P<k>\d+)\.csv$")

def load_csv(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({k: float(v) for k, v in row.items()})
            except (ValueError, TypeError):
                continue
    return rows

def force_metrics(path):
    rs = [r for r in load_csv(path) if r["time"] >= WARMUP]
    if not rs:
        return None
    xy_se = [(r["ee_x"]-r["tgt_x"])**2 + (r["ee_y"]-r["tgt_y"])**2 for r in rs]
    xy_rms = math.sqrt(sum(xy_se)/len(xy_se))
    z_err  = st.mean([r["ee_z"]-r["tgt_z"] for r in rs])
    f_press = st.mean([r["Fz"] - EE_WEIGHT_N for r in rs])
    return dict(n=len(rs), xy_rms_mm=1000*xy_rms, z_err_mm=1000*z_err,
                f_press_N=f_press)

def diag_metrics(path):
    rs = [r for r in load_csv(path) if r.get("time", 0) >= WARMUP]
    if not rs:
        return None
    # FlowMPPI diag has min_mppi / rollouts_ms; sampling diag has min_cost / rollouts_ms.
    if "min_mppi" in rs[0]:
        mins = [r["min_mppi"] for r in rs]
    elif "min_cost" in rs[0]:
        mins = [r["min_cost"] for r in rs]
    else:
        mins = []
    ms = [r.get("rollouts_ms", 0) for r in rs]
    return dict(cost_min_med=st.median(mins) if mins else float("nan"),
                cpu_ms_med=st.median(ms) if ms else float("nan"))

rows = []
for fpath in sorted(glob.glob(os.path.join(OUTDIR, "force_*.csv"))):
    m = FORCE_PATTERN.search(fpath)
    if not m: continue
    tag = dict(plan=m["plan"], h=float(m["h"]), t=int(m["t"]), k=int(m["k"]))
    fm = force_metrics(fpath)
    diag_path = fpath.replace("/force_", "/diag_")
    dm = diag_metrics(diag_path) if os.path.exists(diag_path) else None
    rows.append({**tag, **(fm or {}), **(dm or {})})

if not rows:
    print(f"No force_*.csv under {OUTDIR}", file=sys.stderr)
    sys.exit(1)

print(f"{'planner':>8} {'horiz':>6} {'N_traj':>6} {'knots':>6}  "
      f"{'n_pts':>6} {'xy_RMS(mm)':>11} {'z_err(mm)':>10} {'F_press(N)':>11}  "
      f"{'cost_min':>10} {'cpu_ms':>8}")
print("-" * 110)
for r in sorted(rows, key=lambda x: (x["plan"], -x["h"])):
    print(f"{r['plan']:>8} {r['h']:>6.2f} {r['t']:>6d} {r['k']:>6d}  "
          f"{r.get('n','-'):>6} {r.get('xy_rms_mm', float('nan')):>11.2f} "
          f"{r.get('z_err_mm', float('nan')):>10.2f} "
          f"{r.get('f_press_N', float('nan')):>11.2f}  "
          f"{r.get('cost_min_med', float('nan')):>10.0f} "
          f"{r.get('cpu_ms_med', float('nan')):>8.2f}")
