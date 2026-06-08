#!/usr/bin/env python3
"""MPPI 3-seed analyzer with phase 1/2 + wipe metrics."""
import csv, math, os, statistics as st

ROOT = "sweep_mppi_3seeds"
F_THRESH = 1.0
WIPE_STAB = 5.0  # wipe starts after this

def stats_one(path):
    rs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try: rs.append({k: float(v) for k, v in r.items()})
            except (ValueError, TypeError): pass
    if not rs: return None

    # Phase split: hybrid flag
    phase1 = [r for r in rs if r['hybrid'] < 0.5]
    phase2 = [r for r in rs if r['hybrid'] >= 0.5]
    if not phase2: return None  # never transitioned

    # Phase transition time (first hybrid=1 sample)
    t_transition = phase2[0]['time']

    # Phase 1 contact (did Fz > threshold during approach?)
    p1_contact_samples = sum(1 for r in phase1 if r['Fz'] > F_THRESH)
    p1_contact_any = p1_contact_samples > 0
    p1_fz_peak = max((r['Fz'] for r in phase1), default=0.0)

    # Wipe phase (t > WIPE_STAB), with hybrid=1
    wipe = [r for r in rs if r['time'] > WIPE_STAB and r['hybrid'] >= 0.5]
    if not wipe: return None
    n = len(wipe)
    fz = [r['Fz'] for r in wipe]
    contact = 100.0 * sum(1 for v in fz if v > F_THRESH) / n
    fz_mean = st.mean(fz)
    fz_p95 = sorted(fz)[int(0.95*n)]
    xy = 1000 * math.sqrt(sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in wipe)/n)

    # Transient: phase 2 first 1s peak Fz (just after switch, may not be in wipe)
    t2_init = [r for r in rs if r['hybrid'] >= 0.5 and r['time'] - t_transition < 1.0]
    t2_peak = max((r['Fz'] for r in t2_init), default=0.0)

    return dict(
        t_transition=t_transition,
        p1_contact_any=p1_contact_any,
        p1_fz_peak=p1_fz_peak,
        t2_peak=t2_peak,
        contact=contact, fz_mean=fz_mean, fz_p95=fz_p95, xy=xy,
    )

def fmt(vals, dec=1):
    if not vals: return "-"
    if len(vals) < 2: return f"{vals[0]:.{dec}f}"
    return f"{st.mean(vals):.{dec}f}±{st.stdev(vals):.{dec}f}"

print(f"{'T':>4} {'H':>5}  {'t_swich [s]':>11}  {'P1 contact':>11}  {'P1 Fz peak':>11}  "
      f"{'P2 Fz peak':>11}  {'wipe contact %':>14}  {'wipe Fz [N]':>14}  {'wipe xy [mm]':>14}")
print("-"*125)
for T in [8, 16, 32, 64, 128]:
    for H in ["0.05", "0.10", "0.20", "0.30"]:
        tt, p1c, p1p, p2p, cs, fzs, fp95s, xys = [], [], [], [], [], [], [], []
        for s in [1, 2, 3]:
            p = f"{ROOT}/T{T}_H{H}_s{s}.csv"
            if not os.path.exists(p): continue
            r = stats_one(p)
            if not r: continue
            tt.append(r['t_transition'])
            p1c.append(1 if r['p1_contact_any'] else 0)
            p1p.append(r['p1_fz_peak'])
            p2p.append(r['t2_peak'])
            cs.append(r['contact']); fzs.append(r['fz_mean']); fp95s.append(r['fz_p95']); xys.append(r['xy'])
        if not cs: continue
        p1_any_count = sum(p1c)  # how many seeds had contact in P1
        print(f"{T:>4} {H:>5}  {fmt(tt,2):>11}  {p1_any_count}/{len(p1c):>3}{'':>4}  "
              f"{fmt(p1p):>11}  {fmt(p2p):>11}  "
              f"{fmt(cs):>14}  {fmt(fzs):>14}  {fmt(xys):>14}")
