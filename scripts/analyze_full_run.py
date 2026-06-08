#!/usr/bin/env python3
"""Full-run analyzer — includes initial transient."""
import csv, glob, math, os, statistics as st, sys

F_THRESH = 1.0

def stats_full(path):
    rs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try: rs.append({k: float(v) for k, v in r.items()})
            except (ValueError, TypeError): pass
    if not rs: return None
    n_all = len(rs)
    fz_all = [r['Fz'] for r in rs]
    xy_all = [math.sqrt((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2)*1000 for r in rs]

    # full-window metrics
    contact_full = 100.0 * sum(1 for v in fz_all if v > F_THRESH) / n_all
    xy_full = math.sqrt(sum(x*x for x in xy_all)/n_all)

    # initial transient (first 5s)
    init = [r for r in rs if r['time'] <= 5.0]
    init_fz_peak = max((r['Fz'] for r in init), default=0)
    # time to first contact
    t_first = next((r['time'] for r in rs if r['Fz'] > F_THRESH), None)

    # post-5s window (original metric)
    post = [r for r in rs if r['time'] > 5.0]
    if post:
        fz_post = [r['Fz'] for r in post]
        contact_post = 100.0 * sum(1 for v in fz_post if v > F_THRESH) / len(post)
        xy_post = 1000 * math.sqrt(sum((r['ee_x']-r['tgt_x'])**2 + (r['ee_y']-r['tgt_y'])**2 for r in post)/len(post))
    else:
        contact_post = 0; xy_post = 0

    return dict(
        contact_full=contact_full, xy_full=xy_full,
        contact_post=contact_post, xy_post=xy_post,
        init_fz_peak=init_fz_peak,
        t_first=t_first if t_first is not None else float('nan'),
    )

if __name__ == "__main__":
    ROOT = sys.argv[1]
    LABEL = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(ROOT)
    pattern = sys.argv[3] if len(sys.argv) > 3 else "T{T}_H{H}_s1.csv"
    T_LIST = [8, 16, 32, 64, 128]
    H_LIST = ["0.05", "0.10", "0.20", "0.30"]
    print(f"=== {LABEL} ===")
    print(f"{'T':>4} {'H':>5}  {'c_full':>7} {'c_post':>7}  {'xy_full':>7} {'xy_post':>7}  {'Fz_peak':>8} {'t_first':>7}")
    for T in T_LIST:
        for H in H_LIST:
            p = f"{ROOT}/{pattern.format(T=T, H=H)}"
            if not os.path.exists(p): continue
            s = stats_full(p)
            if not s: continue
            print(f"{T:>4} {H:>5}  {s['contact_full']:6.1f}  {s['contact_post']:6.1f}   "
                  f"{s['xy_full']*1000 if s['xy_full']<1 else s['xy_full']:6.2f}  {s['xy_post']:6.2f}   "
                  f"{s['init_fz_peak']:+7.1f}  {s['t_first']:6.2f}")
