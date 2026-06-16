#!/usr/bin/env python3
"""Sample efficiency with Phase 1 + worst-seed filter.
A (mode, plan, K, H) cell is VALID only if ALL 3 seeds pass:
  1. Phase 1 max |Fz| < FZ_P1_LIMIT
  2. contact_t < CONTACT_T_LIMIT (Phase 2 entry not lost in fail)
  3. Phase 2 wipe xy RMS < XY_LIMIT
The "best" (K, H) is the smallest (K + lex H) cell whose worst-seed contact%
still meets MPPI baseline's worst-seed contact%.
"""
import csv, math, os, statistics as st

DIR = "sweep_syncasync"
K_LIST = [8, 16, 32, 64, 128]
H_LIST = ["0.05", "0.10", "0.20", "0.30"]
SEEDS = [1, 2, 3]
MODES = ["mppi", "cost", "wta3", "mlp"]
LABEL = {"mppi":"MPPI baseline", "cost":"FM-DiT cost",
         "wta3":"FM-DiT WTA",   "mlp":"MLP cost"}

# Strict filter thresholds.
FZ_P1_LIMIT     = 1.0     # N — Phase 1 must be near-zero force (no premature contact)
CONTACT_T_LIMIT = 5.0     # s — must enter Phase 2 by 5s
XY_LIMIT        = 5.0     # mm — wipe xy RMS
FZ_CONTACT_THR  = 1.0     # N — counts as contact in Phase 2
WIPE_T          = 5.0     # s — Phase 2 wipe analysis window starts here

def seed_stats(p):
    """Return per-seed dict {p1_max_fz, contact_t, xy_rms_mm, contact_pct,
    plan_ms, fm_ms} or None if file missing / no Phase 1."""
    if not os.path.exists(p): return None
    p1_max=0.0; contact_t=None
    p2_n=0; p2_cn=0; p2_sq=0.0
    pms=[]; fms=[]
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                t=float(r['time']); h=float(r['hybrid']); fz=float(r['Fz'])
                pm=float(r.get('plan_ms','0'));
                fm=float(r.get('fm_ms','0'))
            except (ValueError, KeyError):
                continue
            if pm>0: pms.append(pm)
            if fm>0: fms.append(fm)
            if h < 0.5:
                if abs(fz) > p1_max: p1_max = abs(fz)
            else:
                if contact_t is None: contact_t = t
                if t > WIPE_T:
                    p2_n += 1
                    if fz > FZ_CONTACT_THR: p2_cn += 1
                    try:
                        dx=float(r['ee_x'])-float(r['tgt_x'])
                        dy=float(r['ee_y'])-float(r['tgt_y'])
                        p2_sq += dx*dx + dy*dy
                    except (ValueError, KeyError):
                        pass
    return {
        'p1_max_fz': p1_max,
        'contact_t': contact_t,        # None means never entered Phase 2
        'p2_n': p2_n,
        'contact_pct': (100.0 * p2_cn / p2_n) if p2_n>0 else 0.0,
        'xy_rms_mm': (1000.0 * math.sqrt(p2_sq / p2_n)) if p2_n>0 else float('inf'),
        'plan_ms': st.mean(pms) if pms else 0.0,
        'fm_ms': st.mean(fms) if fms else 0.0,
    }

def cell_seeds(mode, plan, K, H):
    out=[]
    for s in SEEDS:
        r = seed_stats(f'{DIR}/{mode}_{plan}_K{K}_H{H}_s{s}.csv')
        out.append(r)
    return out

def cell_valid(seeds):
    """All-seeds validity per the three thresholds."""
    if any(s is None for s in seeds): return False
    for s in seeds:
        if s['contact_t'] is None: return False
        if s['contact_t'] > CONTACT_T_LIMIT: return False
        if s['p1_max_fz'] > FZ_P1_LIMIT: return False
        if s['xy_rms_mm'] > XY_LIMIT: return False
    return True

def cell_worst_contact(seeds):
    return min(s['contact_pct'] for s in seeds)

def cell_worst_xy(seeds):  # higher = worse
    return max(s['xy_rms_mm'] for s in seeds)

def cell_mean(seeds, key):
    return st.mean(s[key] for s in seeds)

print("="*92)
print(f"Strict filter: p1_max_fz<{FZ_P1_LIMIT}N  contact_t<{CONTACT_T_LIMIT}s  "
      f"xy_rms<{XY_LIMIT}mm  (all 3 seeds must pass)")
print("="*92)

# Step 1: dump validity grid
for plan in ["async","sync"]:
    print(f"\n[{plan.upper()}]   columns: H={', '.join(H_LIST)}")
    for mode in MODES:
        print(f"  {LABEL[mode]:>16}")
        for K in K_LIST:
            row = f"    K={K:>3}  "
            for H in H_LIST:
                seeds = cell_seeds(mode, plan, K, H)
                if all(s is None for s in seeds):
                    row += f"  {'--':>15}"
                    continue
                if not cell_valid(seeds):
                    # show why
                    reasons=[]
                    for i,s in enumerate(seeds,1):
                        if s is None: reasons.append(f"s{i}:no-file"); continue
                        if s['contact_t'] is None: reasons.append(f"s{i}:no-P2")
                        elif s['contact_t']>CONTACT_T_LIMIT: reasons.append(f"s{i}:tC={s['contact_t']:.1f}")
                        elif s['p1_max_fz']>FZ_P1_LIMIT: reasons.append(f"s{i}:Fz={s['p1_max_fz']:.0f}")
                        elif s['xy_rms_mm']>XY_LIMIT: reasons.append(f"s{i}:xy={s['xy_rms_mm']:.1f}")
                    row += f"  X[{','.join(reasons[:1])}]".ljust(17)
                else:
                    worst = cell_worst_contact(seeds)
                    row += f"   OK {worst:5.1f}%  "
            print(row)

# Step 2: all valid cells per mode — with full xy info
print()
print("="*100)
print("All VALID cells per mode (worst-seed contact% AND worst-seed xy_rms)")
print("="*100)
h2f = {h: float(h) for h in H_LIST}
for plan in ["async","sync"]:
    print(f"\n[{plan.upper()}]")
    print(f"  {'mode':>16}  {'K':>4} {'H':>5} | "
          f"{'contact_worst':>14} {'xy_worst':>10} {'xy_mean':>9} "
          f"{'P1_max':>7} {'tC_max':>7} {'plan_ms':>8} {'fm_ms':>7}")
    for mode in MODES:
        for K in K_LIST:
            for H in H_LIST:
                seeds = cell_seeds(mode, plan, K, H)
                if not cell_valid(seeds): continue
                worst_c = cell_worst_contact(seeds)
                worst_xy = cell_worst_xy(seeds)
                mean_xy = cell_mean(seeds, 'xy_rms_mm')
                max_p1  = max(s['p1_max_fz'] for s in seeds)
                max_tc  = max(s['contact_t'] for s in seeds)
                mean_pm = cell_mean(seeds, 'plan_ms')
                mean_fm = cell_mean(seeds, 'fm_ms')
                print(f"  {LABEL[mode]:>16}  {K:>4} {H:>5} | "
                      f"{worst_c:>13.1f}% {worst_xy:>8.2f}mm {mean_xy:>7.2f}mm "
                      f"{max_p1:>6.0f}N {max_tc:>6.2f}s {mean_pm:>7.2f} {mean_fm:>6.2f}")

# Step 3: head-to-head at MPPI baseline's valid configs
print()
print("="*92)
print("Iso-performance — for each MPPI baseline VALID config, smallest guided (K,H)")
print("with worst-seed contact% ≥ baseline worst-seed contact%")
print("="*92)
for plan in ["async","sync"]:
    print(f"\n[{plan.upper()}]")
    base_valid=[]
    for K in K_LIST:
        for H in H_LIST:
            seeds = cell_seeds("mppi", plan, K, H)
            if cell_valid(seeds):
                base_valid.append((K, H, seeds))
    if not base_valid:
        print("  MPPI baseline has NO valid cell — cannot set a bar")
        continue
    # take the one with lowest worst-xy as the bar (xy is the primary task metric)
    # contact% saturates ~75-80% so xy is the real differentiator
    base_valid.sort(key=lambda r: cell_worst_xy(r[2]))
    bK, bH, bs = base_valid[0]
    bar_c  = cell_worst_contact(bs)
    bar_xy = cell_worst_xy(bs)
    bpm = cell_mean(bs, 'plan_ms')
    print(f"  baseline bar = K={bK} H={bH}  "
          f"worst_contact={bar_c:.1f}%  worst_xy={bar_xy:.2f}mm  plan_ms={bpm:.2f}")
    for mode in ["cost","wta3","mlp"]:
        # Pareto: worst contact >= bar_c-0.5 AND worst xy <= bar_xy+0.1
        matches=[]
        for K in K_LIST:
            for H in H_LIST:
                seeds = cell_seeds(mode, plan, K, H)
                if not cell_valid(seeds): continue
                if (cell_worst_contact(seeds) >= bar_c - 0.5 and
                    cell_worst_xy(seeds)      <= bar_xy + 0.1):
                    matches.append((K, h2f[H], H, seeds))
        if not matches:
            print(f"    {LABEL[mode]:>16}: NO valid config matches both bars")
            continue
        matches.sort(key=lambda r: (r[0], r[1]))
        K,_,H,seeds = matches[0]
        worst_c = cell_worst_contact(seeds)
        worst_xy = cell_worst_xy(seeds)
        mean_pm = cell_mean(seeds, 'plan_ms')
        mean_fm = cell_mean(seeds, 'fm_ms')
        max_p1  = max(s['p1_max_fz'] for s in seeds)
        max_tc  = max(s['contact_t'] for s in seeds)
        dK = (1 - K/bK)*100
        dH = (1 - h2f[H]/h2f[bH])*100
        print(f"    {LABEL[mode]:>16}: K={K:>3} H={H}  "
              f"worst_contact={worst_c:5.1f}%  worst_xy={worst_xy:5.2f}mm  "
              f"P1_max={max_p1:5.0f}N  tC_max={max_tc:4.2f}s  "
              f"plan_ms={mean_pm:5.2f}  fm_ms={mean_fm:5.2f}  "
              f"(K↓{dK:+.0f}%  H↓{dH:+.0f}%)")
