#!/usr/bin/env python3
"""Phase 1 (hybrid==0) EE velocity: FM-only vs WTA #1.

FM-only: K, H 무관 (rollout 사용 안 함) → sweep_TH_fmonly_p1/T16_H0.10_s1.csv 1개를 baseline 으로 사용.
WTA #1: planner=9, fm_frac=1.0, K×H 영향 → K=16,32 × H=0.10,0.20,0.30 seed=1.

Phase 1 정의: CSV `hybrid` 컬럼 < 0.5 인 모든 행.
"""
import csv, math, os, statistics as st

def load_phase1(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try: rows.append({k: float(v) for k, v in r.items() if v != ''})
            except ValueError: pass
    rows.sort(key=lambda r: r['time'])
    # drop init garbage row (t≈0 with ee_z≈0)
    rows = [r for r in rows if r['time'] > 0.005 and r['ee_z'] > 0.05]
    return [r for r in rows if r['hybrid'] < 0.5]

def phase1_stats(rows):
    if len(rows) < 2: return None
    vz_list, vxy_list, v3d_list = [], [], []
    for i in range(1, len(rows)):
        dt = rows[i]['time'] - rows[i-1]['time']
        if dt <= 1e-5: continue
        dx = rows[i]['ee_x'] - rows[i-1]['ee_x']
        dy = rows[i]['ee_y'] - rows[i-1]['ee_y']
        dz = rows[i]['ee_z'] - rows[i-1]['ee_z']
        vz_list.append(dz / dt)
        vxy_list.append(math.hypot(dx, dy) / dt)
        v3d_list.append(math.sqrt(dx*dx + dy*dy + dz*dz) / dt)
    return dict(
        t_p1=rows[-1]['time'] - rows[0]['time'],
        z_start=rows[0]['ee_z'], z_end=rows[-1]['ee_z'],
        vz_mean=st.mean(vz_list), vz_min=min(vz_list),
        vxy_mean=st.mean(vxy_list), vxy_max=max(vxy_list),
        v3d_mean=st.mean(v3d_list), v3d_max=max(v3d_list),
    )

print("Phase 1 (hybrid==0) EE 속도 비교 — FM-only vs WTA #1")
print("FM-only: K,H 무관 → baseline 1 run.  WTA #1: K=16,32 × H=0.10,0.20,0.30 (s=1)")
print()
hdr = (f"{'mode':>10} {'K':>4} {'H':>5}  {'t_p1':>5}  {'z0→z1':>14}  "
       f"{'vz_mean':>8}  {'vz_min':>8}  {'vxy_mean':>9}  {'vxy_max':>8}  "
       f"{'v3d_mean':>9}  {'v3d_max':>8}")
print(hdr); print("-"*len(hdr))

# FM-only baseline
fm_path = "sweep_TH_fmonly_p1/T16_H0.10_s1.csv"
m = phase1_stats(load_phase1(fm_path))
print(f"{'FM-only':>10} {'-':>4} {'-':>5}  {m['t_p1']:>5.2f}  "
      f"{m['z_start']:.3f}→{m['z_end']:.3f}  "
      f"{m['vz_mean']:>8.3f}  {m['vz_min']:>8.3f}  "
      f"{m['vxy_mean']:>9.3f}  {m['vxy_max']:>8.3f}  "
      f"{m['v3d_mean']:>9.3f}  {m['v3d_max']:>8.3f}")
print()

# WTA #1 grid
for H in ["0.10", "0.20", "0.30"]:
    for K in [16, 32]:
        p = f"sweep_flowmppi_4modes_3seeds/wta1_T{K}_H{H}_s1.csv"
        if not os.path.exists(p):
            print(f"{'WTA #1':>10} {K:>4} {H:>5}  (missing)"); continue
        m = phase1_stats(load_phase1(p))
        if m is None:
            print(f"{'WTA #1':>10} {K:>4} {H:>5}  (no phase1)"); continue
        # Truncate t_p1 display if WTA collided (never reached phase 2 → huge t_p1)
        t_disp = f"{m['t_p1']:5.2f}"
        print(f"{'WTA #1':>10} {K:>4} {H:>5}  {t_disp}  "
              f"{m['z_start']:.3f}→{m['z_end']:.3f}  "
              f"{m['vz_mean']:>8.3f}  {m['vz_min']:>8.3f}  "
              f"{m['vxy_mean']:>9.3f}  {m['vxy_max']:>8.3f}  "
              f"{m['v3d_mean']:>9.3f}  {m['v3d_max']:>8.3f}")
    print()
