#!/usr/bin/env python3
"""Generate Korean SUMMARY.md for the auto MLP-cost pipeline.

Usage: write_mlp_cost_summary.py <ROOT> [picked_scales_str]
"""
import datetime
import importlib.util
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "auto_mlp_cost_run"
PICKED = sys.argv[2] if len(sys.argv) > 2 else ""

spec = importlib.util.spec_from_file_location(
    "a", "scripts/analyze_mlp_cost_tuning.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

scale_dir = os.path.join(ROOT, "01_scale")
timing_dir = os.path.join(ROOT, "02_timing")
dirs = [d for d in (scale_dir, timing_dir) if os.path.isdir(d)]
bucket, meta = mod.collect(dirs)
rows = mod.aggregate(bucket)


def fmt(r):
    md = meta[r["key"]]
    la = "—" if md["la"] is None else f"{md['la']:.2f}"
    ci = "—" if md["ci"] is None else f"{md['ci']}"
    return (f"K={md['K']:>3d} H={md['H']:.2f} scale={md['scale']:>3.1f} "
            f"la={la:>4} ci={ci:>2} | seeds={r['n_seeds']} "
            f"xy=**{r['xy_rms_mm']:.2f}mm** contact={r['contact_pct']:.1f}% "
            f"Fz_p95={r['fz_p95']:.1f} peak={r['fz_peak']:.1f} "
            f"plan={r['plan_ms']:.2f}ms fm={r['fm_ms']:.2f}ms "
            f"**total={r['total_ms']:.2f}ms**")


def subset(K=None, H=None, min_contact=None, kind=None):
    out = rows
    if K is not None:
        out = [r for r in out if meta[r["key"]]["K"] == K]
    if H is not None:
        out = [r for r in out if abs(meta[r["key"]]["H"] - H) < 1e-6]
    if min_contact is not None:
        out = [r for r in out if r["contact_pct"] >= min_contact]
    if kind == "scale":
        out = [r for r in out if meta[r["key"]]["la"] is None]
    elif kind == "timing":
        out = [r for r in out if meta[r["key"]]["la"] is not None]
    return out


n_scale = len(subset(kind="scale"))
n_timing = len(subset(kind="timing"))
total_runs = sum(r["n_seeds"] for r in rows)

hits = [r for r in rows if mod.passes_highlight(r)]
hits_sorted = sorted(hits, key=lambda r: (r["total_ms"], r["xy_rms_mm"]))

print("# MLP-cost 자동 sweep 결과 요약")
print()
print(f"- 작성 시각: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"- 결과 디렉토리: `{ROOT}/`")
print(f"- Git HEAD: `{os.popen('git rev-parse --short HEAD').read().strip() or 'n/a'}`")
print()
print("## 목적")
print()
print("MLP guide + FlowMPPI cost 모드(WTA 미사용)가 적은 rollout 수(K=32 또는 K=64)로")
print("MPPI K=128 베이스라인(xy ≈ 1.7 mm, contact ≈ 75–76%, plan ≈ 2.7 ms)에")
print("근접하면서 총 wall-clock(plan_ms + fm_ms)을 더 낮출 수 있는지 검증.")
print()
print("→ Highlight 통과 기준: **xy ≤ 2.0 mm AND contact ≥ 76% AND total_ms < 2.2 ms**")
print()
print("## 무엇을 했나")
print()
print("1. **Scale sweep** (`sweep_mlp_cost_scale.sh`) — K∈{16,32,64} × H∈{0.05,0.10}")
print("   × scale∈{0.5,1.0,1.5,2.0,3.0} × 3 seed.")
print("2. **Top scale 선택** — stage 1 결과에서 contact≥75 & H=0.10 기준 xy 최저")
print(f"   상위 2개 scale 자동 추출 → 선택된 scale = `{PICKED or '(미기록)'}`.")
print("3. **Timing sweep** (`sweep_mlp_cost_timing.sh`) — K∈{32,64} × H=0.10")
print("   × 선택 scale × lookahead∈{0.12,0.18,0.24,0.30}")
print("   × chunk_idx∈{6,9,12} × 3 seed.")
print("   (lookahead·chunk_idx는 MLP가 출력한 chunk trajectory 중 어떤 시점을")
print("   `q_fm_target`으로 쓸지 결정하는 controller-side 파라미터. MLP는 one-shot.)")
print("4. **분석** — `analyze_mlp_cost_tuning.py`로 두 디렉토리 통합 집계.")
print()
print(f"수집된 config(시드 평균 후): scale={n_scale}, timing={n_timing}, 총 run={total_runs}")
print()
print("## 핵심 결과")
print()
print(f"### Highlight 통과 ({len(hits)}개) — total_ms 오름차순")
print()
if hits_sorted:
    for r in hits_sorted[:15]:
        print(f"- {fmt(r)}")
    if len(hits_sorted) > 15:
        print(f"- … (외 {len(hits_sorted) - 15}개)")
else:
    print("- (Highlight 통과 config 없음)")
print()
print("### 전체 contact ≥ 75% 중 xy 최저 TOP 10")
print()
top = sorted(subset(min_contact=75), key=lambda r: r["xy_rms_mm"])[:10]
for r in top:
    print(f"- {fmt(r)}")
if not top:
    print("- (해당 config 없음)")
print()
print("### K=32, H=0.10 (xy 오름차순 TOP 10)")
print()
k32 = sorted(subset(K=32, H=0.10), key=lambda r: r["xy_rms_mm"])[:10]
for r in k32:
    print(f"- {fmt(r)}")
if not k32:
    print("- (해당 config 없음)")
print()
print("### K=64, H=0.10 (xy 오름차순 TOP 10)")
print()
k64 = sorted(subset(K=64, H=0.10), key=lambda r: r["xy_rms_mm"])[:10]
for r in k64:
    print(f"- {fmt(r)}")
if not k64:
    print("- (해당 config 없음)")
print()
print("## 참고 베이스라인 (사용자 제공치, 본 sweep 외부)")
print()
print("| 모드 | K | H | xy | contact | plan_ms | fm_ms |")
print("|---|---|---|---|---|---|---|")
print("| MPPI baseline | 128 | 0.10 | ~1.7 mm | ~75–76% | ~2.7 ms | — |")
print("| MLP-cost (이전) | 32 | 0.10 | ~2.7 mm | ~78% | ~1.36 ms | ~0.19 ms |")
print("| MLP-cost (이전) | 64 | 0.10 | ~2.0 mm | ~78–79% | ~1.7 ms | ~0.19 ms |")
print()
print("## 파일")
print()
print(f"- `{ROOT}/01_scale/` — stage 1 CSV (scale sweep)")
print(f"- `{ROOT}/02_timing/` — stage 2 CSV (timing sweep)")
print(f"- `{ROOT}/03_analyze_scale.txt` — scale sweep 분석 raw")
print(f"- `{ROOT}/04_analyze_timing.txt` — timing sweep 분석 raw")
print(f"- `{ROOT}/05_analyze_combined.txt` — 두 sweep 합본 분석 raw")
print(f"- `{ROOT}/00_run.log` — 파이프라인 stdout/stderr 전체 로그")
print(f"- `{ROOT}/pipeline.pid` — 백그라운드 프로세스 PID")
print()
print("## 재현 명령")
print()
print("```bash")
print("# 분석만 다시")
print(f"python3 scripts/analyze_mlp_cost_tuning.py {ROOT}/01_scale {ROOT}/02_timing")
print()
print("# 특정 highlight config 단발 재실행 예시 (env override)")
print("env MJPC_PLANNER=9 MJPC_FM_MODE=cost MJPC_GUIDE_TYPE=mlp \\")
print("    MJPC_MLP_CKPT=$HOME/tmp/flow-matching-robot-control/checkpoints/student_mlp_v26/student.onnx \\")
print("    MJPC_MLP_STATS=$HOME/tmp/flow-matching-robot-control/checkpoints/student_mlp_v26/normalization_stats.npz \\")
print("    MJPC_FM_TRACK_SCALE=<scale> MJPC_FM_LOOKAHEAD=<la> MJPC_FM_CHUNK_IDX=<ci> \\")
print("    MJPC_HORIZON=0.10 MJPC_TRAJECTORIES=<K> \\")
print("    MJPC_AUTORUN=1 MJPC_FORCE_LOG=/tmp/check.csv \\")
print("  timeout --signal=TERM 30 ./build/bin/mjpc")
print("```")
