#!/usr/bin/env python3
"""Plot MLP+MPPI best vs MPPI baseline 비교 figures.

Input  : out/compare_mlp_vs_mppi/{mppi,mlp}_*.csv  (record_mlp_vs_mppi.sh 산출물)
Output : out/compare_mlp_vs_mppi/
           fig_xy_2d.png    EE 궤적 2D (target wipe-circle 위에 시드별 overlay)
           fig_contact.png  Fz 타임라인(seed1) + contact% bar(mean±std)
           fig_timing.png   plan_ms / fm_ms stacked bar (mean±std)
         + 콘솔에 per-seed 요약 테이블 출력

Phase 필터: time > 5.0 AND hybrid >= 0.5 (analyze 스크립트와 동일)
Contact 정의: Fz > 1.0 N

Usage:
  python3 scripts/plot_mlp_vs_mppi.py [ROOT_DIR]
"""
import csv
import glob
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "out/compare_mlp_vs_mppi"
OUT = ROOT

WARMUP = 5.0
F_THRESH = 1.0

MODES = {
    "mppi": {
        "label": "MPPI baseline (K=128, H=0.10)",
        "short": "MPPI\n(K=128)",
        "color": "#1f77b4",
        "pattern": "mppi_K128_H0.10_s*.csv",
    },
    "mlp": {
        "label": "MLP+MPPI step-indexed (K=32, scale=1.0, la=0.30)",
        "short": "MLP+MPPI\n(K=32)",
        "color": "#d62728",
        "pattern": "mlp_K32_H0.10_sc1.0_la0.30_stepidx_s*.csv",
    },
}


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items() if v != ""})
            except (ValueError, TypeError):
                pass
    return rows


def filter_phase2(rows):
    return [r for r in rows
            if r.get("time", 0) > WARMUP and r.get("hybrid", 0) >= 0.5]


def aggregate(rs):
    n = len(rs)
    if n == 0:
        return dict(contact_pct=float("nan"), xy_rms_mm=float("nan"),
                    plan_ms=0.0, fm_ms=0.0, n=0)
    fz = [r["Fz"] for r in rs]
    contact_pct = 100.0 * sum(1 for v in fz if v > F_THRESH) / n
    xy_rms_mm = 1000.0 * math.sqrt(
        sum((r["ee_x"] - r["tgt_x"]) ** 2 + (r["ee_y"] - r["tgt_y"]) ** 2
            for r in rs) / n
    )
    plan_ms = float(np.mean([r["plan_ms"] for r in rs])) if any("plan_ms" in r for r in rs) else 0.0
    fm_vals = [r["fm_ms"] for r in rs if r.get("fm_ms", 0) > 0]
    fm_ms = float(np.mean(fm_vals)) if fm_vals else 0.0
    return dict(contact_pct=contact_pct, xy_rms_mm=xy_rms_mm,
                plan_ms=plan_ms, fm_ms=fm_ms, n=n)


def load_all():
    data = {}
    for mode, conf in MODES.items():
        paths = sorted(glob.glob(os.path.join(ROOT, conf["pattern"])))
        entries = []
        for p in paths:
            rows = load_csv(p)
            if not rows:
                continue
            rs = filter_phase2(rows)
            agg = aggregate(rs)
            # extract seed from filename
            stem = os.path.basename(p).rsplit(".csv", 1)[0]
            seed = int(stem.rsplit("_s", 1)[1])
            entries.append({"path": p, "seed": seed,
                            "rows": rows, "rs": rs, "agg": agg})
        if entries:
            data[mode] = entries
        else:
            print(f"[warn] no csv matched for {mode} ({conf['pattern']})")
    return data


# ============================================================================
def fig_xy_2d(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
    for ax, (mode, conf) in zip(axes, MODES.items()):
        entries = data.get(mode, [])
        if not entries:
            ax.set_title(f"{conf['label']}\n(no data)")
            continue
        # target wipe-circle: take seed=1 (same trajectory across replicates)
        e1 = next((e for e in entries if e["seed"] == 1), entries[0])
        rs1 = e1["rs"]
        tgt_x = np.array([r["tgt_x"] for r in rs1]) * 1000
        tgt_y = np.array([r["tgt_y"] for r in rs1]) * 1000
        ax.plot(tgt_x, tgt_y, color="0.4", lw=1.6, linestyle="--",
                label="target (mocap)")
        # EE per seed
        for e in entries:
            ex = np.array([r["ee_x"] for r in e["rs"]]) * 1000
            ey = np.array([r["ee_y"] for r in e["rs"]]) * 1000
            lbl = f"EE (s={e['seed']})" if e["seed"] == 1 else None
            ax.plot(ex, ey, color=conf["color"], lw=1.0, alpha=0.75, label=lbl)
        xy_means = [e["agg"]["xy_rms_mm"] for e in entries]
        ax.set_title(f"{conf['label']}\n"
                     f"xy_rms = {np.mean(xy_means):.2f} ± {np.std(xy_means):.2f} mm "
                     f"(n={len(entries)} seeds)")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("EE xy trajectory  (phase 2: t > 5 s, hybrid = 1)")
    fig.tight_layout()
    out = os.path.join(OUT, "fig_xy_2d.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def fig_contact(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={"width_ratios": [2, 1]})
    # (L) Fz timeline (seed=1)
    axL = axes[0]
    for mode, conf in MODES.items():
        entries = data.get(mode, [])
        if not entries:
            continue
        e1 = next((e for e in entries if e["seed"] == 1), entries[0])
        t = np.array([r["time"] for r in e1["rs"]])
        fz = np.array([r["Fz"] for r in e1["rs"]])
        axL.plot(t, fz, color=conf["color"], lw=1.0, label=conf["label"])
    axL.axhline(F_THRESH, color="k", lw=0.6, linestyle=":",
                label=f"contact threshold {F_THRESH:.0f} N")
    axL.set_xlabel("time [s]")
    axL.set_ylabel(r"$F_z$ [N]")
    axL.set_title("Contact force timeline (seed 1)")
    axL.grid(alpha=0.3)
    axL.legend(fontsize=8)

    # (R) contact% bar (mean ± std over seeds)
    axR = axes[1]
    xs, means, stds, colors, labels = [], [], [], [], []
    for i, (mode, conf) in enumerate(MODES.items()):
        entries = data.get(mode, [])
        if not entries:
            continue
        cs = [e["agg"]["contact_pct"] for e in entries]
        xs.append(i)
        means.append(float(np.mean(cs)))
        stds.append(float(np.std(cs)))
        colors.append(conf["color"])
        labels.append(conf["short"])
    axR.bar(xs, means, yerr=stds, color=colors, capsize=5,
            edgecolor="black", linewidth=0.5)
    for x, m in zip(xs, means):
        axR.text(x, m + 1.5, f"{m:.1f}%", ha="center", fontsize=10)
    axR.set_xticks(xs)
    axR.set_xticklabels(labels)
    axR.set_ylabel("contact %  ($F_z > 1\\,N$)")
    axR.set_title("Contact rate (mean ± std, n seeds)")
    axR.set_ylim(0, max(100, max(means) * 1.15 if means else 100))
    axR.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = os.path.join(OUT, "fig_contact.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def fig_timing(data):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    xs, labels = [], []
    plan_m, plan_s, fm_m, fm_s = [], [], [], []
    for mode, conf in MODES.items():
        entries = data.get(mode, [])
        if not entries:
            continue
        pm = [e["agg"]["plan_ms"] for e in entries]
        fm = [e["agg"]["fm_ms"]   for e in entries]
        xs.append(len(xs))
        labels.append(conf["short"])
        plan_m.append(float(np.mean(pm))); plan_s.append(float(np.std(pm)))
        fm_m.append(float(np.mean(fm)));   fm_s.append(float(np.std(fm)))

    # MLP는 동기 호출이라 fm_ms 가 plan_ms 안에 이미 포함됨.
    # 따라서 막대 높이 = plan_ms (= total), 그 안을
    #   (plan_ms − fm_ms) = MPPI rollout/cost 순수
    #   fm_ms              = MLP inference (inline)
    # 로 분리한다. MPPI baseline 은 fm_ms=0 이라 단일 segment.
    mppi_pure_m = [max(p - m, 0.0) for p, m in zip(plan_m, fm_m)]
    totals = plan_m  # plan_ms 자체가 실제 wall-clock
    tot_err = plan_s

    ax.bar(xs, mppi_pure_m, color="#4c72b0", edgecolor="black", linewidth=0.5,
           label="MPPI rollout/cost  (plan_ms − mlp_ms)")
    ax.bar(xs, fm_m, bottom=mppi_pure_m, color="#dd8452", edgecolor="black",
           linewidth=0.5, label="MLP inference  (mlp_ms)")
    ax.errorbar(xs, totals, yerr=tot_err, fmt="none", ecolor="black", capsize=5)

    bar_w = 0.8  # matplotlib default
    for x, mp, m, tot in zip(xs, mppi_pure_m, fm_m, totals):
        # MPPI pure segment label.
        if mp >= 0.25:
            ax.text(x, mp / 2, f"{mp:.2f}", ha="center", va="center",
                    fontsize=9, color="white", weight="bold")
        elif mp > 0:
            ax.annotate(f"{mp:.2f}", xy=(x, mp / 2),
                        xytext=(x + bar_w * 0.55, mp / 2),
                        fontsize=9, color="black", weight="bold",
                        va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.6))
        # MLP segment label.
        if m >= 0.25:
            ax.text(x, mp + m / 2, f"{m:.2f}", ha="center", va="center",
                    fontsize=9, color="white", weight="bold")
        elif m > 0:
            ax.annotate(f"{m:.2f}", xy=(x, mp + m / 2),
                        xytext=(x + bar_w * 0.55, mp + m / 2),
                        fontsize=9, color="black", weight="bold",
                        va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.6))
        ax.text(x, tot + max(totals) * 0.04,
                f"total\n{tot:.2f} ms", ha="center", fontsize=10, weight="bold")

    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("computation time [ms]")
    ax.set_title("Computation time per planning iteration")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(totals) * 1.35 if totals else 1.0)

    fig.tight_layout()
    out = os.path.join(OUT, "fig_timing.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def fig_perf_compute(data):
    """1x3: (L) xy 2D overlay  |  (M) contact% bar  |  (R) computation stacked bar."""
    fig, axes = plt.subplots(
        1, 3, figsize=(16.0, 5.2),
        gridspec_kw={"width_ratios": [1.6, 1.0, 1.3]})

    # ---- Left: xy 2D trajectory overlay (both modes on one axis) ----
    axXY = axes[0]
    # Target circle from any seed (same across replicates within a mode; identical mocap across modes).
    drew_target = False
    for mode, conf in MODES.items():
        entries = data.get(mode, [])
        if not entries:
            continue
        if not drew_target:
            rs1 = next((e for e in entries if e["seed"] == 1), entries[0])["rs"]
            tgt_x = np.array([r["tgt_x"] for r in rs1]) * 1000
            tgt_y = np.array([r["tgt_y"] for r in rs1]) * 1000
            axXY.plot(tgt_x, tgt_y, color="0.25", lw=5.0, linestyle="--",
                      label="target (mocap)")
            drew_target = True
        # All seeds overlaid, first one carries the legend label.
        for k, e in enumerate(entries):
            ex = np.array([r["ee_x"] for r in e["rs"]]) * 1000
            ey = np.array([r["ee_y"] for r in e["rs"]]) * 1000
            axXY.plot(ex, ey, color=conf["color"], lw=1.0, alpha=0.7,
                      label=conf["label"] if k == 0 else None)
    axXY.set_xlabel("x [mm]")
    axXY.set_ylabel("y [mm]")
    axXY.set_aspect("equal")
    axXY.set_title("EE xy trajectory  (phase 2)")
    axXY.grid(alpha=0.3)
    axXY.legend(loc="best", fontsize=8)

    # ---- Middle: contact rate bar (mean ± std) ----
    axL = axes[1]
    xs, means, stds, colors, labels = [], [], [], [], []
    for i, (mode, conf) in enumerate(MODES.items()):
        entries = data.get(mode, [])
        if not entries:
            continue
        cs = [e["agg"]["contact_pct"] for e in entries]
        xs.append(i)
        means.append(float(np.mean(cs)))
        stds.append(float(np.std(cs)))
        colors.append(conf["color"])
        labels.append(conf["short"])
    axL.bar(xs, means, yerr=stds, color=colors, capsize=5,
            edgecolor="black", linewidth=0.5)
    for x, m in zip(xs, means):
        axL.text(x, m + 1.5, f"{m:.1f}%", ha="center", fontsize=10)
    axL.set_xticks(xs); axL.set_xticklabels(labels)
    axL.set_ylabel("contact %  ($F_z > 1\\,N$)")
    axL.set_title("Contact rate")
    axL.set_ylim(0, max(100, max(means) * 1.15 if means else 100))
    axL.grid(alpha=0.3, axis="y")

    # ---- Right: computation-time stacked bar (plan_ms = MPPI + MLP) ----
    axR = axes[2]
    xs2, labels2 = [], []
    plan_m, plan_s, fm_m, fm_s = [], [], [], []
    for mode, conf in MODES.items():
        entries = data.get(mode, [])
        if not entries:
            continue
        pm = [e["agg"]["plan_ms"] for e in entries]
        fm = [e["agg"]["fm_ms"]   for e in entries]
        xs2.append(len(xs2))
        labels2.append(conf["short"])
        plan_m.append(float(np.mean(pm))); plan_s.append(float(np.std(pm)))
        fm_m.append(float(np.mean(fm)));   fm_s.append(float(np.std(fm)))

    mppi_pure_m = [max(p - m, 0.0) for p, m in zip(plan_m, fm_m)]
    totals = plan_m
    tot_err = plan_s

    axR.bar(xs2, mppi_pure_m, color="#4c72b0", edgecolor="black",
            linewidth=0.5, label="MPPI rollout/cost  (plan_ms − mlp_ms)")
    axR.bar(xs2, fm_m, bottom=mppi_pure_m, color="#dd8452",
            edgecolor="black", linewidth=0.5,
            label="MLP inference  (mlp_ms)")
    axR.errorbar(xs2, totals, yerr=tot_err, fmt="none",
                 ecolor="black", capsize=5)

    bar_w = 0.8
    for x, mp, m, tot in zip(xs2, mppi_pure_m, fm_m, totals):
        if mp >= 0.25:
            axR.text(x, mp / 2, f"{mp:.2f}", ha="center", va="center",
                     fontsize=9, color="white", weight="bold")
        elif mp > 0:
            axR.annotate(f"{mp:.2f}", xy=(x, mp / 2),
                         xytext=(x + bar_w * 0.55, mp / 2),
                         fontsize=9, color="black", weight="bold",
                         va="center", ha="left",
                         arrowprops=dict(arrowstyle="-", color="black", lw=0.6))
        if m >= 0.25:
            axR.text(x, mp + m / 2, f"{m:.2f}", ha="center", va="center",
                     fontsize=9, color="white", weight="bold")
        elif m > 0:
            axR.annotate(f"{m:.2f}", xy=(x, mp + m / 2),
                         xytext=(x + bar_w * 0.55, mp + m / 2),
                         fontsize=9, color="black", weight="bold",
                         va="center", ha="left",
                         arrowprops=dict(arrowstyle="-", color="black", lw=0.6))
        axR.text(x, tot + max(totals) * 0.04,
                 f"total\n{tot:.2f} ms", ha="center", fontsize=10, weight="bold")

    axR.set_xticks(xs2); axR.set_xticklabels(labels2)
    axR.set_ylabel("computation time [ms]")
    axR.set_title("Computation time per planning iteration")
    axR.legend(loc="upper right", fontsize=8)
    axR.grid(alpha=0.3, axis="y")
    axR.set_ylim(0, max(totals) * 1.35 if totals else 1.0)

    fig.tight_layout()
    out = os.path.join(OUT, "fig_perf_compute.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def print_summary(data):
    print()
    print(f"{'mode':<6} {'seed':>4}  {'contact%':>8}  {'xy_mm':>6}  "
          f"{'plan_ms':>7}  {'fm_ms':>5}  {'total':>5}  {'n':>5}")
    print("-" * 60)
    for mode, conf in MODES.items():
        for e in data.get(mode, []):
            a = e["agg"]
            print(f"{mode:<6} {e['seed']:>4}  {a['contact_pct']:>7.1f}%  "
                  f"{a['xy_rms_mm']:>6.2f}  {a['plan_ms']:>7.2f}  "
                  f"{a['fm_ms']:>5.2f}  {a['plan_ms']+a['fm_ms']:>5.2f}  "
                  f"{a['n']:>5d}")
    print()
    print("mean ± std over seeds:")
    for mode, conf in MODES.items():
        entries = data.get(mode, [])
        if not entries:
            continue
        cs = [e["agg"]["contact_pct"] for e in entries]
        xy = [e["agg"]["xy_rms_mm"]   for e in entries]
        pm = [e["agg"]["plan_ms"]     for e in entries]
        fm = [e["agg"]["fm_ms"]       for e in entries]
        tot = [p + m for p, m in zip(pm, fm)]
        print(f"  {conf['label']}")
        print(f"    contact = {np.mean(cs):.1f} ± {np.std(cs):.1f} %")
        print(f"    xy_rms  = {np.mean(xy):.2f} ± {np.std(xy):.2f} mm")
        print(f"    plan_ms = {np.mean(pm):.2f} ± {np.std(pm):.2f}")
        print(f"    fm_ms   = {np.mean(fm):.2f} ± {np.std(fm):.2f}")
        print(f"    total   = {np.mean(tot):.2f} ± {np.std(tot):.2f} ms")


def main():
    if not os.path.isdir(ROOT):
        print(f"ERROR: {ROOT} not found", file=sys.stderr)
        sys.exit(1)
    data = load_all()
    if not data:
        print("ERROR: no CSV loaded", file=sys.stderr)
        sys.exit(1)
    fig_xy_2d(data)
    fig_contact(data)
    fig_timing(data)
    fig_perf_compute(data)
    print_summary(data)


if __name__ == "__main__":
    main()
