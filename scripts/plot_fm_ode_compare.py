#!/usr/bin/env python3
"""FM cost-mode ODE × K sweep figures.

Input  : sweep_fm_ode_cost/fm_cost_ode{ODE}_K{K}_H0.10_s{SEED}.csv
         out/compare_mlp_vs_mppi/{mppi_K128_H0.10_s*.csv, mlp_K32_H0.10_sc1.0_la0.30_stepidx_s*.csv}
Output : sweep_fm_ode_cost/
           fig_fm_ode_compare.png    1x2 (perf scatter | compute line)

Metrics: phase 2 (time > 5s and hybrid >= 0.5)
  xy_rms_mm, contact_pct, plan_ms, fm_ms
"""
import csv
import glob
import math
import os
import re
import statistics as st
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

FM_DIR = "sweep_fm_ode_cost"
REF_DIR = "out/compare_mlp_vs_mppi"
OUT = FM_DIR

WARMUP = 5.0
F_THRESH = 1.0

ODE_LIST = [12, 8, 5, 3, 1]
K_LIST   = [8, 32]
K_COLOR  = {8: "#1f77b4", 32: "#d62728"}
ODE_MARK = {12: "o", 8: "D", 5: "^", 3: "s", 1: "v"}


def load_metrics(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items() if v != ""})
            except (ValueError, TypeError):
                pass
    rs = [r for r in rows
          if r.get("time", 0) > WARMUP and r.get("hybrid", 0) >= 0.5]
    if not rs:
        return None
    n = len(rs)
    fz = [r["Fz"] for r in rs]
    contact = 100.0 * sum(1 for v in fz if v > F_THRESH) / n
    xy = 1000.0 * math.sqrt(
        sum((r["ee_x"] - r["tgt_x"]) ** 2 + (r["ee_y"] - r["tgt_y"]) ** 2
            for r in rs) / n
    )
    plan_ms = st.mean(r["plan_ms"] for r in rs) if rs else 0.0
    fm_vals = [r["fm_ms"] for r in rs if r.get("fm_ms", 0) > 0]
    fm_ms = st.mean(fm_vals) if fm_vals else 0.0
    return dict(xy=xy, contact=contact, plan_ms=plan_ms, fm_ms=fm_ms)


def collect_fm():
    """returns dict[(K, ODE)] -> list of per-seed metric dicts"""
    bucket = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(FM_DIR, "fm_cost_ode*_K*_H0.10_s*.csv"))):
        m = re.search(r"ode(\d+)_K(\d+)_H[\d.]+_s(\d+)", p)
        if not m:
            continue
        ode = int(m.group(1)); K = int(m.group(2))
        mm = load_metrics(p)
        if mm:
            bucket[(K, ode)].append(mm)
    return bucket


def collect_ref(pattern):
    rs = []
    for p in sorted(glob.glob(os.path.join(REF_DIR, pattern))):
        mm = load_metrics(p)
        if mm:
            rs.append(mm)
    return rs


def agg(rs):
    if not rs:
        return None
    out = {k: st.mean(r[k] for r in rs) for k in rs[0]}
    if len(rs) > 1:
        for k in rs[0]:
            out[k + "_std"] = st.stdev(r[k] for r in rs)
    else:
        for k in rs[0]:
            out[k + "_std"] = 0.0
    return out


def fmt_label(K, ode):
    return f"K={K} ODE={ode}"


# ----------------------------------------------------------------------------
def make_compare_fig(fm, mppi_ref, mlp_ref, out_path, perf_errorbar=False):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # ---------------- (L) Performance scatter: xy vs contact ----------------
    axL = axes[0]
    # Per-point label offset overrides to avoid overlaps with other markers.
    label_offset = {
        (8, 3):  (-10, 0),    # K=8 ODE=3 — push label to the left
        (8, 1):  (-10, 4),    # K=8 ODE=1 — top-left of the point
        (32, 3): (7, -10),    # K=32 ODE=3 — bottom-right of the point
    }
    for K in K_LIST:
        for ode in ODE_LIST:
            rs = fm.get((K, ode), [])
            if not rs:
                continue
            a = agg(rs)
            if perf_errorbar:
                axL.errorbar(a["xy"], a["contact"],
                             xerr=a["xy_std"], yerr=a["contact_std"],
                             fmt="o", ms=10, mfc=K_COLOR[K],
                             mec="black", mew=0.7,
                             ecolor=K_COLOR[K], elinewidth=0.8, capsize=3)
            else:
                axL.plot(a["xy"], a["contact"],
                         marker="o", ms=10, mfc=K_COLOR[K],
                         mec="black", mew=0.7, ls="")
            off = label_offset.get((K, ode), (7, 4))
            ha = "right" if off[0] < 0 else "left"
            axL.annotate(f"ODE={ode}",
                         xy=(a["xy"], a["contact"]),
                         xytext=off, textcoords="offset points",
                         fontsize=8, color=K_COLOR[K], weight="bold",
                         ha=ha, va="center")

    # references
    if mppi_ref:
        axL.plot(mppi_ref["xy"], mppi_ref["contact"],
                 marker="*", ms=18, mfc="gold", mec="black", mew=0.8, ls="",
                 label="MPPI baseline (K=128)")
    if mlp_ref:
        axL.plot(mlp_ref["xy"], mlp_ref["contact"],
                 marker="P", ms=14, mfc="#2ca02c", mec="black", mew=0.8, ls="",
                 label="MLP+MPPI step-indexed (K=32)")

    # K legend (color)
    k_handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                            mfc=K_COLOR[k], mec="black", ms=10,
                            label=f"FM K={k}") for k in K_LIST]
    other_handles, other_labels = axL.get_legend_handles_labels()
    axL.legend(handles=k_handles + other_handles,
               labels=[h.get_label() for h in k_handles] + other_labels,
               loc="best", fontsize=8)

    axL.set_xlabel("xy_rms [mm]  (→ worse)")
    axL.set_ylabel(r"contact %  ($F_z > 1\,N$, ↑ better)")
    axL.set_title("Performance scatter — phase 2 (3 seeds, mean)")
    axL.grid(alpha=0.3)

    # ---------------- (R) Compute lines: ODE vs total time ----------------
    # total = plan_ms + fm_thread  (plan thread + FM thread CPU 합)
    axR = axes[1]
    # Per-point label offsets (K=8 line below labels, K=32 above).
    point_off = {8: (0, -14), 32: (0, 9)}
    # Per-(K, ODE) overrides to dodge axis edges / other markers.
    specific_off = {
        (32, 12): (14, 12),    # K=32 ODE=12 — top-right (avoid axis clip)
        (8, 12):  (12, -14),   # K=8 ODE=12 — push right so it doesn't clip
        (32, 1):  (0, 16),     # K=32 ODE=1 — slightly higher to clear MPPI axhline
    }
    for K in K_LIST:
        xs, tot_means, tot_stds = [], [], []
        for ode in sorted(ODE_LIST, reverse=True):  # plot 12 -> 1
            rs = fm.get((K, ode), [])
            if not rs:
                continue
            a = agg(rs)
            xs.append(ode)
            tot = a["plan_ms"] + a["fm_ms"]
            tot_means.append(tot)
            tot_stds.append(math.sqrt(a["plan_ms_std"]**2 + a["fm_ms_std"]**2))
        if not xs:
            continue
        axR.errorbar(xs, tot_means, yerr=tot_stds, marker="o", ms=8,
                     color=K_COLOR[K], lw=2.0, capsize=3,
                     label=f"FM K={K}  total (plan + fm_thread)")
        for x, y in zip(xs, tot_means):
            off = specific_off.get((K, x), point_off[K])
            ha = "left" if off[0] > 0 else ("right" if off[0] < 0 else "center")
            va = "bottom" if off[1] > 0 else "top"
            axR.annotate(f"{y:.2f} ms",
                         xy=(x, y), xytext=off,
                         textcoords="offset points",
                         ha=ha, va=va,
                         fontsize=8, color=K_COLOR[K], weight="bold")

    if mlp_ref:
        axR.axhline(mlp_ref["plan_ms"], color="#2ca02c", lw=2.0,
                    linestyle="-", label="MLP+MPPI K=32  total")
        axR.text(0.995, mlp_ref["plan_ms"], f"{mlp_ref['plan_ms']:.2f} ms ",
                 transform=axR.get_yaxis_transform(),
                 va="bottom", ha="right",
                 color="#2ca02c", fontsize=9, weight="bold")
    if mppi_ref:
        axR.axhline(mppi_ref["plan_ms"], color="gold", lw=2.0,
                    linestyle="-", label="MPPI K=128  total")
        # 그래프 오른쪽 안쪽, K=32 ODE=1 점과 겹치지 않게 약간 왼쪽으로.
        axR.text(0.88, mppi_ref["plan_ms"], f"{mppi_ref['plan_ms']:.2f} ms",
                 transform=axR.get_yaxis_transform(),
                 va="bottom", ha="right",
                 color="#b58900", fontsize=9, weight="bold")

    axR.set_xlabel("FM ODE steps")
    axR.set_ylabel("total computation time [ms]")
    axR.set_title("Per-iteration total compute  (plan_ms + fm_thread)")
    axR.grid(alpha=0.3, which="both")
    axR.set_xticks(ODE_LIST)
    axR.invert_xaxis()  # 12 left, 1 right
    axR.legend(fontsize=8, loc="best")
    axR.set_yscale("log")
    # Make sure the topmost annotation (K=32 ODE=12) fits inside the axis.
    axR.set_ylim(top=35.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    fm = collect_fm()
    mppi_ref = agg(collect_ref("mppi_K128_H0.10_s*.csv"))
    mlp_ref  = agg(collect_ref("mlp_K32_H0.10_sc1.0_la0.30_stepidx_s*.csv"))

    # (1) FM ODE × K with reference points / lines.
    make_compare_fig(fm, mppi_ref, mlp_ref,
                     os.path.join(OUT, "fig_fm_ode_compare.png"))
    # (2) FM ODE × K only — references stripped (no MPPI/MLP overlays).
    #     Show xy/contact errorbars on the performance scatter.
    make_compare_fig(fm, None, None,
                     os.path.join(OUT, "fig_fm_ode_only.png"),
                     perf_errorbar=True)

    # short text summary
    print()
    print("=== References ===")
    if mppi_ref:
        print(f"  MPPI baseline (K=128): xy={mppi_ref['xy']:.2f}mm  contact={mppi_ref['contact']:.1f}%  plan_ms={mppi_ref['plan_ms']:.2f}")
    if mlp_ref:
        print(f"  MLP+MPPI step-idx K=32: xy={mlp_ref['xy']:.2f}mm  contact={mlp_ref['contact']:.1f}%  plan_ms={mlp_ref['plan_ms']:.2f}  (mlp_ms inline {mlp_ref['fm_ms']:.2f})")


if __name__ == "__main__":
    main()
