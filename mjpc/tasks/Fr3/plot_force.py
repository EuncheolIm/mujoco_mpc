#!/usr/bin/env python3
"""Plot force + position tracking from MJPC_FORCE_LOG csv.

Usage:
    python3 plot_force.py <csv_path> [--show] [--out <png>]

CSV columns:
    time, Fx, Fy, Fz, F_task_z, ee_x, ee_y, ee_z, tgt_x, tgt_y, tgt_z, hybrid

2x2 layout:
    (0,0) Force tracking (z) — F_task_z vs F_des
    (0,1) Position tracking (xyz) — ee.x/y/z vs target.x/y/z
    (1,0) Force tracking error — F_task_z − F_des
    (1,1) xy top-down trace — ee path vs target path

Hybrid-on instant is marked with a vertical dashed line on every time-axis
subplot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="MJPC_FORCE_LOG csv path")
    ap.add_argument("--out", type=Path, default=None,
                    help="output png (default: <csv>.png)")
    ap.add_argument("--show", action="store_true",
                    help="display interactively")
    ap.add_argument("--f-des", type=float, default=None,
                    help="F_des.z line on the force plot. "
                         "If omitted, auto-detect from task.xml next to "
                         "this script (numeric 'F_des' z component).")
    args = ap.parse_args()

    # Auto-detect F_des from sibling task.xml when not given explicitly.
    if args.f_des is None:
        task_xml = Path(__file__).resolve().parent / "task.xml"
        f_des_val = -10.0
        if task_xml.exists():
            import re
            txt = task_xml.read_text()
            # Find non-commented F_des numeric and parse last float on that line.
            for ln in txt.splitlines():
                s = ln.strip()
                if s.startswith("<!--"):
                    continue
                if 'name="F_des"' in ln and "data=" in ln:
                    m = re.search(r'data="([^"]+)"', ln)
                    if m:
                        nums = m.group(1).split()
                        if len(nums) >= 3:
                            try:
                                f_des_val = float(nums[2])  # z component
                                break
                            except ValueError:
                                pass
        args.f_des = f_des_val
        print(f"auto-detected F_des.z = {args.f_des} from {task_xml}")

    df = pd.read_csv(args.csv)
    hyb_on = None
    if "hybrid" in df.columns and (df["hybrid"] >= 1).any():
        hyb_on = df.loc[df["hybrid"] >= 1, "time"].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # (0,0) Force z: F_task_z + F_des reference.
    ax = axes[0, 0]
    ax.plot(df.time, df.F_task_z, label="F_task_z (intent)",
            linewidth=1.2, color="tab:orange")
    ax.axhline(args.f_des, color="r", ls=":", label=f"F_des={args.f_des}")
    ax.set_ylabel("Force z [N]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Force tracking (z)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # (1,0) Force tracking error.
    ax = axes[1, 0]
    ax.plot(df.time, df.F_task_z - args.f_des,
            label="F_err = F_task_z − F_des", color="tab:red")
    ax.axhline(0, color="k", ls=":", alpha=0.5)
    ax.set_ylabel("F error [N]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Force tracking error")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # (0,1) Position xyz tracking — three axes overlaid.
    ax = axes[0, 1]
    colors = {"x": "tab:blue", "y": "tab:green", "z": "tab:purple"}
    for axis_name, color in colors.items():
        ax.plot(df.time, df[f"ee_{axis_name}"],
                label=f"ee.{axis_name}", linewidth=1.2, color=color)
        ax.plot(df.time, df[f"tgt_{axis_name}"], "--",
                label=f"tgt.{axis_name}", linewidth=1.0, alpha=0.7, color=color)
    ax.set_ylabel("Position [m]")
    ax.set_xlabel("Time [s]")
    ax.set_title("Position tracking (xyz)")
    ax.legend(loc="best", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Force distribution (post-stabilize). Histogram of F_task_z
    # restricted to the steady-state window so the descent transient
    # doesn't dominate the bins. mean ± std overlaid.
    ax = axes[1, 1]
    ss_t = 3.0  # steady-state window start [s]
    ss = df[df.time >= ss_t]
    if len(ss) > 1:
        vals = ss.F_task_z.to_numpy()
        mean = float(vals.mean())
        std  = float(vals.std(ddof=1))
        ax.hist(vals, bins=40, color="tab:orange", alpha=0.85,
                edgecolor="black", linewidth=0.4)
        ax.axvline(args.f_des, color="r", ls=":", linewidth=1.6,
                   label=f"F_des={args.f_des}")
        ax.axvline(mean, color="tab:blue", ls="-", linewidth=1.4,
                   label=f"mean={mean:+.2f}")
        ax.axvspan(mean - std, mean + std, color="tab:blue", alpha=0.12,
                   label=f"±1σ ({std:.2f})")
        ax.set_xlabel("F_task_z [N]")
        ax.set_ylabel("count")
        ax.set_title(f"Force distribution (t ≥ {ss_t:.0f} s)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no steady-state data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Force distribution")

    # Hybrid-on vertical line on time-axis plots only.
    if hyb_on is not None:
        for ax in (axes[0, 0], axes[1, 0], axes[0, 1]):
            ax.axvline(hyb_on, color="g", ls="--", alpha=0.6,
                       label=f"hybrid on @ {hyb_on:.2f}s")
        # add the legend entry only on the top-left
        axes[0, 0].legend(loc="best")

    fig.suptitle(f"Force + Position tracking — {args.csv.name}",
                 fontsize=12)
    fig.tight_layout()

    out = args.out or args.csv.with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
