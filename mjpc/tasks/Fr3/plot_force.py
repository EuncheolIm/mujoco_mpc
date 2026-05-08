#!/usr/bin/env python3
"""Plot force + position tracking from MJPC_FORCE_LOG csv.

Usage:
    python3 plot_force.py <csv_path> [--show] [--out <png>]

CSV columns:
    time, Fx, Fy, Fz, F_task_z, ee_x, ee_y, ee_z, tgt_x, tgt_y, tgt_z, hybrid

Two-column layout:
    Left column (force):
      Fz vs F_task_z vs F_des, Fx, Fy
    Right column (position):
      ee_x vs tgt_x, ee_y vs tgt_y, ee_z vs tgt_z

Hybrid-on instant is marked with a vertical dashed line on every subplot.
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
    ap.add_argument("--f-des", type=float, default=-10.0,
                    help="F_des.z line on the force plot (default -10)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # First time hybrid flag flips to 1 — visualize the position→force
    # phase transition on every subplot.
    hyb_on = None
    if "hybrid" in df.columns and (df["hybrid"] >= 1).any():
        hyb_on = df.loc[df["hybrid"] >= 1, "time"].iloc[0]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)

    # --- left column: force ---
    axL = axes[:, 0]
    axL[0].plot(df.time, df.Fz, label="F_sensor.z (raw, gravity biased)",
                linewidth=1.4, color="tab:blue")
    axL[0].plot(df.time, df.F_task_z, "--", label="F_task_z (intent)",
                linewidth=1.0, alpha=0.8, color="tab:orange")
    if "F_press_z" in df.columns:
        axL[0].plot(df.time, df.F_press_z, ":",
                    label="F_press_z (gravity removed)",
                    linewidth=1.4, color="tab:green")
    axL[0].axhline(args.f_des, color="r", ls=":", label=f"F_des={args.f_des}")
    axL[0].set_ylabel("Force z [N]"); axL[0].legend(loc="best")
    axL[0].set_title("Force tracking (z)")

    axL[1].plot(df.time, df.Fx, label="F_sensor.x", color="tab:blue")
    axL[1].plot(df.time, df.Fy, label="F_sensor.y", color="tab:orange")
    axL[1].set_ylabel("Force xy [N]"); axL[1].legend(loc="best")
    axL[1].set_title("Lateral force (should stay near 0)")

    axL[2].plot(df.time, df.Fz - args.f_des, label="F_err = Fz − F_des",
                color="tab:red")
    axL[2].axhline(0, color="k", ls=":", alpha=0.5)
    axL[2].set_ylabel("F error [N]"); axL[2].legend(loc="best")
    axL[2].set_title("Force tracking error")
    axL[2].set_xlabel("time [s]")

    # --- right column: position ---
    axR = axes[:, 1]
    for ax_, axis_name in zip(axR, ["x", "y", "z"]):
        ax_.plot(df.time, df[f"ee_{axis_name}"], label=f"ee.{axis_name}",
                 linewidth=1.4)
        ax_.plot(df.time, df[f"tgt_{axis_name}"], "--",
                 label=f"target.{axis_name}", linewidth=1.0, alpha=0.8)
        ax_.set_ylabel(f"{axis_name} [m]"); ax_.legend(loc="best")
        ax_.set_title(f"Position tracking ({axis_name})")
    axR[2].set_xlabel("time [s]")

    if hyb_on is not None:
        for ax_ in axes.ravel():
            ax_.axvline(hyb_on, color="g", ls="--", alpha=0.6,
                        label=f"hybrid on @ {hyb_on:.2f}s"
                        if ax_ is axes[0, 0] else None)
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
