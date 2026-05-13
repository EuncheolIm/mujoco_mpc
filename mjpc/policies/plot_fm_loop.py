#!/usr/bin/env python3
"""Plot output of fm_closed_loop_test.

Usage:
    python3 plot_fm_loop.py <csv> [--out <png>] [--wipe-start 5.0]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wipe-start", type=float, default=5.0)
    args = ap.parse_args()

    try:
        font_manager.fontManager.addfont(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        mpl.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    df = pd.read_csv(args.csv)
    t = df["time"].to_numpy()
    mask = t >= args.wipe_start
    ee = df[["ee_x", "ee_y", "ee_z"]].to_numpy()
    tg = df[["tgt_x", "tgt_y", "tgt_z"]].to_numpy()
    F  = df[["Fx", "Fy", "Fz"]].to_numpy()

    err_xy = np.linalg.norm(ee[mask, :2] - tg[mask, :2], axis=1)
    err_z  = ee[mask, 2] - tg[mask, 2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # xy top-down
    ax = axes[0, 0]
    ax.plot(tg[:, 0], tg[:, 1], "r--", label="target", lw=1.4)
    ax.plot(ee[:, 0], ee[:, 1], "b-", label="EE", lw=1.0)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Top-down xy trajectory"); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3); ax.legend()

    # xyz over time
    ax = axes[0, 1]
    colors = {"x": "C0", "y": "C1", "z": "C2"}
    for i, n in enumerate(["x", "y", "z"]):
        ax.plot(t, tg[:, i], "--", color=colors[n], label=f"tgt.{n}", lw=1.0,
                alpha=0.7)
        ax.plot(t, ee[:, i], "-", color=colors[n], label=f"ee.{n}",  lw=1.0)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Position [m]")
    ax.set_title("Position vs time"); ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)

    # forces
    ax = axes[1, 0]
    ax.plot(t, F[:, 0], color="C0", label="Fx", lw=0.9)
    ax.plot(t, F[:, 1], color="C1", label="Fy", lw=0.9)
    ax.plot(t, F[:, 2], color="C2", label="Fz", lw=1.2)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Force [N]")
    ax.set_title("EE force (ee_force sensor)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # tracking error
    ax = axes[1, 1]
    if mask.any():
        ax.plot(t[mask], err_xy * 1000.0, color="tab:red", lw=1.0,
                label="xy err [mm]")
        ax.plot(t[mask], err_z  * 1000.0, color="tab:purple", lw=1.0,
                label="z err [mm]")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Error [mm]")
    ax.set_title(f"Tracking error (t ≥ {args.wipe_start}s)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # stats annotation
    txt = []
    if mask.any():
        txt.append(f"pos xy mean={1000*err_xy.mean():.1f}mm  "
                   f"max={1000*err_xy.max():.1f}mm")
        txt.append(f"z err  mean={1000*err_z.mean():+.1f}mm  "
                   f"max={1000*np.abs(err_z).max():.1f}mm")
        txt.append(f"Fz   mean={F[mask, 2].mean():+.2f}N  "
                   f"min={F[mask, 2].min():+.2f}N  "
                   f"max={F[mask, 2].max():+.2f}N")
    fig.text(0.01, 0.005, "\n".join(txt), fontsize=9,
             family="monospace", va="bottom")

    fig.suptitle(f"FM closed-loop (in MJPC) — {args.csv.name}", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out = args.out or args.csv.with_suffix(".png")
    fig.savefig(out, dpi=140)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
