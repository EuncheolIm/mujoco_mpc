#!/usr/bin/env python3
"""3 FlowMPPI modes (warm-start #2, #3, FM as cost).
Row 1: xy 2D wipe trajectory (3 panels)
Row 2: bar chart — contact %, peak Fz, mean |Fz| during contact (3 modes side-by-side)
"""
import csv, math, statistics as st
import matplotlib.pyplot as plt

DATA = "out/videos/4modes_K128_H0.10"
MODES = ["wta2", "wta3", "cost"]
TITLES = {
    "wta2": "Warm start #2 (shared softmax, half-half)",
    "wta3": "Warm start #3 (per-group, half-half)",
    "cost": "FM as cost",
}
COLORS = {"wta2": "#ff7f0e", "wta3": "#2ca02c", "cost": "#1f77b4"}

NEEDED = ("time","Fz","ee_x","ee_y","tgt_x","tgt_y","hybrid")
F_THRESH = 1.0
WIPE_T = 5.0

def load(path):
    out = {k: [] for k in NEEDED}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                vals = {k: float(r[k]) for k in NEEDED}
            except (ValueError, KeyError):
                continue
            for k in NEEDED: out[k].append(vals[k])
    return out

def metrics(d):
    """Compute wipe-phase metrics: contact%, peak Fz, mean |Fz| during contact, xy [mm]."""
    wipe_idx = [i for i,(t,h) in enumerate(zip(d["time"], d["hybrid"]))
                if t > WIPE_T and h >= 0.5]
    if not wipe_idx: return None
    fz = [d["Fz"][i] for i in wipe_idx]
    contact_n = sum(1 for v in fz if v > F_THRESH)
    contact_pct = 100 * contact_n / len(fz)
    peak_fz = max(fz) if fz else 0
    fz_in = [v for v in fz if v > F_THRESH]
    fz_mean = st.mean(fz_in) if fz_in else 0
    sq = sum((d["ee_x"][i]-d["tgt_x"][i])**2 + (d["ee_y"][i]-d["tgt_y"][i])**2
             for i in wipe_idx) / len(wipe_idx)
    xy = 1000 * math.sqrt(sq)
    return dict(contact_pct=contact_pct, peak=peak_fz, fz_mean=fz_mean, xy=xy)

data = {m: load(f"{DATA}/{m}_K128_H0.10_s1.csv") for m in MODES}
mtr  = {m: metrics(data[m]) for m in MODES}

fig = plt.figure(figsize=(15, 9))
gs  = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])

# Row 1: xy 2D trajectory
for col, mode in enumerate(MODES):
    ax = fig.add_subplot(gs[0, col])
    d = data[mode]
    ex = [x for x,h in zip(d["ee_x"], d["hybrid"]) if h >= 0.5]
    ey = [y for y,h in zip(d["ee_y"], d["hybrid"]) if h >= 0.5]
    tx = [x for x,h in zip(d["tgt_x"], d["hybrid"]) if h >= 0.5]
    ty = [y for y,h in zip(d["tgt_y"], d["hybrid"]) if h >= 0.5]
    ax.plot(tx, ty, "--", color="gray", lw=1.2, alpha=0.7, label="target")
    ax.plot(ex, ey, color=COLORS[mode], lw=1.0, label="EE")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{TITLES[mode]}\nxy error = {mtr[mode]['xy']:.2f} mm",
                 fontsize=13)
    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

# Row 2: bar chart — 3 metrics × 3 modes
METRIC_KEYS = ["contact_pct", "fz_mean", "peak"]
METRIC_LABEL = {
    "contact_pct": "Contact % (Fz>1N)",
    "fz_mean":     "Mean Fz during contact [N]",
    "peak":        "Peak Fz [N]",
}
F_DES = 10.0  # target magnitude (|F_des_z|)

for col, key in enumerate(METRIC_KEYS):
    ax = fig.add_subplot(gs[1, col])
    vals  = [mtr[m][key] for m in MODES]
    bars  = ax.bar(range(len(MODES)), vals,
                   color=[COLORS[m] for m in MODES],
                   edgecolor="k", linewidth=0.8)
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels(["#2", "#3", "cost"], fontsize=11)
    ax.set_ylabel(METRIC_LABEL[key], fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f" {v:.1f}", ha="center", va="bottom", fontsize=10)
    # F_des reference line for mean Fz panel
    if key == "fz_mean":
        ax.axhline(F_DES, color="red", linestyle="--", lw=1.0,
                   alpha=0.7, label=f"|F_des| = {F_DES:.0f} N")
        ax.legend(fontsize=9, loc="upper right")

fig.suptitle("K=128, H=0.10 — wipe trajectory (top) and wipe-phase metrics (bottom)",
             fontsize=15, y=0.995)
plt.tight_layout()
out = "3modes_xyz_force.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
print("metrics:")
for m in MODES:
    print(f"  {m}: {mtr[m]}")
