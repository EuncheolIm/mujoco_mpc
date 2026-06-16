#!/usr/bin/env python3
"""3-way comparison: MPPI baseline vs FM-only vs FM-as-cost.
Row 1: xy 2D wipe trajectory (3 panels)
Row 2: bar charts — contact %, peak Fz, mean Fz during contact (3 modes side-by-side)
"""
import csv, math, statistics as st
import matplotlib.pyplot as plt

DATA = "out/videos/4modes_K128_H0.10"
MODES = [
    ("mppi",   "MPPI baseline",     "mppi_K128_H0.10_s1.csv",   "#7f7f7f"),
    ("fmonly", "FM only",           "fmonly_K1_H0.2_s1.csv",    "#9467bd"),
    ("cost",   "FM as cost",        "cost_K128_H0.10_s1.csv",   "#1f77b4"),
]

F_THRESH = 1.0
WIPE_T   = 5.0
F_DES    = 10.0  # |F_des_z| target

NEEDED = ("time","Fz","ee_x","ee_y","tgt_x","tgt_y","hybrid")
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
    # phase 2 transition time
    t_trans = None
    for t, h in zip(d["time"], d["hybrid"]):
        if h >= 0.5:
            t_trans = t
            break
    # P2 transient peak: max Fz in first 1s after phase 2 transition
    peak_fz = 0.0
    if t_trans is not None:
        for t, h, fz in zip(d["time"], d["hybrid"], d["Fz"]):
            if h >= 0.5 and t - t_trans < 1.0 and fz > peak_fz:
                peak_fz = fz
    # Wipe metrics (t > WIPE_T, hybrid)
    wipe_idx = [i for i,(t,h) in enumerate(zip(d["time"], d["hybrid"]))
                if t > WIPE_T and h >= 0.5]
    if not wipe_idx:
        return None
    fz = [d["Fz"][i] for i in wipe_idx]
    contact_n = sum(1 for v in fz if v > F_THRESH)
    contact_pct = 100 * contact_n / len(fz)
    fz_in = [v for v in fz if v > F_THRESH]
    fz_mean = st.mean(fz_in) if fz_in else 0
    sq = sum((d["ee_x"][i]-d["tgt_x"][i])**2 + (d["ee_y"][i]-d["tgt_y"][i])**2
             for i in wipe_idx) / len(wipe_idx)
    xy = 1000 * math.sqrt(sq)
    return dict(contact_pct=contact_pct, peak=peak_fz, fz_mean=fz_mean, xy=xy)

data = {tag: load(f"{DATA}/{fname}") for tag, _, fname, _ in MODES}
mtr  = {tag: metrics(data[tag]) for tag, _, _, _ in MODES}

fig = plt.figure(figsize=(15, 9))
gs  = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])

# Row 1: xy 2D trajectory (wipe phase)
for col, (tag, title, _, color) in enumerate(MODES):
    ax = fig.add_subplot(gs[0, col])
    d = data[tag]
    ex = [x for x,h in zip(d["ee_x"], d["hybrid"]) if h >= 0.5]
    ey = [y for y,h in zip(d["ee_y"], d["hybrid"]) if h >= 0.5]
    tx = [x for x,h in zip(d["tgt_x"], d["hybrid"]) if h >= 0.5]
    ty = [y for y,h in zip(d["tgt_y"], d["hybrid"]) if h >= 0.5]
    if not ex:
        # FM-only may not transition to phase 2; use entire trajectory after warmup
        idx = [i for i,t in enumerate(d["time"]) if t > 1.0]
        ex = [d["ee_x"][i] for i in idx]
        ey = [d["ee_y"][i] for i in idx]
        tx = [d["tgt_x"][i] for i in idx]
        ty = [d["tgt_y"][i] for i in idx]
    ax.plot(tx, ty, "--", color="gray", lw=1.2, alpha=0.7, label="target")
    ax.plot(ex, ey, color=color, lw=1.0, label="EE")
    ax.set_aspect("equal", adjustable="datalim")
    xy_val = mtr[tag]['xy'] if mtr[tag] else float('nan')
    ax.set_title(f"{title}\nxy error = {xy_val:.2f} mm", fontsize=13)
    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

# Row 2: bar charts — 3 metrics × 3 modes
METRIC_KEYS = ["contact_pct", "peak", "fz_mean"]
METRIC_LABEL = {
    "contact_pct": "Contact % (Fz > 1 N)",
    "peak":        "P2 transient peak Fz [N]\n(max in first 1s after contact)",
    "fz_mean":     "Mean Fz during contact [N]",
}

for col, key in enumerate(METRIC_KEYS):
    ax = fig.add_subplot(gs[1, col])
    tags = [m[0] for m in MODES]
    short = ["MPPI", "FM-only", "FM-cost"]
    colors = [m[3] for m in MODES]
    vals = [mtr[t][key] if mtr[t] else 0 for t in tags]
    bars = ax.bar(range(len(tags)), vals, color=colors,
                  edgecolor="k", linewidth=0.8)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(short, fontsize=11)
    ax.set_ylabel(METRIC_LABEL[key], fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f" {v:.1f}", ha="center", va="bottom", fontsize=10)
    # (no F_des reference line — only MPPI baseline tracks F_des explicitly;
    #  FM-only has no force objective, FM-as-cost uses an upper-bound hinge.)

fig.suptitle("MPPI baseline vs FM-only vs FM-as-cost — wipe trajectory (top) and wipe-phase metrics (bottom)",
             fontsize=15, y=0.995)
plt.tight_layout()
out = "3way_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
print("metrics:")
for tag, title, _, _ in MODES:
    print(f"  {title}: {mtr[tag]}")
