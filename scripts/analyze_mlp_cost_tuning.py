#!/usr/bin/env python3
"""Analyze MLP-cost tuning sweeps.

Handles both filename schemes:
  scale  : mlp_cost_scale{SCALE}_K{K}_H{H}_s{SEED}.csv
  timing : mlp_cost_K{K}_H{H}_sc{SCALE}_la{LA}_ci{CIDX}_s{SEED}.csv

Metrics (per-CSV):
  rows used     : time > 5.0 AND (hybrid >= 0.5 if column present)
  contact_pct   : 100 * #(Fz > 1.0) / n
  xy_rms_mm     : 1000 * sqrt(mean((ee_x-tgt_x)^2 + (ee_y-tgt_y)^2))
  fz_p95        : 95th percentile of Fz
  fz_peak       : max(Fz)
  plan_ms       : mean of plan_ms
  fm_ms         : mean of fm_ms where fm_ms > 0
  total_ms      : plan_ms + fm_ms

Configs aggregate across seeds (simple mean).
Usage:
  scripts/analyze_mlp_cost_tuning.py sweep_mlp_cost_scale
  scripts/analyze_mlp_cost_tuning.py sweep_mlp_cost_timing
  scripts/analyze_mlp_cost_tuning.py sweep_mlp_cost_scale sweep_mlp_cost_timing
"""
import csv
import math
import os
import re
import statistics as st
import sys
from collections import defaultdict

WARMUP = 5.0
F_THRESH = 1.0

# Highlight thresholds (target: match MPPI K=128 at lower compute).
XY_TARGET   = 2.0
CONT_TARGET = 76.0
TOTAL_TARGET = 2.2

# Filename patterns ----------------------------------------------------------
RE_SCALE  = re.compile(
    r"^mlp_cost_scale(?P<sc>[\d.]+)_K(?P<K>\d+)_H(?P<H>[\d.]+)_s(?P<seed>\d+)\.csv$"
)
RE_TIMING = re.compile(
    r"^mlp_cost_K(?P<K>\d+)_H(?P<H>[\d.]+)_sc(?P<sc>[\d.]+)"
    r"_la(?P<la>[\d.]+)_ci(?P<ci>\d+)_s(?P<seed>\d+)\.csv$"
)


def parse_name(name):
    m = RE_TIMING.match(name)
    if m:
        return ("timing", {
            "K": int(m["K"]), "H": float(m["H"]),
            "scale": float(m["sc"]), "la": float(m["la"]),
            "ci": int(m["ci"]), "seed": int(m["seed"]),
        })
    m = RE_SCALE.match(name)
    if m:
        return ("scale", {
            "K": int(m["K"]), "H": float(m["H"]),
            "scale": float(m["sc"]), "la": None, "ci": None,
            "seed": int(m["seed"]),
        })
    return (None, None)


def csv_metrics(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items() if v != ""})
            except (ValueError, TypeError):
                pass
    if not rows:
        return None
    has_hybrid = "hybrid" in rows[0]
    rs = [r for r in rows
          if r.get("time", 0) > WARMUP
          and (not has_hybrid or r.get("hybrid", 0) >= 0.5)]
    if not rs:
        return None
    n = len(rs)
    fz = [r["Fz"] for r in rs]
    contact_pct = 100.0 * sum(1 for v in fz if v > F_THRESH) / n
    fz_sorted = sorted(fz)
    fz_p95 = fz_sorted[min(n - 1, int(0.95 * n))]
    fz_peak = max(fz)
    xy_rms_mm = 1000.0 * math.sqrt(
        sum((r["ee_x"] - r["tgt_x"]) ** 2 + (r["ee_y"] - r["tgt_y"]) ** 2
            for r in rs) / n
    )
    plan_vals = [r["plan_ms"] for r in rs if "plan_ms" in r]
    fm_vals   = [r["fm_ms"]   for r in rs if r.get("fm_ms", 0) > 0]
    plan_ms = st.mean(plan_vals) if plan_vals else 0.0
    fm_ms   = st.mean(fm_vals)   if fm_vals   else 0.0
    return {
        "contact_pct": contact_pct,
        "xy_rms_mm":   xy_rms_mm,
        "fz_p95":      fz_p95,
        "fz_peak":     fz_peak,
        "plan_ms":     plan_ms,
        "fm_ms":       fm_ms,
        "total_ms":    plan_ms + fm_ms,
        "n":           n,
    }


def collect(dirs):
    """Returns dict[config_key] -> list of per-seed metric dicts.
    config_key omits seed so seeds aggregate together."""
    bucket = defaultdict(list)
    config_meta = {}
    for d in dirs:
        if not os.path.isdir(d):
            print(f"# warn: {d} is not a directory, skipping", file=sys.stderr)
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".csv"):
                continue
            kind, fields = parse_name(name)
            if kind is None:
                continue
            m = csv_metrics(os.path.join(d, name))
            if m is None:
                continue
            key = (kind, fields["K"], fields["H"], fields["scale"],
                   fields["la"], fields["ci"])
            bucket[key].append(m)
            config_meta[key] = fields
    return bucket, config_meta


def aggregate(bucket):
    """Returns list of dicts: one per config, mean over seeds + n_seeds."""
    out = []
    for key, ms in bucket.items():
        if not ms:
            continue
        agg = {k: st.mean(m[k] for m in ms) for k in ms[0] if k != "n"}
        agg["n_seeds"] = len(ms)
        agg["key"] = key
        out.append(agg)
    return out


def fmt_row(meta, agg):
    K, H, sc = meta["K"], meta["H"], meta["scale"]
    la = "—" if meta["la"] is None else f"{meta['la']:.2f}"
    ci = "—" if meta["ci"] is None else f"{meta['ci']:>2d}"
    return (f"K={K:>3d} H={H:.2f} sc={sc:.1f} la={la:>4} ci={ci:>2}  "
            f"seeds={agg['n_seeds']}  "
            f"xy={agg['xy_rms_mm']:5.2f}mm  "
            f"cont={agg['contact_pct']:5.1f}%  "
            f"Fz_p95={agg['fz_p95']:5.1f}  "
            f"peak={agg['fz_peak']:6.1f}  "
            f"plan={agg['plan_ms']:5.2f}ms  "
            f"fm={agg['fm_ms']:4.2f}ms  "
            f"tot={agg['total_ms']:5.2f}ms")


def passes_highlight(agg):
    return (agg["xy_rms_mm"] <= XY_TARGET
            and agg["contact_pct"] >= CONT_TARGET
            and agg["total_ms"] < TOTAL_TARGET)


def print_section(title, items, meta_lookup, limit=None):
    print(f"\n=== {title} ===")
    if not items:
        print("  (no rows)")
        return
    shown = items if limit is None else items[:limit]
    for agg in shown:
        marker = "  *" if passes_highlight(agg) else "   "
        print(marker, fmt_row(meta_lookup[agg["key"]], agg))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    dirs = sys.argv[1:]
    bucket, meta_lookup = collect(dirs)
    rows = aggregate(bucket)
    if not rows:
        print("No data parsed from:", dirs)
        sys.exit(1)

    print(f"Loaded {len(rows)} configs across {sum(a['n_seeds'] for a in rows)} runs")
    print(f"Highlight rule (*): xy_rms_mm <= {XY_TARGET}  AND  "
          f"contact >= {CONT_TARGET}  AND  total_ms < {TOTAL_TARGET}")

    # 1. Top configs over the full sweep, filtered by contact >= 75.
    filt = [a for a in rows if a["contact_pct"] >= 75.0]
    rank = sorted(filt, key=lambda a: (a["xy_rms_mm"], a["fz_peak"]))
    print_section("Top configs (contact >= 75, sort: xy asc, fz_peak asc)",
                  rank, meta_lookup, limit=20)

    # 2. K=32, H=0.10 detail.
    k32 = [a for a in rows
           if meta_lookup[a["key"]]["K"] == 32
           and abs(meta_lookup[a["key"]]["H"] - 0.10) < 1e-6]
    k32_rank = sorted(k32, key=lambda a: a["xy_rms_mm"])
    print_section("K=32, H=0.10 (sort: xy asc)", k32_rank, meta_lookup)

    # 3. K=64, H=0.10 detail.
    k64 = [a for a in rows
           if meta_lookup[a["key"]]["K"] == 64
           and abs(meta_lookup[a["key"]]["H"] - 0.10) < 1e-6]
    k64_rank = sorted(k64, key=lambda a: a["xy_rms_mm"])
    print_section("K=64, H=0.10 (sort: xy asc)", k64_rank, meta_lookup)

    # 4. Explicit highlight list.
    hits = [a for a in rows if passes_highlight(a)]
    hits_rank = sorted(hits, key=lambda a: (a["total_ms"], a["xy_rms_mm"]))
    print_section(
        f"HIGHLIGHT  xy<={XY_TARGET} & contact>={CONT_TARGET} & "
        f"total_ms<{TOTAL_TARGET}  (sort: total_ms asc)",
        hits_rank, meta_lookup)


if __name__ == "__main__":
    main()
