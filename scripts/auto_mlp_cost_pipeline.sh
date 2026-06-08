#!/usr/bin/env bash
# Hands-off MLP-cost tuning pipeline.
#
#   stage 1  scale sweep      ~ 90 runs × 30s ≈ 45 min
#   stage 2  analyze scale
#   stage 3  pick top 2 scales (contact >= 75, H=0.10, xy asc) from stage 1
#   stage 4  timing sweep with narrowed SCALE_LIST
#                             ~144 runs × 30s ≈ 72 min  (2 scales × 4 LA × 3 CI × 2 K × 3 seed)
#   stage 5  analyze timing + combined
#   stage 6  write SUMMARY.md (Korean)
#
# Output root: $ROOT (default: auto_mlp_cost_run).
# Designed to be launched with `nohup setsid` so it survives shell exit.
set -u
cd "$(dirname "$0")/.."

ROOT=${ROOT:-auto_mlp_cost_run}
RUN_S=${RUN_S:-30}
mkdir -p "$ROOT"
LOG="$ROOT/00_run.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "pipeline start (PID=$$, ROOT=$ROOT, RUN_S=$RUN_S)"
log "host=$(hostname) user=$(whoami) git=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

# ---------------- stage 1: scale sweep ----------------
if [ -d "$ROOT/01_scale" ] && [ -n "$(find "$ROOT/01_scale" -name '*.csv' -print -quit 2>/dev/null)" ]; then
  log "stage 1: SKIP (already has CSV in $ROOT/01_scale)"
else
  log "stage 1: scale sweep -> $ROOT/01_scale ..."
  OUTDIR="$ROOT/01_scale" RUN_S="$RUN_S" \
    bash scripts/sweep_mlp_cost_scale.sh >> "$LOG" 2>&1
  log "stage 1: done ($(find "$ROOT/01_scale" -name '*.csv' | wc -l) csv)"
fi

# ---------------- stage 2: analyze scale ----------------
log "stage 2: analyze scale -> 03_analyze_scale.txt ..."
python3 scripts/analyze_mlp_cost_tuning.py "$ROOT/01_scale" \
  > "$ROOT/03_analyze_scale.txt" 2>&1 || true
log "stage 2: done"

# ---------------- stage 3: pick top scales ----------------
log "stage 3: pick top scales from stage 1 ..."
PICK=$(ROOT="$ROOT" python3 - <<'PY'
import os, importlib.util
spec = importlib.util.spec_from_file_location("a", "scripts/analyze_mlp_cost_tuning.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
b, meta = mod.collect([os.path.join(os.environ["ROOT"], "01_scale")])
rows = mod.aggregate(b)
filt = [r for r in rows
        if r["contact_pct"] >= 75
        and abs(meta[r["key"]]["H"] - 0.10) < 1e-6]
rank = sorted(filt, key=lambda r: r["xy_rms_mm"])
seen = []
for r in rank:
    s = meta[r["key"]]["scale"]
    if s not in seen:
        seen.append(s)
    if len(seen) >= 2:
        break
# fallback if stage 1 found nothing useful
print(" ".join(f"{s:g}" for s in seen) if seen else "1.0 1.5")
PY
)
log "stage 3: top scales = '$PICK'"

# ---------------- stage 4: timing sweep ----------------
if [ -d "$ROOT/02_timing" ] && [ -n "$(find "$ROOT/02_timing" -name '*.csv' -print -quit 2>/dev/null)" ]; then
  log "stage 4: SKIP (already has CSV in $ROOT/02_timing)"
else
  log "stage 4: timing sweep (SCALE_LIST='$PICK') -> $ROOT/02_timing ..."
  SCALE_LIST="$PICK" OUTDIR="$ROOT/02_timing" RUN_S="$RUN_S" \
    bash scripts/sweep_mlp_cost_timing.sh >> "$LOG" 2>&1
  log "stage 4: done ($(find "$ROOT/02_timing" -name '*.csv' | wc -l) csv)"
fi

# ---------------- stage 5: analyze timing + combined ----------------
log "stage 5: analyze timing + combined ..."
python3 scripts/analyze_mlp_cost_tuning.py "$ROOT/02_timing" \
  > "$ROOT/04_analyze_timing.txt" 2>&1 || true
python3 scripts/analyze_mlp_cost_tuning.py "$ROOT/01_scale" "$ROOT/02_timing" \
  > "$ROOT/05_analyze_combined.txt" 2>&1 || true
log "stage 5: done"

# ---------------- stage 6: SUMMARY.md ----------------
log "stage 6: write SUMMARY.md ..."
python3 scripts/write_mlp_cost_summary.py "$ROOT" "$PICK" \
  > "$ROOT/SUMMARY.md" 2> >(tee -a "$LOG" >&2) || true
log "pipeline DONE — see $ROOT/SUMMARY.md"
