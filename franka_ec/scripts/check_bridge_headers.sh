#!/usr/bin/env bash
# Every copy of mjpc_bridge_dual.h must define the SAME struct. They all map the same
# /mjpc_bridge_dual region BY NAME, so a field added in one copy and not another
# misaligns the rest -- the classic version of this failure had both processes logging
# success while the arms sat still.
#
# The magic/struct_size preamble now catches that AT RUNTIME too, so a mismatch fails
# loudly instead of silently. This script still exists because catching it before a
# hardware session is cheaper than catching it during one, and because it names the
# offending file.
#
#   ./check_bridge_headers.sh                     reference = franka_ec's own copy
#   ./check_bridge_headers.sh ~/ws ~/other_ws     search these roots instead
#   ./check_bridge_headers.sh -r /path/to/mjpc_bridge_dual.h ~/ws
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
# franka_ec owns the region, so its copy is the reference by definition.
REF="$HERE/../include/franka_ec/mjpc_bridge_dual.h"

while [ $# -gt 0 ]; do
  case "$1" in
    -r|--ref) REF="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's|^# \{0,1\}||'; exit 0 ;;
    *) break ;;
  esac
done

if [ ! -f "$REF" ]; then
  echo "reference header not found: $REF" >&2
  echo "pass one with -r /path/to/mjpc_bridge_dual.h" >&2
  exit 2
fi

# Default roots: the workspace src tree that contains this package. That covers both
# franka_ec and any mujoco-mpc checkout under it (including franka_ec/tmp/). Roots at
# or above $HOME are dropped -- scanning a home directory is slow and turns up copies
# belonging to other projects.
if [ $# -gt 0 ]; then
  ROOTS=("$@")
else
  PKG=$(cd "$HERE/.." && pwd)
  ROOTS=()
  cand=$PKG
  for _ in 1 2 3; do
    case "$cand" in
      "$HOME"|"$HOME"/|/|"") break ;;
    esac
    ROOTS+=("$cand")
    cand=$(dirname "$cand")
  done
  [ ${#ROOTS[@]} -eq 0 ] && ROOTS=("$PKG")
fi

# Compare the STRUCT and the two protocol constants only. Comments and whitespace
# legitimately differ between copies and none of that affects the layout -- but a
# changed shm NAME or MAGIC breaks the link just as thoroughly as a changed field, so
# both are part of the signature.
strip() {
  {
    sed -n '/^struct MjpcBridgeDual/,/^};/p' "$1" | sed 's|//.*||'
    grep -E '^#define (MJPC_DUAL_SHM_NAME|MJPC_DUAL_MAGIC) ' "$1"
  } | tr -d '[:space:]'
}

REF_SIG=$(strip "$REF")
case "$REF_SIG" in
  *structMjpcBridgeDual*) ;;
  *) echo "could not find 'struct MjpcBridgeDual' in the reference: $REF" >&2; exit 2 ;;
esac

REF_REAL=$(readlink -f "$REF" 2>/dev/null || echo "$REF")
echo "reference: $REF"
for r in "${ROOTS[@]}"; do echo "searching: $r"; done
echo

found=0
rc=0
seen=""
for r in "${ROOTS[@]}"; do
  [ -d "$r" ] || continue
  while IFS= read -r f; do
    real=$(readlink -f "$f" 2>/dev/null || echo "$f")
    case " $seen " in *" $real "*) continue ;; esac   # same file reached twice
    seen="$seen $real"
    [ "$real" = "$REF_REAL" ] && continue
    found=$((found + 1))
    if [ "$(strip "$f")" = "$REF_SIG" ]; then
      printf '  OK        %s\n' "$f"
    else
      printf '  MISMATCH  %s\n' "$f"
      rc=1
    fi
  done < <(find "$r" -name mjpc_bridge_dual.h -not -path '*/build/*' -not -path '*/install/*' 2>/dev/null | sort)
done

echo
if [ "$found" -eq 0 ]; then
  echo "no other copies found. If the mjpc checkout lives outside these roots, pass it"
  echo "as an argument:  ./check_bridge_headers.sh /path/to/mujoco-mpc"
  exit 0
fi
if [ $rc -ne 0 ]; then
  echo "A mismatching copy must not be run against this controller. Sync the struct in"
  echo "every copy, bump MJPC_DUAL_MAGIC if the layout really changed, and rebuild BOTH"
  echo "sides."
else
  echo "all $found other copies agree"
fi
exit $rc
