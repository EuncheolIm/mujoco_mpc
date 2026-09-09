#!/usr/bin/env bash
# Every copy of mjpc_bridge.h must define the SAME struct. They all map the same
# /mjpc_bridge region BY NAME, so a field added in one copy and not another misaligns
# the rest silently: the arm stops moving while both processes log success.
#
#   ./check_headers.sh                            reference = the copy beside this script
#   ./check_headers.sh ~/ws ~/other_ws            search these roots instead
#   ./check_headers.sh -r /path/to/mjpc_bridge.h ~/ws
#
# NO PATH ASSUMPTIONS. An earlier version looked for
# franka_ec/include/franka_ec/mjpc_bridge.h, which only exists in one particular
# workspace layout -- on another machine the controller package may be named or placed
# differently, or may not be present at all. The reference is therefore the copy
# shipped next to this script, which is a byte copy of the controller's.
#
# That is sufficient. The question is whether all copies AGREE; if every copy matches
# this one then they match each other. A controller whose copy differs shows up as a
# MISMATCH row like any other, with no need to know which path it lives at.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
REF="$HERE/mjpc_bridge.h"

while [ $# -gt 0 ]; do
  case "$1" in
    -r|--ref) REF="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's|^# \{0,1\}||'; exit 0 ;;
    *) break ;;
  esac
done

if [ ! -f "$REF" ]; then
  echo "reference header not found: $REF" >&2
  echo "pass one with -r /path/to/mjpc_bridge.h" >&2
  exit 2
fi

# Default roots: this repo, its parent, and its grandparent. The parent catches sibling
# checkouts of other mjpc trees (the usual hiding place for a stale copy) and the
# grandparent catches the controller package, which typically sits beside the directory
# holding those checkouts rather than inside it. Roots at or above $HOME are dropped --
# scanning a home directory is slow and finds copies that belong to other projects.
# Override by listing roots explicitly.
if [ $# -gt 0 ]; then
  ROOTS=("$@")
else
  REPO=$(cd "$HERE/../.." && pwd)
  ROOTS=()
  cand=$REPO
  for _ in 1 2 3; do
    case "$cand" in
      "$HOME"|"$HOME"/|/|"") break ;;
    esac
    ROOTS+=("$cand")
    cand=$(dirname "$cand")
  done
  [ ${#ROOTS[@]} -eq 0 ] && ROOTS=("$REPO")
fi

# Compare the STRUCT only: comments and whitespace legitimately differ between copies
# (each carries notes about its own task) and none of that affects the layout.
strip() { sed -n '/^struct MjpcBridge/,/^};/p' "$1" | sed 's|//.*||' | tr -d '[:space:]'; }

REF_SIG=$(strip "$REF")
if [ -z "$REF_SIG" ]; then
  echo "could not find 'struct MjpcBridge' in the reference: $REF" >&2
  exit 2
fi

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
  done < <(find "$r" -name mjpc_bridge.h -not -path '*/build/*' -not -path '*/install/*' 2>/dev/null | sort)
done

echo
if [ "$found" -eq 0 ]; then
  echo "no other copies found. If the controller lives outside these roots, pass its"
  echo "workspace as an argument:  ./check_headers.sh ~/your_ws"
  exit 0
fi
if [ $rc -ne 0 ]; then
  echo "A mismatching copy must not be run against this controller. Either sync the"
  echo "struct in every copy and rebuild every binary, or keep that tree away from"
  echo "/mjpc_bridge. See MJPC_ROS2_BRIDGE_GUIDE.md section 2.2."
else
  echo "all $found other copies agree"
fi
exit $rc
