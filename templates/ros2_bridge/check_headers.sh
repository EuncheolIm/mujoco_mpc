#!/usr/bin/env bash
# Every copy of mjpc_bridge.h must define the SAME struct: they all map the same
# /mjpc_bridge region by name, so a field added in one place and not another
# misaligns the rest silently -- the arm stops moving while both sides log success.
#
# This compares each copy's STRUCT (comments stripped) against franka_ec's, which is
# the one the controller is compiled with and therefore the authority.
#
#   ./check_headers.sh [search-root]
#
# With no argument it walks UP from this script looking for a directory that holds
# include/franka_ec/mjpc_bridge.h, so it works from wherever the template was copied
# to instead of hard-coding a number of "..".
set -u
if [ $# -ge 1 ]; then
  ROOT="$1"
else
  ROOT=$(cd "$(dirname "$0")" && pwd)
  while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/include/franka_ec/mjpc_bridge.h" ]; do
    ROOT=$(dirname "$ROOT")
  done
  [ "$ROOT" = "/" ] && ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
fi
REF=$(find "$ROOT" -path '*/franka_ec/include/franka_ec/mjpc_bridge.h' -not -path '*/build/*' 2>/dev/null | head -1)
[ -z "$REF" ] && [ -f "$ROOT/include/franka_ec/mjpc_bridge.h" ] && REF="$ROOT/include/franka_ec/mjpc_bridge.h"

if [ -z "$REF" ]; then
  echo "reference franka_ec/include/franka_ec/mjpc_bridge.h not found under $ROOT" >&2
  exit 2
fi

strip() { sed -n '/^struct MjpcBridge/,/^};/p' "$1" | sed 's|//.*||' | tr -d '[:space:]'; }
REF_SIG=$(strip "$REF")
echo "reference: $REF"
echo

rc=0
while IFS= read -r f; do
  [ "$f" = "$REF" ] && continue
  if [ "$(strip "$f")" = "$REF_SIG" ]; then
    printf '  OK        %s\n' "${f#$ROOT/}"
  else
    printf '  MISMATCH  %s\n' "${f#$ROOT/}"
    rc=1
  fi
done < <(find "$ROOT" -name mjpc_bridge.h -not -path '*/build/*' 2>/dev/null | sort)

echo
if [ $rc -ne 0 ]; then
  echo "A mismatching copy must not be run against this controller. Either sync the"
  echo "struct in every copy and rebuild every binary, or keep that tree away from"
  echo "/mjpc_bridge. See MJPC_ROS2_BRIDGE_GUIDE.md section 2.2."
else
  echo "all copies agree"
fi
exit $rc
