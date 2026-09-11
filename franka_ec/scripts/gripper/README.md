# Hyundai 3-finger gripper — ROS 2 bridge

Two files, copied from `prior_mppi_judo/ours/`, so the gripper can be driven on a
machine that does not have the judo checkout.

| file | role |
|---|---|
| `gripper_bridge_node.py` | rclpy node: `/judo_gripper` shm ↔ `/ag_*` services + status topic |
| `gripper_shm.py` | the 40 B shm layout. Imported by name from the same directory. |

Move them together. The layout must stay identical to judo's copy **and** to mjpc's
`mjpc/tasks/Fr3HGripperCarry/gripper_shm.h`, because all three map `/judo_gripper`.

## Prerequisites, not in this repo

- `art_gripper` + `art_gripper_interfaces` (the vendor driver, e.g. `~/gripper_ws`)
- `rclpy` — so run with the **system python3**, not a venv. The judo venv is
  deliberately isolated and has no rclpy.

## Run

One terminal per block. **The namespace has to be the same in all three** — the
launch file name picks it, and everything after must use the matching `/ag_*`.

```bash
# ── terminal 1: gripper server ───────────────────────────────────────────
source /opt/ros/humble/setup.bash
source ~/gripper_ws/install/setup.bash

ros2 launch art_gripper gripper_ecat_right.launch.py    # -> /ag_right/*
# ros2 launch art_gripper gripper_ecat_left.launch.py   # -> /ag_left/*
# ros2 launch art_gripper gripper_ecat_dual.launch.py   # -> both

# check which one actually came up before going on:
ros2 service list | grep ag_
```

```bash
# ── terminal 2: motor ON, then verify by hand ───────────────────────────
source /opt/ros/humble/setup.bash
source ~/gripper_ws/install/setup.bash

# MUST be first. Without it the width commands are accepted (result=0) and nothing
# moves. Note the quotes: `on` is a Python keyword, and "{on: 1}" fails with
# "Failed to populate field: getattr(): attribute name must be string".
ros2 service call /ag_right/motor_on \
  art_gripper_interfaces/srv/MotorOn "{'on': 1}"

# open / close by hand, to confirm the hardware before any planner is involved.
# 0 = closed, 100 = open -- the OPPOSITE of mjpc's slide. See "Sign and travel".
ros2 service call /ag_right/set_target_finger_width \
  art_gripper_interfaces/srv/SetTargetFingerWidth "{finger_width: 95}"
ros2 service call /ag_right/set_target_finger_width \
  art_gripper_interfaces/srv/SetTargetFingerWidth "{finger_width: 32}"

# with a speed, and the grip force
ros2 service call /ag_right/set_target_finger_width_with_speed \
  art_gripper_interfaces/srv/SetTargetFingerWidthWithSpeed \
  "{finger_width: 90, finger_width_speed: 30}"
ros2 service call /ag_right/set_gripping_force \
  art_gripper_interfaces/srv/SetGrippingForce "{force: 20}"

# measured width + status word
ros2 topic echo /ag_right/gripper_status

# motor off when finished
ros2 service call /ag_right/motor_on \
  art_gripper_interfaces/srv/MotorOn "{'on': 0}"
```

```bash
# ── terminal 3: this bridge ─────────────────────────────────────────────
# SYSTEM python3 -- it needs rclpy, which the judo venv does not have.
source /opt/ros/humble/setup.bash
source ~/gripper_ws/install/setup.bash
cd <repo root>

# one gripper
python3 franka_ec/scripts/gripper/gripper_bridge_node.py --ns /ag_right

# BOTH grippers, one process: repeat --ns
python3 franka_ec/scripts/gripper/gripper_bridge_node.py \
    --ns /ag_left --ns /ag_right
```

It owns `/judo_gripper` (creates it, unlinks it on exit), so **start it before the
planner**. On startup it issues `motor_on(1)`, `set_target_finger_pose(180)` and
`set_gripping_force` once.

```bash
# ── terminal 4: the planner, with the gripper enabled ───────────────────
MJPC_GRIPPER_SHM=1 MJPC_GRIP_OPEN_MM=95 MJPC_GRIP_CLOSE_MM=32 MJPC_GRIP_FORCE_N=20 \
  ./build/bin/mjpc --task <task>
```

`MJPC_GRIPPER_SHM=1` makes the planner command the fingers.
`MJPC_GRIPPER_MIRROR=1` is read-only — the measured width drives the sim fingers and
the real ones are never commanded, which is the safe way to check the mapping first.
Only a task that carries the gripper-shm block reacts to these (see "Not included").

### Order, and what goes wrong out of order

| | why |
|---|---|
| 1. server | nothing else can resolve `/ag_*` |
| 2. `motor_on` | width commands return `result=0` and do nothing without it |
| 3. bridge | it is the OWNER of `/judo_gripper`; the planner only attaches |
| 4. planner | attaches; if the region is missing it degrades to sim-only |

`--ns` defaults to `/ag_left`. If the driver came up as `right`, only `/ag_right/*`
exists and every call logs "service not ready" — the warning is rate-limited to one
line per service per 2 s and names the namespace it tried.

### Both grippers

Repeat `--ns`. One process, one 50 Hz timer, one `ArmBridge` per gripper; each holds
its own service clients, its own shm region and its own state, so a command arriving
for one arm moves that arm only. Log lines are prefixed `[/ag_left]` / `[/ag_right]`
— without that, "motor is OFF" would not say which gripper.

**One shm region per gripper.** A region carries a single `cmd_seq`/`ack_seq`
handshake, so two arms sharing it would overwrite each other's commands. The names are
derived from the namespaces unless `--shm` overrides them:

| `--ns` | region |
|---|---|
| one, e.g. `/ag_right` | `/judo_gripper` — unchanged, so the single-arm path is untouched |
| `/ag_left` `/ag_right` | `/judo_gripper_left`, `/judo_gripper_right` |

```bash
--ns /ag_left --ns /ag_right --shm /a --shm /b     # explicit, same order
ls /dev/shm | grep judo                            # verify what actually exists
```

Duplicate namespaces, duplicate region names, and a `--shm` count that does not match
`--ns` are all rejected before rclpy starts.

The mjpc side matches with `mjpc_gripper_open("/judo_gripper_left")`; the argument
defaults to `/judo_gripper`, so existing single-gripper tasks need no edit.

**Shared fate, by construction.** One process means one EtherCAT stall blocks the
other arm's poll too: the service calls are `call_async` and never awaited, but
`read_cmd()` is not. Two processes (`--ns` once each, distinct `--shm`) trade the
single-command convenience for independent failure.

> The example in `CMD.md` launches `gripper_ecat_left.launch.py` and then calls
> `/ag_right/motor_on`. That pair cannot work: the left launch publishes `/ag_left/*`
> only. Pick one side and use it in all three terminals.

Options: `--ns`, `--dry-run`, `--contact-sensitivity`, `--width-speed`.

## Shm layout — 40 B, ten int32

| off | field | dir |
|---|---|---|
| 0 | `want_open` | → node (1 = send `width_open_mm`, 0 = `width_close_mm`) |
| 4 | `finger_pose_deg` | → node (180 = 2-finger mode) |
| 8 | `width_open_mm` | → node |
| 12 | `width_close_mm` | → node |
| 16 | `grip_force_n` | → node |
| 20 | `motor_on` | → node |
| 24 | `cmd_seq` | → node, bumped **last** |
| 28 | `status_word` | ← node (`GripperStatus.gripper_status`) |
| 32 | `finger_width_mm` | ← node, **measured** |
| 36 | `ack_seq` | ← node, bumped last |

An arbitrary width is sent by setting `want_open = 0` and putting the value in
`width_close_mm`; the node only issues a service call when the value **changes**, so a
steady width costs no EtherCAT traffic.

## Sign and travel — worth knowing before trusting a timing

Opposite conventions: mjpc's `grab_motor` ctrl is a slide target in metres over
0..0.05 where **0.05 is closed**; the hardware takes a width in mm where **0 is
closed**. Measured on the sim model, the pad gap is 108.2 mm at slide 0 and 8.2 mm at
0.05 — exactly the 100 mm the hardware reports as `finger_width`.

Measured travel is only **~18 mm/s**, so 95 → 32 mm is 63 mm ≈ **3.5 s**. Any
sequence that assumes the fingers are shut sooner will move the arm while the object
is still loose.

## The mjpc side

`mjpc/tasks/Fr3HGripperDual/gripper_shm.h` is the C++ end of the same region, copied
from `Fr3HGripperCarry/`. Layout cross-checked against the Python side: ten `int32_t`
vs `FMT = "10i"` / `STRUCT_SIZE = 40`.

A task only reaches the real fingers if it carries the gripper-shm block that reads
`grab_motor`'s ctrl and writes a width — `Fr3HGripperCarry` and `Fr3HGripperPick` do;
`Fr3HGripperDual` does **not** yet, so the header is present but nothing calls it.
Until that block is added, `MJPC_GRIPPER_SHM=1` has no effect on the dual task and the
real gripper only moves from the service calls in terminal 2.

All three copies of the layout — this header, `gripper_shm.py`, and judo's original —
map the same `/judo_gripper` and must stay identical.
