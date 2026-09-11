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

```bash
# 1. driver
source /opt/ros/humble/setup.bash
source ~/gripper_ws/install/setup.bash
ros2 launch art_gripper gripper_ecat_right.launch.py

# 2. motor on. 'on' is a Python keyword, so the quotes are required -- without them
#    you get "Failed to populate field: getattr(): attribute name must be string".
ros2 service call /ag_right/motor_on art_gripper_interfaces/srv/MotorOn "{'on': 1}"

# 3. this bridge (owns /judo_gripper, so start it BEFORE the planner)
python3 franka_ec/scripts/gripper/gripper_bridge_node.py --ns /ag_right
```

`--ns` defaults to `/ag_left`. If the driver came up as `right`, only `/ag_right/*`
exists and every call logs "service not ready" — check with
`ros2 service list | grep ag_`.

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

## Not included

The mjpc-side C++ header `gripper_shm.h` is **not** on this branch — it lives in
`mjpc/tasks/Fr3HGripperCarry/` on the single-arm branch. A dual task that wants to
command the real grippers needs it (or its own copy), and the 40 B layout above is
what it must match.
