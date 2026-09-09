# Connecting mujoco-mpc to the real FR3 — integration guide

How `mjpc` and `franka_ec`'s ROS 2 controller exchange data, what had to be added on
each side, and what another agent needs to do to wire a **new** mjpc task or a
**different** mjpc checkout to the same robot.

Written from the code as it stands. Every number here was read out of the source or
measured, not recalled; where a claim comes from a specific line it is cited.

---

## 1. Architecture

Three independent POSIX shared-memory regions. No ROS dependency inside `mjpc`, no
mjpc dependency inside `franka_ec`: neither tree includes the other's headers, and
`franka_ec` never learns about `art_gripper_interfaces`.

```
┌─ terminal 1 ────────────────────────────────────────────────────────────┐
│ ros2 launch franka_bringup mppi_track_controller.launch.py              │
│   franka_ec/MPPITrackController        1 kHz, torque mode              │
│      owns  /mjpc_bridge  (120 B)  ── arm state out, torque in           │
└─────────────────────────────────────────────────────────────────────────┘
┌─ terminal 2+3 ──────────────────────────────────────────────────────────┐
│ ros2 launch art_gripper gripper_ecat_{left,right}.launch.py             │
│ python3 ours/gripper_bridge_node.py --ns /ag_right   (system python3)    │
│      owns  /judo_gripper (40 B)   ── width command out, measured in     │
└─────────────────────────────────────────────────────────────────────────┘
┌─ terminal 4 (optional) ─────────────────────────────────────────────────┐
│ object_zmq_bridge.py --endpoint tcp://HOST:5557                         │
│      owns  /mjpc_object  (36 B)   ── camera object pose                 │
└─────────────────────────────────────────────────────────────────────────┘
┌─ terminal 5 ────────────────────────────────────────────────────────────┐
│ ./build/bin/mjpc --task FR3_H_Gripper_Pick                              │
│      attaches to all three as a NON-owner; each one missing is a        │
│      normal state that degrades gracefully, never an error              │
└─────────────────────────────────────────────────────────────────────────┘
```

**Ownership rule.** The ROS side creates and unlinks; mjpc only ever attaches. So the
controller can be restarted under a running mjpc and vice versa. mjpc retries every
2 s, which removes the start-order requirement entirely — an earlier version attached
once and a session started in the wrong order stayed disconnected with only one
startup line to say so.

---

## 2. `/mjpc_bridge` — the arm link

### 2.1 Layout

`franka_ec/include/franka_ec/mjpc_bridge.h`, duplicated verbatim into each mjpc task
directory that uses it.

| offset | field | dir | note |
|---|---|---|---|
| 0 | `float q[7]` | controller → mjpc | measured joint angles, rad |
| 28 | `float dq[7]` | controller → mjpc | measured joint velocities, rad/s |
| 56 | `float ee_pos[3]` | controller → mjpc | EE position from FK, world frame |
| 68 | `int32 state_seq` | controller → mjpc | bumped **after** writing q/dq/ee_pos |
| 72 | `float action[7]` | mjpc → controller | feedforward torque, N·m |
| 100 | `int32 action_seq` | mjpc → controller | bumped **after** writing action |
| 104 | `float target_pos[3]` | external → mjpc | optional EE target override |
| 116 | `int32 target_seq` | external → mjpc | bumped after writing target_pos |

**`sizeof(MjpcBridge) == 120`.** Verified by compiling the struct and printing
`sizeof`/`offsetof`.

### 2.2 THE TRAP: three copies of this header exist and they are not all the same

```
05e4cb98  franka_ec/include/franka_ec/mjpc_bridge.h          120 B
05e4cb98  tmp/lerobot-prior-mppi/.../Fr3OodSim2Real/         120 B
0263d250  tmp/mujoco_mpc-hgripper/.../Fr3HGripperCarry/      120 B   (comments differ only)
0263d250  tmp/mujoco_mpc-hgripper/.../Fr3HGripperReach/      120 B
02522559  tmp/mujoco_mpc/.../Fr3/mjpc_bridge.h                92 B   <-- DIFFERENT STRUCT
02522559  tmp/mujoco_mpc/.../mppi_track/mjpc_bridge.h         92 B   <-- DIFFERENT STRUCT
```

The `tmp/mujoco_mpc` copies have **no `ee_pos` and no `target_pos`/`target_seq`**:

|  | size | `state_seq` | `action` | `action_seq` |
|---|---|---|---|---|
| franka_ec / hgripper | 120 B | @68 | @72 | @100 |
| old `mujoco_mpc` | 92 B | @56 | @60 | @88 |

They share the same shm **name**, so attaching the old tree's binary to the region the
current controller created misaligns everything:

- the old tree writes `action` into bytes 60–87, which is the controller's `ee_pos` +
  `state_seq` + `action[0..3]` — **it corrupts the controller's own sequence counter**
- the old tree never writes byte 100, so the controller's `action_seq` never advances
  and it sits in gravity compensation forever

Failure mode: the arm does nothing while both processes report success. **Before
running any mjpc tree against this controller, diff its `mjpc_bridge.h` against
`franka_ec/include/franka_ec/mjpc_bridge.h`.** If a struct field is added, every copy
has to change together and every binary has to be rebuilt.

### 2.3 Synchronisation

A plain sequence counter, not a lock. Each side writes its payload first and bumps its
counter last; the reader compares against the value it last saw. There is no
`state_seq` re-read after the payload, so a torn read is possible in principle — at
1 kHz against a 100 Hz planner it has never been observed, and a single stale sample
costs nothing because the next tick supersedes it.

---

## 3. The controller side (`franka_ec/src/mppi_track_controller.cpp`)

`franka_ec/MPPITrackController`, spawned by
`ros2 launch franka_bringup mppi_track_controller.launch.py robot_ip:=172.16.0.2`
(`controllers.yaml` maps the name to the type). **Torque mode**: the command
interfaces are joint efforts, so `action` is read as N·m.

`update()` at 1 kHz, in order:

1. **Publish state** — `q`, `dq` as float, then `state_seq++`.
2. **Read action** — only when `action_seq > last_action_seq_`; otherwise
   `stale_count_++`.
3. **Low-pass** — `tau_filtered = 0.2*tau_filtered + 0.8*tau_new`.
   The source comment says "≈160 Hz cutoff"; **the comment is wrong**. 0.2 is the
   retention coefficient, so the pole is at `-ln(0.2)` per ms and the cutoff is
   ~350 Hz — i.e. this barely filters. Do not size the planner's jitter budget
   against 160 Hz.
4. **Rate limit** — 1.0 N·m per step = **1000 N·m/s**.
5. **Timeout** — `stale_count_ > 100` (100 ms) → gravity compensation.
6. **Torque ceiling** — `{87,87,87,87,12,12,12}` N·m; exceeding it → gravity
   compensation and a red log line.
7. **Command** — either the limited torque, or `0.0` on every joint, which for this
   hardware interface *is* gravity compensation.

Both fallbacks latch `action_ready_ = false`, so recovery needs a fresh `action_seq`.

Lifecycle: `on_activate` calls `mjpc_bridge_open(true)` (creates + zeroes),
`on_deactivate` closes and **unlinks**. Stopping the controller therefore removes the
region under a running mjpc; mjpc's 2 s retry picks the new one up.

---

## 4. The mjpc side — what a task has to add

All of this lives in `TransitionLocked()`, which runs on the GUI/physics `mjData`
only — never inside a rollout. See `mjpc/tasks/Fr3HGripperPick/fr3.cc`.

### 4.1 Attach, lazily and repeatedly

```cpp
if (!bridge_ && !bridge_tried_) {
  bridge_tried_ = true;
  if (const char* e = std::getenv("MJPC_BRIDGE_DRYRUN")) dry_run_ = (std::atoi(e) != 0);
  bridge_ = mjpc_bridge_open(false);          // NON-owner
  ...
}
if (!bridge_ && data->time - bridge_retry_t_ >= 2.0) { /* retry */ }
```

Snapshot `state_seq`/`target_seq` on attach so leftovers from a previous session are
not mistaken for fresh data. No bridge → pure sim, unchanged behaviour.

### 4.2 Mirror the real arm into the sim

```cpp
const bool state_fresh = (bridge_->state_seq != last_state_seq_);
if (state_fresh) {
  for (int i = 0; i < 7; ++i) {
    data->qpos[i] = bridge_->q[i];
    data->qvel[i] = bridge_->dq[i];
  }
}
```

Only `qpos[0..6]`. This requires **the arm to occupy the first 7 qpos slots** — in
`Fr3HGripperPick` the layout is arm(0..6), fingers(7..9), free box(10..16). A model
that declares the object body first (as judo's does) would need the offset instead.

### 4.3 Publish torque — and get the gravity branch right

```cpp
const int jbody = model->dof_bodyid[i];
const bool gc = model->body_gravcomp[jbody] > 0.0;
double tau_ff = data->ctrl[i];
if (!gc) tau_ff -= data->qfrc_bias[i];
if (!dry_run_) bridge_->action[i] = tau_ff;
```

| model `gravcomp` | code | torque sent |
|---|---|---|
| **1** | no subtraction | `tau = ctrl` |
| 0 | subtract `qfrc_bias` | `tau = ctrl − (C q̇ + g)` |

The robot applies its own `g(q)`. So a model whose arm bodies carry `gravcomp="1"`
must **not** have `qfrc_bias` subtracted — subtracting would command `−g(q)` and the
arm sags the moment `ctrl ≈ 0`. This was a live bug in the Carry task: its model had
`gravcomp="0"` on the arm, so `tau = −g(q)` at rest. The branch is per-DOF because
`body_gravcomp` is per body, so one code path serves both conventions.

`Fr3HGripperPick`'s model sets `gravcomp="1"` on 11 bodies (7 arm links + `hand` + 3
finger links) and leaves `sugar_box` at 0 — a free body must fall.

### 4.4 Dry run — always the first hardware step

`MJPC_BRIDGE_DRYRUN=1` computes and prints the torque but never bumps `action_seq`,
so the controller's 100 ms timeout holds gravity compensation and the arm cannot
move. At rest the printed `|tau|` must be near zero; tens of N·m on joints 2/4 means
gravity is being subtracted twice and releasing dry-run would drop the arm. The
gripper is unaffected by dry-run, so grasp timing can be checked at the same time.

### 4.5 Actuators must be torque

```xml
<motor name="actuator1" joint="fr3_joint1" ctrllimited="true" ctrlrange="-87.0 87.0"/>
```

`ctrl` is N·m. Pairing a `<position>`-actuator model with this controller sends
radians where newton-metres are expected — ±87 rad of commanded angle. The sibling
`mppi_pos_controller` is the one that wants position setpoints.

---

## 5. `/judo_gripper` — the Hyundai gripper link

40 B, ten `int32`. Layout is dictated by
`prior_mppi_judo/ours/gripper_shm.py`; `gripper_bridge_node.py` (rclpy, **system
python3** — the judo venv has no rclpy) is the other end and judo's `run_real.py`
also writes here. **Do not reorder or grow it.**

| offset | field | dir |
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

The node calls a ROS service only when the value **changes**, so a steady width costs
no EtherCAT traffic.

### 5.1 Sign and scale

Opposite conventions. `grab_motor`'s `ctrl` is a slide target in metres over
0..0.05 where 0.05 is fully **closed**; the hardware takes a width in mm where 0 is
closed. **Measured** on this model: the pad gap is 108.2 mm at slide 0 and 8.2 mm at
0.05 — exactly the 100 mm the hardware reports as `finger_width`, because both jaws
close symmetrically. So the conversion is a straight line, not a rescale into the
operator's band:

```
width_mm = 108.2 − (slide / 0.05) × 100        then clamped to [close_mm, open_mm]
```

The clamp limits how far the command may go **without distorting what it means**.
Rescaling instead meant a slide that just held the box in sim came out as a different
width on hardware, which is the one thing this mapping exists to get right.

Rate limiting matters more than for the arm: every change is a service call plus an
EtherCAT round trip and the measured travel is only ~18 mm/s. The command is
quantised to 2 mm and capped at 10 Hz.

**Consequence for timing:** 95 → 32 mm is 63 mm ≈ **3.5 s**. Any task that assumes
the fingers are shut sooner than that will lift while the real box is still loose.

### 5.2 Namespace

`--ns` defaults to `/ag_left`. If the driver was launched as `right`, only
`/ag_right/*` exists and every service call fails with "service not ready". Check
with `ros2 service list | grep ag_`.

---

## 6. `/mjpc_object` — camera object pose

36 B: `float pos[3]`, `float quat[4]` (w-first), `int32 seq`, `int32 valid`. Written
by `object_zmq_bridge.py`, which owns both the socket and the region so that
`libmjpc` never links libzmq and the wire format stays in one place. That parser is
not obvious and was worked out on hardware: **7 raw float32, no framing, quaternion
XYZW not w-first, transport is PUSH so the reader must PULL, and the position needs a
hand-measured frame offset.**

mjpc opts in with `MJPC_OBJECT_SHM=1`. Without it the object keeps the pose the XML
gives it, and any hold/injection latch never executes at all — a detail worth knowing
before concluding that a latch is broken.

---

## 7. Driving the gripper from a task without letting MPPI plan it

`Fr3HGripperPick` removes the gripper from the search: `sampling_std_per_joint`'s last
entry is 0 and no cost term mentions the channel. The phase machine issues the
command instead. That needs one hook, because of ordering:

`Task::Transition` runs **before** `mj_step`, and `mjcb_control` runs **inside**
`mj_step` and writes **all** `nu` channels via `ActionFromPolicy`. So anything a task
puts in `data->ctrl` is overwritten. The hook therefore sits after that call, in
`mjpc/app.cc`:

```cpp
if (m->nuserdata >= 2 && data->userdata[0] != 0.0) {
  const int ga = mj_name2id(m, mjOBJ_ACTUATOR, "grab_motor");
  if (ga >= 0 && ga < m->nu) data->ctrl[ga] = data->userdata[1];
}
```

`userdata[0]` is the opt-in and `userdata[1]` the command, so every other task is
inert. `userdata` was chosen over a new Task virtual because `trajectory.cc` already
copies it into rollout data, and it needs no signature change.

Note the one-step lag this creates for §5: the gripper shm block reads `data->ctrl[7]`
from *before* `mj_step`, i.e. the previous tick's value. At dt = 0.01 that is 10 ms,
against 18 mm/s of travel and a 10 Hz send cap — immaterial.

---

## 8. What rollouts do and do not see

Worth knowing before assuming a task's plan matches its physics.

`trajectory.cc` seeds each rollout with **only** `qpos`, `qvel`, `act`, `userdata`,
`time`. Consequently:

- **`mjData::eq_active` is never propagated.** A weld flipped at runtime binds the GUI
  physics and *not* the rollouts. `Fr3HGripperPick` accepts this deliberately: no cost
  term reads the object, so the only consequence is that rollouts do not feel the
  0.5 kg payload. A task whose cost *does* read the object must instead keep the weld
  `active="true"` and drive `mjModel::eq_data` — the model is shared
  (`agent.cc` copies it once and the planner holds that pointer), so model writes
  *are* visible to rollouts.
- `mjModel` edits (`eq_data`, `body_gravcomp`, numerics) reach rollouts; `mjData`
  state outside the five fields above does not.

---

## 9. Wiring up a new task or a new mjpc checkout

1. **Diff the header.** `mjpc_bridge.h` against
   `franka_ec/include/franka_ec/mjpc_bridge.h`. Same struct or nothing works — see
   §2.2. Copy it into the task directory; do not include across trees.
2. **Torque actuators.** `<motor>` with `ctrlrange` matching
   `{87,87,87,87,12,12,12}`.
3. **Arm first in qpos.** Or adjust the mirror loop in §4.2.
4. **Decide the gravity convention** and let §4.3's branch handle it. Put
   `gravcomp="1"` on the arm links unless there is a reason not to; leave free bodies
   at 0.
5. **Copy the four blocks** from `Fr3HGripperPick/fr3.cc`: attach + retry, state
   mirror, torque publish with the gravcomp branch, dry-run gate. They are
   self-contained and depend only on the members declared in `fr3.h`.
6. **Register** in `mjpc/tasks/tasks.cc` and `mjpc/CMakeLists.txt`.
7. **Model assets.** If the task includes a model that references meshes, symlink
   `assets -> ../Fr3/assets`; without it the mesh paths do not resolve and the model
   fails to load with a bare "Error opening file".
8. **Bring up in this order**, then verify each line before the next:

```bash
# 1  arm controller (creates /mjpc_bridge)
ros2 launch franka_bringup mppi_track_controller.launch.py robot_ip:=172.16.0.2

# 2  gripper driver
ros2 launch art_gripper gripper_ecat_right.launch.py

# 3  gripper motor on   ('on' is a Python keyword -> the quotes are required)
ros2 service call /ag_right/motor_on art_gripper_interfaces/srv/MotorOn "{'on': 1}"

# 4  gripper bridge (system python3, owns /judo_gripper)
cd .../prior_mppi_judo
source /opt/ros/humble/setup.bash && source ~/gripper_ws/install/setup.bash
python3 ours/gripper_bridge_node.py --ns /ag_right

# 5  raise the collision thresholds if cartesian_reflex trips (force = the 6-vectors;
#    the torque 7-vectors are joint reflexes and are a different error)
ros2 service call /service_server/set_full_collision_behavior \
  franka_msgs/srv/SetFullCollisionBehavior "{
  lower_torque_thresholds_nominal:      [25.0,25.0,22.0,20.0,19.0,17.0,14.0],
  upper_torque_thresholds_nominal:      [35.0,35.0,32.0,30.0,29.0,27.0,24.0],
  lower_torque_thresholds_acceleration: [25.0,25.0,22.0,20.0,19.0,17.0,14.0],
  upper_torque_thresholds_acceleration: [35.0,35.0,32.0,30.0,29.0,27.0,24.0],
  lower_force_thresholds_nominal:       [48.0,48.0,48.0,40.0,40.0,40.0],
  upper_force_thresholds_nominal:       [64.0,64.0,64.0,56.0,56.0,56.0],
  lower_force_thresholds_acceleration:  [48.0,48.0,48.0,40.0,40.0,40.0],
  upper_force_thresholds_acceleration:  [64.0,64.0,64.0,56.0,56.0,56.0]}"

# 6  mjpc, DRY RUN first — the arm must not move and |tau| must be ~0 at rest
cd .../mujoco_mpc-hgripper
MJPC_TASKS_DIR=$PWD/mjpc/tasks MJPC_BRIDGE_DRYRUN=1 \
MJPC_GRIPPER_SHM=1 MJPC_GRIP_OPEN_MM=95 MJPC_GRIP_CLOSE_MM=32 MJPC_GRIP_FORCE_N=20 \
MJPC_PLANNER_THREADS=6 nice -n 5 ./build/bin/mjpc --task FR3_H_Gripper_Pick

# 7  drop MJPC_BRIDGE_DRYRUN=1
```

---

## 10. Failure table

| symptom | cause | check |
|---|---|---|
| `SIM ONLY` on startup | controller not up | mjpc retries every 2 s; no action needed |
| arm never moves, both sides look fine | header struct mismatch | §2.2, diff and rebuild |
| arm sags when dry-run is released | gravity subtracted twice | dry-run `|tau|` at rest must be ~0 |
| `action timeout — fallback to gravity comp` | > 100 ms without a new `action_seq` | plan rate, or mjpc paused/stopped |
| `torque limit exceeded` | a channel over `{87,87,87,87,12,12,12}` | cost weights, `ctrlrange` |
| `cartesian_reflex` | external force over threshold | §9 step 5; also check what the arm is pushing on |
| gripper never moves | wrong namespace | `ros2 service list | grep ag_`, then `--ns` |
| `Failed to populate field: getattr()` | `"{on: 1}"` | `on` is a keyword: `"{'on': 1}"` |
| sim grasps but the real box is pushed away | weld/lift fired before the fingers closed | §5.1: 63 mm of travel ≈ 3.5 s |
| latch/injection "not working" | `/mjpc_object` absent | §6: the whole block is gated on it |
| model fails to load, "Error opening file" | missing `assets` symlink | §9 step 7 |

---

## 11. Deliberate non-goals

- **No ROS in mjpc, no mjpc in franka_ec.** Adding a field to `MjpcBridge` breaks
  every copy and every binary at once, so new signals go in a *new* region (that is
  why `/judo_gripper` and `/mjpc_object` are separate) rather than by growing this one.
- **The gripper bridge is Python.** Both ends of `/judo_gripper` are Python, so there
  is no C header to keep in sync and `franka_ec` never depends on
  `art_gripper_interfaces`.
- **No grasp detection on the mjpc side of Pick.** Command-based tests cannot see the
  box (the command saturates) and measured-slide tests depend on friction and force
  the real gripper does not share. The real hand does not drop the box, so the sim is
  told the same via a weld on a timer. If that timer ever needs to go, the honest
  signal is the Hyundai drive's Contact bit, already carried in `status_word`.
