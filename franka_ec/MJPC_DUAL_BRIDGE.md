# Connecting mujoco-mpc `FR3_H_Gripper_Dual` to the real dual FR3

Dual-arm counterpart to `MJPC_ROS2_BRIDGE_GUIDE.md` in the mujoco-mpc tree
(branch `fr3-hgripper-tasks`). That guide is the reference for the protocol and its
failure modes; **this file records only what is different for two arms**, plus the
bring-up order.

Every number below was read from the source or measured by loading the model — none
of it is recalled. Where a claim was checked, the check is named.

**Arm only.** The grippers are not bridged yet; see [What is deliberately not done].

---

## 1. What was added

| file | side | role |
|---|---|---|
| `franka_ec/include/franka_ec/mjpc_bridge_dual.h` | both | **canonical** 184 B struct. Byte-identical copy in the mjpc task dir. |
| `franka_ec/include/franka_ec/mjpc_dual_bridge_controller.hpp` | ROS | controller declaration |
| `franka_ec/src/mjpc_dual_bridge_controller.cpp` | ROS | the whole protocol, ~250 lines |
| `franka_ec/scripts/bridge_probe_dual.cc` | tool | standalone region tester; needs only g++ and librt |
| `franka_ec/scripts/check_bridge_headers.sh` | tool | diffs every copy of the header |
| `franka_bringup/launch/real/mjpc_dual_bridge_controller.py` | ROS | brings both arms up and spawns the controller |
| `franka_bringup/config/dual_controllers.yaml` | ROS | type + params (edited) |
| `mjpc/tasks/Fr3HGripperDual/mjpc_bridge_dual.h` | mjpc | the copy |
| `mjpc/tasks/Fr3HGripperDual/fr3.{h,cc}` | mjpc | the four bridge blocks (edited) |

`.bak_prebridge` copies sit beside every edited file.

Nothing in `mjpc/CMakeLists.txt` needed changing: `fr3.cc` was already built and the
new header is included, not compiled separately.

---

## 2. Why one 14-DOF region and not two 7-DOF ones

`FR3_H_Gripper_Dual` plans both arms in **one** MPPI tick — `nu = 16`, per-arm softmax
but a single planner. Two independent regions would give each arm its own sequence
counter, so the left arm could act on tick N while the right acted on N−1. For a task
whose two arms share a load that tearing is a real hazard, so both arms share one
region and one pair of counters: every state and action snapshot is self-consistent.

The single-arm `/mjpc_bridge` (120 B) is untouched and can run alongside this.

### 2.1 Layout — `/mjpc_bridge_dual`, 184 B

Index order in every 14-vector: **`left_joint1..7` then `right_joint1..7`.**

| offset | field | dir |
|---|---|---|
| 0 | `int32 magic` | owner stamps `0x4D4A4432` |
| 4 | `int32 struct_size` | owner stamps `184` |
| 8 | `float q[14]` | controller → mjpc, rad |
| 64 | `float dq[14]` | controller → mjpc, rad/s |
| 120 | `int32 state_seq` | controller → mjpc, bumped **after** q/dq |
| 124 | `float action[14]` | mjpc → controller, N·m |
| 180 | `int32 action_seq` | mjpc → controller, bumped **after** action |

Verified by compiling the struct and printing `sizeof`/`offsetof`:
`./bridge_probe_dual size`.

No `ee_pos`, no `target_pos`. Both were dead weight in the single-arm struct (nothing
wrote `ee_pos`, no task read it) and this struct has no backwards layout to preserve.

### 2.2 The header trap is now self-reporting

The single-arm bridge's worst failure was two different structs sharing one shm
**name**: layouts misaligned, one side corrupted the other's counter, and both
processes logged success while the arm sat still.

Two things make that impossible here. The name is different (`/mjpc_bridge_dual`), and
the `magic`/`struct_size` preamble is checked on attach — a non-owner built from a
different header **refuses to attach and says so**. Tested by building the probe
against a header with a bumped magic:

```
[mjpc_bridge_dual] header MISMATCH: region magic=0x4D4A4432 size=184, this build
expects magic=0x4D4A4433 size=184. The two sides were built from different copies
of mjpc_bridge_dual.h -- diff them and rebuild BOTH.
```

`check_bridge_headers.sh` still exists because catching this before a hardware session
beats catching it during one, and because it names the offending file. Its signature
covers the struct **and** `MJPC_DUAL_SHM_NAME`/`MJPC_DUAL_MAGIC` — a changed name
breaks the link just as thoroughly as a changed field. Both cases were negative-tested.

---

## 3. THE DUAL-SPECIFIC TRAP: ctrl index ≠ dof index

The single-arm guide could assume the arm owns the first 7 `qpos` slots and that
`ctrl` index equals `dof` index. **Neither holds here.** Measured by loading
`task.xml` with mujoco 3.3.3:

```
nq=27  nv=26  nu=16

idx  joint          qposadr  dofadr  actuator      ctrl
0-6  l_fr3_joint1-7    0-6     0-6   l_actuator1-7   0-6      <- coincidentally equal
7-13 r_fr3_joint1-7   10-16   10-16  r_actuator1-7   8-14     <- DIFFERENT
                                     l_grab_motor      7
                                     r_grab_motor     15
```

The right arm's dofs are 10–16 but its ctrl channels are 8–14, because each gripper's
`grab_motor` sits between the two arms in actuator order.

A single index used for both — which is what the template does, correctly, for one arm
— would make the right arm read `qfrc_bias` for `r_fr3_joint3` while writing the
torque of `r_fr3_joint1`, wrong by two joints, and would run two channels past the end
of `ctrl` (`nu = 16`). On hardware, with no error message.

So `ResolveArmAddresses()` keeps **three** tables (`qadr_`, `dadr_`, `cadr_`) and
resolves them from the model **by name**, not by arithmetic. `aid + j` would assume the
actuators are declared contiguously — true today, and exactly the kind of assumption a
model edit breaks silently.

It also refuses to drive anything it cannot verify:

- every joint and actuator name must resolve
- each actuator must have `biastype == mjBIAS_NONE`, i.e. be a `<motor>`. **Measured:
  the 14 arm actuators are `biastype 0`; both `grab_motor`s are `biastype 1`
  (`<position>`).** `action` is N·m, so a position actuator would send radians —
  guide §4.5 asks for this to be eyeballed in the xml; checking it in code means a
  model swap cannot get it wrong.
- each actuator must actually drive that joint (`actuator_trnid`), which ties the ctrl
  channel to the joint whose `qfrc_bias` and gravcomp flag are used for it

Any failure prints why, leaves `addr_ok_ = false`, and publishes nothing — so the
controller times out and both arms hold gravity compensation.

On attach it logs the resolved map, which is the first thing to read:

```
[Fr3HGripperDual] arm map  L qpos[0..6] dof[0..6] ctrl[0..6] | R qpos[10..16] dof[10..16] ctrl[8..14]
```

---

## 4. Gravity: this model needs NO subtraction

**Measured: `body_gravcomp == 1.0` on all 14 arm bodies** (136 bodies carry
`gravcomp="1"` in the xml). The robot applies its own `g(q)`, so `ctrl` is already
gravity-free and `qfrc_bias` must **not** be subtracted. Backwards, this commands
`−g(q)` and the arm sags the moment `ctrl ≈ 0` — a bug that shipped once in the Carry
task.

The branch is still per-DOF (`model->body_gravcomp[dof_bodyid[dadr_[k]]]`), so a mixed
or edited model keeps working. Do not "simplify" it to an unconditional pass-through
just because this model happens to be all-1.

---

## 4a. The pot is sim-only — `MJPC_DUAL_NO_POT=1`

`FR3_H_Gripper_Dual` needs no gripper command at all, which is what makes it the right
first task for hardware. **Verified by loading the model:** `neq = 4` and all four are
finger-slide joint couplings — there is no grasp weld; the task has no phase machine;
`ctrl[7]` and `ctrl[15]` (`l_/r_grab_motor`) sit at 0.0 forever, because
`sampling_std_per_joint` is 0 on both channels and nothing else writes them. 0 on a
`ctrlrange` of ±0.1 is fully open. So no gripper command is ever produced, and the real
gripper needs none.

The pot is the exception that does matter. No cost term reads it — the cross-arm
collision term counts only `l_`/`r_` body pairs, so `pot` never matches — but it is
still **1.0 kg with 9 collision geoms** at `(0.55, 0, 0.12)`, real physics in the GUI
model and in every rollout.

BLOCK 2 force-writes the arm `qpos` from the robot every tick and does **not** mirror
the pot's free joint. So if the real arm stands where the sim pot is, the sim resolves a
deep penetration and launches the pot, and it never comes back — from then on the
rollouts plan around a phantom obstacle in the wrong place. Measured: parking the pot on
the left arm produces **130 contacts, all of them pot contacts**.

`MJPC_DUAL_NO_POT=1` turns off those 9 geoms and sets `body_gravcomp[pot] = 1.0` so the
pot does not sink once nothing holds it up. Measured after the switch: **0 contacts, and
0.0 m of drift over 2 s** of integration. `mjModel` edits are the right lever because
the planner holds the model pointer, so they reach rollouts — `mjData` state outside the
five fields `trajectory.cc` propagates would not.

Leave it unset if a real pot sits at the same pose, or in pure sim.

---

## 5. The controller side

`franka_ec/MjpcDualBridgeController`. **Torque mode** — command interfaces are
`<arm_id>_joint1..7/effort` for both arms, so `action` is read as N·m.

This is **not** a variant of `MultiJointMPPIController`: that one receives joint
position/velocity references over ZMQ and closes a PD loop locally. This one receives
feedforward torque over shared memory and passes it through. Neither replaces the
other; both remain registered.

`update()` at 1 kHz:

1. **Publish state** — `q`, `dq` for all 14 as float, then `state_seq++`.
2. **Read action** — only when `action_seq > last_action_seq_`; else `stale_count_++`.
3. **Low-pass** — `tau_f = 0.2*tau_f + 0.8*tau_new`. 0.2 is the **retention**
   coefficient, so at 1 kHz the −3 dB point is ≈352 Hz against a 500 Hz Nyquist: this
   **barely filters**. Do not size the planner's jitter budget against it. (The
   single-arm source comment claiming ≈160 Hz is wrong; the guide's ≈350 Hz is right.)
4. **Slew limit** — 1.0 N·m per step = 1000 N·m/s. The ceiling in 6 is therefore
   checked on the limited value, not the raw one.
5. **Timeout** — `stale_count_ > 100` (100 ms).
6. **Torque ceiling** — `{87,87,87,87,12,12,12}` per arm.
7. **Command, or fall back** — zero effort on **both** arms.

**The fallback always zeroes both arms, never one.** A dual task whose arms share a
load must not have one arm keep driving while the other goes slack. Both fallbacks
latch `action_ready_ = false`, so recovery needs a fresh `action_seq`, not merely the
offending condition going away.

Zero effort *is* gravity compensation on this hardware interface, because the robot's
own controller holds `g(q)`. On an arm where zero effort means the arm falls, this
fallback would have to become an explicit gravity torque first.

Lifecycle: `on_activate` creates + zeroes + stamps the region; `on_deactivate` closes
and **unlinks**. So the controller can be restarted under a running mjpc — mjpc's 2 s
retry reattaches.

---

## 6. One-step lag, and why it is fine

`Task::TransitionLocked` runs **before** `mj_step`, and `mjcb_control` writes `ctrl`
**inside** `mj_step`. So the torque published on a tick is the one computed on the
*previous* tick. **Measured: `timestep = 0.001 s`**, so the lag is 1 ms. The
single-arm task has the identical structure; this is noted so it is not rediscovered
as a bug.

---

## 7. Bring-up order

Each step is verifiable before the next. Do not skip step 3.

```bash
# 0  headers agree (cheapest check there is)
cd franka_ec/scripts && ./check_bridge_headers.sh          # every copy must be OK

# 1  build both sides
cd ~/franka_ros2_ws && colcon build --packages-select franka_ec && source install/setup.bash
# mujoco-mpc: build as usual; fr3.cc is already in the task's source list

# 2  arm controller — creates /mjpc_bridge_dual
ros2 launch franka_bringup mjpc_dual_bridge_controller.py \
    robot_ip_1:=172.16.0.2 robot_ip_2:=172.16.0.3
#    Both arms are on the SAME /24 (this PC holds 172.16.0.1 on the robot NIC).
#    load_gripper_1/2 default to FALSE here: the 3finger xacro is not namespaced and
#    loading it twice duplicates base_link, killing robot_state_publisher.
#    expect: "activated -- /mjpc_bridge_dual created (184 B, 14 DOF)"

# 3  PROVE THE CONTROLLER WORKS, WITHOUT MJPC
cd franka_ec/scripts
g++ -O2 -std=c++17 -I../include/franka_ec -o bridge_probe_dual bridge_probe_dual.cc -lrt
./bridge_probe_dual size     # 184 B, and the offsets above
./bridge_probe_dual watch    # state_seq must RISE; L and R q must both look sane
./bridge_probe_dual zero     # no "action timeout" in the controller log
#    Only once `zero` runs clean is a missing motion the task's fault.

# 4  raise the collision thresholds if cartesian_reflex trips (per arm; see
#    MJPC_ROS2_BRIDGE_GUIDE.md §9 step 5 for the full service call)

# 5  mjpc, DRY RUN FIRST — the arms must not move and |tau| must be ~0 at rest
cd franka_ec/tmp/mujoco-mpc
MJPC_TASKS_DIR=$PWD/mjpc/tasks MJPC_BRIDGE_DRYRUN=1 MJPC_DUAL_NO_POT=1 \
  ./build/bin/mjpc --task FR3_H_Gripper_Dual
#    MJPC_DUAL_NO_POT=1 unless a real pot is on the table at the sim pose -- see 4a
#    expect the arm map line, then once a second:
#      [Fr3HGripperDual] DRY  |tau|max L=  0.03 R=  0.02 Nm
#    Tens of N·m on joints 2/4 => gravity subtracted twice; releasing dry-run would
#    drop that arm. Fix gravcomp in the model, not the code.

# 6  drop MJPC_BRIDGE_DRYRUN=1
```

`MJPC_DUAL_PRIM=1` selects the primitive-collision model variant. It keeps `nu = 16`
and the same joint/actuator names, so the bridge is unaffected — but the arm map line
is still worth reading after switching.

---

## 8. Failure table (dual-specific)

| symptom | cause | check |
|---|---|---|
| `SIM ONLY` on startup | controller not up | mjpc retries every 2 s; nothing to do |
| `header MISMATCH` on attach | two sides built from different headers | `check_bridge_headers.sh`, then rebuild **both** |
| `bridge DISABLED: model has no joint/actuator …` | wrong model loaded | the task expects `l_/r_fr3_joint1..7` |
| `bridge DISABLED: actuator … is not a <motor>` | position-actuator model | `action` is N·m; fix the xml |
| `is N B, expected >= 184 B` | stale region from an older build | stop the controller (it unlinks), or `rm /dev/shm/mjpc_bridge_dual` |
| one arm moves, the other does not | ctrl/dof mapping | read the arm map line; §3 |
| both arms sag when dry-run is released | gravity subtracted twice | dry-run `|tau|` at rest must be ~0; §4 |
| arms detour around nothing / plans get worse over time | sim pot knocked out of place, never mirrored back | `MJPC_DUAL_NO_POT=1`; §4a |
| `action timeout … BOTH arms` | >100 ms without a new `action_seq` | plan rate, or mjpc paused |
| `torque limit exceeded on right_joint4` | that channel over its ceiling | cost weights, `ctrlrange` |

---

## 9. What is deliberately not done

- **Forwarding a gripper command to the real hand.** `FR3_H_Gripper_Dual` never
  produces one (§4a), which is precisely why it is the first task to bring up. The
  tasks that DO produce one are `Fr3HGripperPotDual` and `Fr3HGripperCoCarry`: their
  phase machines call `set_grip()` (which drives `l_/r_grab_motor` by clamping
  `actuator_ctrlrange`) and `set_weld()` at the pre-grasp -> grasp transition. Bridging
  that command to the real gripper is the next step, and it reads from those two tasks,
  not from this one.
- **The grippers.** The template's advice is explicit: get the arms moving first, then
  add the gripper, so failures do not stack. It is also more work than it looks — the
  dual model's `l_/r_grab_motor` are `<position>` actuators on
  `l_/r_finger_A_slide_joint` with `ctrlrange -0.1..0.1`, so the single-arm guide's
  `slide 0..0.05 → width_mm` line (§5.1) does **not** transfer and needs re-measuring;
  and `/judo_gripper` is a single 40 B region, so two hands need a left/right split
  designed first.
- **`/mjpc_object`** (camera object pose). Separate region, separate opt-in.
- **Growing this struct** to carry either of the above. A field added here invalidates
  every copy of the header and every binary at once. New signals go in a new region —
  which is exactly why `/judo_gripper` and `/mjpc_object` are separate.

## 10. Verification status

Verified: struct size/offsets by compilation; the address map, actuator types and
gravcomp by loading `task.xml` with mujoco 3.3.3; `fr3.cc` syntax-clean against real
mujoco + abseil headers; `franka_ec` builds clean and exports the plugin; the shm
protocol end to end against a stand-in owner (state_seq rose at 1 kHz, 100 Hz actions
arrived, the 100 ms timeout fired, the region unlinked); header mismatch rejected at
runtime; `check_bridge_headers.sh` negative-tested on a changed field and a changed
shm name.

Also verified for the no-gripper question: `neq`/equality types, gripper `ctrl` values
and ranges, and the pot contact count before and after `MJPC_DUAL_NO_POT=1`, all by
loading `task.xml` with mujoco 3.3.3.

**Never run on the robot.** Treat it as reviewed, tested-in-loopback source — not as
tested behaviour. Step 3 and the dry run in step 5 exist because of that.
