# ros2_bridge — template for wiring an mjpc task to the real FR3

Copy-paste starting point for `MJPC_ROS2_BRIDGE_GUIDE.md` (repo root). The guide
explains the protocol and the failure modes; this directory is the minimum code.

| file | what it is |
|---|---|
| `mjpc_bridge.h` | **canonical** copy of the 120 B struct. Byte-identical to `franka_ec/include/franka_ec/mjpc_bridge.h`. Copy, never edit. |
| `example_task.h` / `.cc` | minimal task: the four bridge blocks and nothing else |
| `task.xml` | minimal task xml, with the two settings that are not optional called out |
| `bridge_probe.cc` | standalone `/mjpc_bridge` tester. No mjpc, no ROS. |
| `check_headers.sh` | compares every `mjpc_bridge.h` in the workspace against `franka_ec`'s |

Nothing here is in `mjpc/CMakeLists.txt`, so the templates cannot break the build.

---

## Start here: prove the controller works, without mjpc

`bridge_probe` attaches exactly the way a task does, which separates "the controller
is broken" from "my task is broken" before any task exists.

```bash
g++ -O2 -o bridge_probe bridge_probe.cc -lrt

./bridge_probe size     # offsets + a hard check against 120 B
./bridge_probe watch    # state_seq must RISE -> the controller is publishing
./bridge_probe zero     # publish zero torque at 100 Hz = gravity comp, safe
```

`zero` running without `action timeout` in the controller's log means the whole action
path works end to end. Only then is a missing motion your task's fault.

`hold <joint> <Nm>` commands real torque; it caps itself at ±10 N·m (±3 on joints
5–7). Killing the probe is a safe stop — the controller drops to gravity compensation
after 100 ms of silence.

---

## Then: bring up a task

1. `cp -r templates/ros2_bridge mjpc/tasks/ExampleBridge` — keep `mjpc_bridge.h`,
   drop `bridge_probe.cc` / `check_headers.sh` / this README.
2. `./check_headers.sh` — every copy must report OK. A MISMATCH here is the failure
   that costs the most time to find, because both processes report success while the
   arm does nothing.
3. Rename `ExampleBridgeTask` / `Example_Bridge` / the include guards, and fill in
   `task.xml`'s arm include, its sensors, and your cost.
4. Register it:
   - `mjpc/tasks/tasks.cc`: add the include and `std::make_shared<YourTask>()`
   - `mjpc/CMakeLists.txt`: add `tasks/YourTask/fr3.cc` and `.h` to the source list
5. Build, then run **dry** first:

```bash
MJPC_TASKS_DIR=$PWD/mjpc/tasks MJPC_BRIDGE_DRYRUN=1 \
  ./build/bin/mjpc --task Example_Bridge
```

`|tau|max` must be near zero with the arm at rest. Tens of N·m on joints 2/4 means the
gravity branch picked the wrong side and releasing dry-run would drop the arm — fix
`gravcomp` in the model, not the code.

---

## The four blocks, and why each one is shaped that way

Each is marked `BLOCK n` in `example_task.cc`.

**1 — attach, lazily and repeatedly.** Non-owner; the ROS side creates and unlinks. The
2 s retry exists because an attach-once version left the arm disconnected for a whole
session when mjpc started first, with a single startup line as the only symptom.

**2 — mirror the real arm.** `qpos[0..6]` / `qvel[0..6]` only. **Assumes the arm is the
first 7 qpos entries.** Check your model: judo's declares the object body first, which
would need an offset here.

**3 — publish torque, with the gravity branch.** The robot adds its own `g(q)`, so
`gravcomp="1"` on the arm means `ctrl` is already gravity-free and `qfrc_bias` must not
be subtracted. Backwards, this commands `−g(q)` and the arm sags the instant `ctrl ≈ 0`
— a real bug once shipped in the Carry task. The branch reads `body_gravcomp` per DOF,
so a mixed model still works. Bump `action_seq` **after** the payload.

**4 — the dry-run report.** The only cheap test of block 3, and the reason dry-run
exists at all.

---

## What this template deliberately leaves out

The gripper (`/judo_gripper`, 40 B) and the camera (`/mjpc_object`, 36 B) are separate
regions with separate blocks — guide sections 5 and 6. Add them after the arm moves.
Growing `MjpcBridge` to carry them would break every copy of the header and every
binary at once, which is exactly why they are separate.
