# macOS arm64 build + grasp-model fixes

## 1. Build

Seven things block a first build on macOS arm64. Only the second is a repo bug; the
rest are platform differences.

| # | symptom | fix |
|---|---|---|
| 1 | `clang++: unsupported option '-msse4.1' for arm64` | abseil emits `-Xarch_x86_64 -maes -msse4.1`, and `-Xarch_x86_64` binds to `-maes` ALONE, so `-msse4.1` reaches the arm64 compile. Empty `ABSL_RANDOM_HWAES_X64_FLAGS` in `build/_deps/abseil-cpp-src/absl/copts/GENERATED_AbseilCopts.cmake` (regenerated on a clean build) |
| 2 | `Error copying directory from /home/kkomji/tmp/...` | **repo bug**: `mjpc/tasks/CMakeLists.txt:49` hardcodes another machine's path. Changed to `${CMAKE_CURRENT_SOURCE_DIR}/Fr3/assets`, which exists and holds the 86 fr3 meshes. Commit this. |
| 3 | `onnxruntime_cxx_api.h not found` | brew puts it under `include/onnxruntime/`, not `include/`. Pass the cache vars explicitly |
| 4 | `-Werror` on an unused variable in `tasks/Fr3/fr3.cc` | `-DMJPC_DISABLE_WERROR=ON` (the option already exists) |
| 5 | `eigen3/Eigen/Dense not found` | `-isystem /opt/homebrew/include` |
| 6 | `xmmintrin.h: only for x86` | `planners/FlowMPPI{,Rpy}/planner.cc` reset MXCSR FTZ/DAZ because ORT sets it in worker threads and it changes rollout costs. **Ported to arm64 FPCR bit 24 (FZ) rather than deleted** -- the determinism hole is real on arm too |
| 7 | `cnpy.h not found` | build github.com/rogersce/cnpy (2 files) into `~/.local` |

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DMJPC_DISABLE_WERROR=ON \
  -DONNXRUNTIME_INCLUDE_DIR=/opt/homebrew/Cellar/onnxruntime/<ver>/include/onnxruntime \
  -DONNXRUNTIME_LIBRARY=/opt/homebrew/lib/libonnxruntime.dylib \
  -DCMAKE_CXX_FLAGS="-isystem /opt/homebrew/include"
ninja -C build mjpc
```

Run: `MJPC_FM_CONFIG` selects the config; **without it the default is
`tasks/Fr3/fm_config.yaml`** (horizon 0.05, K=16), not the task's own.

```bash
MJPC_FM_CONFIG=mjpc/tasks/Fr3HGripperPotDual/fm_config.yaml MJPC_PLANNER=9 \
MJPC_POTD_DBG=1.0 ./build/bin/mjpc --task="FR3_H_Gripper_PotDual"
```

## 2. Grasp model

Measured on `Fr3HGripperPotDual` (pot ~1 kg = 10 N, grasped by a 20 mm-radius handle).

**`condim` appears nowhere in the xml, so every contact defaulted to 3: sliding
friction only, no torsional.** A point contact on a cylinder then has zero resistance
to spin about the handle axis.

| configuration | holds |
|---|---|
| as shipped (condim 3, 10 N) | **0.5 N** -- cannot lift a 10 N pot at all |
| condim 3, 30 N | 2.4 N -- more grip force does not help |
| condim 4, 10 N | 23.1 N |
| **condim 4, 30 N (real gripper force)** | **102-285 N static, 4.2 g in a carry** |

Sliding friction is NOT the limit here: mu 1.0 vs 2.0 changes nothing. What was
limiting was first the missing torsional friction, then the grip force.

**The 48 `group="1"` display meshes on the H-gripper omit `contype`/`conaffinity`,
so MuJoCo's default of 1 made all of them collidable** -- 342k vertices duplicating
the 6.7 mm proxy spheres that are the intended collision model.

| | us/mj_step while grasping | carry rotation |
|---|---|---|
| meshes collidable | 246.8 | 2.6-50 deg |
| meshes disabled | **102.7 (2.4x)** | **0.2-0.4 deg** |

Faster AND firmer: the redundant mesh contacts were fighting the sphere contacts
(36 contacts vs 18), and the solver jittered.

Applied: `condim="4"` on 46 finger geoms + 10 pot geoms, `forcerange="-30 30"`,
`contype="0" conaffinity="0"` on the 48 display meshes. Originals in `.bak`.

## 3. Can the weld go?

`set_weld()` is now gated on `MJPC_NO_WELD=1`. Two reasons the weld looked necessary
turned out not to hold:

- it is **not** cheaper: 98.2 vs 98.0 us/mj_step
- the rollout **can** represent a friction grasp. mjpc sets the planning model's
  timestep to `agent_timestep` (`agent.cc:334`), so rollouts step at 0.03 s, but with
  u=0 (gravcomp holds the arm) the pot drifts only 2.9 mm over a 0.4 s horizon at
  that step -- versus 0.5-2.8 mm for the weld.

The weld force readout is only inside the `MJPC_POTD_DBG` block, so removing the weld
does not affect the cost or the FSM. It does remove that readout, and the code already
notes it is sim-only ("on hardware it would have to come from joint torques").

Watch one thing: `phase_ 2.5 -> 3` is gated on `t_conf_ >= hold_s`, a TIMER, with the
comment "weld가 파지이므로 시간 기반". Without a weld, time no longer implies grasp.
Replace it with a contact criterion -- `mj_contactForce` on the finger<->handle pairs
gives normal and tangential separately, so the slip margin is `mu*F_n - |F_t|`.

## 4. Open, not diagnosed

On an M1 Pro (8 cores) the task never leaves `phase_ 1` and the sim runs at ~30% of
real time. Two observations for whoever picks this up:

- **the two arms behave differently.** At default sigma the RIGHT arm converges
  (errR 229 -> 88 mm) while the LEFT diverges (errL 220 -> 435 -> 377 mm). With
  per-arm softmax each arm optimises its own terms, so this points at the left
  target or the left cost, not at the planner.
- raising `MJPC_STD_ARM` only saturates torque (`sat` 0.24 -> 0.67 -> 1.00) without
  converging, so this is not under-exploration.
