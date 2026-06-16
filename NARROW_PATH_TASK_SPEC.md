# FR3 Narrow Passage Task Spec

## Purpose

Create a new FR3 task for testing whether a learned free-space reaching prior can reduce MPPI rollout budget while MPPI handles unseen workspace constraints.

This task must be separate from the existing FR3 wipe / `MPPI_Force` task. Do not repurpose the current wipe task as the narrow-passage task.

The intended narrative is:

- The learned FM/MLP prior is not obstacle-aware.
- The prior only provides free-space reaching guidance toward a target.
- MPPI must satisfy the unseen corridor/gate constraints online through cost.
- If the prior is useful, MLP+MPPI should solve the task with fewer rollouts than MPPI alone.

This is not a contact or force-control task.

## Required First Steps

1. Read `AGENTS.md`.
2. Read `CLAUDE.md`.
3. Inspect the existing FR3 task implementation:
   - `mjpc/tasks/Fr3/fr3.cc`
   - `mjpc/tasks/Fr3/task.xml`
   - `mjpc/tasks/Fr3/cost_fn.cc`
   - `mjpc/tasks/Fr3/cost_fn.h`
   - `mjpc/tasks/Fr3/fm_config.yaml`
   - `mjpc/tasks/tasks.cc`
4. Preserve the existing FR3 wipe task behavior.

## New Task

Create a new task directory, for example:

```text
mjpc/tasks/Fr3Narrow/
```

Use a new task name, for example:

```text
FR3_Narrow
```

Do not keep the task name as `MPPI_Force`.

Suggested files:

```text
mjpc/tasks/Fr3Narrow/fr3_narrow.cc
mjpc/tasks/Fr3Narrow/fr3_narrow.h
mjpc/tasks/Fr3Narrow/cost_fn.cc
mjpc/tasks/Fr3Narrow/cost_fn.h
mjpc/tasks/Fr3Narrow/task.xml
mjpc/tasks/Fr3Narrow/fm_config.yaml
```

It is acceptable to reuse shared FR3 assets/model files from `mjpc/tasks/Fr3/` if that avoids duplication.

Register the new task in `mjpc/tasks/tasks.cc`, without removing the existing FR3 task.

## Task Concept

This is a fixed-base arm workspace narrow-passage task.

The robot base does not move. Only the end-effector path is constrained.

Use a small S-shaped virtual corridor in the reachable FR3 workspace. Start with soft virtual gates, not physical collision walls.

The end-effector should move from a start pose to a final target pose while passing through two offset gates:

```text
Top view, x to the right, y upward.

            Gate 1                    Gate 2
          #########                 #########
          ###   ###                 #########
          ###   ###                 #########
S --------#########-----------------#########-------- T
          #########-----------------###   ###
          #########                 ###   ###
          #########                 #########

Gate 1 gap is above center.
Gate 2 gap is below center.
The straight path y=0 violates both gates.
```

Start with a small reachable workspace setup:

```text
z_ref   = 0.34 or 0.35
start   = (0.42,  0.00, z_ref)
gate1   = (0.48, +0.06, z_ref)
gate2   = (0.54, -0.06, z_ref)
target  = (0.60,  0.00, z_ref)
```

Suggested initial gate settings:

```text
gate_half_width = 0.04 to 0.06
gate_sigma_x    = 0.025 to 0.04
```

Do not start with a very narrow or collision-based passage. FR3's end-effector is large, and real wall collision will likely create unnecessary tuning problems.

## Cost Design

The new task should be free-space reaching plus virtual corridor constraints.

Recommended cost terms:

```text
Reach_pos
Reach_ori
Gate1
Gate2
joint_vel_penalty
u_reg
FM_track
```

Do not include force-control terms from the wipe task:

```text
EE_Force
EE_zvel
```

Do not use table contact as part of this task.

### Reach Position

Track the final target position. Initially keep the target fixed.

For the first implementation, use a mocap target or numeric target similar to the existing FR3 task, but do not copy the wipe circular motion behavior.

### Gate Cost

Implement two soft virtual gates. Each gate is active only when the end-effector is near the gate's x position.

For gate `i`:

```text
gate_active = exp(-((ee_x - gate_x_i)^2) / sigma_x^2)
violation   = max(0, abs(ee_y - gap_y_i) - gap_half_width)
residual_i  = gate_active * violation
```

Use separate residuals for gate 1 and gate 2.

Suggested gate values:

```text
gate1_x = 0.48
gate1_y = +0.06
gate2_x = 0.54
gate2_y = -0.06
```

This should make the straight-line path from start to target invalid, because y=0 is outside both gate gaps.

### Visual Geometry

Optional: add transparent visual wall/gate geoms to `task.xml` to help inspect the task.

If visual geoms are added, set collision off:

```xml
contype="0" conaffinity="0"
```

The constraint should come from the cost function, not physical collision.

## Guide Integration

Reuse the existing guide infrastructure where possible:

```text
MJPC_GUIDE_TYPE=mlp
MJPC_FM_MODE=cost
MJPC_FM_TRACK_SCALE=<scale>
```

The learned prior is a free-space reaching prior. It is not trained on gates or obstacles. This is intentional.

The expected behavior is:

- MLP/FМ prior pulls the robot toward the target.
- Gate costs bend/refine the path through the corridor.
- MPPI handles the unseen constraints online.

Do not implement WTA/proposal mode as the default for this task. Start with cost mode only.

## Experiments

Create sweep scripts after the task builds and a short smoke test works.

Suggested comparison:

```text
MPPI baseline:
K = 16 32 64 128
H = 0.10 0.20 0.30

MLP+MPPI:
K = 8 16 32 64
H = 0.10 0.20 0.30
scale = 0.5 1.0 1.5
```

Use 3 seeds if possible.

## Metrics

Log enough data to compute:

```text
success:
  final position error < 1 cm
  gate violation below threshold

final_pos_error_mm
max_gate_violation_mm
mean_gate_violation_mm
min/mean clearance proxy
plan_ms
guide_ms or fm_ms
total_ms
```

If the existing CSV logger is copied, update columns so gate residuals/violations are visible.

## Expected Research Question

The task should answer:

```text
Can a generic free-space learned prior reduce MPPI rollout budget even when the task contains unseen workspace constraints?
```

Desired result:

```text
MPPI alone requires larger K/H to reliably find the S-shaped corridor.
MLP+MPPI succeeds with smaller K/H because the prior gives target-directed guidance and MPPI only needs to refine around the gates.
```

Important caveat:

If `FM_track` scale is too high, the prior may force straight-line reaching and hurt gate satisfaction. Sweep scale and report this if it happens.

## Implementation Constraints

- Do not break existing FR3 wipe task.
- Do not remove or rename existing `MPPI_Force`.
- Do not add physical collision walls initially.
- Keep the first version simple and inspectable.
- Build after changes:

```bash
cmake --build build --target mjpc -j2
```

- Run only short smoke tests first. Do not start a full sweep until the task can visibly reach the target and gate residuals behave sensibly.

## Suggested Smoke Tests

MPPI baseline:

```bash
env MJPC_PLANNER=0 MJPC_HORIZON=0.20 MJPC_TRAJECTORIES=64 \
    MJPC_AUTORUN=1 MJPC_FORCE_LOG=/tmp/fr3_narrow_mppi.csv \
  timeout --signal=TERM 10 ./build/bin/mjpc
```

MLP+MPPI:

```bash
env MJPC_PLANNER=9 MJPC_FM_MODE=cost MJPC_GUIDE_TYPE=mlp \
    MJPC_MLP_CKPT=$HOME/tmp/flow-matching-robot-control/checkpoints/student_mlp_v26/student.onnx \
    MJPC_MLP_STATS=$HOME/tmp/flow-matching-robot-control/checkpoints/student_mlp_v26/normalization_stats.npz \
    MJPC_FM_TRACK_SCALE=1.0 \
    MJPC_HORIZON=0.20 MJPC_TRAJECTORIES=32 \
    MJPC_AUTORUN=1 MJPC_FORCE_LOG=/tmp/fr3_narrow_mlp.csv \
  timeout --signal=TERM 10 ./build/bin/mjpc
```

Adjust task selection command/env if the repository uses a specific task id or task name mechanism.

## Final Report Required

After implementation, write a short summary file, for example:

```text
FR3_NARROW_IMPLEMENTATION_SUMMARY.md
```

Include:

- What task was created.
- Which files were added/modified.
- How the gates are defined.
- How to run MPPI and MLP+MPPI smoke tests.
- Whether build passed.
- Whether smoke tests passed.
- Any observed failure modes.
- What should be swept next.

