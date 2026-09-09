# MLP student guide — export contract

`guide_type: mlp` is already wired (`mjpc/policies/mlp_policy.{h,cc}`, used from
`FlowMPPIRpy/planner.cc:1723`). A student that satisfies the contract below drops in
with **no C++ changes**. Everything here was read out of the loader, not assumed.

## Why the MLP path differs from FM (the reason to bother)

| | FM (`guide_type: fm`) | MLP student (`guide_type: mlp`) |
|---|---|---|
| call site | async thread; planner takes the last ready chunk | **synchronous, inline in the planner** |
| cost per query | 20 sequential ORT Runs | 1 ORT Run |
| staleness | 10-40 ms depending on core placement | **zero — the chunk is for the current state** |

Measured on this NUC (i7-1360P, 12 planner threads, K=32): FM refreshes at 30 Hz with the
default core placement and 94 Hz with cores reserved, against a 50 Hz replan rate. A
synchronous student removes the question entirely: ~0.8 ms per forward against a plan
budget of 20 ms, where plan itself measures 9-13 ms.

## Inputs — matched by POSITION, not by name

The loader reads whatever names the model declares (`GetInputNameAllocated`), so name them
freely. Only the **count** and the shapes matter, and only 2 or 4 are accepted:

    4 inputs:  state(1,17)  prev_state(1,17)  prev_action(1,7)  goal(1,6)
    2 inputs:  state(1,17)  goal(1,6)                            <- needs_history_=false

Any other count refuses to load with "expected 2 or 4".

`goal` is **hardcoded to 6** in the loader; it is not read from the stats file.

Since the current stats carry `drop_history=1`, the C++ side feeds prev_state/prev_action
as **zeros** even in the 4-input form. So export the **2-input** student and skip them.

## Output — exactly one

    (1, H*action_dim)  or  (1, H, action_dim)

`H` is inferred as `out_total / action_dim`, so `action_dim` must divide the flattened
size. Match `H=10` to stay aligned with `fm_chunk_dt: 0.02` (10 x 0.02 = 0.2 s, against
the task horizon 0.225); a different H still loads but shifts the time alignment that
`fm_step_indexed` lookup assumes.

## Normalization — the C++ side normalizes in and denormalizes out

Inputs are normalized before the Run and the output is denormalized as
`y * action_std + action_mean`. **Train in normalized space with these same stats.**

`normalization_stats.npz` must contain:

| key | dtype | shape | note |
|---|---|---|---|
| `state_mean` / `state_std` | float64 | (17,) | |
| `goal_mean` / `goal_std` | float64 | (6,) | always 6 |
| `action_mean` / `action_std` | float64 | (7,) | |
| `state_dim` | int64 | (1,) | optional; absent -> `include_ee ? 17 : 14` |
| `action_dim` | int64 | (1,) | optional |
| `include_ee`, `relative_goal`, `drop_history`, `action_type` | int64 | (1,) | optional flags |

Values in the existing flow checkpoint (`Fr3HGripper/checkpoints/flow_h_grip/`), i.e. the
numbers to match:

    state_dim 17    action_dim 7    goal_dim 6    horizon 10
    action_type 2   drop_history 1  include_ee 1  subsample 20

`model_info.npz` is optional for the MLP path (the loader infers H from the output shape),
but keeping it costs nothing.

## Config

```yaml
guide_type: mlp
mlp_checkpoint: mjpc/tasks/Fr3HGripper/checkpoints/<name>/student.onnx
mlp_stats:      mjpc/tasks/Fr3HGripper/checkpoints/<name>/normalization_stats.npz
fm_track_scale: 1.0      # MUST be > 0 -- see below
```

Env `MJPC_MLP_CKPT` / `MJPC_MLP_STATS` take precedence over the yaml.

## Three traps

1. **`fm_track_scale: 0.0` silently disables the guide** regardless of `guide_type`.
   `CostFMTrack` returns all-zero residual when the scale is 0, so the student loads,
   runs, and changes nothing. `Fr3HGripperReach/fm_config.yaml` has `0.0` and an empty
   `fm_checkpoint` — feeding that config to Carry runs vanilla MPPI at Reach's K=128 and
   H=0.4. That mistake has already been made once.

2. **The pinning env vars become inert.** `MJPC_FM_CPUS`, `MJPC_FM_INTRAOP`,
   `MJPC_FM_INTRAOP_AFFINITY`, `fm_ode_steps` all belong to the FM thread, which is never
   created under `guide_type: mlp`. Drop them from the launch line. `MJPC_PLANNER_THREADS`
   still applies (the GUI reads that, **not** `MJPC_THREADS`).

3. **`action_dim = 7`, so the guide does not cover the gripper.** `CostFMTrack` is
   `residual = SCALE * (qpos[0..6] - q_fm_target)` — arm joints only. `ctrl[7]` is left
   entirely to MPPI's own sampling. No student, however good, can address the release
   failure (the gripper opening mid-carry); that needs the gripper taken out of MPPI's
   hands. See the `Grip_hold` note in `task.xml` for the two cost-based attempts that
   failed.

## Verifying it actually loaded

    [MLPGuide] Loaded MLP student: inputs=2 (needs_history=0) state_dim=17 action_dim=7 horizon=10
    [MLPGuide] Input names : ...
    [MLPGuide] Output names: ...

No `[MLPGuide]` line means the guide is not running. A `[FM Timing]` line means
`guide_type` is still `fm`.
