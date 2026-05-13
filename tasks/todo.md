# Force-control demo materials — 4 scenarios on force-control-table-wip

## Goal
Visual + plot evidence for the force-control timeline that preceded
t-bar push. Each scenario must be runnable in GUI (video) and produce a
CSV (plot).

Branch: `force-control-table-wip` (just checked out).
Backup of t-bar work: `t-bar-push-wip` branch.

---

## Current code summary (force-control-table-wip)

`task.xml`:
- panda + **table** at (0.55, 0.10, 0.10) box 0.4×1.6×0.4
- hand_copy mocap @ (0.5, 0.3, 0.5) quat (0,1,0,0) — descent target
- F_des = (0, 0, -10) numeric
- approach_time 2.0, hybrid_switch_dist 0.02, _angle 0.087, _delay 0.5
- MPPI: horizon 0.12, samples 128, knots 40 zero-spline, σ {2,2,2,2,1,1,1},
  λ=0.1, dc_noise=1, exploration 1.0
- User costs: Reach_pos 1e6, Reach_ori 5e4, joint_cent 1000, joint_vel 500,
  EE_Force 30, u_reg 0.01

`cost_fn.cc`:
- CostPosition: hybrid → z=0 (xy only); approach → full 3D
- CostOrientation: full 3-axis quat error
- CostJointCentralize: null-space projected (q − q_mid)
- CostJointVelocity: smooth hinge at |qdot| > 1.0 rad/s
- **CostForce: uses `hand_force` sensor, hybrid only, z component only.
  In free space hand_force = 0, so cost = F_des = -10 (constant penalty).**
- CostControl: tau (weight 0.01)

`fr3.cc`:
- Auto-trajectory mocap lerp from home EE to xml mocap pos over approach_time
- After approach + reach + delay → `userdata[3] = 1` (hybrid on)
- stderr/CSV log: F_sensor (hand_force) and F_task_z (J#^T·(τ−qfrc_bias))

`hand_force` sensor = MuJoCo `force` sensor on `hand_site` — measures
constraint forces at the site's parent body. Not the same as a touch
sensor at the probe sphere; hand_force in EE frame is gravity-biased.

---

## Scenario plan (sequential)

### S1 — Free-space hybrid: position xy + force z (no contact)

What it shows: in free space, hybrid mode is "harmless". Position xy
holds home, z is delegated to force cost. F_sensor=0 (no contact), so
the cost cannot drive the EE. F_task_z (intent) and F_sensor diverge —
just plot side by side.

Changes from current xml:
- **Remove `<body name="table">`** from worldbody.
- That's it. Everything else (mocap, MPPI, costs) stays.

Expected:
- approach: EE moves from home keyframe pose to mocap pos (0.5, 0.3, 0.5).
- t > approach + delay: hybrid on. CostPosition switches to xy-only.
- Free space → F_sensor = 0 stays. F_task_z fluctuates around the value
  the controller commands.
- Plot: F_sensor.z ≈ 0, F_task_z ≈ noisy. xy position locked.

Save as `task_s1_free.xml`. Run via `mjpc --task=MPPI_Force` (after
swapping the active xml).

Success criterion: video shows EE descended to mocap, hovering, no
contact. CSV shows hybrid switch at ~2.5s, F_sensor z stays ≈ 0,
F_task_z fluctuating.

### S2 — Table present, same `hand_force` cost: force tracking fails

What it shows: putting a table back in (so the EE actually contacts
something) but using the **same wrong force sensor** (hand_force on
hand_site, which reads constraint forces in EE frame, gravity-biased)
gives unreliable tracking.

Changes from S1:
- Add the table body back.
- Lower mocap z so the probe ends up touching the table top
  (table top z = 0.30; mocap z = 0.40 puts hand_site z ≈ 0.30).
- Adjust hybrid_switch_dist for the new mocap.

Expected:
- EE descends, contacts table.
- F_sensor reads contact reaction projected to EE frame, with gravity
  bias mixed in. Noisy / wrong sign / not tracking F_des = −10.

Save as `task_s2_table.xml`. Decision point: leave CostForce
unchanged (uses hand_force) — that's the whole point of the demo.

Success criterion: video shows EE pressing on table; CSV shows F_sensor
oscillating, not converging to −10.

### S3 — Add touch sensor on probe sphere; track that

What it shows: switch the cost to read a `touch` sensor at a small
probe sphere added to the panda hand. Now the cost measures actual
contact normal force; tracking becomes clean.

Code changes:
- Patch `panda_modified.xml` to add a probe sphere geom + site at the
  hand tip (this already exists on the standard-mjpc-sampling branch as
  `probe_sphere` / `probe_site`; cherry-pick or recreate).
- Add `<touch name="probe_touch" site="probe_site"/>` to the sensor list.
- Change CostForce to read `probe_touch` instead of `hand_force` (1-D
  scalar: residual = F_des_z − F_touch). Touch sensors return scalar
  ≥ 0; sign convention: F_touch = |reaction|. F_des = −10 means we want
  10 N of pushing, so residual = 10 − F_touch.

Save as `task_s3_touch.xml`.

Success criterion: F_touch settles at ~10 N, EE stable on table.

### S4 — Stage 4-A wiping (limits demo)

What it shows: extend S3 with a circle xy mocap motion under the same
hybrid mode. EE slides on table while pressing. Friction binarity +
contact flicker + sensor flicker show up — smooth low-magnitude press
while sliding does not happen reliably.

Code changes (from S3):
- Add circle xy mocap mode (already in standard-mjpc-sampling
  fr3.cc — `target_circle_enable`, contact_z, descent_t, etc).
- Cherry-pick that block of TransitionLocked back to this branch, or
  reimplement minimally.
- Tune contact_z so press is light (~3–10 N).

Save as `task_s4_wiping.xml`.

Success criterion: video shows the rapid wiping motion + the failure
modes (jackhammer / stick-slip / force flicker). CSV shows force trace
jagged with bursts.

---

## Implementation order

1. **S1 first** (smallest delta from current xml — just remove table).
   Build, run GUI, save video, save CSV. Confirm with user.
2. **S2** (add table back, lower mocap). Build, run, save.
3. **S3** (add probe sphere + touch sensor + cost rewire). Build, run,
   save.
4. **S4** (port circle mocap + wiping cost stack). Build, run, save.

Common infrastructure:
- Each scenario gets its own `task_sN_*.xml` under `mjpc/tasks/Fr3/`.
- A small switch (e.g. `XmlPath()` argument or a separate Task subclass
  per scenario) so the demo selector in mjpc UI can pick. Simplest: use
  separate task class per scenario, all sharing the same FR3 ResidualFn.
- CSV log filename per run via `MJPC_FORCE_LOG`.
- Python plot script per scenario (existing `plot_cost_compare.py` may
  help).

## Open decisions

1. Single class FR3 + dynamic XML switch, vs four separate task classes
   (FR3_S1 .. FR3_S4)? **Default**: separate classes — clean, no flag
   plumbing. Each subclass overrides only `Name()` and `XmlPath()`.

2. S1 — do we keep CostForce active (with `hand_force` sensor reading 0)
   or set EE_Force weight to 0? **Default**: keep weight 30 so the
   user sees the cost is "trying" but has no signal in free space.

3. S4 details (circle radius, period, contact_z) — defer; tune in GUI.

## Review (TBD)
TBD
