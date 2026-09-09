# GCS for the discrete decisions MPPI cannot make

Handoff note. Written after a long session that tested the *other* way of combining
GCS with MPPI and refuted it. Read section 1 before proposing anything, so the dead
ends are not walked again.

Target system: `mujoco_mpc` task `FR3_H_Gripper_PotDual` -- two FR3 arms with H-grippers,
torque-sampling MPPI (`FlowMPPI`, planner 9/14), 166 Hz, horizon 0.4 s. This is the
bimanual + mujoco-mpc extension of the IROS torque-sampling MPPI paper.

---

## 1. What was already tested and refuted

Do not rebuild any of these expecting a different answer.

### 1.1 GCS regions as the MPPI sampling support (`supp(Q) ⊆ F`)

The original research plan. **Refuted in four independent settings.**

| experiment | result |
|---|---|
| 2R planar arm | support restriction collapses (alpha 0.02); corridor freezes the arm |
| single FR3 (7 dof) | 95% of the joint box is already self-collision-free, so support vs penalty is not a distinction |
| bimanual, velocity level | penalty 0 collisions; region methods 2-4x worse task error, ESS 0.04 |
| bimanual, torque level | CBF on certified regions **collided [7,0,4]**; penalty **[0,0,0]**. Task error 0.245 vs 0.166 |

The reason it fails is not tuning. **The regions are not certificates.** `IrisZo` with
`epsilon=0.01` produced regions whose interiors are **7.16% in collision** (worst
region 13.25%), measured by hit-and-run sampling inside each region. A hard constraint
onto a 93%-safe set pushes that 7% straight into the executed trajectory, because
**control has no verification step**:

|  | the region answers | cost of being wrong |
|---|---|---|
| planning | "where should I look for a path?" | re-plan |
| control  | "is this state safe?" | **collision** |

Same regions, both uses, opposite outcomes -- measured: 90/90 GCS *planning* queries
collision-free (each path verified afterwards) vs [7,0,4] collisions under *control*.

The Science Robotics paper says this about its own polytopes:

> "While these polytopes **could be rigorously certified** to be collision free, for the
> experiments reported here we use a fast implementation based on nonconvex
> optimization that **does not provide a rigorous certification**, but that appears to
> be very reliable in practice."

Fine for planning. Not fine as a control invariant.

### 1.2 GCS trajectory as an L2 prior in the cost ("proxpi")

GCS returns `q_d`, MPPI samples torque, so the prior cannot warm-start the controls;
it enters as `w * ||q_rollout - q_d||^2`. Tested on a start/goal pair whose straight
line in joint space collides in 52 of 60 samples and for which GCS finds a free 4.73 s
detour -- i.e. a case where a prior *should* pay.

```
prior alone, best (w=1)     collided [3, 50, 59]   vs no-prior [0, 142, 158]
distance penalty alone      collided [0,  0,  0]
```

- The prior **does** help: 65% fewer collisions on its own.
- It **cannot replace** the collision term. It is a soft bias in a scalar objective,
  with no mechanism forbidding a rollout from entering collision.
- There is a **safety ceiling on the prior weight**: `w <= 1.0` keeps collisions at 0,
  `w = 3.0` breaks it ([2, 9, 17]). The prior competes with the safety term inside the
  same scalar and wins if it is large enough.
- Inside the safe range there is **no measurable task benefit** (6 seeds):

```
distance penalty .02      err 0.3104 +- 0.0073  (sem 0.0030)
prior w=0.1 + dist .02    err 0.3051 +- 0.0182  (sem 0.0074)
                          difference 0.0053, combined SE 0.0080 -> 0.66 sigma
```

The prior also **triples the run-to-run variance**.

### 1.3 The baseline any new method has to beat

Bimanual FR3, torque level, sigma 0.15, 3 seeds, benchmark = drive each TCP at its own
link 1-3 (a condition the user has used for years to provoke self-collision):

```
no penalty            collided [46, 54, 50]   min gap -0.006 .. -0.010   err 0.077
contact penalty .03   collided [ 0,  0,  0]   min gap +0.042 .. +0.048   err 0.179
distance penalty .02  collided [ 0,  0,  0]   min gap +0.038 .. +0.048   err 0.166
```

Zero collisions on every seed with 3-5 cm of real margin. **Self-collision is not the
open problem.** A distance penalty solves it, and no region-based method beat that.

---

## 2. The idea that survives

### 2.1 GCS was not built for collision regions

The foundational paper (Marcucci, Umenberger, Parrilo, Tedrake, *Shortest Paths in
Graphs of Convex Sets*) states its main target directly:

> "Optimal control of discrete-time hybrid dynamical systems is **a main application
> that we target in this paper**"

> "given a finite collection {D_nu} of compact convex subsets of the **state and control
> space**, a PWA system has dynamics s_{tau+1} = A_nu s_tau + B_nu a_tau + c_nu **if
> (s_tau, a_tau) in D_nu**"

The convex sets are **dynamics modes**; a path is a **mode sequence**. Collision-free
motion planning is one instantiation, added later in the Science Robotics paper.

| paper | the convex sets are | a path is |
|---|---|---|
| [Shortest Paths in GCS](https://groups.csail.mit.edu/robotics-center/public_papers/Marcucci21.pdf) | PWA dynamics modes in state-AND-control space | a mode sequence |
| [Motion Planning around Obstacles](https://www.science.org/doi/10.1126/scirobotics.adf7843) | collision-free C-space regions from IRIS | a collision-free trajectory |
| [Towards Tight Convex Relaxations for Contact-Rich Manipulation](https://arxiv.org/abs/2402.10312) (RSS 2024) | **contact modes** -- quasi-static dynamics under a fixed mode | a **contact sequence** |
| [Multi-Query GCS](https://arxiv.org/pdf/2409.19543) | regions + an offline SDP cost-to-go bound | repeated online queries, ~100x faster |

**GCS's binary variables choose which convex set you are in. What the set *means* is a
modelling choice.**

### 2.2 What MPPI is actually bad at

Not self-collision -- a distance penalty handles that for free (1.3). What torque
sampling cannot search is a **combinatorial choice**: when to make and break contact,
which arm holds, how many regrasps, in what order. Gaussian noise on torque explores a
continuous neighbourhood; it cannot jump between discrete mode sequences.

This is worse in the deployed configuration, which uses **DC noise** (one Gaussian per
rollout+joint, held across the whole horizon -- `sampling_dc_noise=1`, with the xml
noting per-knot noise diverges). A rollout is essentially "one constant torque for
0.4 s". That cannot represent "release, move, re-grasp".

### 2.3 Why a mode injects better than a trajectory does

This is the crux, and it is what makes this attempt different from 1.2.

```
q_d    ->  ADDS a term to a scalar cost   ->  competes with every other term
                                              -> weight tuning, safety ceiling
mode   ->  SELECTS which terms are active ->  no competition
```

Under mode `m` the objective is "left hand at contact point p_L on the object, normal
force f"; under `m+1` those terms are off and the right hand's are on. Nothing is
added to a sum, so there is no weight to trade against safety. The measured ceiling
in 1.2 (`w<=1` safe, `w=3` unsafe) simply does not arise.

It also means **the region impurity of 1.1 stops being fatal**: a mode sequence is a
*proposal*, and the MPPI rollout is the verification step that control was missing.

---

## 3. Concrete design

### 3.1 The current task has no decision to make

`Fr3HGripperPotDual` (`mjpc/tasks/Fr3HGripperPotDual/fr3.{h,cc}`) is a cooperative
grasp: both arms take the two handles of one pot and carry it to a target.

```
phase_ : 1 pre-grasp -> 2 approach -> 2.5 close (squeeze_) -> 3 transport -> 4 delivered
```

**Fixed, linear, no branching.** Transitions are threshold+dwell conditions
(`err_pre < enter_tol && v_hand < settle_v` sustained for `enter_dw`), with ~40 `pd_*`
numerics tuning them. GCS chooses a path in a graph; here the graph is a line, so
attaching GCS as-is buys nothing. What is hand-tuned are thresholds and intermediate
waypoints -- continuous parameters, not a combinatorial search.

**The task has to change before GCS has a job.**

### 3.2 Minimal change that creates a real choice

Transport currently freezes the grasp and carries rigidly:

```cpp
mju_mulMatTVec(rel_p_[a], R0, dp, 3, 3);
mju_mulQuat(rel_q_[a], q0c, freeze_q_[a]);
frozen_ = true;
```

**Give the goal a rotation larger than the wrist range allows in one grasp.** The
frozen grasp then cannot reach it and the arms must release and re-grasp. That makes
these genuine decisions:

- how many regrasps
- at which intermediate object orientations to release
- which arm releases first, or both

which is exactly a shortest path in a graph.

Alternative task with the same property, if the pot is inconvenient: move an object
from a pose only the left arm can reach to a pose only the right arm can reach. No
single-grasp solution exists; a handover or place-and-repick is mandatory.

### 3.3 What the convex sets should be

**Work in object space, not the 14-dof C-space.** This is the single most important
design choice.

```
GCS state  =  object pose (SE(2) -> 3 dof, or SE(3) -> 6)  x  mode
mode       =  m0 free,  m1 left holds,  m2 both hold / on table,  m3 right holds
set        =  the object poses reachable while in that mode
edge       =  a regrasp transition; connect modes that share a feasible object pose
cost       =  number of regrasps + travel time
path       =  the mode sequence
```

Three things fall out of the low dimension:

- IRIS in 3-6 dof is far cheaper than the 14 dof used earlier (which took ~42 s per
  region and still produced 7% impure sets)
- impurity is no longer fatal, because the rollout verifies (2.3)
- **object dynamics enter naturally**, which is what the IROS paper's future work asks
  for ("extend this framework to include object dynamics ... in-hand manipulation and
  multi-arm collaborative tasks")

### 3.4 Interface

```
GCS (low rate / offline)                 MPPI (166 Hz, unchanged)
  mode m_k and switch times      ->      which contact cost terms are active
  object pose trajectory x_obj   ->      task cost: drive the object to x_obj
  contact locations p_contact    ->      that hand's target contact point + normal force
```

`C_constraint = C_joint + C_collision` and the force estimate stay as they are. The
mode changes only *which* task terms are switched on.

Implementation point: `phase_` in `fr3.h` is already the mode variable. Today a human
fixed it to `1->2->2.5->3->4`. GCS would emit e.g.
`1->2->2.5->3->2.5->3->...->4` with regrasps inserted. **That substitution is the
contribution.**

### 3.5 Grasp confirmation has to become physical

`phase_ 2.5 -> 3` is currently gated on a **timer**:

```cpp
t_conf_ += dt;                 // "weld가 파지이므로 시간 기반"
if (t_conf_ >= hold_s) { set_weld(true); ... }
```

That is only valid because the weld *is* the grasp. Once modes are real contact states
this must be a contact criterion. MuJoCo gives it directly:

```cpp
mj_contactForce(m, d, i, f6);   // f6[0] = normal, f6[1..2] = tangential
slip_margin_i = mu * f6[0] - hypot(f6[1], f6[2]);
```

Confirm the grasp when every finger<->object contact has margin above a threshold.
The same quantity belongs in the rollout cost:

```
C_slip = w * sum_i max(0, d_margin - slip_margin_i)^2
```

which makes MPPI **decelerate before it slips** rather than discover slip afterwards.
Measured on this model, the grasp holds to about 4.2 g; that cost term is what would
keep the controller inside it. The CUDA version could not do this (no contact sensing)
and a weld makes it meaningless.

### 3.6 On `F_ee = J̄ᵀτ`

The IROS paper used it because CUDA-only rollouts had no way to measure interaction
force. Two notes for the mujoco-mpc port:

- `J̄ᵀτ` is the **commanded** task wrench and mixes the arm's own inertia with what is
  transmitted. The external part needs `tau_ext = tau - (M qddot + C qdot + g)` first.
- MPPI's cost is evaluated **inside rollouts, which are simulations even on hardware**.
  Using MuJoCo contact forces there introduces no sim2real gap in the controller
  structure. Only the *current-state* force estimate needs an observer or F/T sensor.

So the paper's formula keeps its job (state estimate) and contact forces take the
prediction job.

---

## 4. Implementation order, with gates

Each step has a falsifiable gate. **Do not proceed past a failed gate by tuning.**

1. **Apply the model fixes** (section 5). Without them the grasp cannot be modelled at
   all and every downstream number is meaningless.

2. **Gate A -- make the baseline fail.** Set the goal rotation beyond one-grasp reach
   and run the existing phase machine. *If it succeeds, the task is not hard enough;
   enlarge the rotation.* Record how it fails.

3. **Gate B -- hand-code one regrasp sequence** and run it. *If MPPI cannot execute a
   sequence a human wrote, the problem is not the mode sequence and GCS will not help.*
   Fix that first. This is the upper bound GCS is measured against.

4. **Build the object-space GCS** (3.3). Verify offline that the returned mode sequence
   matches the hand-coded one on the case from step 3.

5. **Gate C -- let GCS drive `phase_`.** Compare against step 3 on success rate, number
   of regrasps, and time. *Only here is there a contribution.*

6. Only then vary: more object poses, obstacles that force different sequences,
   friction instead of weld.

---

## 5. Model fixes to apply first (verified, independent of everything above)

Measured on `Fr3HGripperPotDual`; originals kept as `.bak`, fixed as `.fixed`.

**`condim` appears nowhere in the xml, so every contact defaulted to 3: sliding
friction only, no torsional.** A point contact on a cylindrical handle then has zero
resistance to spin about the handle axis.

| configuration | holds |
|---|---|
| as shipped (condim 3, 10 N) | **0.5 N** -- cannot lift a 10 N pot at all |
| condim 3, 30 N | 2.4 N -- more grip force does not help |
| condim 4, 10 N | 23.1 N |
| **condim 4, 30 N (real gripper force)** | **102-285 N static, 4.2 g in a carry** |

Sliding friction is not the limit: mu 1.0 vs 2.0 changes nothing.

**The 48 `group="1"` display meshes on the H-gripper omit `contype`/`conaffinity`, so
MuJoCo's default of 1 made all of them collidable** -- 342k vertices duplicating the
6.7 mm proxy spheres that are the intended collision model.

| | us/mj_step while grasping | carry rotation |
|---|---|---|
| meshes collidable | 246.8 | 2.6-50 deg |
| meshes disabled | **102.7 (2.4x)** | **0.2-0.4 deg** |

Faster *and* firmer -- 36 contacts fighting each other became 18. In the running task
this moved the real-time factor from 0.36 to 0.82.

Applied: `condim="4"` on 46 finger geoms + 10 pot geoms, `forcerange="-30 30"`,
`contype="0" conaffinity="0"` on the 48 display meshes.

**The weld can go.** `set_weld()` is gated on `MJPC_NO_WELD=1`. It is not cheaper
(98.2 vs 98.0 us/mj_step), and the rollout *can* represent a friction grasp: mjpc sets
the planning model's timestep to `agent_timestep` (`agent.cc:334`) so rollouts step at
0.03 s, and with u=0 (gravcomp holds the arm) the pot drifts 2.9 mm over a 0.4 s
horizon at that step, versus 0.5-2.8 mm for the weld. The weld force readout lives
only inside the `MJPC_POTD_DBG` block, so nothing in the cost or FSM depends on it.

---

## 6. Pitfalls that cost time in this session

Every one of these produced a confident wrong conclusion before being caught.

- **Match the softmax temperature to the cost scale.** `lambda=1000` on an O(10) cost
  makes every weight equal; the update becomes the mean of zero-mean noise and the
  controller learns nothing while *looking* busy (large joint travel). Use mjpc's
  min-max normalisation (`MJPC_MINMAX_NORM=1`) or size lambda to the cost.
- **Poll contacts every physics substep.** One control step is 30 of them. Polling only
  the last reported **0** self-collisions where there were 34.
- **A safety metric of 0 is ambiguous** -- "prevented" and "not observed" look
  identical. Verify the detector can see the event before celebrating a zero. Worse,
  if the cost and the metric share code, the blind spot appears in both and they
  confirm each other.
- **Never conclude from n=1.** A left/right asymmetry that looked systematic was noise;
  so was a "steady-state offset".
- **Do not hand-edit generated files.** `dual_build.build()` rewrites its output on
  every import; a `gravcomp` fix applied to the output vanished silently and every
  torque number after it was taken without gravity compensation.
- **Verify simulator options empirically.** `geom_margin` was assumed not to change the
  dynamics if `gap == margin`; it does (u=0 drifted 0.40 rad in 1 s). Sense distance in
  a *separate* model.
- **Read the code before modelling the compute budget.** A budget was computed assuming
  30 physics substeps per rollout step; `agent.cc:334` sets the planning model's
  timestep to the control period, so it is 1. The estimate was off 30x.
- **A conclusion that kills an idea needs the same evidence as one that saves it.** The
  "penalty is enough, drop the certificate" call was made on undersampled contacts.

## 7. Open issue on the current checkout

The pot task never leaves `phase_ 1` on the machine used here (M1 Pro, 8 cores, RTF
0.82). Each arm individually reaches the 15 mm pre-grasp gate -- left held 5.8-12.2 mm
for ~5 s in one run, right reached 34 mm in another -- but **never both at the same
time**, and which arm succeeds varies run to run. `corr(errL, errR) = +0.46`, so they
are not fighting each other.

Ruled out: the section 5 model fixes (the original xml fails identically), the arm64
FPCR port, weld vs no-weld (the weld never activates in phase 1), planner 9 vs 14,
planner threads 2/4/8, `lambda` 1000 vs 300, adaptive sigma on/off (off is worse -- the
arms stop moving entirely), K = 64/128/256, `perarm_res_off` (verified: L is residual
0..40, R is 41..81, matching the xml), reachability (both targets solve to 0.1 mm IK
residual with no joint at a limit), mocap propagation into rollouts, source/build xml
sync, and compute rate.

This reproduces on the machine here but is reported to work elsewhere, so treat it as
environment-specific and resolve it before starting section 4.

`fr3.cc`'s `MJPC_POTD_DBG` line was extended with `ncon=` and the left arm's 7 joint
angles; revert if unwanted.
