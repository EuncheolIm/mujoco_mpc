# FR3_H_Gripper_Carry on the NUC13 — what to change, what to run, why

Handoff note for whoever runs this task on the NUC13. Everything below is
measured on a 20-core desktop unless stated otherwise; the NUC numbers that are
known are called out explicitly.

Repo state assumed on the NUC: `origin/fr3-hgripper-tasks` (what is pushed).
All cost terms and the multi-target episode code are ALREADY in that branch
(`CostCarryVel`, `CostGraspAlign`, `CostObjectOri`, `CostNullspaceVel`,
`CostObjectVel`, the gated `CostControl`, and `Fr3HGripperCarry/fr3.cc`'s
spawn/deliver/respawn loop). Nothing needs porting except the values in §1.

---

## 1. Difference vs the pushed branch — three values in one file

`mjpc/tasks/Fr3HGripperCarry/task.xml`

| numeric / term | pushed | use this | why (measured) |
|---|---|---|---|
| `agent_timestep` | 0.01 | **0.02** | rollout cost is H/dt: 22 steps -> 11, i.e. half the compute. Latency ratio (replan/dt) 1.02 -> 0.30 at 4 threads. Deliveries/90 s went UP: 3.75 -> 7.00 |
| `Object_tgt` weight | 1e6 | **3e6** | cycle 13.65 s -> 5.30 s; the carry+settle part 11.99 s -> 4.32 s |
| `Carry_vel` weight | 2e4 | **5e4** | pairs with the above. ALONE it is catastrophic (0/6 seeds delivered, object stalls 47.8 mm out) because it damps the transport too; together the pair is the best measured config |

task.xml only, so **no rebuild** is needed - just run with
`MJPC_TASKS_DIR=$PWD/mjpc/tasks` so the source xml is read instead of the
build copy.

```bash
X=mjpc/tasks/Fr3HGripperCarry/task.xml
sed -i 's|<numeric name="agent_timestep" data="0.01"/>|<numeric name="agent_timestep" data="0.02"/>|' $X
sed -i 's|user="2 1000000 0 5000000  0.01"|user="2 3000000 0 50000000 0.01"|' $X   # Object_tgt
sed -i 's|user="2 20000   0 5000000  0.01"|user="2 50000   0 5000000  0.01"|' $X   # Carry_vel
```

Optional (needs a rebuild): `mjpc/app.cc` on the desktop adds `MJPC_PLAN_LOG`,
which prints the compute cost every N seconds. Strongly recommended on the NUC -
see §4. Ask for that commit if it is not in the branch yet.

---

## 2. Compute budget — the reason this note exists

Two independent costs:

| | desktop (20 cores) | NUC13 |
|---|---|---|
| MPPI replan, K=32, dt 0.02, 4 threads | 6.0 ms | measure it |
| MPPI replan, K=32, dt 0.02, GUI (~18 threads) | 2.0-3.1 ms | measure it |
| FM prior inference (ODE 20, ONNX 1 thread) | 7.5-8.9 ms | **~69-70 ms** |

The prior runs on its OWN thread, so it does not enter the replan time; it costs
a core and it makes the injected reference STALE. With `fm_chunk_dt = 0.02` the
chunk is consumed in one control period, so on the NUC the prior is roughly
3-4 periods behind.

What follows from that:
- **K is the compute knob, `dt` is the free win.** K 32 -> 64 doubles the replan
  (6.0 -> 11.2 ms). dt 0.01 -> 0.02 HALVES it and improves throughput. Do dt
  first, and do not raise K on the NUC.
- dt 0.03 is a cliff, not a continuation: 0 deliveries in 4/4 seeds at both
  K=32 and K=64. A 30 ms control period cannot place the object inside 5 mm.

Deliveries per 90 s at 4 threads (4 seeds each), for reference:

| dt / K | deliveries | replan | ratio | closest |
|---|---|---|---|---|
| 0.01 / 32 (pushed) | 3.75 | 10.2 ms | 1.02 | 1.4 mm |
| **0.02 / 32** | **7.00** | **6.0 ms** | **0.30** | 0.7 mm |
| 0.02 / 64 | 15.00 | 11.2 ms | 0.56 | 1.0 mm |
| 0.01 / 64 | 26.50 | 20.8 ms | 2.08 | 0.8 mm |
| 0.03 / 32 or 64 | 0.00 | 4.4 / 8.2 ms | 0.15 / 0.27 | 3.3-5.4 mm |

---

## 3. Running WITHOUT the prior

`fm_mode: cost` means the prior is injected ONLY through the `FM_track` cost
residual - the warm-start path is disabled (`N_fm = 0`, verified in
`FlowMPPIRpy/planner.cc:346,379`). So there are two different "off"s:

**(a) stop injecting, keep inferring** - the FM thread still burns a core:
```bash
MJPC_FM_TRACK_SCALE=0
```

**(b) do not load the model at all** - no FM thread, no core cost. There is NO
env for this: `MJPC_FM_CKPT` is read as `e && e[0]`, so an empty value is
treated as "unset" and the yaml wins. Copy the yaml and blank it:
```bash
cp mjpc/tasks/Fr3HGripperCarry/fm_config.yaml /tmp/fm_carry_noprior.yaml
sed -i 's|^fm_checkpoint:.*|fm_checkpoint:|; s|^fm_stats:.*|fm_stats:|' /tmp/fm_carry_noprior.yaml
```
Confirm it worked by the ABSENCE of `[ONNXPolicy]` / `[FM Timing]` lines.

Is the prior worth 70 ms on the NUC? The measurements do not justify it:

| dt | prior | deliveries/90 s | closest | \|v_obj\| |
|---|---|---|---|---|
| 0.01 | ON | 5.50 (6 seeds) | 0.7 mm | 0.030 |
| 0.01 | OFF | 6.00 | 1.2 mm | 0.084 |
| 0.02 | ON | 6.25 (4 seeds) | 1.1 mm | 0.075 |
| 0.02 | OFF | 4.75 | 3.3 mm | 0.048 |

Throughput is indistinguishable (the per-seed spreads overlap completely). The
prior helps ACCURACY (closest distance) at both dt. Its effect on object
smoothness FLIPS SIGN with dt - it damps at 0.01 and adds motion at 0.02 - which
is consistent with the chunk/period alignment changing. At 70 ms of staleness on
the NUC there is no measured reason to expect the accuracy benefit to survive.
**Start prior-OFF on the NUC**, and only re-enable it if the delivery accuracy
is visibly worse there.

If the prior IS wanted on the NUC, cut its cost first:
- `MJPC_FM_ODE_STEPS` (20 -> 8 or 4). Inference is ~linear in ODE steps:
  0.44 ms/step here, so 8 steps ~= 28 ms on the NUC, 4 steps ~= 14 ms. NOTE: ODE
  reduction was abandoned on a different task/checkpoint once (FR3 wipe, an early
  631 N collision), so re-verify grasp quality if you use it.
- `onnx_policy.cc:26` hardcodes `SetIntraOpNumThreads(1)`. Raising it is untested
  and would compete with the rollout threads, but on the NUC it is the obvious
  first thing to try.
- There is NO distilled MLP student for the H-gripper (`guide_type: mlp` exists,
  the checkpoint does not).

---

## 4. Commands

```bash
# prior OFF (recommended starting point on the NUC)
cp mjpc/tasks/Fr3HGripperCarry/fm_config.yaml /tmp/fm_carry_noprior.yaml
sed -i 's|^fm_checkpoint:.*|fm_checkpoint:|; s|^fm_stats:.*|fm_stats:|' /tmp/fm_carry_noprior.yaml

MJPC_TASKS_DIR=$PWD/mjpc/tasks \
MJPC_FM_CONFIG=/tmp/fm_carry_noprior.yaml \
MJPC_CARRY_MULTI=1 MJPC_PLAN_LOG=2 \
build/bin/mjpc --task=FR3_H_Gripper_Carry

# prior ON
MJPC_TASKS_DIR=$PWD/mjpc/tasks \
MJPC_FM_CONFIG=$PWD/mjpc/tasks/Fr3HGripperCarry/fm_config.yaml \
MJPC_CARRY_MULTI=1 MJPC_PLAN_LOG=2 \
build/bin/mjpc --task=FR3_H_Gripper_Carry
```

`fm_config.yaml` holds an ABSOLUTE `fm_checkpoint` path
(`/home/kkomji/Euncheol/...`) - fix it for the NUC's layout if the prior is used.

`fm_config.yaml` also carries K, H, knots and lambda
(`trajectories: 32`, `horizon: 0.225`, `knots: 30`, `lambda: 1000`), and env
wins over it (`MJPC_TRAJECTORIES`, `MJPC_HORIZON`, `MJPC_LAMBDA`).

### What to read in the output
```
[COMPUTE t=  4.00] plan   6.12 ms  (dt 20 ms, ratio 0.31)   fm   7.68 ms
[CARRY-MULTI] delivery 3 at t=21.91  cycle=7.67s (reach+grasp 5.78s, carry+settle 1.88s)
```
- `ratio` must stay **below 1**. Above 1 the arm is executing a stale plan; that
  is what broke tracking on the Reach and Dual tasks in this repo.
- `cycle` is the delivery period. reach+grasp is normally 1-2 s (8-13%);
  carry+settle is the rest, so that is where any remaining time goes.
- A delivery requires: object within `carry_success_tol` (5 mm) of the target
  AND the gripper closed AND held for `carry_success_dwell` (0.3 s).

---

## 5. Axes already tried and rejected (do not spend NUC time on these)

| axis | result |
|---|---|
| pad friction 1.0 -> 2.0 / 3.0 | worse at 6 seeds (1.67 deliveries vs 3.00); a 3-seed win did not reproduce |
| `carry_ureg_hi` 1e4 -> 3e4 | worse at 6 seeds (1.00-1.50 vs 3.00) |
| `Nullspace_vel` (weight 5000) | no benefit; settle amplitude got worse on one seed |
| `Object_vel` term | penalising the object's own velocity blocks the GRASP entirely |
| adaptive sigma | recorded oscillation: object to 0.6 mm, sigma to 0.05, object slips 30 mm, sigma snaps to 1.0 |
| horizon 0.225 -> 0.35 | deliveries halve; replan 14-26 ms |
| dt 0.03 | 0 deliveries at any K |

## 6. Measurement discipline (learned the hard way)

- This task's run-to-run spread routinely covers the whole effect size. Two
  configs were promoted on 3 seeds and both failed at 6. **Use >= 6 seeds** and
  print the per-seed numbers, not just the mean.
- `GetNumberOrDefault(env_value, model, name)` lets the xml numeric SHADOW the
  env. One sweep silently ran at 30 mm while claiming 5 mm because of this.
  Check which one wins before trusting a sweep.
- Keep the CPU budget at `nproc/2` (concurrent runs x threads) and `nice -n 15`.

---

## 7. "Same code, different behaviour" — how to tell CPU from everything else

MuJoCo physics is deterministic: a slower CPU does not change the result for a
given control sequence. CPU speed can only change behaviour through
(a) the GUI's real-time plan rate, (b) FM prior staleness, (c) thread scheduling.

Two facts to keep in mind before blaming the CPU:

**This task is NOT reproducible even on one machine.** Measured here: same seed
(`MJPC_SEED=7`), prior OFF, headless, 5 threads, two consecutive runs gave
4 deliveries / 240 mm and 0 deliveries / 421 mm. A third run at 2 threads gave
1 delivery. Multi-threaded rollout reduction makes the fixed seed insufficient.
So a single NUC run vs a single desktop run proves nothing - compare
DISTRIBUTIONS over >= 6 seeds.

**A CONSISTENT qualitative difference, on the other hand, is systematic**, and
the likely causes are not all compute:

| cause | how to check |
|---|---|
| the xml actually loaded differs | compare the planner banner (K / H / lambda / alpha / weight list) printed at startup on both machines |
| `MJPC_TASKS_DIR` unset -> stale `build/mjpc/tasks/` copy is read | check the variable is exported; the repo has been bitten by this before |
| MuJoCo / onnxruntime version, Debug vs Release, `-mavx` | compare build type and library versions |
| real-time plan rate (this one IS compute) | `MJPC_PLAN_LOG=2` -> `plan` ms and `ratio` |

### Decisive split
The headless harness plans on a FIXED schedule (`i % steps_per_plan`), so it has
no real-time component:

```bash
for s in 0 1 2 3 4 5; do
  MJPC_TASKS_DIR=$PWD/mjpc/tasks \
  MJPC_FM_CONFIG=/tmp/fm_carry_noprior.yaml \
  MJPC_THREADS=4 MJPC_SEED=$s MJPC_EVAL_TASK=FR3_H_Gripper_Carry \
  MJPC_SETTLE_KEEP_GOAL=1 MJPC_CARRY_MULTI=1 MJPC_CARRY_MULTI_LOG=1 \
  build/bin/hgripper_settle_eval 90 7 2>&1 | grep -c delivery
done
```
- distributions match across machines, GUI differs -> it is the plan rate (CPU)
- distributions differ systematically -> it is config / version / build, NOT CPU

Desktop reference for that exact command (dt 0.02, K 32, prior OFF, 4 seeds):
4.75 deliveries per 90 s on average, per-seed 11 / 1 / 5 / 2.
