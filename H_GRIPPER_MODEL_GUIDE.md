# H-gripper model variants — what was simplified, why, and what it cost

The Hyundai 3-finger gripper ships as a 60-body subtree. On the NUC that made `mj_step`
the bottleneck, so two reductions were made. Both are exact in the quantities that
matter and both were verified rather than assumed; this file records the numbers, the
method, and the three traps that were paid for on the way.

Every figure below is reproducible with `scripts/verify_gripper_models.py`.

---

## 1. The variants

| model | used by | nbody | nmesh | nmeshface | ngeom | nv | nu |
|---|---|---|---|---|---|---|---|
| `Fr3HGripperCarry/fr3_H_gripper.xml` | (original, kept for reference) | 69 | 89 | **718,149** | 142 | 16 | 8 |
| `Fr3HGripperReach/fr3_H_gripper_rigid.xml` | Reach | **9** | 28 | 232,298 | 54 | 7 | 7 |
| `Fr3HGripperCarry/fr3_H_gripper_lite.xml` | Carry | **13** | 29 | 251,958 | 79 | 16 | 8 |
| `Fr3HGripperPick/fr3_H_gripper_pick.xml` | Pick | 13 | 29 | 251,958 | 79 | 16 | 8 |

`mj_step`, 2000 steps after a 200-step warm-up:

```
original  (69 bodies, 718k faces)    58.10 us
pick      (13 bodies, 252k faces)    19.42 us      2.99x
```

Three runs of the same check gave 2.77×, 2.81× and 2.99×; the absolute microseconds
move with machine load, the ratio does not. Compare ratios, and re-run rather than
trusting the number printed here.

**rigid** collapses the whole gripper to one body — the fingers are welded, so `nv = 7`
and the 7-DOF null-space helpers in `Fr3Reach/cost_fn.cc` are valid. Reach never
grasps, so it loses nothing.

**lite** collapses to four bodies — palm plus three fingers, each keeping its slide
joint — so grasping still works. `nv = 16` (7 arm + 3 slides + 6 free box), which is
why Carry and Pick must use the `fr3hgrip::` cost functions and not the `fr3reach::`
ones: `fr3reach::CostJointCentralize` goes through `GetHandManipulatorJacobian`, which
returns without writing anything unless `nv == 7`.

---

## 2. Reduction one — collapsing bodies

Each retained body gets the composite mass, centre of mass and inertia of the subtree
it replaced, so the dynamics are unchanged rather than approximated.

`rigid`, the whole gripper as one body:

```xml
<inertial pos="-0.009863237 -0.000073661 0.000882385" mass="1.467288630"
          quat="0.703430392 -0.694244096 0.099431435 0.115430534"
          diaginertia="3.022759965e-03 5.677399593e-03 7.220569646e-03"/>
```

`lite`, four bodies:

| body | mass (kg) | com (m) |
|---|---|---|
| `hand` | 1.103393800 | −0.0035758, 0.0003736, −0.0180203 |
| `finger_A_slide_link` | 0.117427410 | 0.0823484, −0.0001514, 0.0583951 |
| `finger_B_slide_link` | 0.123233710 | −0.0597930, −0.0020421, 0.0581053 |
| `finger_C_slide_link` | 0.123233710 | −0.0597902, −0.0020411, 0.0581053 |

### Verified

All three are asserted by `scripts/verify_gripper_models.py`, which is the source of
truth; the figures below are what it reported last.

- **total mass identical**: 17.462102 kg both ways, differing by `3.55e-15` kg — float
  noise, not an approximation.
- **gravity torque** at the home pose: max difference **4.0e-07 N·m** across the 7 arm
  joints. This is the check that matters for hardware, because the arm bridge sends
  `qfrc_bias`-corrected torque and an error here would be a standing offset on the real
  robot.
- **pad world positions** match at fully open **and** at half-closed, to
  **4.1e-07 m** — so the retained slide joints move the pads where the full chain did,
  not merely at one configuration. Checking a single slide value would have proved
  nothing about the joint.

Collision primitives (spheres and boxes) were kept verbatim. The 24 *mesh* collision
geoms were dropped: at this scale a mesh-mesh contact costs far more than the box that
replaces it, and the boxes were already in the model.

> An attempt to fit a capsule shell instead gave a 62 mm radius, because it summed
> `geom_rbound` — the bounding-sphere radius of an elongated mesh — with a
> perpendicular offset. That was discarded. The model's own primitives needed no
> fitting at all.

---

## 3. Reduction two — deleting unreferenced mesh assets

**`mj_collision` cost scales with `nmeshface` even for meshes no geom references.**
That is the non-obvious part and it is where most of the saving came from.

Controlled experiment — same `ngeom`, only the declared mesh assets differ:

```
232k faces  ->  12.2 us
698k faces  ->  35.9 us
```

So 60 mesh assets that nothing pointed at were costing roughly 3× the collision time.
They are declared by the vendor model for the visual subtree that the collapse removed.

**Consequence for anyone editing these files:** deleting a `<body>` is not enough. The
`<asset><mesh>` entries it used have to go too, or the compute stays. Conversely,
adding a mesh asset "just in case" is not free.

---

## 4. Trap one — the hand is yawed −90°, and the jaws separate along Y

```xml
<body name="hand" childclass="H_gripper" pos="0 0 0.220" gravcomp="1" euler="0 0 -1.5708">
```

`hand_site` is attached to `fr3_link7`; the `hand` body sits 90° around Z from it. So
**in the `hand_site` frame the jaws open along Y, not X.** Measured on the lite model:

```
gripper_pad_1   x=+0.0000  y=-0.0541  z=+0.0530      (single-finger side)
gripper_pad_2   x=+0.0321  y=+0.0541  z=+0.0530      (two-finger side)
gripper_pad_3   x=-0.0321  y=+0.0541  z=+0.0530
```

This has already caused two separate bugs:

1. **Generating the lite model, the `hand` tag was retyped and the `euler` was lost.**
   Every *relative* position matched exactly and the *world* pad positions were 76 mm
   off. Fixed by copying the original tag verbatim — which is now the rule: when
   collapsing, copy the parent tag, never retype it.
2. **Pick's grasp orientation aligned the jaws with `hand_site`'s X**, so they tried to
   straddle the box's 94 mm face instead of its 40 mm one. Fixed by building the target
   rotation from measured axes: Z down, Y along the box's short axis, X = Y × Z.

---

## 5. Trap two — `rigid` and `lite` do not have the same gripper orientation

`Fr3HGripperReach/fr3_H_gripper_rigid.xml` declares

```xml
<body name="hand" pos="0 0 0.220" gravcomp="1">      <!-- no euler -->
```

so its gripper is yawed 90° relative to the real robot and to `lite`. Measured, same
pose, same site frame:

| | pad separation axis | `gripper_site` ahead of `hand_site` |
|---|---|---|
| `rigid` (Reach) | **x** = ±0.0541 | **55.0 mm** |
| `lite` (Carry, Pick) | **y** = ±0.0541 | **53.0 mm** |

Reach is unaffected in practice: it welds the fingers, never grasps, and its cost reads
`hand_site`, whose orientation comes from `link7` and is therefore correct either way.

**But do not port pad geometry, a grasp pose, or a `gripper_site` offset between the two
models.** Anything measured on `rigid` is 90° and 2 mm away from the real robot. The
2 mm is the other half of this: `rigid` still carries judo's `gripper_site pos 0.145`,
while Carry and Pick were corrected to **0.143**, the measured midpoint between the
opposing jaws.

> The jaw midpoint is the *opposing-jaw* midpoint, not the mean of the three pads —
> that mean is biased toward the two-finger side. The correct point is
> (−0.000022, 0.000019, 0.14301) in the hand frame, and it is constant across the whole
> closing range.

---

## 6. Trap three — sites live inside the bodies being collapsed

`fr3_H_gripper.xml`'s `<sensor>` block declares touch sensors on
`gripper_pad_1/2/3_site`, and those sites lived inside the sub-bodies the collapse
removed. The first lite model simply failed to load. The three sites are now
re-declared at their measured positions on the retained bodies.

Same class of problem, worth checking before deleting any body: **sites, sensors,
equality constraints and contact excludes can all reference a body by name.**

---

## 7. Numbers that follow from the geometry

Useful when reasoning about grasps, and all measured on `lite`:

- **jaw gap** = `108.2 − 100 × (slide / 0.05)` mm. 108.2 mm at slide 0, 8.2 mm fully
  closed — exactly the 100 mm of travel the hardware reports as `finger_width`, since
  both jaws move. The 8.2 mm floor is pad thickness.
- a **40 mm box** therefore stops the slide at **0.030**; a running sim measured 0.0302.
- the gripper reaches **68.7 mm** from `hand_site`, pads centred at 53.0 mm, so there is
  only 15.7 mm of structure past the pad centre.
- **grasp depth is not limited to the top of a tall box.** With the box centre placed at
  the jaw midpoint there are 4 pad contacts and **0** non-pad contacts. The
  `finger_*_sub_bolt_link_3` geoms sit 0.8 mm outboard of the pad faces and only bite
  if the pads are driven ~4 mm into the box; real contact penetration runs 0.2–0.3 mm.
  An earlier "10.3 mm limit" claim came from measuring at an over-closed slide and was
  wrong.

---

## 8. Reproducing all of it

```bash
scripts/verify_gripper_models.py          # needs a python with the mujoco package
```

It prints the variants table, the `mj_step` benchmark, the mass / gravity-torque / pad
equivalence checks, the pad axes and `gripper_site` offsets for both variants, and the
jaw-gap fit. Any claim in this file that stops being true will show up there.

---

## 9. Rules for editing these models

1. **Copy the parent tag verbatim when collapsing.** Retyping loses attributes like
   `euler` and the failure is a world-frame offset with correct relative positions,
   which reads like a completely different bug.
2. **Delete the mesh assets too**, not just the bodies — §3.
3. **Re-declare sites, sensors and equalities** that pointed into removed bodies — §6.
4. **Verify with `scripts/verify_gripper_models.py`**, and verify pads at more than one
   slide value; one configuration matching proves nothing about the joint.
5. **Keep the original.** `fr3_H_gripper.xml` is the reference every check compares
   against; without it none of §2 is checkable.
6. **Measure, do not port.** Between `rigid` and `lite` the gripper differs by 90° and
   `gripper_site` by 2 mm — §5.
