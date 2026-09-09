#!/usr/bin/env python3
"""Re-check every claim in H_GRIPPER_MODEL_GUIDE.md.

    scripts/verify_gripper_models.py

Needs a python with the `mujoco` package (the judo venv has one:
../prior_mppi_judo/.venv/bin/python). Run from the repo root.

Why a script and not a table in the doc: the collapsed models are only useful if they
are still EQUIVALENT to the original, and equivalence is not something a comment can
keep true. Anything here that stops holding is a real regression in the models.
"""
import os
import sys
import time

import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("no mujoco module. Try:\n"
             "  ../prior_mppi_judo/.venv/bin/python scripts/verify_gripper_models.py")

ORIG = "mjpc/tasks/Fr3HGripperCarry/fr3_H_gripper.xml"
VARIANTS = [
    ("original", ORIG),
    ("rigid  (Reach)", "mjpc/tasks/Fr3HGripperReach/fr3_H_gripper_rigid.xml"),
    ("lite   (Carry)", "mjpc/tasks/Fr3HGripperCarry/fr3_H_gripper_lite.xml"),
    ("pick   (Pick)", "mjpc/tasks/Fr3HGripperPick/fr3_H_gripper_pick.xml"),
]
TASKS = [
    ("rigid (Reach)", "mjpc/tasks/Fr3HGripperReach/task.xml"),
    ("lite  (Pick)", "mjpc/tasks/Fr3HGripperPick/task.xml"),
]
PADS = ("gripper_pad_1", "gripper_pad_2", "gripper_pad_3")
SLIDES = ("finger_A_slide_joint", "finger_B_slide_joint", "finger_C_slide_joint")


def load(path):
    if not os.path.exists(path):
        return None, None
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def set_slides(m, d, value):
    for j in SLIDES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = value
    mujoco.mj_forward(m, d)


def pad_world(m, d):
    """Pad positions in the world frame, or None if a pad is missing."""
    out = []
    for nm in PADS:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm)
        if g < 0:
            return None
        out.append(d.geom_xpos[g].copy())
    return np.array(out)


def section(title):
    print("\n" + title)
    print("-" * len(title))


fails = []


def check(label, ok, detail=""):
    print("  %-52s %s%s" % (label, "OK" if ok else "FAIL", ("   " + detail) if detail else ""))
    if not ok:
        fails.append(label)


# ── 1. the variants table ───────────────────────────────────────────────
section("1. variants  (guide section 1)")
print("  %-16s %6s %6s %10s %6s %4s %3s" %
      ("model", "nbody", "nmesh", "nmeshface", "ngeom", "nv", "nu"))
loaded = {}
for tag, path in VARIANTS:
    m, d = load(path)
    if m is None:
        print("  %-16s  MISSING: %s" % (tag, path))
        continue
    loaded[tag] = (m, d, path)
    print("  %-16s %6d %6d %10d %6d %4d %3d" %
          (tag, m.nbody, m.nmesh, m.nmeshface, m.ngeom, m.nv, m.nu))

# ── 2. mj_step ──────────────────────────────────────────────────────────
section("2. mj_step  (guide section 1)")
timings = {}
for tag in ("original", "pick   (Pick)"):
    if tag not in loaded:
        continue
    m, _, path = loaded[tag]
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(200):
        mujoco.mj_step(m, d)
    t0 = time.perf_counter()
    for _ in range(2000):
        mujoco.mj_step(m, d)
    timings[tag] = (time.perf_counter() - t0) / 2000 * 1e6
    print("  %-16s %8.2f us" % (tag, timings[tag]))
if len(timings) == 2:
    ratio = timings["original"] / timings["pick   (Pick)"]
    check("speedup >= 2.0x", ratio >= 2.0, "measured %.2fx" % ratio)

# ── 3. equivalence: mass, gravity torque, pads ──────────────────────────
section("3. collapsed vs original  (guide section 2)")
mo, do, _ = loaded.get("original", (None, None, None))
if mo is None:
    print("  original missing; cannot check equivalence")
else:
    for tag in ("lite   (Carry)", "pick   (Pick)"):
        if tag not in loaded:
            continue
        m, d, _ = loaded[tag]
        dm = abs(m.body_mass.sum() - mo.body_mass.sum())
        check("%s total mass == original" % tag, dm < 1e-9, "diff %.2e kg" % dm)

        # gravity torque on the 7 arm joints at the shared home pose
        do2 = mujoco.MjData(mo)
        d2 = mujoco.MjData(m)
        for mm, dd in ((mo, do2), (m, d2)):
            if mm.nkey:
                mujoco.mj_resetDataKeyframe(mm, dd, 0)
            dd.qvel[:] = 0.0
            mujoco.mj_forward(mm, dd)
        dg = np.abs(do2.qfrc_bias[:7] - d2.qfrc_bias[:7]).max()
        check("%s gravity torque == original" % tag, dg < 1e-5, "max diff %.1e Nm" % dg)

        # pads at two slide values -- one configuration proves nothing about the joint
        worst = 0.0
        for s in (0.0, 0.025):
            do3 = mujoco.MjData(mo)
            d3 = mujoco.MjData(m)
            for mm, dd in ((mo, do3), (m, d3)):
                if mm.nkey:
                    mujoco.mj_resetDataKeyframe(mm, dd, 0)
                set_slides(mm, dd, s)
            a, b = pad_world(mo, do3), pad_world(m, d3)
            if a is None or b is None:
                worst = float("nan")
                break
            worst = max(worst, np.abs(a - b).max())
        check("%s pads == original (slide 0 and 0.025)" % tag,
              worst == worst and worst < 1e-5, "max diff %.1e m" % worst)

# ── 4. the two variants are NOT interchangeable ─────────────────────────
section("4. rigid vs lite orientation  (guide section 5)")
for tag, path in TASKS:
    m, d = load(path)
    if m is None:
        print("  %-14s MISSING: %s" % (tag, path))
        continue
    hs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "hand_site")
    hR = d.site_xmat[hs].reshape(3, 3)
    hp = d.site_xpos[hs]
    loc = {}
    for nm in PADS:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm)
        loc[nm] = hR.T @ (d.geom_xpos[g] - hp) if g >= 0 else None
    p1 = loc["gripper_pad_1"]
    axis = "x" if abs(p1[0]) > abs(p1[1]) else "y"
    gs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_site")
    off = np.linalg.norm(d.site_xpos[gs] - hp) * 1e3 if gs >= 0 else float("nan")
    print("  %-14s jaw axis = %s (%+.4f %+.4f)   gripper_site %.1f mm ahead"
          % (tag, axis, p1[0], p1[1], off))
print("  -> different by design/history: do NOT port pad geometry or grasp poses")
print("     between them. See guide section 5.")

# ── 5. jaw gap fit ──────────────────────────────────────────────────────
section("5. jaw gap vs slide  (guide section 7)")
if "pick   (Pick)" in loaded:
    m, _, path = loaded["pick   (Pick)"]
    rows = []
    for s in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05):
        d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        set_slides(m, d, s)
        hs = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "hand_site")
        hR = d.site_xmat[hs].reshape(3, 3)
        hp = d.site_xpos[hs]
        g1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "gripper_pad_1")
        g2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "gripper_pad_2")
        a = hR.T @ (d.geom_xpos[g1] - hp)
        b = hR.T @ (d.geom_xpos[g2] - hp)
        rows.append((s, abs(a[1] - b[1]) * 1e3))
    for s, gap in rows:
        print("    slide %.3f -> gap %6.1f mm" % (s, gap))
    pred = [108.2 - 100.0 * (s / 0.05) for s, _ in rows]
    err = max(abs(p - g) for p, (_, g) in zip(pred, rows))
    check("gap == 108.2 - 100*(slide/0.05) mm", err < 0.2, "max err %.2f mm" % err)
    s40 = next((s for s, g in rows if abs(g - 40.0) < 1.0), None)
    print("    -> a 40 mm box stops the slide near 0.030"
          + ("" if s40 is None else " (bracketed here)"))

# ── verdict ─────────────────────────────────────────────────────────────
print()
if fails:
    print("%d CHECK(S) FAILED:" % len(fails))
    for f in fails:
        print("  - " + f)
    print("\nThe collapsed models are no longer equivalent to the original, or the")
    print("guide is out of date. Do not use them on hardware until this is resolved.")
    sys.exit(1)
print("all checks passed")
