#!/usr/bin/env python3
"""Print the robot's CURRENT hand_site pose, and the MJPC_TARGET_* line that puts the
reach target exactly there.

Why: Fr3HGripperReach starts its target at (0.5, 0, 0.5). If the arm is nowhere near
that, MPPI opens at full effort -- measured 53 Nm sustained on joints 1-4 (limits 87) in
dry run. Starting with the target ON the current EE means the first live moment is a
near-zero command, and you can then drag the target where you want it.

Reads q straight out of /mjpc_bridge (written by franka_ec's mppi_track_controller) and
runs FK on this task's own model, so the answer is in the same frame the task uses.

    # with the controller running:
    env -u PYTHONPATH ../prior_mppi_judo/.venv/bin/python ee_now.py
"""
import os
import struct
import sys
import time

import mujoco
import numpy as np

SHM = "/dev/shm/mjpc_bridge"
XML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "mjpc/tasks/Fr3HGripperReach/fr3_H_gripper_single.xml")

# 120-byte MjpcBridge (franka_ec/include/franka_ec/mjpc_bridge.h):
#   q@0 dq@28 ee_pos@56 state_seq@68 action@72 action_seq@100 target@104 target_seq@116
Q_OFF, STATE_SEQ_OFF, SIZE = 0, 68, 120


def read_q():
    """Seqlock read: q is only trusted when state_seq is unchanged around it."""
    with open(SHM, "rb") as f:
        buf = f.read(SIZE)
    if len(buf) < SIZE:
        raise RuntimeError(f"{SHM} is {len(buf)} B, expected {SIZE}")
    for _ in range(8):
        (s0,) = struct.unpack_from("i", buf, STATE_SEQ_OFF)
        q = np.array(struct.unpack_from("7f", buf, Q_OFF), dtype=float)
        with open(SHM, "rb") as f:
            buf2 = f.read(SIZE)
        (s1,) = struct.unpack_from("i", buf2, STATE_SEQ_OFF)
        if s0 == s1 and abs(q[3]) > 1e-9:   # joint4 range excludes 0 -> 0 means unwritten
            return q, s1
        buf = buf2
    raise RuntimeError("no consistent q snapshot")


def main() -> int:
    if not os.path.exists(SHM):
        print(f"!! {SHM} not found. Start the controller first:\n"
              f"   ros2 launch franka_bringup mppi_track_controller.launch.py "
              f"robot_ip:=172.16.0.2")
        return 2
    s_a = read_q()[1]
    time.sleep(0.2)
    q, s_b = read_q()
    if s_a == s_b:
        print(f"!! state_seq frozen at {s_b} -- the controller is not running (stale shm)")
        return 2

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    d.qpos[:7] = q
    mujoco.mj_forward(m, d)

    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "hand_site")
    pos = d.site_xpos[sid].copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, d.site_xmat[sid])

    np.set_printoptions(precision=4, suppress=True)
    print(f"state_seq   {s_a} -> {s_b}  (live)")
    print(f"q           {q}")
    print(f"hand_site   pos {pos}   quat(wxyz) {quat}")
    print(f"            |pos| {np.linalg.norm(pos):.4f} m")
    print()
    print("Start mjpc with the target ON the current EE:")
    print(f"  MJPC_TARGET_X={pos[0]:.4f} MJPC_TARGET_Y={pos[1]:.4f} "
          f"MJPC_TARGET_Z={pos[2]:.4f} \\")
    print(f"    MJPC_BRIDGE_DRYRUN=1 ./build/bin/mjpc --task FR3_H_Gripper_Reach")
    print()
    print("NOTE: the task also forces the target ORIENTATION to gripper-down (0,1,0,0),")
    print(f"      and the hand is currently at {np.array2string(quat, precision=3)}.")
    print("      If those differ, the orientation term alone will command torque even")
    print("      with the position error at zero -- check ori_err before going live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
