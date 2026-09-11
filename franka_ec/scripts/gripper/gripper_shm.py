"""Shared memory for the Hyundai gripper command, separate from /mjpc_bridge.

WHY A SECOND REGION instead of extending MjpcBridge: mjpc_bridge.h is duplicated
in four places (franka_ec/include, mujoco_mpc/mjpc/tasks/Fr3 and .../mppi_track,
lerobot-prior-mppi/.../Fr3OodSim2Real) and every mjpc binary mmaps
sizeof(MjpcBridge) == 120 on the same /mjpc_bridge name. Growing the struct would
break all of them. A separate region touches nothing that exists.

Both ends are Python (run_real.py and gripper_bridge_node.py), so there is no C
header to keep in sync and no C++ rebuild anywhere.

Layout: 10 x int32 ("10i", 40 bytes). All-int32 on purpose -- no padding to
reason about, and every field here is naturally integral (the gripper services
take uint8 mm / deg / N).

  offset  field              direction          note
       0  want_open          planner -> node    1 = open, 0 = closed
       4  finger_pose_deg    planner -> node    0..180 (180 = 2-finger mode)
       8  width_open_mm      planner -> node    0..100, sent when want_open
      12  width_close_mm     planner -> node    0..100, sent when closing
      16  grip_force_n       planner -> node    1..100
      20  motor_on           planner -> node    1 = request motor on
      24  cmd_seq            planner -> node    bumped LAST, after the payload
      28  status_word        node -> planner    GripperStatus.gripper_status raw
      32  finger_width_mm    node -> planner    MEASURED width, for grasp confirm
      36  ack_seq            node -> planner    bumped LAST, after a service call

Seqlock discipline, same as mjpc_shm.py: the writer fills the payload and then
bumps its sequence counter, so a reader that sees the same sequence before and
after its payload read knows no write overlapped it.
"""
from __future__ import annotations

import mmap
import os
import struct
import time

SHM_NAME = "/judo_gripper"
SHM_PATH = f"/dev/shm{SHM_NAME}"
STRUCT_SIZE = 40
FMT = "10i"

WANT_OPEN = 0
FINGER_POSE = 4
WIDTH_OPEN = 8
WIDTH_CLOSE = 12
GRIP_FORCE = 16
MOTOR_ON = 20
CMD_SEQ = 24
STATUS_WORD = 28
FINGER_WIDTH = 32
ACK_SEQ = 36

MAX_READ_RETRIES = 8

# GripperStatus.gripper_status bits, per art_gripper_interfaces/msg/GripperStatus.msg:
#   "Ready(BIT: 0)/Fault(BIT: 1)/In-motion(BIT: 2)/Contact(BIT: 3)"
# MEASURED on this unit (2026-08-13), and the surprising part is worth writing down:
#   word 1  = motor OFF -- but width commands are STILL executed and the fingers do
#             travel (98 -> 80 -> 10 mm was observed at word 1). Only the holding
#             force is missing, so "it moved" does NOT prove the motor is on.
#   word 3  = 1|2 = FAULT (2026-08-13). In this state motor_on is ACCEPTED by the
#             service (result=0, the node logs "sent motor_on(1)") but never takes
#             effect: the word stays 3 and bit4 never appears. The fingers still
#             travel, so the only symptom is that every grasp slips and CLOSE loops
#             on its 2.5 s timeout forever. The fault has to be cleared first --
#             see the FAULT branch in gripper_bridge_node._poll().
#   word 17 = motor ON, normal. This is what the README calls the running state.
# In-motion and CONTACT were never seen in either state, so neither bit can be relied
# on. Grasp detection therefore works off the measured finger_width alone -- see
# hardware_grasp_confirm below -- and motor_on has to be verified from the word.
BIT_READY = 1 << 0
BIT_FAULT = 1 << 1
BIT_MOTION = 1 << 2
BIT_CONTACT = 1 << 3
BIT_MOTOR_ON = 1 << 4     # not in the msg comment; deduced from word 17 = 16|1 after
                          # a successful motor_on, vs word 1 before it


def motor_is_on(status_word: int) -> bool:
    """True when the drive is energised (word 17 = BIT_MOTOR_ON|BIT_READY).

    Needed because the fingers move either way: with the motor off the position
    command is still followed, it just has no holding force -- which is exactly the
    state you do NOT want to grasp in."""
    return bool(status_word & BIT_MOTOR_ON)


class GripperShmError(RuntimeError):
    pass


class GripperShm:
    def __init__(self, mm: mmap.mmap, fd: int) -> None:
        self._mm = mm
        self._fd = fd
        self.torn_reads = 0

    # ---- lifecycle ----
    @classmethod
    def create(cls) -> "GripperShm":
        """Owner (gripper_bridge_node.py): create + zero the region."""
        fd = os.open(SHM_PATH, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, STRUCT_SIZE)
        mm = mmap.mmap(fd, STRUCT_SIZE)
        mm.seek(0)
        mm.write(b"\x00" * STRUCT_SIZE)
        return cls(mm, fd)

    @classmethod
    def open_existing(cls, timeout: float | None = 0.0) -> "GripperShm":
        """Attach to the node's region. timeout=0 fails immediately, None waits."""
        t0 = time.time()
        while not os.path.exists(SHM_PATH):
            if timeout is not None and time.time() - t0 >= timeout:
                raise GripperShmError(
                    f"{SHM_NAME} not found. Start the gripper bridge first:\n"
                    f"  source /opt/ros/humble/setup.bash && "
                    f"source ~/gripper_ws/install/setup.bash\n"
                    f"  python3 ours/gripper_bridge_node.py")
            time.sleep(0.05)
        fd = os.open(SHM_PATH, os.O_RDWR)
        return cls(mmap.mmap(fd, STRUCT_SIZE), fd)

    def close(self) -> None:
        self._mm.close()
        os.close(self._fd)

    def unlink(self) -> None:
        try:
            os.unlink(SHM_PATH)
        except FileNotFoundError:
            pass

    def __enter__(self) -> "GripperShm":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- planner -> node ----
    def write_cmd(self, *, want_open: bool, finger_pose_deg: int, width_open_mm: int,
                  width_close_mm: int, grip_force_n: int, motor_on: bool) -> int:
        """Publish a command, bumping cmd_seq last. Returns the new cmd_seq."""
        struct.pack_into("6i", self._mm, WANT_OPEN,
                         1 if want_open else 0, int(finger_pose_deg),
                         int(width_open_mm), int(width_close_mm),
                         int(grip_force_n), 1 if motor_on else 0)
        (seq,) = struct.unpack_from("i", self._mm, CMD_SEQ)
        seq += 1
        struct.pack_into("i", self._mm, CMD_SEQ, seq)
        return seq

    def read_cmd(self) -> dict:
        """Node side: a consistent snapshot of the command block."""
        for _ in range(MAX_READ_RETRIES):
            (s0,) = struct.unpack_from("i", self._mm, CMD_SEQ)
            vals = struct.unpack_from("6i", self._mm, WANT_OPEN)
            (s1,) = struct.unpack_from("i", self._mm, CMD_SEQ)
            if s0 == s1:
                return {
                    "want_open": bool(vals[0]), "finger_pose_deg": vals[1],
                    "width_open_mm": vals[2], "width_close_mm": vals[3],
                    "grip_force_n": vals[4], "motor_on": bool(vals[5]),
                    "cmd_seq": s1,
                }
            self.torn_reads += 1
        raise GripperShmError("no consistent command snapshot")

    # ---- node -> planner ----
    def write_status(self, status_word: int, finger_width_mm: int) -> int:
        """Publish measured state, bumping ack_seq last."""
        struct.pack_into("2i", self._mm, STATUS_WORD,
                         int(status_word), int(finger_width_mm))
        (seq,) = struct.unpack_from("i", self._mm, ACK_SEQ)
        seq += 1
        struct.pack_into("i", self._mm, ACK_SEQ, seq)
        return seq

    def read_status(self) -> dict:
        """Planner side: a consistent snapshot of the measured state."""
        for _ in range(MAX_READ_RETRIES):
            (s0,) = struct.unpack_from("i", self._mm, ACK_SEQ)
            sw, fw = struct.unpack_from("2i", self._mm, STATUS_WORD)
            (s1,) = struct.unpack_from("i", self._mm, ACK_SEQ)
            if s0 == s1:
                return {"status_word": sw, "finger_width_mm": fw, "ack_seq": s1}
            self.torn_reads += 1
        raise GripperShmError("no consistent status snapshot")


def hardware_grip_busy(shm: GripperShm, quiet_s: float = 0.35, move_eps_mm: int = 1):
    """Build a grip_busy() for GraspPhases: True while the fingers are still travelling.

    The CLOSE timeout exists to catch a gripper that is STUCK, but it was measuring
    total elapsed time, so a gripper that is merely SLOW looked identical to a stuck
    one. 95 -> 10 mm is 85 mm of travel on this unit and that can exceed the 2.5 s
    budget, so the machine reopened mid-close and the late close landed on the next
    approach -- the "it only grips on the second or third try" symptom.

    Busy = the measured width changed within the last quiet_s. Time-based rather than
    "changed since the last call" because this is polled at ~200 Hz while
    gripper_status arrives far slower, so almost every individual call sees no change.
    """
    state = {"last_w": None, "changed_at": None}

    def busy() -> bool:
        try:
            w = shm.read_status()["finger_width_mm"]
        except GripperShmError:
            return False                       # no data -> do not extend the deadline
        now = time.monotonic()
        if state["last_w"] is None or abs(w - state["last_w"]) >= move_eps_mm:
            state["last_w"] = w
            state["changed_at"] = now
        return (now - (state["changed_at"] or now)) < quiet_s

    return busy


def hardware_grasp_confirm(shm: GripperShm, get_close_mm, get_open_mm, margin_mm: int = 3,
                           travel_min_mm: int = 5, stall_s: float = 0.35,
                           require_ack: bool = True, min_still_samples: int = 2, log=None):
    """Build a confirm_grasp() for GraspPhases, backed by the real gripper.

    The sim test counts pad<->box contacts; hardware has no such thing, so we infer
    it from the measured finger_width: the fingers must have moved INWARD from the
    open width and then stalled ABOVE the commanded close width. Told to go to
    10 mm, starting from 80, stopping at 35 -> something is in the way, and that
    something is the object.

    BOTH bounds matter. An earlier version tested only "width > close + margin",
    which is trivially true when the gripper does not move at all -- a gripper stuck
    at 69 mm reported a successful grasp, and the arm lifted with empty fingers. So
    a grasp now also requires width < open - margin, i.e. evidence of real travel.

    CONTACT is accepted as an alternative signal but is not sufficient on its own
    for the same reason: the bit semantics are documented inconsistently (see the
    note on BIT_CONTACT above), so it is only trusted together with travel.
    """
    state = {"warned": False, "baseline": None, "was_open": True,
             "last_w": None, "stable_since": None, "fault_warned": False,
             "last_ack": None, "still_n": 0}

    def confirm() -> bool:
        try:
            st = shm.read_status()
            cmd = shm.read_cmd()
        except GripperShmError:
            return False
        if require_ack and st["ack_seq"] <= 0:
            return False                      # node has not reported anything yet
        if st["status_word"] & BIT_FAULT:
            # A fault VETOES the grasp on purpose: the drive has dropped out, so whatever
            # the fingers are resting on is held by friction alone and will slip the moment
            # the arm lifts. But say so -- silently returning False here made a successful
            # squeeze that faulted look identical to "the fingers never reached the object",
            # and the machine just retried forever with no hint why.
            if log is not None and not state["fault_warned"]:
                state["fault_warned"] = True
                base = state["baseline"]
                trav = "unknown" if base is None else f"{base - st['finger_width_mm']} mm"
                log(f"gripper FAULTED during CLOSE (status={st['status_word']}) at width "
                    f"{st['finger_width_mm']} mm after {trav} of travel. If that width is "
                    f"about the object's thickness then the grasp itself was FINE and the "
                    f"drive tripped on the stall -- it is being commanded "
                    f"{st['finger_width_mm'] - get_close_mm()} mm PAST the object. Raise "
                    f"--width-close to just under the object width and/or lower "
                    f"--grip-force, then clear the fault (motor_on 0 then 1).")
            return False
        state["fault_warned"] = False
        w = st["finger_width_mm"]

        # Latch the width at the moment CLOSE was commanded. Travel has to be judged
        # against what the fingers ACTUALLY were, not against the open setpoint: a
        # gripper stuck at 69 mm that never reached the commanded 80 mm still looks
        # "moved in" if you compare to 80, and that false positive had the arm lift
        # with empty fingers.
        now = time.monotonic()
        if cmd["want_open"]:
            state.update(was_open=True, baseline=None, warned=False,
                         last_w=None, stable_since=None, still_n=0)
            return False                      # not even trying to grasp
        if state["was_open"]:
            state.update(was_open=False, baseline=w, last_w=w, stable_since=now,
                         still_n=0)

        base = state["baseline"] if state["baseline"] is not None else w
        travelled = base - w                   # mm the fingers have actually closed

        # STALL detection, not just "has moved". Requiring only travel >= a few mm
        # fires while the fingers are still sweeping through open air -- at 94 mm
        # commanded to 10, the test passed at 89 mm and the arm lifted before the
        # fingers ever reached the object. A grasp is the fingers STOPPING early.
        #
        # Judged across DISTINCT STATUS SAMPLES, not wall time. The driver's launch file
        # sets gripper_status_publish_rate_hz = 1, so the width readback only changes once a
        # second; a wall-clock stall test shorter than that interval reports "stalled" for
        # the ~0.65 s between every pair of samples, even mid-travel. That fired a false
        # grasp while the fingers were still closing -> LIFT -> CARRY with nothing held.
        # Requiring the SAME width on >= min_still_samples fresh samples makes this correct
        # at any publish rate (and much faster at 20 Hz -- see --contact-sensitivity notes).
        fresh = st["ack_seq"] != state["last_ack"]
        state["last_ack"] = st["ack_seq"]
        if fresh:
            if state["last_w"] is None or abs(w - state["last_w"]) > 1:
                state["last_w"] = w
                state["stable_since"] = now
                state["still_n"] = 1          # this sample is the first at this width
            else:
                state["still_n"] += 1
        stalled_for = now - (state["stable_since"] or now)

        if travelled < travel_min_mm:
            if log is not None and not state["warned"]:
                state["warned"] = True
                log(f"gripper is not closing: width {w} mm, only {travelled} mm of travel "
                    f"since CLOSE was commanded (need {travel_min_mm}). Check that the "
                    f"motor is on (gripper_status must be 17, not 1 -- the fingers move "
                    f"either way but have no holding force at 1) and that "
                    f"gripper_bridge_node.py is NOT in --dry-run.")
            return False
        if w <= get_close_mm() + margin_mm:
            return False                       # reached the target -> nothing in there
        if stalled_for < stall_s or state["still_n"] < min_still_samples:
            return False                       # still closing; not a grasp yet
        return True

    return confirm
