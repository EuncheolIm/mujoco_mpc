#!/usr/bin/env python3
"""Bridge /judo_gripper shared memory <-> the Hyundai gripper's ROS 2 interface.

Run with the SYSTEM python3, not the judo venv -- the venv is deliberately
isolated and has no rclpy:

    source /opt/ros/humble/setup.bash
    source ~/gripper_ws/install/setup.bash          # art_gripper + interfaces
    python3 franka_ec/scripts/gripper/gripper_bridge_node.py --ns /ag_right

--ns DEFAULTS TO /ag_left. If the driver was launched as `right`, only /ag_right/*
exists and every call fails with "service not ready". Check with:
    ros2 service list | grep ag_

This node OWNS /judo_gripper (creates and unlinks it), so start it before the
planner. It exists so that neither franka_ec nor the mjpc trees have to learn
anything about the gripper: no MjpcBridge change, no C++ rebuild, and franka_ec
never depends on art_gripper_interfaces.

Copied here from prior_mppi_judo/ours/ so the gripper can be run on a machine that
does not have the judo checkout. gripper_shm.py sits beside it and is imported by
name -- the two must be moved together, and the 40 B layout must stay identical to
judo's copy and to mjpc's gripper_shm.h, since all three map /judo_gripper.

Direction of travel:
  shm  -> services : on every cmd_seq change, set_target_finger_width(open|close)
  topic -> shm     : gripper_status gives the measured finger_width and status
                     word, which is what run_real.py uses to confirm a grasp

Services are called with call_async and never waited on: EtherCAT round trips
would otherwise stall the poll loop, and the planner does not need the return
value -- it watches the measured width instead.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402

from art_gripper_interfaces.msg import GripperStatus  # noqa: E402
from art_gripper_interfaces.srv import (  # noqa: E402
    MotorOn,
    SetContactSensitivity,
    SetGrippingForce,
    SetTargetFingerPose,
    SetTargetFingerWidth,
    SetTargetFingerWidthWithSpeed,
)

from gripper_shm import (  # noqa: E402
    SHM_NAME,
    BIT_CONTACT,
    BIT_FAULT,
    BIT_READY,
    GripperShm,
    GripperShmError,
    motor_is_on,
)

POLL_HZ = 50.0


def clamp_u8(v, lo, hi):
    return int(max(lo, min(hi, int(v))))


class _PrefixLogger:
    """Tags every line with the arm's namespace.

    With two arms in ONE node the messages interleave, and almost every warning in
    here is about a specific gripper -- an unprefixed "motor is OFF" would not say
    which. Wrapping the logger means the existing call sites need no edit.
    """

    def __init__(self, logger, prefix: str):
        self._log = logger
        self._p = prefix

    def info(self, m):  self._log.info(f"{self._p}{m}")
    def warn(self, m):  self._log.warn(f"{self._p}{m}")
    def error(self, m): self._log.error(f"{self._p}{m}")


class ArmBridge:
    """One gripper: its namespace, its shm region, its clients, its state.

    This was the whole Node. It is now a plain object so ONE node can hold several,
    which is what lets both grippers come up from a single process and act
    independently as commands arrive for each. The per-arm logic below is unchanged;
    only the three things that cannot be per-arm moved out -- rclpy entity creation,
    the logger, and the timer, which the node now owns and fans out.
    """

    def __init__(self, node: Node, ns: str, shm_name: str, dry_run: bool,
                 contact_sensitivity=None, width_speed=None):
        self.node = node
        self.ns = ns.rstrip("/")
        self.shm_name = shm_name
        self._logger = _PrefixLogger(node.get_logger(), f"[{self.ns}] ")
        self._warned = {}     # srv_name -> last warn time, for the throttle in _call
        self.dry_run = dry_run
        self.contact_sensitivity = contact_sensitivity
        self.width_speed = width_speed
        self.shm = GripperShm.create(shm_name)
        self.get_logger().info(f"created /dev/shm{shm_name} (owner)")

        self.cli_width = node.create_client(SetTargetFingerWidth,
                                            f"{self.ns}/set_target_finger_width")
        self.cli_pose = node.create_client(SetTargetFingerPose,
                                           f"{self.ns}/set_target_finger_pose")
        self.cli_motor = node.create_client(MotorOn, f"{self.ns}/motor_on")
        self.cli_force = node.create_client(SetGrippingForce,
                                            f"{self.ns}/set_gripping_force")
        self.cli_sens = node.create_client(SetContactSensitivity,
                                          f"{self.ns}/set_contact_sensitivity")
        self.cli_width_spd = node.create_client(
            SetTargetFingerWidthWithSpeed, f"{self.ns}/set_target_finger_width_with_speed")
        node.create_subscription(GripperStatus, f"{self.ns}/gripper_status",
                                 self._on_status, 10)

        self._last_cmd_seq = 0
        self._configured = False      # motor_on / pose / force sent once
        self._last_width_sent = None
        self._status_n = 0
        self._last_status = None
        self._last_width = None
        self._quiet_polls = 0
        self._motor_warned = False
        self._motor_off_polls = 0
        self._motor_off_warned = False
        self._fault_warned = False
        self._configured_at = None    # when motor_on/pose/force went out
        self.get_logger().info(f"namespace {self.ns}  shm {shm_name}  dry_run={dry_run}")
        if dry_run:
            self.get_logger().warn("DRY RUN: no service calls will be made")

    def get_logger(self):
        return self._logger

    # ---- topic -> shm ----
    def _on_status(self, msg: GripperStatus) -> None:
        self._status_n += 1
        self._last_status = int(msg.gripper_status)
        self._last_width = int(msg.finger_width)
        try:
            self.shm.write_status(int(msg.gripper_status), int(msg.finger_width))
        except GripperShmError as e:
            self.get_logger().warn(f"status write failed: {e}")
        if self._status_n == 1:
            flags = []
            if msg.gripper_status & BIT_FAULT:
                flags.append("FAULT")
            if msg.gripper_status & BIT_CONTACT:
                flags.append("CONTACT")
            self.get_logger().info(
                f"first status: word={msg.gripper_status} "
                f"width={msg.finger_width}mm pose={msg.finger_pose}deg "
                f"{' '.join(flags)}")

    # ---- shm -> services ----
    def _call(self, client, req, what: str) -> bool:
        """Dispatch a service call. Returns False if it could not be sent."""
        if self.dry_run:
            self.get_logger().info(f"[dry-run] {what}")
            return True
        if not client.service_is_ready():
            # THROTTLED. The poll loop runs at 50 Hz and _configure() emits three of
            # these per tick, so an unavailable driver used to bury every other message
            # under thousands of identical lines -- including the one naming the wrong
            # namespace, which was the actual fault. One line per service per 2 s.
            now = time.monotonic()
            if now - self._warned.get(client.srv_name, -1e9) >= 2.0:
                self._warned[client.srv_name] = now
                self.get_logger().warn(
                    f"{what}: service not ready ({client.srv_name}) -- is the driver up, "
                    f"and is --ns {self.ns} the right namespace?")
            return False
        client.call_async(req)      # deliberately not awaited; see module docstring
        self.get_logger().info(f"sent {what}")
        return True

    def _configure(self, cmd) -> None:
        """Setup: motor on, 2-finger pose, gripping force.

        RETRIES until every call actually goes out. The first version marked itself
        configured even when service_is_ready() was false, so starting this node
        before the gripper driver left the motor off forever -- the width commands
        were then accepted (result=0) while nothing moved, and gripper_status stayed
        at 1 (Ready only) instead of the running value."""
        ok = True
        if cmd["motor_on"]:
            r = MotorOn.Request()
            r.on = 1
            ok &= self._call(self.cli_motor, r, "motor_on(1)")
        r = SetTargetFingerPose.Request()
        r.finger_pose = clamp_u8(cmd["finger_pose_deg"], 0, 180)
        ok &= self._call(self.cli_pose, r, f"finger_pose({r.finger_pose})")
        r = SetGrippingForce.Request()
        r.gripping_force = clamp_u8(cmd["grip_force_n"], 1, 100)   # field is gripping_force, not force
        ok &= self._call(self.cli_force, r, f"gripping_force({r.gripping_force})")
        # Contact sensitivity decides how soon the drive treats resistance as "arrived"
        # instead of pushing on. Left alone it keeps forcing toward the commanded width and
        # trips into FAULT on the stall -- observed at 34-40 mm on a 35 mm box while
        # commanded to 10 mm. Lower = trips earlier = gentler stop.
        if self.contact_sensitivity is not None:
            r = SetContactSensitivity.Request()
            r.contact_sensitivity = clamp_u8(self.contact_sensitivity, 1, 100)
            ok &= self._call(self.cli_sens, r,
                             f"contact_sensitivity({r.contact_sensitivity})")
        if ok:
            self._configured = True
            self._configured_at = time.monotonic()
        else:
            now = time.monotonic()
            if now - self._warned.get("_cfg", -1e9) >= 2.0:
                self._warned["_cfg"] = now
                self.get_logger().warn("configuration incomplete -- will retry on the "
                                       "next command. Is the gripper driver up?")

    def poll(self) -> None:
        try:
            cmd = self.shm.read_cmd()
        except GripperShmError as e:
            self.get_logger().warn(f"cmd read failed: {e}")
            return
        if cmd["cmd_seq"] == 0:
            return                      # planner has not published anything yet

        # Retry configuration on EVERY poll until it sticks. Doing it only on a
        # cmd_seq change was not enough: the planner sends a command only when
        # open/close actually flips, so a failed motor_on could sit unretried.
        if not self._configured:
            self._configure(cmd)

        # FAULT first: it MASKS motor_on. Observed word=3 (Ready|Fault) where motor_on is
        # accepted (result=0, "sent motor_on(1)" logged) but bit4 never appears, so the
        # fingers travel with no holding force and every CLOSE loops on its 2.5 s timeout.
        # Retrying motor_on cannot fix this, so say so instead of quietly retrying forever.
        if (self._last_status is not None and (self._last_status & BIT_FAULT)
                and not self._fault_warned):
            self._fault_warned = True
            self.get_logger().error(
                f"gripper is in FAULT (gripper_status={self._last_status}, bit1 set). "
                f"motor_on will be accepted but IGNORED -- the fingers still move, with no "
                f"holding force, so no grasp can ever be confirmed. Clear the fault before "
                f"running:\n"
                f"    ros2 service call {self.ns}/motor_on "
                f"art_gripper_interfaces/srv/MotorOn \"{{'on': 0}}\"   # then 1 again\n"
                f"  If the word stays faulted, read the raw drive words to see which axis:\n"
                f"    ros2 topic echo --once {self.ns}/gripper_status\n"
                f"  and if needed power-cycle the SMPS and redo the EtherCAT setup "
                f"(README section 9).")
        elif self._last_status is not None and not (self._last_status & BIT_FAULT):
            self._fault_warned = False        # re-arm once the fault actually clears

        # Two independent checks, because on this unit they fail differently:
        #  * width not tracking the command  -> the gripper is not responding at all
        #  * word without bit4 (i.e. not 17) -> motor OFF. Measured: the fingers still
        #    TRAVEL with the motor off (98 -> 10 mm at word 1), they just have no
        #    holding force, so motion alone never proves motor_on succeeded.
        if (self._configured and self._last_status is not None
                and not self._motor_off_warned and not motor_is_on(self._last_status)):
            self._motor_off_polls += 1
            if self._motor_off_polls > int(2.0 * POLL_HZ):
                self._motor_off_warned = True
                self.get_logger().warn(
                    f"motor is OFF (gripper_status={self._last_status}, need bit4 set "
                    f"i.e. 17). The fingers will still MOVE but with no holding force, "
                    f"so a grasp will slip. Retrying motor_on.")
                self._configured = False      # forces _configure() to fire again
        elif self._configured and self._last_status is not None and motor_is_on(self._last_status):
            self._motor_off_polls = 0

        if (self._configured and self._status_n > 0 and not self._motor_warned
                and self._last_width_sent is not None and self._last_width is not None):
            if abs(self._last_width - self._last_width_sent) > 5:
                self._quiet_polls += 1
                if self._quiet_polls > int(3.0 * POLL_HZ):
                    self._motor_warned = True
                    self.get_logger().warn(
                        f"width has stayed at {self._last_width} mm for 3 s after "
                        f"commanding {self._last_width_sent} mm. Service calls return "
                        f"result=0 even with the motor off, so check it explicitly:\n"
                        f"    ros2 service call {self.ns}/motor_on "
                        f"art_gripper_interfaces/srv/MotorOn \"{{'on': 1}}\"\n"
                        f"  (quote the key -- bare  on:  is a YAML boolean and the call "
                        f"fails with 'attribute name must be string')\n"
                        f"  Then: python3 ours/gripper_test.py")
            else:
                self._quiet_polls = 0
                self._motor_warned = False

        # Do NOT send a width in the same breath as motor_on. The configure calls go out
        # with call_async and are never awaited, so the drive is still energising when the
        # width lands 4 ms later -- and it silently ignores it. Symptom: "sent
        # finger_width(60)" in the log, fingers at 98 mm, and the next command (whenever
        # something else changed) working fine. Give the drive a moment.
        if self._configured_at is not None and time.monotonic() - self._configured_at < 1.0:
            return

        if cmd["cmd_seq"] == self._last_cmd_seq:
            return
        self._last_cmd_seq = cmd["cmd_seq"]

        width = (cmd["width_open_mm"] if cmd["want_open"] else cmd["width_close_mm"])
        width = clamp_u8(width, 0, 100)
        if width != self._last_width_sent:
            self._last_width_sent = width
            tag = "OPEN" if cmd["want_open"] else "CLOSE"
            if self.width_speed is not None:
                # The driver hardcodes finger_width_speed = 150 mm/s at startup
                # (art_gripper procCtrl.cpp initProcess == 1), but the measured travel was
                # ~18 mm/s, so 95 -> 10 mm took over 4 s and the phase machine moved on
                # before the fingers arrived. Command the speed explicitly.
                req = SetTargetFingerWidthWithSpeed.Request()
                req.finger_width = width
                req.finger_width_speed = clamp_u8(self.width_speed, 1, 200)
                self._call(self.cli_width_spd, req,
                           f"finger_width({width}) @{req.finger_width_speed}mm/s [{tag}]")
            else:
                req = SetTargetFingerWidth.Request()
                req.finger_width = width
                self._call(self.cli_width, req, f"finger_width({width}) [{tag}]")

    def shutdown(self) -> None:
        try:
            self.shm.close()
            self.shm.unlink()
            self.get_logger().info(f"unlinked /dev/shm{self.shm_name}")
        except Exception as e:                       # teardown must not mask others
            self.get_logger().warn(f"shm teardown: {e}")


class GripperBridge(Node):
    """One node, one timer, N grippers.

    Chosen over N processes so that "both grippers up" is a single command. The cost
    is shared fate: one EtherCAT stall blocks the other arm's poll too, because the
    service calls are call_async and never awaited but read_cmd() is not.
    """

    def __init__(self, specs, dry_run: bool, contact_sensitivity=None,
                 width_speed=None):
        super().__init__("judo_gripper_bridge")
        self.arms = [
            ArmBridge(self, ns, shm, dry_run, contact_sensitivity, width_speed)
            for ns, shm in specs
        ]
        # ONE timer for all arms: each poll() only touches its own shm and its own
        # clients, so they stay independent -- a command arriving for one arm moves
        # that arm and nothing else.
        self.create_timer(1.0 / POLL_HZ, self._tick)
        self.get_logger().info(
            f"{len(self.arms)} gripper(s): "
            + ", ".join(f"{a.ns} -> {a.shm_name}" for a in self.arms))

    def _tick(self) -> None:
        for a in self.arms:
            a.poll()

    def destroy_node(self):
        for a in self.arms:
            a.shutdown()
        return super().destroy_node()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", action="append", default=None, metavar="NS",
                   help="gripper namespace. Repeat it for both grippers:\n"
                        "  --ns /ag_left --ns /ag_right\n"
                        "Default /ag_left. The launch file picks which one exists --\n"
                        "check with `ros2 service list | grep ag_` before assuming.")
    p.add_argument("--shm", action="append", default=None, metavar="NAME",
                   help="shm region per --ns, same order. Omit and it is derived:\n"
                        "  one  --ns  -> /judo_gripper        (unchanged, single-arm)\n"
                        "  many --ns  -> /judo_gripper_<suffix>, e.g. /judo_gripper_left\n"
                        "One region per gripper is required, not a style choice: a\n"
                        "region carries ONE cmd_seq/ack_seq handshake and two arms\n"
                        "sharing it would overwrite each other's commands.")
    p.add_argument("--dry-run", action="store_true",
                   help="log what would be sent but call no services (gripper stays put)")
    p.add_argument("--contact-sensitivity", type=int, default=None, metavar="1..100",
                   help="how soon the drive gives up pushing and calls it contact (driver "
                        "default 80). Lower it if the gripper FAULTS while closing on the "
                        "object: it stops forcing sooner instead of stalling into an "
                        "overload. Try 40-60. Omit to leave the driver's value alone.")
    p.add_argument("--width-speed", type=int, default=None, metavar="MM/S",
                   help="finger open/close speed, 1..200 mm/s. The driver sets 150 at "
                        "startup but measured travel was ~18 mm/s, which made CLOSE outlast "
                        "the phase machine's patience. Try 100-150 to close quickly. Omit to "
                        "use the plain set_target_finger_width service.")
    args, ros_args = p.parse_known_args()

    nss = args.ns if args.ns else ["/ag_left"]
    nss = [n.rstrip("/") for n in nss]
    if len(set(nss)) != len(nss):
        p.error(f"--ns repeated with the same namespace: {nss}")

    if args.shm:
        if len(args.shm) != len(nss):
            p.error(f"got {len(nss)} --ns and {len(args.shm)} --shm; "
                    f"pass one --shm per --ns, in the same order")
        shms = args.shm
    elif len(nss) == 1:
        shms = [SHM_NAME]                      # single arm: unchanged default
    else:
        # /ag_left -> /judo_gripper_left. Derived from the namespace so the pairing is
        # visible in `ls /dev/shm` and cannot be mismatched by argument order.
        shms = [f"{SHM_NAME}_{n.rsplit('/', 1)[-1].removeprefix('ag_')}" for n in nss]
    if len(set(shms)) != len(shms):
        p.error(f"--shm names must differ, got {shms}")

    rclpy.init(args=ros_args)
    node = GripperBridge(list(zip(nss, shms)), args.dry_run,
                         args.contact_sensitivity, args.width_speed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("interrupted")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
