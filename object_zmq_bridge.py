#!/usr/bin/env python3
"""Camera object pose: ZMQ -> /mjpc_object shared memory, for mjpc's C++ tasks.

Owns the socket and the shm region so that libmjpc never has to link libzmq, and
so the ZMQ wire-format handling stays in ONE place. That parser is not obvious and
was worked out on hardware: the feed sends 7 RAW float32 (no framing, no JSON),
the quaternion is XYZW rather than MuJoCo's w-first, the transport is PUSH so we
must PULL (not SUB), and the position needs a hand-measured frame offset. All of
that lives in prior_mppi_judo/ours/object_source.py and is imported here rather
than reimplemented.

    # judo's venv has pyzmq; the system python3 may not
    cd .../mujoco_mpc-hgripper
    env -u PYTHONPATH ../prior_mppi_judo/.venv/bin/python object_zmq_bridge.py \
        --endpoint tcp://161.122.114.37:5557 --offset 0.02 0.30 0.01

Then start mjpc with MJPC_OBJECT_SHM=1 and it will drive the task's free-jointed
object from this feed instead of the model's default pose.

Verify the numbers BEFORE trusting them: ours/view_object_zmq.py in the judo tree
draws the pose in a viewer, which is the only way to confirm the offset sign.
"""
import argparse
import mmap
import os
import signal
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
JUDO_OURS = os.path.abspath(os.path.join(HERE, "..", "prior_mppi_judo", "ours"))
if JUDO_OURS not in sys.path:
    sys.path.insert(0, JUDO_OURS)
try:
    import object_source as OS          # noqa: E402
except ImportError as e:                # pragma: no cover
    sys.exit(f"cannot import object_source from {JUDO_OURS}: {e}\n"
             f"  That module owns the ZMQ wire format. If the judo tree moved, pass its\n"
             f"  'ours' directory on PYTHONPATH or fix JUDO_OURS in this file.")

SHM_PATH = "/dev/shm/mjpc_object"
# Must match struct MjpcObject in mjpc/tasks/Fr3HGripperCarry/object_shm.h
FMT = "<3f4fii"                          # pos[3], quat[4] (w first), seq, valid
SIZE = struct.calcsize(FMT)              # 36
POS_OFF, QUAT_OFF, SEQ_OFF, VALID_OFF = 0, 12, 28, 32


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", default="tcp://161.122.114.37:5557")
    p.add_argument("--offset", type=float, nargs=3, default=[0.02, 0.30, 0.01],
                   metavar=("DX", "DY", "DZ"),
                   help="added to every incoming position to reach the robot base "
                        "frame. Confirm the SIGN with view_object_zmq.py first.")
    p.add_argument("--sub", action="store_true",
                   help="feed is PUB/SUB rather than PUSH/PULL")
    p.add_argument("--quat-wxyz", action="store_true",
                   help="feed sends w-first; default assumes xyzw, as the camera does")
    p.add_argument("--stale", type=float, default=0.5, metavar="S",
                   help="mark the region invalid after this long with no message, so "
                        "mjpc stops moving the object instead of freezing it at a "
                        "pose that may already be wrong (default 0.5 s)")
    p.add_argument("--rate", type=float, default=100.0, metavar="HZ")
    args = p.parse_args()

    # pkill sends SIGTERM, whose default action skips `finally`, so the region was
    # left behind and the next start refused with "already exists". Turn it into
    # KeyboardInterrupt so the normal cleanup path runs.
    def _term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGHUP, _term)

    if os.path.exists(SHM_PATH):
        print(f"!! {SHM_PATH} already exists -- another bridge is probably running.\n"
              f"   Two writers on one region interleave and produce nonsense poses.")
        return 2

    fd = os.open(SHM_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    os.ftruncate(fd, SIZE)
    mm = mmap.mmap(fd, SIZE)
    mm.seek(0); mm.write(b"\x00" * SIZE)

    src = OS.ZmqObjectPose(args.endpoint, offset=args.offset,
                           quat_xyzw=not args.quat_wxyz, pull=not args.sub,
                           stale_after_s=args.stale)
    print(f"feed {args.endpoint}  {'SUB' if args.sub else 'PULL'}  "
          f"quat={'wxyz' if args.quat_wxyz else 'xyzw'}")
    print(f"offset {args.offset}   -> {SHM_PATH} ({SIZE} B)")
    print("start mjpc with MJPC_OBJECT_SHM=1 to use it.  Ctrl+C to stop.\n")

    seq, last_report, n_valid = 0, time.monotonic(), 0
    try:
        period = 1.0 / max(args.rate, 1.0)
        while True:
            got = src.get()
            # Payload first, THEN seq: a reader that sees the same seq before and
            # after its read knows the payload did not change under it.
            if got is None:
                struct.pack_into("i", mm, VALID_OFF, 0)
            else:
                pos, quat = got
                struct.pack_into("3f", mm, POS_OFF, *pos)
                struct.pack_into("4f", mm, QUAT_OFF, *quat)   # already w-first
                struct.pack_into("i", mm, VALID_OFF, 1)
                n_valid += 1
            seq += 1
            struct.pack_into("i", mm, SEQ_OFF, seq)

            now = time.monotonic()
            if now - last_report >= 1.0:
                last_report = now
                if got is not None:
                    print(f"pos {got[0].round(4)}  quat {got[1].round(4)}  "
                          f"seq {seq}  [{src.stats}]")
                else:
                    print(f"no fresh pose (feed quiet > {args.stale}s)  seq {seq}  "
                          f"[{src.stats}]")
            time.sleep(period)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        # Mark invalid before going away, so mjpc stops trusting the last pose even
        # if it keeps its mapping alive (mmap survives unlink).
        try:
            struct.pack_into("i", mm, VALID_OFF, 0)
            struct.pack_into("i", mm, SEQ_OFF, seq + 1)
        except Exception:
            pass
        src.close(); mm.close(); os.close(fd)
        try:
            os.unlink(SHM_PATH)
            print(f"unlinked {SHM_PATH}")
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
