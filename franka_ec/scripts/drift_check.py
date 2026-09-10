#!/usr/bin/env python3
"""Measure which joint drifts under zero commanded torque, and how fast.

Eyeballing a slow drift tells you it happens; it does not tell you WHICH joint or
whether the residual torque explains it. This samples q and tau_ext_hat_filtered at
the start and end of a window and prints the per-joint drift next to the residual, so
the two can be read together.

Run it while a zero-torque controller is active (gravity_compensation_example_controller,
or MjpcDualBridgeController with no planner attached):

    python3 drift_check.py                      # 30 s, single-arm topic
    python3 drift_check.py --seconds 60
    python3 drift_check.py --topic /franka_right_robot_state_broadcaster/robot_state

Nothing is commanded -- this only listens.
"""
import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from franka_msgs.msg import FrankaState


class DriftCheck(Node):
    def __init__(self, topic):
        super().__init__('drift_check')
        self.first = None
        self.last = None
        self.count = 0
        self.create_subscription(FrankaState, topic, self._cb, 10)

    def _cb(self, msg):
        sample = (list(msg.q), list(msg.tau_ext_hat_filtered))
        if self.first is None:
            self.first = sample
        self.last = sample
        self.count += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/franka_robot_state_broadcaster/robot_state')
    ap.add_argument('--seconds', type=float, default=30.0)
    args = ap.parse_args()

    rclpy.init()
    node = DriftCheck(args.topic)
    print(f'listening on {args.topic} for {args.seconds:.0f} s ...', flush=True)

    end = node.get_clock().now().nanoseconds + int(args.seconds * 1e9)
    while rclpy.ok() and node.get_clock().now().nanoseconds < end:
        rclpy.spin_once(node, timeout_sec=0.2)

    if node.count == 0:
        print(f'no messages on {args.topic}.\n'
              '  - is a controller running?  ros2 topic list | grep robot_state',
              file=sys.stderr)
        rclpy.shutdown()
        return 1

    q0, t0 = node.first
    q1, t1 = node.last
    print(f'{node.count} samples\n')
    print(f'{"joint":>6} {"drift [deg]":>12} {"q start":>10} {"q end":>10} '
          f'{"tau_ext start":>14} {"tau_ext end":>12}')
    worst, worst_j = 0.0, 0
    for i in range(7):
        d = math.degrees(q1[i] - q0[i])
        if abs(d) > abs(worst):
            worst, worst_j = d, i + 1
        print(f'{i+1:>6} {d:>12.3f} {q0[i]:>10.4f} {q1[i]:>10.4f} '
              f'{t0[i]:>14.3f} {t1[i]:>12.3f}')

    rate = worst / args.seconds * 60.0
    print(f'\nlargest drift: joint{worst_j}  {worst:+.3f} deg over {args.seconds:.0f} s '
          f'({rate:+.2f} deg/min)')
    print(f'its tau_ext_hat_filtered: {t0[worst_j-1]:+.3f} -> {t1[worst_j-1]:+.3f} Nm')
    print('\nRead the two together: a joint that drifts AND carries the largest residual\n'
          'points at that joint\'s model/calibration. A joint that drifts with a small\n'
          'residual points at friction/brake/drive instead.')
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
