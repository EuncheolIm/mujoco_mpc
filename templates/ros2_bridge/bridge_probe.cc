// Standalone /mjpc_bridge probe. NO mjpc, NO ROS -- just this file and librt.
//
//   g++ -O2 -o bridge_probe bridge_probe.cc -lrt
//
//   ./bridge_probe watch          print q/dq/seq at 5 Hz, verify the controller writes
//   ./bridge_probe zero           publish zero torque at 100 Hz (= gravity comp, safe)
//   ./bridge_probe hold 0 1.5     publish 1.5 Nm on joint 0, everything else zero
//   ./bridge_probe size           print sizeof/offsetof and compare against 120 B
//
// WHY THIS EXISTS. When the arm does not move there are two suspects -- the controller
// and the planner -- and separating them by editing either one is slow. This attaches
// as a non-owner exactly the way a task does, so:
//   * `watch` shows a rising state_seq  => the controller is alive and publishing
//   * `zero` accepted without a timeout => the action path works end to end
//   * `size` mismatching 120 B          => the header trap in the guide, section 2.2
//
// SAFETY. `hold` commands real torque on a real arm. Start from `zero`, keep the
// magnitude small, and remember the controller rate-limits to 1 Nm/step and drops to
// gravity compensation after 100 ms without a fresh action_seq -- so killing this
// program is itself a safe stop.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <ctime>
#include <csignal>

#include "mjpc_bridge.h"

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int) { g_stop = 1; }

static void sleep_ms(long ms) {
  struct timespec ts{ms / 1000, (ms % 1000) * 1000000L};
  nanosleep(&ts, nullptr);
}

static void print_size() {
  printf("sizeof(MjpcBridge) = %zu B  (franka_ec expects 120)\n", sizeof(MjpcBridge));
  printf("  q          @ %3zu\n", offsetof(MjpcBridge, q));
  printf("  dq         @ %3zu\n", offsetof(MjpcBridge, dq));
  printf("  ee_pos     @ %3zu\n", offsetof(MjpcBridge, ee_pos));
  printf("  state_seq  @ %3zu\n", offsetof(MjpcBridge, state_seq));
  printf("  action     @ %3zu\n", offsetof(MjpcBridge, action));
  printf("  action_seq @ %3zu\n", offsetof(MjpcBridge, action_seq));
  printf("  target_pos @ %3zu\n", offsetof(MjpcBridge, target_pos));
  printf("  target_seq @ %3zu\n", offsetof(MjpcBridge, target_seq));
  if (sizeof(MjpcBridge) != 120) {
    printf("\n  *** MISMATCH: this header is not the one the controller was built with.\n"
           "      Every copy of mjpc_bridge.h must be identical -- see the guide, 2.2.\n");
  }
}

int main(int argc, char** argv) {
  const char* mode = (argc > 1) ? argv[1] : "watch";
  if (!std::strcmp(mode, "size")) { print_size(); return 0; }

  signal(SIGINT, on_sigint);

  // NON-owner, exactly as a task attaches. The controller creates the region in
  // on_activate and unlinks it in on_deactivate.
  MjpcBridge* b = mjpc_bridge_open(false);
  if (!b) {
    fprintf(stderr, "/mjpc_bridge not present. Start the controller first:\n"
                    "  ros2 launch franka_bringup mppi_track_controller.launch.py "
                    "robot_ip:=172.16.0.2\n");
    return 1;
  }
  printf("attached (%zu B). Ctrl-C to stop.\n", sizeof(MjpcBridge));

  if (!std::strcmp(mode, "watch")) {
    int32_t last = b->state_seq - 1;
    while (!g_stop) {
      const int32_t s = b->state_seq;
      printf("state_seq=%-10d %s  q=[% .3f % .3f % .3f % .3f % .3f % .3f % .3f]\n",
             s, (s == last) ? "STALLED" : "ok     ",
             b->q[0], b->q[1], b->q[2], b->q[3], b->q[4], b->q[5], b->q[6]);
      last = s;
      sleep_ms(200);
    }
  } else if (!std::strcmp(mode, "zero") || !std::strcmp(mode, "hold")) {
    int joint = -1;
    double value = 0.0;
    if (!std::strcmp(mode, "hold")) {
      if (argc < 4) { fprintf(stderr, "usage: bridge_probe hold <joint 0-6> <Nm>\n"); return 2; }
      joint = std::atoi(argv[2]);
      value = std::atof(argv[3]);
      if (joint < 0 || joint > 6) { fprintf(stderr, "joint must be 0..6\n"); return 2; }
      // The controller's own ceiling: {87,87,87,87,12,12,12}. Refuse anything near it
      // here too -- a probe has no business exercising the limit path.
      const double cap = (joint < 4) ? 10.0 : 3.0;
      if (value > cap || value < -cap) {
        fprintf(stderr, "refusing %.2f Nm on joint %d: this probe caps at +-%.0f\n",
                value, joint, cap);
        return 2;
      }
      printf("publishing %.2f Nm on joint %d at 100 Hz\n", value, joint);
    } else {
      printf("publishing ZERO torque at 100 Hz (controller reads 0 as gravity comp)\n");
    }
    while (!g_stop) {
      // Payload first, counter last: the controller only reads when action_seq moves.
      for (int i = 0; i < 7; ++i) b->action[i] = 0.0f;
      if (joint >= 0) b->action[joint] = static_cast<float>(value);
      b->action_seq++;
      sleep_ms(10);   // 100 Hz; the controller times out after 100 ms of silence
    }
    // Leave a zero command behind rather than a stale torque.
    for (int i = 0; i < 7; ++i) b->action[i] = 0.0f;
    b->action_seq++;
    printf("\nstopped; published zero on the way out.\n");
  } else {
    fprintf(stderr, "unknown mode '%s' (watch | zero | hold | size)\n", mode);
    mjpc_bridge_close(b);
    return 2;
  }

  mjpc_bridge_close(b);
  return 0;
}
