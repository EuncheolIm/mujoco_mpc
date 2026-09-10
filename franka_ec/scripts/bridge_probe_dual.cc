// Standalone /mjpc_bridge_dual probe. NO mjpc, NO ROS -- just this file and librt.
//
//   g++ -O2 -std=c++17 -I../include/franka_ec -o bridge_probe_dual bridge_probe_dual.cc -lrt
//
//   ./bridge_probe_dual size            offsets + a hard check against 184 B
//   ./bridge_probe_dual watch           print both arms' q and state_seq at 5 Hz
//   ./bridge_probe_dual zero            publish zero torque on all 14 at 100 Hz
//   ./bridge_probe_dual hold l 1 1.5    1.5 Nm on left_joint1, zero everywhere else
//
// WHY THIS EXISTS. When the arms do not move there are two suspects -- the controller
// and the planner -- and separating them by editing either one is slow. This attaches
// as a NON-owner exactly the way the mjpc task does, so:
//   * `watch` shows a rising state_seq  => the controller is alive and publishing
//   * `zero` accepted without a timeout => the action path works end to end
//   * `size` mismatching 184 B          => the two sides were built from different
//                                          copies of mjpc_bridge_dual.h
// Only after `zero` runs clean is a missing motion the task's fault.
//
// SAFETY. `hold` commands real torque on a real arm. Start from `zero`, keep the
// magnitude small, and remember that the controller slew-limits to 1 Nm/step and drops
// BOTH arms to gravity compensation after 100 ms without a fresh action_seq -- so
// killing this program is itself a safe stop.

#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "mjpc_bridge_dual.h"

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int) { g_stop = 1; }

static void sleep_ms(long ms) {
  struct timespec ts {ms / 1000, (ms % 1000) * 1000000L};
  nanosleep(&ts, nullptr);
}

static void print_size() {
  printf("sizeof(MjpcBridgeDual) = %zu B  (both sides expect 184)\n",
         sizeof(MjpcBridgeDual));
  printf("  magic       @ %3zu   (0x%08X)\n", offsetof(MjpcBridgeDual, magic),
         MJPC_DUAL_MAGIC);
  printf("  struct_size @ %3zu\n", offsetof(MjpcBridgeDual, struct_size));
  printf("  q[14]       @ %3zu\n", offsetof(MjpcBridgeDual, q));
  printf("  dq[14]      @ %3zu\n", offsetof(MjpcBridgeDual, dq));
  printf("  state_seq   @ %3zu\n", offsetof(MjpcBridgeDual, state_seq));
  printf("  action[14]  @ %3zu\n", offsetof(MjpcBridgeDual, action));
  printf("  action_seq  @ %3zu\n", offsetof(MjpcBridgeDual, action_seq));
  printf("\nindex order: 0-6 = arm_1 (left) joint1..7, 7-13 = arm_2 (right) joint1..7\n");
  if (sizeof(MjpcBridgeDual) != 184) {
    printf("\n  *** MISMATCH: this header is not the one the controller was built "
           "with.\n      Run check_bridge_headers.sh and rebuild BOTH sides.\n");
  }
}

static void print_arm(const char* tag, const float* v) {
  printf("  %s [% .3f % .3f % .3f % .3f % .3f % .3f % .3f]\n", tag, v[0], v[1], v[2],
         v[3], v[4], v[5], v[6]);
}

int main(int argc, char** argv) {
  const char* mode = (argc > 1) ? argv[1] : "watch";
  if (!std::strcmp(mode, "size")) {
    print_size();
    return 0;
  }

  // SIGTERM as well as SIGINT: this gets run under `timeout` and from scripts,
  // and the exit path (publish zero, then close) is worth taking in both cases.
  signal(SIGINT, on_sigint);
  signal(SIGTERM, on_sigint);

  // NON-owner, exactly as the task attaches. The controller creates the region in
  // on_activate and unlinks it in on_deactivate. A header mismatch is reported by
  // mjpc_bridge_dual_open() itself.
  MjpcBridgeDual* b = mjpc_bridge_dual_open(false);
  if (!b) {
    fprintf(stderr,
            "%s not present (or header mismatch, see above). Start the controller "
            "first:\n"
            "  ros2 launch franka_bringup mjpc_dual_bridge_controller.py \\\n"
            "      robot_ip_1:=172.16.0.2 robot_ip_2:=172.16.1.2\n",
            MJPC_DUAL_SHM_NAME);
    return 1;
  }
  printf("attached to %s (%zu B). Ctrl-C to stop.\n", MJPC_DUAL_SHM_NAME,
         sizeof(MjpcBridgeDual));

  if (!std::strcmp(mode, "watch")) {
    int32_t last = b->state_seq - 1;
    while (!g_stop) {
      const int32_t s = b->state_seq;
      printf("state_seq=%-10d %s\n", s, (s == last) ? "STALLED" : "ok");
      print_arm("L q ", &b->q[0]);
      print_arm("R q ", &b->q[MJPC_DUAL_NJOINT]);
      last = s;
      sleep_ms(200);
    }
  } else if (!std::strcmp(mode, "zero") || !std::strcmp(mode, "hold")) {
    int index = -1;
    double value = 0.0;
    if (!std::strcmp(mode, "hold")) {
      if (argc < 5) {
        fprintf(stderr, "usage: bridge_probe_dual hold <l|r> <joint 1-7> <Nm>\n");
        return 2;
      }
      const char* arm = argv[2];
      const int joint = std::atoi(argv[3]);
      value = std::atof(argv[4]);
      int a;
      if (!std::strcmp(arm, "l")) {
        a = 0;
      } else if (!std::strcmp(arm, "r")) {
        a = 1;
      } else {
        fprintf(stderr, "arm must be 'l' or 'r'\n");
        return 2;
      }
      if (joint < 1 || joint > MJPC_DUAL_NJOINT) {
        fprintf(stderr, "joint must be 1..7\n");
        return 2;
      }
      index = a * MJPC_DUAL_NJOINT + (joint - 1);
      // The controller's ceiling is {87,87,87,87,12,12,12}. Cap far below it here: a
      // probe has no business exercising the limit path.
      const double cap = (joint <= 4) ? 10.0 : 3.0;
      if (value > cap || value < -cap) {
        fprintf(stderr, "refusing %.2f Nm on %s_joint%d: this probe caps at +-%.0f\n",
                value, a == 0 ? "left" : "right", joint, cap);
        return 2;
      }
      printf("publishing %.2f Nm on %s_joint%d (index %d) at 100 Hz\n", value,
             a == 0 ? "left" : "right", joint, index);
    } else {
      printf("publishing ZERO torque on all %d joints at 100 Hz "
             "(the controller reads 0 as gravity compensation)\n", MJPC_DUAL_NDOF);
    }
    while (!g_stop) {
      // Payload first, counter last: the controller only reads when action_seq moves.
      for (int k = 0; k < MJPC_DUAL_NDOF; ++k) b->action[k] = 0.0f;
      if (index >= 0) b->action[index] = static_cast<float>(value);
      b->action_seq++;
      sleep_ms(10);  // 100 Hz; the controller times out after 100 ms of silence
    }
    // Leave a zero command behind rather than a stale torque.
    for (int k = 0; k < MJPC_DUAL_NDOF; ++k) b->action[k] = 0.0f;
    b->action_seq++;
    printf("\nstopped; published zero on the way out.\n");
  } else {
    fprintf(stderr, "unknown mode '%s' (size | watch | zero | hold)\n", mode);
    mjpc_bridge_dual_close(b);
    return 2;
  }

  mjpc_bridge_dual_close(b);
  return 0;
}
