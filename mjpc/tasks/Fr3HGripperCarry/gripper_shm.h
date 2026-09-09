// Real H-gripper command, over the SAME /judo_gripper region judo already uses.
//
// Layout is dictated by prior_mppi_judo/ours/gripper_shm.py -- 10 x int32, 40 B.
// Do not reorder: gripper_bridge_node.py (rclpy) is the other end and judo's
// run_real.py also writes here. Growing the struct would break both.
//
//   offset  field            dir              note
//        0  want_open        -> node          1 = send width_open_mm, 0 = width_close_mm
//        4  finger_pose_deg  -> node          180 = 2-finger mode
//        8  width_open_mm    -> node          0..100
//       12  width_close_mm   -> node          0..100
//       16  grip_force_n     -> node          1..100
//       20  motor_on         -> node          1 = request motor on
//       24  cmd_seq          -> node          bumped LAST (seqlock)
//       28  status_word      <- node          GripperStatus.gripper_status
//       32  finger_width_mm  <- node          MEASURED width
//       36  ack_seq          <- node          bumped LAST
//
// HOW A CONTINUOUS WIDTH IS SENT: the node picks width_open_mm or width_close_mm
// according to want_open, so to command an arbitrary width we set want_open = 0
// and put the value in width_close_mm. The field name then reads oddly, but this
// keeps judo's node and shm layout untouched, which is worth more than a tidier
// name. The node only issues a service call when the value CHANGES, so a steady
// width costs no EtherCAT traffic.

#ifndef MJPC_TASKS_FR3HGRIPPERCARRY_GRIPPER_SHM_H_
#define MJPC_TASKS_FR3HGRIPPERCARRY_GRIPPER_SHM_H_

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define MJPC_GRIPPER_SHM_NAME "/judo_gripper"

struct JudoGripper {
  int32_t want_open;
  int32_t finger_pose_deg;
  int32_t width_open_mm;
  int32_t width_close_mm;
  int32_t grip_force_n;
  int32_t motor_on;
  int32_t cmd_seq;
  int32_t status_word;
  int32_t finger_width_mm;
  int32_t ack_seq;
};
static_assert(sizeof(JudoGripper) == 40, "must match gripper_shm.py's 10i layout");

// The bridge NODE owns the region, so a null return just means it is not running.
inline JudoGripper* mjpc_gripper_open() {
  int fd = shm_open(MJPC_GRIPPER_SHM_NAME, O_RDWR, 0666);
  if (fd < 0) return nullptr;
  void* p = mmap(nullptr, sizeof(JudoGripper), PROT_READ | PROT_WRITE, MAP_SHARED,
                 fd, 0);
  close(fd);
  if (p == MAP_FAILED) return nullptr;
  return static_cast<JudoGripper*>(p);
}

inline void mjpc_gripper_close(JudoGripper* g) {
  if (g) munmap(g, sizeof(JudoGripper));
}

// Publish a width in mm. Payload first, cmd_seq last (seqlock, same discipline as
// the Python side).
inline void mjpc_gripper_write_width(JudoGripper* g, int width_mm, int pose_deg,
                                     int force_n) {
  if (!g) return;
  if (width_mm < 0) width_mm = 0;
  if (width_mm > 100) width_mm = 100;
  g->want_open = 0;                 // -> the node reads width_close_mm
  g->finger_pose_deg = pose_deg;
  g->width_close_mm = width_mm;
  g->width_open_mm = 100;
  g->grip_force_n = force_n;
  g->motor_on = 1;
  g->cmd_seq = g->cmd_seq + 1;      // LAST
}

// Measured width, or -1 when the node has not reported yet.
inline int mjpc_gripper_measured(const JudoGripper* g) {
  if (!g) return -1;
  for (int attempt = 0; attempt < 8; ++attempt) {
    const int32_t a0 = g->ack_seq;
    if (a0 <= 0) return -1;
    const int32_t w = g->finger_width_mm;
    const int32_t a1 = g->ack_seq;
    if (a0 == a1) return static_cast<int>(w);
  }
  return -1;
}

#endif  // MJPC_TASKS_FR3HGRIPPERCARRY_GRIPPER_SHM_H_
