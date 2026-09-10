// CANONICAL dual-arm mjpc <-> franka_ec bridge. 184 B, POSIX shared memory.
//
// This file must stay BYTE-IDENTICAL in every tree that uses it:
//   franka_ec/include/franka_ec/mjpc_bridge_dual.h                 (owner side)
//   mujoco_mpc/mjpc/tasks/Fr3HGripperDual/mjpc_bridge_dual.h        (mjpc side)
// Copy it; never edit one copy alone, and never #include across trees.
// `franka_ec/scripts/check_bridge_headers.sh` diffs every copy.
//
// WHY A SEPARATE STRUCT FROM THE SINGLE-ARM MjpcBridge (120 B, "/mjpc_bridge")
//
// FR3_H_Gripper_Dual plans both arms in ONE MPPI tick (nu = 16, per-arm softmax but a
// single planner). Two independent 7-DOF regions would give each arm its own sequence
// counter, so the left arm could act on tick N while the right acts on N-1. For a
// coordinated carry that tearing is a real hazard, so both arms share one region and
// one pair of counters: a state or action snapshot is always self-consistent.
//
// The name differs from "/mjpc_bridge" ON PURPOSE. The single worst failure mode of the
// single-arm bridge was two DIFFERENT structs sharing one shm NAME: the layouts
// misaligned, one side corrupted the other's sequence counter, and both processes
// reported success while the arm sat still. A distinct name makes that impossible here.
// The magic + struct_size preamble then makes a *version* mismatch self-reporting
// instead of silent -- attach fails loudly rather than misreading the payload.

#ifndef MJPC_BRIDGE_DUAL_H_
#define MJPC_BRIDGE_DUAL_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define MJPC_DUAL_SHM_NAME "/mjpc_bridge_dual"

// 'M' 'J' 'D' '2' -- bump the last digit if the layout below ever changes, so an old
// binary refuses to attach instead of misreading the region.
#define MJPC_DUAL_MAGIC 0x4D4A4432

// Joint index order in every 14-vector below: left_joint1..7 then right_joint1..7.
// "left"/"right" are the franka_ec arm_ids; they map to the model's l_ / r_ prefixes.
#define MJPC_DUAL_NARM 2
#define MJPC_DUAL_NJOINT 7
#define MJPC_DUAL_NDOF 14

struct MjpcBridgeDual {
  // ---- preamble, written once by the owner ----
  int32_t magic;         //   0  MJPC_DUAL_MAGIC
  int32_t struct_size;   //   4  sizeof(MjpcBridgeDual)

  // ---- controller -> mjpc ----
  float q[MJPC_DUAL_NDOF];    //   8  measured joint angles, rad
  float dq[MJPC_DUAL_NDOF];   //  64  measured joint velocities, rad/s
  int32_t state_seq;          // 120  bumped by the controller AFTER writing q/dq

  // ---- mjpc -> controller ----
  float action[MJPC_DUAL_NDOF];  // 124  feedforward torque, N*m
  int32_t action_seq;            // 180  bumped by mjpc AFTER writing action
};                               // 184 B total

// No ee_pos and no target_pos, unlike the single-arm struct. Both were dead weight
// there -- nothing wrote ee_pos and no task read it -- and this struct has no
// backwards layout to preserve, so they are simply absent. New signals (grippers, a
// camera pose) go in a NEW region, never by growing this one: a field added here
// invalidates every copy of the header and every binary at once.

static_assert(sizeof(MjpcBridgeDual) == 184,
              "MjpcBridgeDual layout changed -- update every copy of this header, bump "
              "MJPC_DUAL_MAGIC, and rebuild both sides");

// Open the shared-memory region.
//   owner == true   create, zero, stamp magic/struct_size. The ROS 2 controller only.
//   owner == false  attach to an existing region and VERIFY the preamble. mjpc only.
// Returns nullptr on failure, which for a non-owner is a normal, recoverable state:
// the controller may simply not be running yet. Retry; do not treat it as fatal.
inline MjpcBridgeDual* mjpc_bridge_dual_open(bool owner) {
  const int flags = owner ? (O_CREAT | O_RDWR) : O_RDWR;
  int fd = shm_open(MJPC_DUAL_SHM_NAME, flags, 0666);
  if (fd < 0) return nullptr;

  if (owner) {
    if (ftruncate(fd, sizeof(MjpcBridgeDual)) != 0) {
      close(fd);
      return nullptr;
    }
  } else {
    // A region left over from an older/smaller layout would make mmap() succeed and
    // then SIGBUS on first touch past its end. Check the real size before mapping.
    struct stat st;
    if (fstat(fd, &st) != 0 ||
        static_cast<size_t>(st.st_size) < sizeof(MjpcBridgeDual)) {
      fprintf(stderr,
              "[mjpc_bridge_dual] %s is %lld B, expected >= %zu B -- stale region from "
              "a different build. Stop the controller (it unlinks on deactivate) or "
              "remove /dev/shm%s\n",
              MJPC_DUAL_SHM_NAME, static_cast<long long>(st.st_size),
              sizeof(MjpcBridgeDual), MJPC_DUAL_SHM_NAME);
      close(fd);
      return nullptr;
    }
  }

  void* ptr = mmap(nullptr, sizeof(MjpcBridgeDual), PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd, 0);
  close(fd);  // the mapping keeps the region alive; the fd is not needed
  if (ptr == MAP_FAILED) return nullptr;

  MjpcBridgeDual* bridge = static_cast<MjpcBridgeDual*>(ptr);

  if (owner) {
    std::memset(bridge, 0, sizeof(MjpcBridgeDual));
    // Payload zeroed FIRST, preamble LAST: a non-owner attaching mid-init sees
    // magic == 0, fails its check, and retries -- rather than reading zeros as state.
    bridge->struct_size = static_cast<int32_t>(sizeof(MjpcBridgeDual));
    bridge->magic = MJPC_DUAL_MAGIC;
  } else {
    if (bridge->magic != MJPC_DUAL_MAGIC ||
        bridge->struct_size != static_cast<int32_t>(sizeof(MjpcBridgeDual))) {
      // Either the owner is still initialising (magic == 0) or the two sides were
      // built from different headers. Both are handled by returning nullptr and
      // retrying; the second one prints so it cannot be mistaken for the first.
      if (bridge->magic != 0) {
        fprintf(stderr,
                "[mjpc_bridge_dual] header MISMATCH: region magic=0x%08X size=%d, this "
                "build expects magic=0x%08X size=%zu. The two sides were built from "
                "different copies of mjpc_bridge_dual.h -- diff them and rebuild "
                "BOTH.\n",
                bridge->magic, bridge->struct_size, MJPC_DUAL_MAGIC,
                sizeof(MjpcBridgeDual));
      }
      munmap(ptr, sizeof(MjpcBridgeDual));
      return nullptr;
    }
  }

  return bridge;
}

inline void mjpc_bridge_dual_close(MjpcBridgeDual* bridge) {
  if (bridge) munmap(bridge, sizeof(MjpcBridgeDual));
}

// OWNER ONLY. mjpc must never unlink: a region removed under a running controller
// would be recreated by nobody, and the controller's next write would go to a mapping
// no new mjpc can find by name.
inline void mjpc_bridge_dual_unlink() { shm_unlink(MJPC_DUAL_SHM_NAME); }

#endif  // MJPC_BRIDGE_DUAL_H_
