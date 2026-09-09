// COPIED VERBATIM from franka_ec/include/franka_ec/mjpc_bridge.h -- do not edit the
// layout here independently. The controller mmaps sizeof(MjpcBridge) on the same
// /mjpc_bridge name, so any field added or removed on one side silently shifts every
// offset on the other. mjpc/tasks/mppi_track/mjpc_bridge.h is an OLDER 92-byte variant
// (no ee_pos, no target_pos) and is NOT compatible with the running controller: it puts
// state_seq at byte 56 where the controller writes ee_pos, and action at 60 where the
// controller expects state_seq. Use this 120-byte copy.
#ifndef MJPC_BRIDGE_H_
#define MJPC_BRIDGE_H_

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define MJPC_SHM_NAME "/mjpc_bridge"

struct MjpcBridge {
  // controller → mjpc / act_only
  float q[7];
  float dq[7];
  float ee_pos[3];       // EE position from FK (world frame)
  int32_t state_seq;     // bumped by controller after writing q/dq/ee_pos

  // mjpc / act_only → controller
  float action[7];
  int32_t action_seq;    // bumped by mjpc after writing action

  // external → mjpc (target override)
  float target_pos[3];   // desired EE position (x, y, z) in world frame
  int32_t target_seq;    // bumped by external script after writing target_pos
};

// Open (or create) the shared memory region. Returns pointer, or nullptr on failure.
// owner=true: creates & initializes to zero.  owner=false: opens existing.
inline MjpcBridge* mjpc_bridge_open(bool owner) {
  int flags = owner ? (O_CREAT | O_RDWR) : O_RDWR;
  int fd = shm_open(MJPC_SHM_NAME, flags, 0666);
  if (fd < 0) return nullptr;

  if (owner) {
    if (ftruncate(fd, sizeof(MjpcBridge)) != 0) {
      close(fd);
      return nullptr;
    }
  }

  void* ptr = mmap(nullptr, sizeof(MjpcBridge), PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, 0);
  close(fd);  // fd not needed after mmap

  if (ptr == MAP_FAILED) return nullptr;

  if (owner) {
    std::memset(ptr, 0, sizeof(MjpcBridge));
  }

  return static_cast<MjpcBridge*>(ptr);
}

inline void mjpc_bridge_close(MjpcBridge* bridge) {
  if (bridge) {
    munmap(bridge, sizeof(MjpcBridge));
  }
}

inline void mjpc_bridge_unlink() {
  shm_unlink(MJPC_SHM_NAME);
}

#endif  // MJPC_BRIDGE_H_
