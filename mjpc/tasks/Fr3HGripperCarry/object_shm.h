// Object pose from a camera, handed over in POSIX shared memory.
//
// WHY SHM AND NOT ZMQ DIRECTLY: the wire format on the camera side is not
// self-describing -- it is 7 raw float32 (pos xyz + quat XYZW, not w-first), sent
// over PUSH/PULL, and it needs a hand-measured frame offset. All of that already
// lives, tested, in prior_mppi_judo/ours/object_source.py. Re-implementing the
// parser in C++ would mean re-deriving the byte order, the quaternion convention
// and the offset sign, every one of which was found the hard way on hardware.
// So object_zmq_bridge.py owns the socket and writes here; this header only reads.
// It also keeps libzmq out of libmjpc, which every other task links against.
//
// Layout (36 B, little-endian, all fixed-width):
//   offset  field       note
//        0  pos[3]      metres, ROBOT BASE frame (offset already applied)
//       12  quat[4]     MuJoCo order, w first (the bridge converts from xyzw)
//       28  seq         bumped LAST by the writer; seqlock
//       32  valid       1 = feed is live, 0 = gone quiet (writer sets it)
//
// The writer owns and unlinks the region, so a missing region simply means the
// bridge is not running and the caller should leave the simulated object alone.

#ifndef MJPC_TASKS_FR3HGRIPPERCARRY_OBJECT_SHM_H_
#define MJPC_TASKS_FR3HGRIPPERCARRY_OBJECT_SHM_H_

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define MJPC_OBJECT_SHM_NAME "/mjpc_object"

struct MjpcObject {
  float pos[3];
  float quat[4];      // w, x, y, z
  int32_t seq;        // writer bumps this AFTER the payload
  int32_t valid;
};

// Attach read-only-ish (RDWR so a future writer can share the header). Returns
// nullptr when the bridge is not running -- that is a normal state, not an error.
inline MjpcObject* mjpc_object_open() {
  int fd = shm_open(MJPC_OBJECT_SHM_NAME, O_RDWR, 0666);
  if (fd < 0) return nullptr;
  void* p = mmap(nullptr, sizeof(MjpcObject), PROT_READ | PROT_WRITE, MAP_SHARED,
                 fd, 0);
  close(fd);
  if (p == MAP_FAILED) return nullptr;
  return static_cast<MjpcObject*>(p);
}

inline void mjpc_object_close(MjpcObject* o) {
  if (o) munmap(o, sizeof(MjpcObject));
}

// Seqlock read. Returns false if no consistent snapshot could be taken, if the
// writer has not published yet, or if the feed has gone stale -- in every one of
// those cases the caller must NOT touch the object, so one bool is enough.
inline bool mjpc_object_read(const MjpcObject* o, double pos[3], double quat[4],
                             int32_t* seq_out) {
  if (!o) return false;
  for (int attempt = 0; attempt < 8; ++attempt) {
    const int32_t s0 = o->seq;
    if (s0 == 0) return false;              // nothing published yet
    float p[3], q[4];
    std::memcpy(p, o->pos, sizeof(p));
    std::memcpy(q, o->quat, sizeof(q));
    const int32_t valid = o->valid;
    const int32_t s1 = o->seq;
    if (s0 != s1) continue;                 // torn: writer ran mid-read
    if (!valid) return false;               // feed went quiet
    for (int i = 0; i < 3; ++i) pos[i] = static_cast<double>(p[i]);
    for (int i = 0; i < 4; ++i) quat[i] = static_cast<double>(q[i]);
    if (seq_out) *seq_out = s1;
    return true;
  }
  return false;
}

#endif  // MJPC_TASKS_FR3HGRIPPERCARRY_OBJECT_SHM_H_
