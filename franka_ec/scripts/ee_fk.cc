// Sim-side forward kinematics for one arm of the dual FR3 model, so it can be compared
// against the robot's own O_T_EE at the SAME joint angles.
//
// The bridge mirrors JOINT ANGLES only -- the sim derives the end-effector pose from
// its own kinematic chain. If that chain disagrees with the real arm plus its mounted
// gripper, the two end-effectors sit in different places for identical q, and mjpc
// plans to the wrong point. This prints every candidate frame so the offset can be
// attributed: a constant offset is a frame-definition difference, one that changes with
// pose is a kinematic mismatch.
//
//   g++ -O2 -std=c++17 -I$HOME/.local/include -o ee_fk ee_fk.cc \
//       -L$HOME/.local/lib -lmujoco -Wl,-rpath,$HOME/.local/lib
//   ./ee_fk <task.xml> <l|r> q1 q2 q3 q4 q5 q6 q7
//
// Positions are reported RELATIVE TO THAT ARM'S BASE, which is what the robot's own
// O_T_EE is relative to. The sim world origin sits between the two arms, so comparing
// world coordinates directly would be off by the base offset.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mujoco/mujoco.h>

int main(int argc, char** argv) {
  if (argc < 10) {
    fprintf(stderr, "usage: ee_fk <task.xml> <l|r> q1..q7\n");
    return 2;
  }
  const char* side = argv[2];
  char err[1024] = {0};
  mjModel* m = mj_loadXML(argv[1], nullptr, err, sizeof err);
  if (!m) { printf("LOAD FAILED: %s\n", err); return 1; }
  mjData* d = mj_makeData(m);
  if (m->nkey) mj_resetDataKeyframe(m, d, 0);

  char nm[64];
  for (int j = 1; j <= 7; ++j) {
    snprintf(nm, sizeof nm, "%s_fr3_joint%d", side, j);
    int jid = mj_name2id(m, mjOBJ_JOINT, nm);
    if (jid < 0) { printf("joint %s missing\n", nm); return 1; }
    d->qpos[m->jnt_qposadr[jid]] = atof(argv[2 + j]);
  }
  mj_kinematics(m, d);

  snprintf(nm, sizeof nm, "%s_base", side);
  int bb = mj_name2id(m, mjOBJ_BODY, nm);
  if (bb < 0) { printf("body %s missing\n", nm); return 1; }
  const double* org = d->xpos + 3 * bb;

  auto rel = [&](const char* label, const double* p, const double* R) {
    printf("  %-24s (%+.4f %+.4f %+.4f)", label, p[0] - org[0], p[1] - org[1],
           p[2] - org[2]);
    if (R) {
      // z axis of the frame: the direction the gripper points. Compared against
      // O_T_EE's third column.
      printf("   z_axis (%+.4f %+.4f %+.4f)", R[2], R[5], R[8]);
    }
    printf("\n");
  };
  printf("sim FK, relative to %s (arm base):\n", nm);
  snprintf(nm, sizeof nm, "%s_fr3_link7", side);
  int b7 = mj_name2id(m, mjOBJ_BODY, nm);
  if (b7 >= 0) rel("fr3_link7 origin", d->xpos + 3 * b7, d->xmat + 9 * b7);
  snprintf(nm, sizeof nm, "%s_hand", side);
  int bh = mj_name2id(m, mjOBJ_BODY, nm);
  if (bh >= 0) rel("hand origin (palm)", d->xpos + 3 * bh, d->xmat + 9 * bh);
  snprintf(nm, sizeof nm, "%s_hand_site", side);
  int hs = mj_name2id(m, mjOBJ_SITE, nm);
  if (hs >= 0) rel("hand_site  <- cost uses", d->site_xpos + 3 * hs, d->site_xmat + 9 * hs);
  snprintf(nm, sizeof nm, "%s_gripper_site", side);
  int gs = mj_name2id(m, mjOBJ_SITE, nm);
  if (gs >= 0) rel("gripper_site (jaw mid)", d->site_xpos + 3 * gs, d->site_xmat + 9 * gs);

  mj_deleteData(d); mj_deleteModel(m);
  return 0;
}
