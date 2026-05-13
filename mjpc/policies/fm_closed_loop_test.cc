// Closed-loop FM-only test inside MJPC build.
//
// Loads the wiping scene, runs a closed-loop with ONNXPolicy as the
// reference q_d generator and a joint-space PD as the low-level torque
// controller. Logs CSV (time, q, qdot, ee, target, Fz). Optional viewer.
//
// Usage:
//   MJPC_FM_CKPT=.../flow_policy.onnx \
//   MJPC_FM_STATS=.../normalization_stats.npz \
//   MJPC_SCENE_XML=.../scene_wiping.xml \
//   MJPC_LOG=/tmp/fm_loop.csv \
//   MJPC_MAX_TIME=12 \
//   ./bin/fm_closed_loop_test
//
// Trajectory matches mjpc/tasks/Fr3 wipe: traj_final = (0.4, 0, 0.3),
// wipe_stabilize=5, wipe_radius=0.05, wipe_period=π. EE site at hand_site.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include "mjpc/policies/onnx_policy.h"

// --- minimal GLFW + MuJoCo viewer state (only used if MJPC_VIEW=1) -----
namespace viewer {
GLFWwindow* window = nullptr;
mjvCamera   cam;
mjvOption   opt;
mjvScene    scn;
mjrContext  con;
mjvPerturb  pert;
bool        button_left   = false;
bool        button_middle = false;
bool        button_right  = false;
bool        ctrl_held     = false;
bool        shift_held    = false;
double      lastx = 0, lasty = 0;
const mjModel* g_model = nullptr;
mjData*        g_data  = nullptr;

void mouse_button(GLFWwindow* w, int /*btn*/, int act, int /*mods*/) {
  button_left   = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT)   == GLFW_PRESS;
  button_right  = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT)  == GLFW_PRESS;
  button_middle = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
  glfwGetCursorPos(w, &lastx, &lasty);

  // On Ctrl+press, select body under cursor and start perturbation.
  if (act == GLFW_PRESS && ctrl_held && (button_left || button_right)) {
    int win_w, win_h;
    glfwGetWindowSize(w, &win_w, &win_h);
    mjrRect viewport = {0, 0, win_w, win_h};
    mjtNum relx = lastx / win_w;
    mjtNum rely = 1.0 - (lasty / win_h);   // mujoco y is bottom-origin
    mjtNum aspect = (mjtNum)viewport.width / viewport.height;
    mjtNum selpnt[3]; int geomid[1] = {-1}, flexid[1] = {-1}, skinid[1] = {-1};
    int bid = mjv_select(g_model, g_data, &opt, aspect, relx, rely, &scn,
                         selpnt, geomid, flexid, skinid);
    if (bid > 0) {
      pert.select = bid;
      pert.skinselect = -1;
      pert.flexselect = -1;
      // Only allow perturb on mocap bodies (body_mocapid >= 0).
      if (g_model->body_mocapid[bid] >= 0) {
        pert.active = button_right ? mjPERT_TRANSLATE : mjPERT_ROTATE;
        mjv_initPerturb(g_model, g_data, &scn, &pert);
        std::cout << "[mocap] grabbed body id=" << bid
                  << " name=" << mj_id2name(g_model, mjOBJ_BODY, bid)
                  << " (Ctrl+right-drag=translate, Ctrl+left-drag=rotate)\n";
      } else {
        pert.select = 0;
        pert.active = 0;
      }
    } else {
      pert.select = 0;
      pert.active = 0;
    }
  }
  if (act == GLFW_RELEASE) {
    pert.active = 0;
  }
}
void cursor_pos(GLFWwindow* w, double x, double y) {
  if (!button_left && !button_right && !button_middle) return;
  double dx = x - lastx, dy = y - lasty;
  lastx = x; lasty = y;
  int w_, h_; glfwGetWindowSize(w, &w_, &h_);
  bool mod_shift = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT)  == GLFW_PRESS ||
                    glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
  mjtMouse action = button_right
      ? (mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V)
      : (button_left ? (mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V)
                     : mjMOUSE_ZOOM);

  if (pert.active) {
    // Same dy convention as moveCamera (glfw cursor dy is top-origin).
    mjv_movePerturb(g_model, g_data, action, dx / h_, dy / h_, &scn, &pert);
  } else {
    mjv_moveCamera(g_model, action, dx / h_, dy / h_, &scn, &cam);
  }
}
void scroll(GLFWwindow*, double /*xoff*/, double yoff) {
  mjv_moveCamera(g_model, mjMOUSE_ZOOM, 0, -0.05 * yoff, &scn, &cam);
}

// Manual goal state (used when MJPC_MANUAL=1). Moved by keyboard.
double  g_manual_goal[3] = {0.4, 0.0, 0.3};
double  g_manual_step    = 0.01;  // meters per key press
bool    g_manual_enabled = false;

void key(GLFWwindow* w, int kk, int /*scan*/, int act, int /*mods*/) {
  // Track Ctrl / Shift state for mocap drag.
  if (kk == GLFW_KEY_LEFT_CONTROL || kk == GLFW_KEY_RIGHT_CONTROL) {
    ctrl_held = (act != GLFW_RELEASE);
    return;
  }
  if (kk == GLFW_KEY_LEFT_SHIFT || kk == GLFW_KEY_RIGHT_SHIFT) {
    shift_held = (act != GLFW_RELEASE);
    return;
  }
  if (!g_manual_enabled) return;
  if (act != GLFW_PRESS && act != GLFW_REPEAT) return;
  (void)w;
  double s = g_manual_step;
  switch (kk) {
    case GLFW_KEY_LEFT:      g_manual_goal[0] -= s; break;  // -x
    case GLFW_KEY_RIGHT:     g_manual_goal[0] += s; break;  // +x
    case GLFW_KEY_DOWN:      g_manual_goal[1] -= s; break;  // -y
    case GLFW_KEY_UP:        g_manual_goal[1] += s; break;  // +y
    case GLFW_KEY_PAGE_DOWN: g_manual_goal[2] -= s; break;  // -z
    case GLFW_KEY_PAGE_UP:   g_manual_goal[2] += s; break;  // +z
    default: return;
  }
  std::cout << "[manual] goal = (" << g_manual_goal[0]
            << ", " << g_manual_goal[1]
            << ", " << g_manual_goal[2] << ")\n";
}

bool init(const mjModel* model, mjData* data) {
  g_model = model; g_data = data;
  mjv_defaultPerturb(&pert);
  if (!glfwInit()) {
    std::cerr << "[fm_loop] glfwInit failed\n";
    return false;
  }
  window = glfwCreateWindow(1280, 900, "FM closed-loop (MJPC)", nullptr, nullptr);
  if (!window) { glfwTerminate(); return false; }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  mjv_defaultCamera(&cam);
  mjv_defaultOption(&opt);
  mjv_defaultScene(&scn);
  mjr_defaultContext(&con);
  mjv_makeScene(model, &scn, 4000);
  mjr_makeContext(model, &con, mjFONTSCALE_150);
  // Default-ish view of the table area.
  cam.azimuth = 120; cam.elevation = -20;
  cam.lookat[0] = 0.2; cam.lookat[1] = 0.0; cam.lookat[2] = 0.4;
  cam.distance = 1.6;
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetCursorPosCallback(window, cursor_pos);
  glfwSetScrollCallback(window, scroll);
  glfwSetKeyCallback(window, key);
  return true;
}

void render() {
  if (!window) return;
  mjrRect viewport = {0, 0, 0, 0};
  glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
  mjv_updateScene(g_model, g_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
  mjr_render(viewport, &scn, &con);
  glfwSwapBuffers(window);
  glfwPollEvents();
}

void applyPerturb() {
  if (pert.active) {
    mjv_applyPerturbPose(g_model, g_data, &pert, 0);  // mocap only
  }
}

bool should_close() {
  return window ? glfwWindowShouldClose(window) : true;
}

void shutdown() {
  if (!window) return;
  mjv_freeScene(&scn);
  mjr_freeContext(&con);
  glfwDestroyWindow(window);
  glfwTerminate();
  window = nullptr;
}
}  // namespace viewer

namespace {

const char* env_or(const char* key, const char* fallback) {
  const char* v = std::getenv(key);
  return (v && v[0]) ? v : fallback;
}

double env_d(const char* key, double fallback) {
  const char* v = std::getenv(key);
  return v ? std::atof(v) : fallback;
}

// Convert a 3x3 rotation matrix (row-major as in mujoco site_xmat) to
// roll/pitch/yaw (ZYX convention). Returns false on near-gimbal-lock cases.
[[maybe_unused]] void RotMatToRPY(const double R[9], double rpy[3]) {
  // R is row-major: R[0..2] row0, R[3..5] row1, R[6..8] row2.
  const double sy = std::sqrt(R[0] * R[0] + R[3] * R[3]);
  if (sy > 1e-6) {
    rpy[0] = std::atan2(R[7], R[8]);
    rpy[1] = std::atan2(-R[6], sy);
    rpy[2] = std::atan2(R[3], R[0]);
  } else {
    // gimbal lock
    rpy[0] = std::atan2(-R[5], R[4]);
    rpy[1] = std::atan2(-R[6], sy);
    rpy[2] = 0.0;
  }
}

// Linear interp of circle target (mirrors mjpc fr3 wipe).
void CircleTarget(double t, const double final_xyz[3], double wipe_stabilize,
                  double wipe_radius, double wipe_period, double out[3]) {
  if (t < wipe_stabilize || wipe_period <= 1e-6) {
    out[0] = final_xyz[0];
    out[1] = final_xyz[1];
    out[2] = final_xyz[2];
    return;
  }
  const double w = 2.0 * M_PI / wipe_period;
  const double th = w * (t - wipe_stabilize);
  out[0] = final_xyz[0] + wipe_radius * (std::cos(th) - 1.0);
  out[1] = final_xyz[1] + wipe_radius * std::sin(th);
  out[2] = final_xyz[2];
}

const double HOME_Q[7] = {0.0, -M_PI_4, 0.0, -3.0 * M_PI_4,
                          0.0,  M_PI_2, M_PI_4};

}  // namespace

int main(int argc, char** argv) {
  const char* ckpt = env_or(
      "MJPC_FM_CKPT",
      "/home/kkomji/tmp/flow-matching-robot-control/checkpoints/"
      "flow_v26_6dof_tcp/flow_policy.onnx");
  const char* stats = env_or(
      "MJPC_FM_STATS",
      "/home/kkomji/tmp/flow-matching-robot-control/checkpoints/"
      "flow_v26_6dof_tcp/normalization_stats.npz");
  const char* xml_path = env_or(
      "MJPC_SCENE_XML",
      "/home/kkomji/tmp/flow-matching-robot-control/model/scene_wiping.xml");
  const char* csv_path = env_or("MJPC_LOG", "/tmp/fm_loop.csv");

  const double max_time = env_d("MJPC_MAX_TIME", 12.0);
  const double wipe_stabilize = env_d("MJPC_WIPE_STAB", 5.0);
  const double wipe_radius   = env_d("MJPC_WIPE_R",   0.05);
  const double wipe_period   = env_d("MJPC_WIPE_T",   M_PI);
  const double KP = env_d("MJPC_KP", 400.0);
  const double KD = env_d("MJPC_KD",  40.0);

  const double final_xyz[3] = {env_d("MJPC_TGT_X", 0.4),
                               env_d("MJPC_TGT_Y", 0.0),
                               env_d("MJPC_TGT_Z", 0.3)};
  const double goal_rpy[3] = {env_d("MJPC_TGT_R", -M_PI),
                              env_d("MJPC_TGT_P", 0.0),
                              env_d("MJPC_TGT_YAW", 0.0)};

  // ----- load mujoco model -----
  char err[1024] = {0};
  mjModel* model = mj_loadXML(xml_path, nullptr, err, sizeof(err));
  if (!model) {
    std::cerr << "[fm_loop] failed to load " << xml_path << ": " << err << "\n";
    return 1;
  }
  mjData* data = mj_makeData(model);

  // Init home pose.
  for (int i = 0; i < 7 && i < model->nq; ++i) data->qpos[i] = HOME_Q[i];
  for (int i = 0; i < model->nv; ++i) data->qvel[i] = 0;
  std::fill(data->ctrl, data->ctrl + model->nu, 0.0);
  mj_forward(model, data);

  // Align mocap (link7_t) to current EE pose at HOME so the robot starts
  // *at* its target. Matches eval_mocap_follow_v24.py:init_mocap_to_home_ee.
  // Without this, mocap stays at scene_wiping.xml default (0.5, 0, 0.55) →
  // target_site = (0.5, 0, 0.336) ≠ HOME EE → FM is asked to immediately
  // move robot from a "fresh-at-rest" state, which falls outside its
  // training distribution and leaves the robot stuck.
  {
    int link7t_bid = mj_name2id(model, mjOBJ_BODY, "link7_t");
    int site_bid = mj_name2id(model, mjOBJ_SITE, "hand_site");
    if (site_bid < 0) site_bid = mj_name2id(model, mjOBJ_SITE, "attachment_site");
    if (link7t_bid >= 0 && site_bid >= 0) {
      int mocap_id = model->body_mocapid[link7t_bid];
      if (mocap_id >= 0) {
        const double* ee_pos = data->site_xpos + 3 * site_bid;
        const double* ee_R   = data->site_xmat + 9 * site_bid;
        // mocap_pos = ee_pos - R · (0, 0, 0.214)   (target_site offset)
        double offset[3] = {0.0, 0.0, 0.214};
        double Roff[3] = {ee_R[0]*offset[0] + ee_R[1]*offset[1] + ee_R[2]*offset[2],
                          ee_R[3]*offset[0] + ee_R[4]*offset[1] + ee_R[5]*offset[2],
                          ee_R[6]*offset[0] + ee_R[7]*offset[1] + ee_R[8]*offset[2]};
        data->mocap_pos[3*mocap_id + 0] = ee_pos[0] - Roff[0];
        data->mocap_pos[3*mocap_id + 1] = ee_pos[1] - Roff[1];
        data->mocap_pos[3*mocap_id + 2] = ee_pos[2] - Roff[2];
        double q_ee[4];
        mju_mat2Quat(q_ee, ee_R);
        data->mocap_quat[4*mocap_id + 0] = q_ee[0];
        data->mocap_quat[4*mocap_id + 1] = q_ee[1];
        data->mocap_quat[4*mocap_id + 2] = q_ee[2];
        data->mocap_quat[4*mocap_id + 3] = q_ee[3];
        mj_forward(model, data);
        std::cout << "[fm_loop] mocap link7_t aligned to HOME EE pose\n";
      }
    }
  }

  // Find hand_site, ee_force sensor.
  int hand_sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
  if (hand_sid < 0) hand_sid = mj_name2id(model, mjOBJ_SITE, "attachment_site");
  if (hand_sid < 0) hand_sid = mj_name2id(model, mjOBJ_SITE, "sensor_site");
  int force_sid = mj_name2id(model, mjOBJ_SENSOR, "ee_force");
  int force_adr = (force_sid >= 0) ? model->sensor_adr[force_sid] : -1;
  int force_dim = (force_sid >= 0) ? model->sensor_dim[force_sid] : 0;

  std::cout << "[fm_loop] xml=" << xml_path
            << " ee_site=" << (hand_sid >= 0 ? "OK" : "missing")
            << " ee_force=" << (force_adr >= 0 ? "OK" : "missing") << "\n";

  // ----- load FM policy -----
  ONNXPolicy policy(ckpt, stats);
  if (!policy.isLoaded()) {
    std::cerr << "[fm_loop] policy not loaded\n";
    return 1;
  }
  policy.startFMThread();

  const int sd = policy.getStateDim();
  const int ad = policy.getActionDim();
  if (sd < 14 || ad < 7) {
    std::cerr << "[fm_loop] unsupported FM dimensions: sd=" << sd
              << " ad=" << ad << "\n";
    return 1;
  }
  const bool include_ee = policy.includesEE() && sd >= 17;
  const int horizon_chunk = policy.getHorizon();
  const double fm_dt = 0.020;   // FM action period (50 Hz)

  // ----- csv -----
  FILE* csv = std::fopen(csv_path, "w");
  if (csv) std::fprintf(csv,
      "time,ee_x,ee_y,ee_z,tgt_x,tgt_y,tgt_z,Fx,Fy,Fz,"
      "q1,q2,q3,q4,q5,q6,q7,qd1,qd2,qd3,qd4,qd5,qd6,qd7\n");

  // ----- viewer (optional) -----
  const bool use_view = std::getenv("MJPC_VIEW") &&
                        std::atoi(std::getenv("MJPC_VIEW")) != 0;
  const bool manual_mode = std::getenv("MJPC_MANUAL") &&
                           std::atoi(std::getenv("MJPC_MANUAL")) != 0;
  if (manual_mode) {
    viewer::g_manual_enabled = true;
    viewer::g_manual_goal[0] = final_xyz[0];
    viewer::g_manual_goal[1] = final_xyz[1];
    viewer::g_manual_goal[2] = final_xyz[2];
    if (const char* s = std::getenv("MJPC_MANUAL_STEP"))
      viewer::g_manual_step = std::atof(s);
    std::cout << "[manual] enabled. step=" << viewer::g_manual_step << " m\n";
    std::cout << "  ←/→ : x   ↑/↓ : y   PgUp/PgDn : z   (must hold viewer focus)\n";
    std::cout << "  initial goal = (" << viewer::g_manual_goal[0] << ", "
              << viewer::g_manual_goal[1] << ", "
              << viewer::g_manual_goal[2] << ")\n";
  }
  if (use_view) {
    if (!viewer::init(model, data)) {
      std::cerr << "[fm_loop] viewer init failed, falling back to headless\n";
    }
  }

  // ----- control loop -----
  const double sim_dt = model->opt.timestep;
  const int control_interval = std::max(1, int(std::round(fm_dt / sim_dt)));
  const int max_steps = int(std::round(max_time / sim_dt));
  const bool realtime_pacing = use_view;
  const double render_period = 1.0 / 60.0;
  double next_render_time = 0.0;

  std::vector<std::vector<Eigen::VectorXd>> te_chunks;  // temporal ensemble
  const int te_max = horizon_chunk;
  const double te_decay = 0.01;

  Eigen::VectorXd q_target = Eigen::VectorXd::Zero(7);
  for (int i = 0; i < 7; ++i) q_target(i) = HOME_Q[i];

  Eigen::VectorXd prev_state = Eigen::VectorXd::Zero(sd);
  Eigen::VectorXd prev_action = Eigen::VectorXd::Zero(ad);

  auto t0 = std::chrono::steady_clock::now();
  for (int step = 0; step < max_steps; ++step) {
    const double t_sim = step * sim_dt;

    // Apply user mocap-drag perturbation (only when viewer + Ctrl+drag).
    if (use_view) viewer::applyPerturb();

    // Find target site once: 'target_site' (inside link7_t mocap) preferred,
    // else 'hand_copy_site', else fall back to mocap_pos / manual / circle.
    static int target_site_id = [&model]() {
      int s = mj_name2id(model, mjOBJ_SITE, "target_site");
      if (s < 0) s = mj_name2id(model, mjOBJ_SITE, "hand_copy_site");
      return s;
    }();
    static int target_body_id = [&model]() {
      int b = mj_name2id(model, mjOBJ_BODY, "link7_t");
      if (b < 0) b = mj_name2id(model, mjOBJ_BODY, "hand_copy");
      return b;
    }();
    static int target_mocap_id =
        (target_body_id >= 0) ? model->body_mocapid[target_body_id] : -1;

    double target_pos[3];
    if (target_site_id >= 0) {
      target_pos[0] = data->site_xpos[3 * target_site_id + 0];
      target_pos[1] = data->site_xpos[3 * target_site_id + 1];
      target_pos[2] = data->site_xpos[3 * target_site_id + 2];
    } else if (target_mocap_id >= 0) {
      target_pos[0] = data->mocap_pos[3 * target_mocap_id + 0];
      target_pos[1] = data->mocap_pos[3 * target_mocap_id + 1];
      target_pos[2] = data->mocap_pos[3 * target_mocap_id + 2];
    } else if (manual_mode) {
      target_pos[0] = viewer::g_manual_goal[0];
      target_pos[1] = viewer::g_manual_goal[1];
      target_pos[2] = viewer::g_manual_goal[2];
    } else {
      CircleTarget(t_sim, final_xyz, wipe_stabilize, wipe_radius, wipe_period,
                   target_pos);
    }

    // FM forward at 50 Hz.
    if (step % control_interval == 0) {
      Eigen::VectorXd state = Eigen::VectorXd::Zero(sd);
      for (int i = 0; i < 7; ++i) {
        state(i) = data->qpos[i];
        state(7 + i) = data->qvel[i];
      }
      if (include_ee && hand_sid >= 0) {
        for (int i = 0; i < 3; ++i)
          state(14 + i) = data->site_xpos[3 * hand_sid + i];
      }
      // Goal orientation: prefer target_site's world rotation (follows
      // user-dragged link7_t rotation). Fallback = MJPC_TGT_R/P/YAW.
      double rpy_used[3] = {goal_rpy[0], goal_rpy[1], goal_rpy[2]};
      if (target_site_id >= 0) {
        const double* R = data->site_xmat + 9 * target_site_id;
        RotMatToRPY(R, rpy_used);
      }
      // Roll wrap to match FM training distribution (goal_mean[3] ≈ -π).
      // Without this, atan2 returns +π for the "EE down" pose → 22σ OOD.
      // Matches eval_mocap_follow_v24.py:rotmat_to_rpy_xyz.
      if (rpy_used[0] > 0.0) rpy_used[0] -= 2.0 * M_PI;
      static int rpy_dbg = 0;
      if (rpy_dbg++ % 100 == 0) {
        std::printf("[goal_rpy] step=%d t=%.3f tgt=(%+.3f,%+.3f,%+.3f) "
                    "rpy=(%+.4f,%+.4f,%+.4f)\n",
                    step, t_sim, target_pos[0], target_pos[1], target_pos[2],
                    rpy_used[0], rpy_used[1], rpy_used[2]);
        std::fflush(stdout);
      }
      Eigen::VectorXd goal(6);
      goal << target_pos[0], target_pos[1], target_pos[2],
              rpy_used[0],   rpy_used[1],   rpy_used[2];

      policy.requestPrediction(state, prev_state, prev_action, goal);
      std::vector<Eigen::VectorXd> chunk;
      if (policy.getLatestChunk(chunk)) {
        te_chunks.push_back(chunk);
        if ((int)te_chunks.size() > te_max) te_chunks.erase(te_chunks.begin());
      }

      // Temporal ensemble: blended next action.
      if (!te_chunks.empty()) {
        Eigen::VectorXd result = Eigen::VectorXd::Zero(7);
        double w_sum = 0;
        int n = (int)te_chunks.size();
        for (int i = 0; i < n; ++i) {
          int idx = n - 1 - i;
          if (idx < (int)te_chunks[i].size()) {
            double w = std::exp(-te_decay * i);
            result += w * te_chunks[i][idx];
            w_sum += w;
          }
        }
        if (w_sum > 0) q_target = result / w_sum;
      }

      prev_state = state;
      prev_action = q_target;

      // Debug: full FM input/output comparison.
      static int full_dbg = 0;
      if (full_dbg++ % 50 == 0) {
        std::fprintf(stderr,
                     "[CLT t=%.2f] q=[%+.3f,%+.3f,%+.3f,%+.3f,%+.3f,%+.3f,%+.3f] "
                     "ee=[%+.3f,%+.3f,%+.3f] goal=[%+.3f,%+.3f,%+.3f,%+.3f,%+.3f,%+.3f] "
                     "q_target=[%+.3f,%+.3f,%+.3f,%+.3f,%+.3f,%+.3f,%+.3f]\n",
                     t_sim, state[0], state[1], state[2], state[3],
                     state[4], state[5], state[6],
                     (int)state.size() >= 17 ? state[14] : 0.0,
                     (int)state.size() >= 17 ? state[15] : 0.0,
                     (int)state.size() >= 17 ? state[16] : 0.0,
                     goal[0], goal[1], goal[2], goal[3], goal[4], goal[5],
                     q_target(0), q_target(1), q_target(2), q_target(3),
                     q_target(4), q_target(5), q_target(6));
        std::fflush(stderr);
      }
      // Debug: orientation residual = axis-angle from current EE to target.
      static int ori_dbg = 0;
      if (ori_dbg++ % 50 == 0 && hand_sid >= 0 && target_site_id >= 0) {
        const double* R_now = data->site_xmat + 9 * hand_sid;
        const double* R_tgt = data->site_xmat + 9 * target_site_id;
        double R_err[9];
        // R_err = R_tgt^T * R_now
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
          double s = 0;
          for (int k = 0; k < 3; ++k) s += R_tgt[k*3+i] * R_now[k*3+j];
          R_err[i*3+j] = s;
        }
        double q_err[4];
        mju_mat2Quat(q_err, R_err);
        double aa[3];
        mju_quat2Vel(aa, q_err, 1.0);
        double q7 = data->qpos[6];
        std::fprintf(stderr,
                     "[fmtest_ori] t=%.3f j7=%+.3f ori_res=(%+.4f,%+.4f,%+.4f)\n",
                     t_sim, q7, aa[0], aa[1], aa[2]);
        std::fflush(stderr);
      }
    }

    // Joint-space PD with M-projection — match training data control law:
    //   tau = M @ (Kp*(q_d - q) + Kd*(qdot_d - qdot)) + qfrc_bias
    // (training: collect_ik_data_v3.py line 174). Previously the test used
    // direct per-joint PD without M, which decouples joint dynamics
    // differently and falls outside the FM training distribution.
    {
      const int nv = model->nv;
      std::vector<double> M_full(nv * nv);
      mj_fullM(model, M_full.data(), data->qM);
      double a[7];
      for (int i = 0; i < 7; ++i) {
        a[i] = KP * (q_target(i) - data->qpos[i]) - KD * data->qvel[i];
      }
      for (int i = 0; i < 7 && i < model->nu; ++i) {
        double s = 0.0;
        for (int j = 0; j < 7; ++j) s += M_full[i * nv + j] * a[j];
        data->ctrl[i] = s + data->qfrc_bias[i];
      }
    }

    mj_step(model, data);

    // Render + realtime pacing when viewer is active.
    if (use_view) {
      if (data->time >= next_render_time) {
        viewer::render();
        next_render_time += render_period;
        // sleep to keep ~realtime
        std::this_thread::sleep_for(std::chrono::milliseconds(
            int(1000 * render_period * 0.9)));
        if (viewer::should_close()) break;
      }
    }
    (void)realtime_pacing;

    // Log every 50 Hz (= control rate).
    if (step % control_interval == 0 && csv) {
      double ee[3] = {0, 0, 0};
      if (hand_sid >= 0) {
        ee[0] = data->site_xpos[3 * hand_sid + 0];
        ee[1] = data->site_xpos[3 * hand_sid + 1];
        ee[2] = data->site_xpos[3 * hand_sid + 2];
      }
      double F[3] = {0, 0, 0};
      if (force_adr >= 0 && force_dim >= 3) {
        F[0] = data->sensordata[force_adr + 0];
        F[1] = data->sensordata[force_adr + 1];
        F[2] = data->sensordata[force_adr + 2];
      }
      std::fprintf(csv,
          "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
          "%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,"
          "%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n",
          t_sim, ee[0], ee[1], ee[2],
          target_pos[0], target_pos[1], target_pos[2],
          F[0], F[1], F[2],
          data->qpos[0], data->qpos[1], data->qpos[2], data->qpos[3],
          data->qpos[4], data->qpos[5], data->qpos[6],
          data->qvel[0], data->qvel[1], data->qvel[2], data->qvel[3],
          data->qvel[4], data->qvel[5], data->qvel[6]);
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "[fm_loop] sim " << max_time << "s in " << elapsed
            << "s wall (" << (max_time / elapsed) << "x realtime)\n";

  if (csv) std::fclose(csv);
  std::cout << "[fm_loop] csv=" << csv_path << "\n";

  if (use_view) viewer::shutdown();
  policy.stopFMThread();
  mj_deleteData(data);
  mj_deleteModel(model);
  return 0;
}
