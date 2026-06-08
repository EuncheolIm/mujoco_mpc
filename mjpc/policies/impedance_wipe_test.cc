// Hybrid Impedance closed-loop wipe test (no MPC, no FM).
//
// Validates whether the *mujoco contact model* (solref/solimp/friction) can
// sustain table contact while tracking a circular trajectory, using only an
// operational-space Hybrid Impedance controller (5-DoF position impedance
// + 1-DoF Z-axis force control). Ported from
//   /home/kkomji/Euncheol/MPPI/HybridImpedance/src/controller.cpp
//   :: Hybrid_Impedance_fast_control()
// using mujoco primitives instead of PRBDL.
//
// Logs CSV: time, qpos, qvel, ee_xyz, target_xyz, F_press_z, F_phi.
// Env:
//   MJPC_SCENE_XML  (required, e.g. mjpc/tasks/Fr3/task.xml)
//   MJPC_LOG        (CSV path)
//   MJPC_MAX_TIME   (sim seconds, default 30)
//   MJPC_F_DES      (target press force, N, default 5)
//   MJPC_KP_XY      (default 10000)
//   MJPC_KP_RPY     (default 1000)
//   MJPC_VIEW       (1 = GLFW viewer)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

using Eigen::Matrix;
using Eigen::Vector3d;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// FR3 home pose used as null-space posture target.
static const double HOME_Q[7] = {
    0.0, -0.78539816, 0.0, -2.35619449, 0.0, 1.57079632, 0.78539816};

// ---------- mujoco helpers --------------------------------------------------
static void GetJacSite(const mjModel* m, mjData* d, int sid,
                       double* jacp /*3x nv*/, double* jacr /*3x nv*/) {
  mj_jacSite(m, d, jacp, jacr, sid);
}

// Compute full mass matrix from data->qM (sparse upper-tri storage).
static void FullM(const mjModel* m, const mjData* d, MatrixXd& M) {
  std::vector<double> M_full(m->nv * m->nv);
  mj_fullM(m, M_full.data(), d->qM);
  for (int i = 0; i < m->nv; ++i)
    for (int j = 0; j < m->nv; ++j)
      M(i, j) = M_full[i * m->nv + j];
}

// First-order low-pass filter on derivative: returns filtered derivative.
// y_n = a * (x_n - x_{n-1})/dt + (1-a) * y_{n-1},   a = dt*wc/(1+dt*wc).
static inline double VelLpf(double dt, double wc, double prev_x, double cur_x,
                            double prev_y) {
  double raw = (cur_x - prev_x) / dt;
  double a = dt * wc / (1.0 + dt * wc);
  return a * raw + (1.0 - a) * prev_y;
}

// ---------- viewer (optional) ----------------------------------------------
namespace viewer {
GLFWwindow* window = nullptr;
mjvCamera   cam;
mjvOption   opt;
mjvScene    scn;
mjrContext  con;
bool        button_left=false, button_middle=false, button_right=false;
double      lastx=0, lasty=0;
const mjModel* gm = nullptr;
mjData*        gd = nullptr;

void mouse_button(GLFWwindow* w, int, int, int) {
  button_left   = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT)   == GLFW_PRESS;
  button_right  = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT)  == GLFW_PRESS;
  button_middle = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
  glfwGetCursorPos(w, &lastx, &lasty);
}
void mouse_move(GLFWwindow* w, double x, double y) {
  if (!button_left && !button_right && !button_middle) { lastx=x; lasty=y; return; }
  double dx = x - lastx, dy = y - lasty; lastx=x; lasty=y;
  int wi, hi; glfwGetWindowSize(w, &wi, &hi);
  bool shift = glfwGetKey(w, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
               glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS;
  mjtMouse action;
  if (button_right)       action = shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  else if (button_left)   action = shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  else                    action = mjMOUSE_ZOOM;
  mjv_moveCamera(gm, action, dx/hi, dy/hi, &scn, &cam);
}
void scroll(GLFWwindow*, double, double yo) {
  mjv_moveCamera(gm, mjMOUSE_ZOOM, 0, -0.05*yo, &scn, &cam);
}
void key_callback(GLFWwindow*, int key, int /*sc*/, int act, int /*mods*/) {
  if (act != GLFW_PRESS) return;
  // mujoco-style visualization toggles
  if (key == GLFW_KEY_C) {
    opt.flags[mjVIS_CONTACTPOINT] = !opt.flags[mjVIS_CONTACTPOINT];
    std::printf("[viewer] contact point: %d\n", (int)opt.flags[mjVIS_CONTACTPOINT]);
  } else if (key == GLFW_KEY_F) {
    opt.flags[mjVIS_CONTACTFORCE] = !opt.flags[mjVIS_CONTACTFORCE];
    std::printf("[viewer] contact force: %d\n", (int)opt.flags[mjVIS_CONTACTFORCE]);
  } else if (key == GLFW_KEY_T) {
    opt.flags[mjVIS_TRANSPARENT] = !opt.flags[mjVIS_TRANSPARENT];
  } else if (key == GLFW_KEY_BACKSPACE) {
    // reset view
    cam.distance = 2.0; cam.azimuth = 90; cam.elevation = -20;
    cam.lookat[0]=0.4; cam.lookat[1]=0.0; cam.lookat[2]=0.4;
  }
}
bool init(const mjModel* m, mjData* d) {
  if (!glfwInit()) return false;
  window = glfwCreateWindow(1200, 900, "Impedance Wipe Test", nullptr, nullptr);
  if (!window) return false;
  glfwMakeContextCurrent(window); glfwSwapInterval(1);
  mjv_defaultCamera(&cam); mjv_defaultOption(&opt);
  mjv_defaultScene(&scn); mjr_defaultContext(&con);
  mjv_makeScene(m, &scn, 4000); mjr_makeContext(m, &con, mjFONTSCALE_150);
  gm = m; gd = d;
  cam.distance = 2.0; cam.azimuth = 90; cam.elevation = -20;
  cam.lookat[0]=0.4; cam.lookat[1]=0.0; cam.lookat[2]=0.4;
  // Default ON: contact visualization (point + force vector).
  opt.flags[mjVIS_CONTACTPOINT] = 1;
  opt.flags[mjVIS_CONTACTFORCE] = 1;
  glfwSetMouseButtonCallback(window, mouse_button);
  glfwSetCursorPosCallback (window, mouse_move);
  glfwSetScrollCallback    (window, scroll);
  glfwSetKeyCallback       (window, key_callback);
  return true;
}
void render(const mjModel* m, mjData* d) {
  mjrRect vp = {0,0,0,0};
  glfwGetFramebufferSize(window, &vp.width, &vp.height);
  mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
  mjr_render(vp, &scn, &con);
  glfwSwapBuffers(window); glfwPollEvents();
}
void destroy() {
  mjv_freeScene(&scn); mjr_freeContext(&con);
  if (window) glfwDestroyWindow(window);
  glfwTerminate();
}
}  // namespace viewer

// ---------- main -----------------------------------------------------------
int main(int /*argc*/, char** /*argv*/) {
  const char* xml = std::getenv("MJPC_SCENE_XML");
  if (!xml) {
    std::fprintf(stderr, "MJPC_SCENE_XML required\n"); return 1;
  }
  char error[1000] = "";
  mjModel* m = mj_loadXML(xml, nullptr, error, sizeof(error));
  if (!m) { std::fprintf(stderr, "mj_loadXML: %s\n", error); return 1; }
  mjData* d = mj_makeData(m);

  // Find hand site.
  int sid = mj_name2id(m, mjOBJ_SITE, "hand_site");
  if (sid < 0) { std::fprintf(stderr, "site 'hand_site' not found\n"); return 1; }

  // Init joints to HOME_Q.
  for (int i = 0; i < 7 && i < m->nq; ++i) d->qpos[i] = HOME_Q[i];
  for (int j = 0; j < m->nv; ++j) d->qvel[j] = 0.0;
  mj_forward(m, d);

  // Params (env override).
  const double F_DES   = std::getenv("MJPC_F_DES")   ? std::atof(std::getenv("MJPC_F_DES"))   :   5.0;
  const double KP_XY   = std::getenv("MJPC_KP_XY")   ? std::atof(std::getenv("MJPC_KP_XY"))   : 10000.0;
  const double KP_RPY  = std::getenv("MJPC_KP_RPY")  ? std::atof(std::getenv("MJPC_KP_RPY"))  :  1000.0;
  const double KN      = std::getenv("MJPC_KN")      ? std::atof(std::getenv("MJPC_KN"))      :    10.0;
  const double DN      = std::getenv("MJPC_DN")      ? std::atof(std::getenv("MJPC_DN"))      :     1.0;
  const double MAX_T   = std::getenv("MJPC_MAX_TIME")? std::atof(std::getenv("MJPC_MAX_TIME")): 30.0;
  const double R_CIRC  = std::getenv("MJPC_R_CIRC")  ? std::atof(std::getenv("MJPC_R_CIRC"))  :   0.05;
  const double T_PER   = std::getenv("MJPC_T_PER")   ? std::atof(std::getenv("MJPC_T_PER"))   :   3.14159265358979;  // π
  const bool   VIEW    = std::getenv("MJPC_VIEW")    && std::atoi(std::getenv("MJPC_VIEW"));

  std::fprintf(stderr,
      "[Impedance] F_DES=%.1f KP_XY=%.0f KP_RPY=%.0f KN=%.1f DN=%.1f "
      "r=%.3f T=%.2f s  view=%d  max_t=%.1fs\n",
      F_DES, KP_XY, KP_RPY, KN, DN, R_CIRC, T_PER, (int)VIEW, MAX_T);

  // Initial EE pose (after mj_forward).
  Vector3d init_pos(d->site_xpos[3*sid+0], d->site_xpos[3*sid+1], d->site_xpos[3*sid+2]);
  Matrix3d init_rot;
  for (int r=0; r<3; ++r) for (int c=0; c<3; ++c) init_rot(r,c) = d->site_xmat[9*sid + 3*r + c];
  double t0 = d->time;

  // 5-DoF position+orientation impedance gains (xy + rpy).
  // Z is handled by force control + a soft position anchor inside F_phi.
  Matrix<double,5,1> K_x, D_x;
  for (int i=0;i<5;++i) {
    K_x(i) = (i<2) ? KP_XY : KP_RPY;
    D_x(i) = 2.0 * std::sqrt(K_x(i));
  }
  // Null-space posture target.
  VectorXd q_default(7);
  for (int i=0;i<7;++i) q_default(i) = HOME_Q[i];

  // Jacobian derivative filter state.
  Matrix<double,1,7> pre_J_phi   = Matrix<double,1,7>::Zero();
  Matrix<double,1,7> pre_J_phidt = Matrix<double,1,7>::Zero();
  Matrix<double,5,7> pre_J_x     = Matrix<double,5,7>::Zero();
  Matrix<double,5,7> pre_J_xdt   = Matrix<double,5,7>::Zero();
  bool first_run = true;

  // Viewer.
  if (VIEW) viewer::init(m, d);

  // Logger.
  const char* log = std::getenv("MJPC_LOG");
  std::ofstream lout;
  if (log) {
    lout.open(log, std::ios::out | std::ios::trunc);
    lout << "time,ee_x,ee_y,ee_z,tgt_x,tgt_y,tgt_z,F_press_z,F_phi,Fx_raw,Fy_raw,Fz_raw\n";
  }

  // EE weight for F_press = Fz_world - mg.
  double mg = 7.46;  // fr3 hand subtree weight ~0.76 kg

  const int N = m->nv;
  MatrixXd M(N,N), Mi(N,N);
  std::vector<double> jacp_buf(3*N), jacr_buf(3*N);
  std::vector<double> bias_with_g(N), gravity_only(N);

  while (d->time - t0 < MAX_T) {
    double t = d->time - t0;
    double dt = m->opt.timestep;
    double omega = 2.0 * M_PI / T_PER;

    // ----- target trajectory (controller.cpp lines 459-487) ----------------
    // Wipe in xy circle around init_pos, drop along cylinder surface for z.
    // To match mjpc Fr3 wipe semantics (circle in xy at constant z = init_z),
    // we drop the cylinder-curve part — z stays at init z. Pure xy circle.
    double xm = R_CIRC * (1.0 - std::cos(omega * t));
    double ym = R_CIRC * std::sin(omega * t);
    // Ramp z target down to engage contact: start at init_z, drop by Z_DROP
    // over Z_RAMP seconds, then hold. Force control on top of this drives
    // F_phi → F_DES. Without the z-position anchor, contact bouncing dominates.
    const double Z_DROP = std::getenv("MJPC_Z_DROP") ? std::atof(std::getenv("MJPC_Z_DROP")) : 0.18;
    const double Z_RAMP = std::getenv("MJPC_Z_RAMP") ? std::atof(std::getenv("MJPC_Z_RAMP")) : 2.0;
    double z_drop = std::min(t / Z_RAMP, 1.0) * Z_DROP;
    Vector3d x_des = init_pos + Vector3d(xm, ym, -z_drop);
    Matrix3d R_des = init_rot;

    // ----- mujoco-side dynamics --------------------------------------------
    mj_kinematics(m, d);
    mj_comPos(m, d);
    mj_crb(m, d);
    FullM(m, d, M);
    Mi = M.inverse();

    // bias = C·qdot + g  via mj_rne with current qvel and flg_acc=0
    mj_rne(m, d, 0, bias_with_g.data());
    // gravity only: temporarily zero qvel, call mj_rne, restore
    std::vector<double> qvel_save(d->qvel, d->qvel + m->nv);
    std::fill(d->qvel, d->qvel + m->nv, 0.0);
    mj_rne(m, d, 0, gravity_only.data());
    std::copy(qvel_save.begin(), qvel_save.end(), d->qvel);

    VectorXd Cqdot(N), grav(N);
    for (int i=0;i<N;++i) { grav(i) = gravity_only[i]; Cqdot(i) = bias_with_g[i] - grav(i); }

    // Jacobian (3x N).
    GetJacSite(m, d, sid, jacp_buf.data(), jacr_buf.data());
    Matrix<double,6,Eigen::Dynamic> Jw(6, N);
    for (int r=0;r<3;++r) for (int c=0;c<N;++c) {
      Jw(r,c)   = jacp_buf[r*N + c];
      Jw(3+r,c) = jacr_buf[r*N + c];
    }

    // EE pose, current.
    Vector3d ee_pos(d->site_xpos[3*sid+0], d->site_xpos[3*sid+1], d->site_xpos[3*sid+2]);
    Matrix3d R_curr;
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) R_curr(r,c) = d->site_xmat[9*sid + 3*r + c];
    Matrix3d Rt = R_curr.transpose();

    // qvel for first 7 joints (FR3) — ignore fingers if present.
    VectorXd qdot7(7);
    for (int i=0;i<7;++i) qdot7(i) = d->qvel[i];
    VectorXd q7(7);
    for (int i=0;i<7;++i) q7(i) = d->qpos[i];

    // J_local (6×7): only first 7 columns of Jw, then rotate via Rt.
    Matrix<double,6,7> J_world7;
    for (int r=0;r<6;++r) for (int c=0;c<7;++c) J_world7(r,c) = Jw(r,c);
    Matrix<double,6,7> J_local;
    J_local.block(0,0,3,7) = Rt * J_world7.block(0,0,3,7);
    J_local.block(3,0,3,7) = Rt * J_world7.block(3,0,3,7);

    Matrix<double,1,7> J_phi   = J_local.row(2);                  // local Z
    Matrix<double,2,7> J_xy    = J_local.block(0,0,2,7);          // local X,Y
    Matrix<double,3,7> J_rpy   = J_local.block(3,0,3,7);          // local RPY
    Matrix<double,5,7> J_x;    J_x.block(0,0,2,7)=J_xy; J_x.block(2,0,3,7)=J_rpy;

    if (first_run) {
      pre_J_phi   = J_phi;
      pre_J_phidt.setZero();
      pre_J_x     = J_x;
      pre_J_xdt.setZero();
      first_run = false;
    }
    Matrix<double,1,7> J_phidt;
    for (int j=0;j<7;++j) {
      J_phidt(j) = VelLpf(dt, 2*M_PI*20.0, pre_J_phi(j), J_phi(j), pre_J_phidt(j));
      pre_J_phi(j)   = J_phi(j);
      pre_J_phidt(j) = J_phidt(j);
    }
    Matrix<double,5,7> J_xdt;
    for (int i=0;i<5;++i) for (int j=0;j<7;++j) {
      J_xdt(i,j) = VelLpf(dt, 2*M_PI*20.0, pre_J_x(i,j), J_x(i,j), pre_J_xdt(i,j));
      pre_J_x(i,j)   = J_x(i,j);
      pre_J_xdt(i,j) = J_xdt(i,j);
    }

    // Dynamics on the FR3 7-DoF subsystem.
    // Build M7, Cqdot7, g7 by slicing first 7 rows/cols (FR3 occupies dof[0..6]).
    Matrix<double,7,7> M7;
    for (int i=0;i<7;++i) for (int j=0;j<7;++j) M7(i,j) = M(i,j);
    Matrix<double,7,7> M7i = M7.inverse();
    Matrix<double,7,1> Cqdot7v, g7;
    for (int i=0;i<7;++i) { Cqdot7v(i) = Cqdot(i); g7(i) = grav(i); }

    double lambda_phi = 1.0 / (J_phi * M7i * J_phi.transpose())(0);
    Matrix<double,5,5> Mx_inv = J_x * M7i * J_x.transpose();
    Matrix<double,5,5> Mx = Mx_inv.inverse();

    Matrix<double,5,1> Jdot_qdot_x = J_xdt * qdot7;
    Matrix<double,5,1> C_x_v = Mx * (J_x * M7i * Cqdot7v - Jdot_qdot_x);

    // Errors in local frame.
    Vector3d pos_err_world = x_des - ee_pos;
    Matrix3d R_err = R_des * R_curr.transpose();
    Eigen::AngleAxisd aa(R_err);
    Vector3d rot_err_world = aa.angle() * aa.axis();

    Vector3d pos_err_local = Rt * pos_err_world;
    Vector3d rot_err_local = Rt * rot_err_world;

    Matrix<double,5,1> e_x, e_xdot;
    e_x.head(2)  = pos_err_local.head(2);
    e_x.tail(3)  = rot_err_local;
    // EE vel in world: J · qdot
    Matrix<double,6,1> ee_vel_world = J_world7 * qdot7;
    Vector3d vel_err_world      = -ee_vel_world.head(3);   // desired vel = 0
    Vector3d rot_vel_err_world  = -ee_vel_world.tail(3);
    Vector3d vel_err_local      = Rt * vel_err_world;
    Vector3d rot_vel_err_local  = Rt * rot_vel_err_world;
    e_xdot.head(2) = vel_err_local.head(2);
    e_xdot.tail(3) = rot_vel_err_local;

    Matrix<double,5,1> F_ctrl_x = C_x_v + K_x.asDiagonal() * e_x + D_x.asDiagonal() * e_xdot;

    // ----- Z force control (controller.cpp lines 528-552) ------------------
    // External force in local Z: read hand_force sensor (in site frame already).
    Vector3d F_sensor(0,0,0);
    int fs_adr = -1;
    for (int s = 0; s < m->nsensor; ++s) {
      if (m->sensor_type[s] == mjSENS_FORCE) {
        const char* nm = m->names + m->name_sensoradr[s];
        if (nm && std::string(nm) == "hand_force") { fs_adr = m->sensor_adr[s]; break; }
      }
    }
    if (fs_adr >= 0) {
      F_sensor = Vector3d(d->sensordata[fs_adr+0], d->sensordata[fs_adr+1], d->sensordata[fs_adr+2]);
    }
    // Local F_ext: hand_force sensor is in site frame -> already local.
    Matrix<double,5,1> F_ext_x = Matrix<double,5,1>::Zero();
    F_ext_x.head(2) = F_sensor.head(2);
    // (rotation torque part not directly available from hand_force)

    Matrix<double,1,7> lamJ_M = lambda_phi * J_phi * M7i;
    double Jdot_qdot_phi = (J_phidt * qdot7)(0);
    double term_consistency = lambda_phi * ((J_phi * M7i * Cqdot7v)(0) - Jdot_qdot_phi);

    // Add local-z position+velocity correction so the controller actively
    // pulls EE toward the desired z (ramped down by z_drop). Without this
    // anchor, force control alone causes mass-spring bouncing on contact.
    double z_err_local       = pos_err_local.z();
    double z_vel_err_local   = vel_err_local.z();
    double K_z_anchor = std::getenv("MJPC_KP_Z") ? std::atof(std::getenv("MJPC_KP_Z")) : 2000.0;
    double D_z_anchor = 2.0 * std::sqrt(K_z_anchor);
    double F_phi = F_DES
                 + K_z_anchor * z_err_local + D_z_anchor * z_vel_err_local
                 - (lamJ_M * J_x.transpose() * F_ctrl_x)(0)
                 + (lamJ_M * J_x.transpose() * F_ext_x)(0)
                 + term_consistency;

    // ----- Null-space posture --------------------------------------------
    Matrix<double,6,6> Lambda_task_inv = J_world7 * M7i * J_world7.transpose();
    Matrix<double,6,6> Lambda_task     = Lambda_task_inv.inverse();
    Matrix<double,7,7> Id7 = Matrix<double,7,7>::Identity();
    Matrix<double,7,6> Jbar_T = J_world7.transpose() * Lambda_task;
    Matrix<double,7,7> N_T = Id7 - Jbar_T * J_world7 * M7i;
    Matrix<double,7,1> tau_0 = -KN * (q7 - q_default) - DN * qdot7;
    Matrix<double,7,1> tau_null = N_T * tau_0;

    // ----- Final torque ---------------------------------------------------
    Matrix<double,7,1> tau = g7
                           + J_x.transpose() * F_ctrl_x
                           + J_phi.transpose() * F_phi
                           + tau_null;

    // Write actuator (saturate to torque limits via mujoco's ctrlrange).
    for (int i=0;i<7 && i<m->nu;++i) d->ctrl[i] = tau(i);
    for (int i=7;i<m->nu;++i) d->ctrl[i] = 0;

    mj_step(m, d);

    // ----- Logging (every mj_step by default; throttle via MJPC_LOG_DT) ----
    static double log_dt = std::getenv("MJPC_LOG_DT") ? std::atof(std::getenv("MJPC_LOG_DT")) : 0.0;
    static double next_log = 0.0;
    if (lout.is_open() && d->time - t0 >= next_log) {
      next_log += log_dt;
      Vector3d Fw = R_curr * F_sensor;
      double F_press_z = Fw.z() - mg;
      lout << (d->time - t0) << ',' << ee_pos.x() << ',' << ee_pos.y() << ',' << ee_pos.z()
           << ',' << x_des.x() << ',' << x_des.y() << ',' << x_des.z()
           << ',' << F_press_z << ',' << F_phi << ','
           << F_sensor.x() << ',' << F_sensor.y() << ',' << F_sensor.z() << '\n';
    }

    if (VIEW) {
      static double next_render = 0.0;
      if (d->time - t0 >= next_render) {
        next_render += 1.0/60.0;
        viewer::render(m, d);
        if (glfwWindowShouldClose(viewer::window)) break;
      }
    }
  }

  if (lout.is_open()) lout.close();
  if (VIEW) viewer::destroy();
  mj_deleteData(d); mj_deleteModel(m);
  return 0;
}
