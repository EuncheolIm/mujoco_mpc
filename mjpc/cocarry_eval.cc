// Headless evaluator for FR3_H_Gripper_CoCarry (dual-arm cooperative carry).
//
// 왜 새로 만드는가: PotDual 계열에는 헤드리스 러너가 없었다. `pot_eval.cc` 는
// `FR3_H_Gripper_Pot` 을 찾고(그 task 의 `l_grip`/`l_grasp` 센서에 의존한다) PotDual/
// CoCarry 를 찾지 않는다. 그래서 이 계열의 기록이 GUI 관찰 1 건뿐이었고 성공률이 없었다.
// 기준선 없이는 난이도를 올렸을 때 무엇이 나빠졌는지 말할 수 없다.
//
// 성공 판정은 **task 의 phase 4 도달**을 그대로 쓴다 (새 기준을 발명하지 않는다).
// FSM 은 `parameters[7] = phase_` 로 phase 를 노출하고(`fr3.cc:777`), phase 4 는
// 위치 `pd_done_tol`(10 mm) AND 자세 `pd_done_ang`(0.10 rad)을 `pd_done_dwell`(0.3 s)
// 유지했을 때만 선다(`fr3.cc:682-692`).
//
// 조기 지표 두 개를 함께 낸다. 실패한 뒤의 최종 오차만 보면 "협조가 무너지는 중"을
// 볼 수 없기 때문이다:
//   internal_max  두 손 상대 포즈가 파지 시점 값에서 벗어난 최대량. `Internal` 항이
//                 재는 것과 같은 양이고, weight 0 → 1e7 에서 이 값이 1.8 → 0.5 mm 로
//                 줄면서 운반이 실패한 이력이 있다(research_log_2026-08-26).
//   weldF_max     weld 구속력 최대. 폐사슬 내부 렌치의 시뮬레이션 정답지다
//                 (`fr3.cc:783-800` 이 같은 값을 디버그로만 찍고 있었다).
//
// Usage: cocarry_eval [seconds] [steps_per_plan]
//   MJPC_CC_TAG       행 라벨
//   MJPC_SEED         플래너 시드
//   MJPC_CC_GOAL      "x,y,z"  냄비 목표 위치 (mocap 2)
//   MJPC_CC_GOAL_RPY  "r,p,y"  냄비 목표 자세 [deg]. **난이도 1축**. 기본은 XML 의
//                     시작 자세를 그대로 써서 회전 0 이다.
//   MJPC_CC_OBS       "x,y"    장애물 기둥 위치 (mocap 3). 없으면 멀리 주차된 채.
//   MJPC_CC_MASS      1|2|3    냄비 질량 [kg]. XmlPath() 가 XML 을 갈아 끼운다.
//   MJPC_CC_W         "이름=값,..." weight 덮어쓰기 (예 "Internal=1e7")
//   MJPC_CC_LOG       스텝별 CSV 경로
//   MJPC_CC_DBG       [s]      진단 출력 주기
// 공유 파일을 수정하지 않는다.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/tasks.h"

namespace {

mjpc::Task* g_task = nullptr;
void residual_callback(const mjModel* m, mjData* d, int stage) {
  if (stage == mjSTAGE_ACC) g_task->Residual(m, d, d->sensordata);
}

double OriErrDeg(const double* qa, const double* qb) {
  double conj[4], e[4], v[3];
  mju_negQuat(conj, qb);
  mju_mulQuat(e, conj, qa);
  mju_quat2Vel(v, e, 1.0);
  return mju_norm3(v) * 180.0 / mjPI;
}

// weld 구속력의 병진 성분 크기. 두 팔이 물체를 통해 서로 밀고 있는 힘이다.
double WeldForce(const mjModel* m, const mjData* d, int eq) {
  if (eq < 0) return 0.0;
  double f[3] = {0, 0, 0};
  int k = 0;
  for (int i = 0; i < d->nefc; i++) {
    if (d->efc_type[i] != mjCNSTR_EQUALITY || d->efc_id[i] != eq) continue;
    if (k < 3) f[k] = d->efc_force[i];   // 앞 3 행 = 병진
    k++;
  }
  return mju_norm3(f);
}

}  // namespace

int main(int argc, char** argv) {
  double secs = (argc > 1) ? std::atof(argv[1]) : 55.0;
  // 기본 30 = 이 task 의 설계 재계획 주기. 물리 timestep 은 1 ms
  // (`fr3_H_gripper_dual.xml:3`) 이고 `agent_timestep`(dt_mppi) 은 30 ms
  // (`task.xml:7`) 이다. Handover 에서 쓰던 7 을 그대로 두면 7 ms 마다 재계획을
  // 요구하면서 재계획 1 회가 10~13 ms 걸려(실측) 실시간 미달이 되고, 기준선이
  // 설계 조건이 아닌 조건에서 측정된다. task.xml 의 주석이 기준을 명시한다:
  // "T_c 12.33 ms at K=128/16thr -> ratio 0.41" (= 12.33/30).
  int steps_per_plan = (argc > 2) ? std::atoi(argv[2]) : 30;
  const char* tag = std::getenv("MJPC_CC_TAG");
  if (!tag || !tag[0]) tag = "run";
  int seed = 0;
  if (const char* s = std::getenv("MJPC_SEED"); s && s[0]) seed = std::atoi(s);

  // task 를 고를 수 있게 해 둔다. CoCarry 는 PotDual 의 복사본이므로, "내 복사가
  // 무언가를 깨뜨렸나"와 "원본도 헤드리스에서 안 되나"를 **같은 러너로** 갈라야 한다.
  // 원본은 읽기만 하므로(다른 task 를 선택할 뿐) 무영향이다.
  const char* task_name = std::getenv("MJPC_CC_TASK");
  if (!task_name || !task_name[0]) task_name = "FR3_H_Gripper_CoCarry";

  mjpc::Agent agent;
  agent.SetTaskList(mjpc::GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName(task_name);
  if (agent.gui_task_id == -1) {
    std::fprintf(stderr, "task %s not found\n", task_name);
    return 1;
  }
  std::fprintf(stderr, "[CC] task = %s\n", task_name);
  auto load = agent.LoadModel();
  mjModel* model = load.model.get();
  if (!model) { std::fprintf(stderr, "%s\n", load.error.c_str()); return 1; }

  mjData* data = mj_makeData(model);
  int home = mj_name2id(model, mjOBJ_KEY, "home");
  if (home >= 0) mj_resetDataKeyframe(model, data, home);
  mj_forward(model, data);

  agent.estimator_enabled = false;
  agent.Initialize(model);
  agent.Allocate();
  agent.Reset(data->ctrl);
  agent.plan_enabled = true;
  g_task = agent.ActiveTask();
  mjcb_sensor = &residual_callback;

  // weight 덮어쓰기 (이름으로). `Internal` 처럼 기본 0 인 항을 켜는 데 쓴다.
  if (const char* w = std::getenv("MJPC_CC_W"); w && w[0]) {
    std::string s(w);
    size_t p = 0;
    while (p < s.size()) {
      size_t comma = s.find(',', p);
      if (comma == std::string::npos) comma = s.size();
      std::string kv = s.substr(p, comma - p);
      size_t eq = kv.find('=');
      if (eq != std::string::npos) {
        std::string name = kv.substr(0, eq);
        double val = std::atof(kv.c_str() + eq + 1);
        bool hit = false;
        for (size_t i = 0; i < g_task->weight_names.size(); i++) {
          if (g_task->weight_names[i] == name) {
            g_task->weight[i] = val; hit = true;
            std::fprintf(stderr, "[CC] weight %s = %g\n", name.c_str(), val);
          }
        }
        if (!hit)
          std::fprintf(stderr, "[CC] WARNING unknown weight '%s'\n", name.c_str());
      }
      p = comma + 1;
    }
    g_task->UpdateResidual();
  }

  int nthreads = mjpc::NumAvailableHardwareThreads() - 2;
  if (const char* t = std::getenv("MJPC_THREADS"); t && t[0])
    nthreads = std::max(1, std::atoi(t));
  mjpc::ThreadPool pool(std::max(1, nthreads));

  const int pot_body = mj_name2id(model, mjOBJ_BODY, "pot");
  const int gs[2] = {mj_name2id(model, mjOBJ_SITE, "l_gripper_site"),
                     mj_name2id(model, mjOBJ_SITE, "r_gripper_site")};
  const int weld[2] = {mj_name2id(model, mjOBJ_EQUALITY, "grasp_weld_l"),
                       mj_name2id(model, mjOBJ_EQUALITY, "grasp_weld_r")};
  // pre-grasp 지표용: 파지 프레임 사이트 (FSM 과 같은 정의)
  const int ts[2] = {mj_name2id(model, mjOBJ_SITE, "pot_grasp_l"),
                     mj_name2id(model, mjOBJ_SITE, "pot_grasp_r")};
  const double pre_off = 0.07;   // pd_pre_off
  const int obs_body = mj_name2id(model, mjOBJ_BODY, "obstacle");
  const int obs_mid = (obs_body >= 0) ? model->body_mocapid[obs_body] : -1;
  if (pot_body < 0 || gs[0] < 0 || gs[1] < 0) {
    std::fprintf(stderr, "pot body / gripper sites not found\n");
    return 1;
  }

  // task 가 목표를 놓게 한 뒤 덮어쓴다. FSM 은 mocap 2 를 **읽기만** 하므로
  // (`fr3.cc:577,661`) 여기서 정한 값이 끝까지 유지된다.
  agent.ActiveTask()->Transition(model, data);
  if (const char* g = std::getenv("MJPC_CC_GOAL"); g && g[0]) {
    double v[3];
    if (std::sscanf(g, "%lf,%lf,%lf", &v[0], &v[1], &v[2]) == 3)
      for (int i = 0; i < 3; i++) data->mocap_pos[6 + i] = v[i];
  }
  // 난이도 1축: 목표 자세. XML 의 시작 자세(냄비와 같은 +90 deg z)에 rpy 를 **덧붙인다**
  // -- 절대 자세로 주면 회전 0 을 표현하려고 XML 상수를 외워야 한다.
  if (const char* r = std::getenv("MJPC_CC_GOAL_RPY"); r && r[0]) {
    double rpy[3] = {0, 0, 0};
    if (std::sscanf(r, "%lf,%lf,%lf", &rpy[0], &rpy[1], &rpy[2]) == 3) {
      const double d2r = mjPI / 180.0;
      double qx[4], qy[4], qz[4], qa[4], qd[4], out[4];
      const double ex[3] = {1, 0, 0}, ey[3] = {0, 1, 0}, ez[3] = {0, 0, 1};
      mju_axisAngle2Quat(qx, ex, rpy[0] * d2r);
      mju_axisAngle2Quat(qy, ey, rpy[1] * d2r);
      mju_axisAngle2Quat(qz, ez, rpy[2] * d2r);
      mju_mulQuat(qa, qz, qy);
      mju_mulQuat(qd, qa, qx);                     // R = Rz*Ry*Rx
      mju_mulQuat(out, qd, data->mocap_quat + 8);   // world 축 기준으로 덧붙인다
      mju_copy4(data->mocap_quat + 8, out);
      std::fprintf(stderr, "[CC] goal rpy = %.1f %.1f %.1f deg\n",
                   rpy[0], rpy[1], rpy[2]);
    }
  }
  if (const char* o = std::getenv("MJPC_CC_OBS"); o && o[0] && obs_mid >= 0) {
    double v[2];
    if (std::sscanf(o, "%lf,%lf", &v[0], &v[1]) == 2) {
      data->mocap_pos[3 * obs_mid + 0] = v[0];
      data->mocap_pos[3 * obs_mid + 1] = v[1];
      std::fprintf(stderr, "[CC] obstacle at %.3f %.3f\n", v[0], v[1]);
    }
  }
  double goal[3] = {data->mocap_pos[6], data->mocap_pos[7], data->mocap_pos[8]};
  double goal_q[4] = {data->mocap_quat[8], data->mocap_quat[9],
                      data->mocap_quat[10], data->mocap_quat[11]};
  double obs[3] = {0, 0, 0};
  if (obs_mid >= 0) mju_copy3(obs, data->mocap_pos + 3 * obs_mid);
  const double ang0 = OriErrDeg(data->xquat + 4 * pot_body, goal_q);
  const double d0 = mju_dist3(data->xpos + 3 * pot_body, goal) * 1000.0;

  FILE* log = nullptr;
  if (const char* lp = std::getenv("MJPC_CC_LOG"); lp && lp[0]) {
    log = std::fopen(lp, "w");
    if (log) std::fprintf(log, "t,phase,pot_x,pot_y,pot_z,d_mm,ang_deg,"
                               "int_mm,int_deg,weldF_l,weldF_r\n");
  }

  const int steps = static_cast<int>(secs / model->opt.timestep);
  double plan_ms = 0.0; long plan_n = 0;
  double int_max_mm = 0.0, int_max_deg = 0.0, weldF_max = 0.0;
  double d_min = 1e9, ang_at_dmin = -1.0;
  double t_done = -1.0;
  int phase_max = 0, retries = 0, prev_phase = 1;
  // pre-grasp 난이도의 **연속 지표**. phase 1 은 양팔 AND 조건이므로 "두 팔이 동시에
  // 얼마나 가까워졌는가" = min over t of max(errL, errR) 가 통과 난이도를 직접 잰다.
  // 통과/실패 이분법은 이 task 의 실행간 편차(같은 시드로도 결과가 갈린다) 앞에서
  // 정보가 너무 적어 n=6 으로 아무것도 가리지 못했다.
  double pre_minmax = 1e9, pre_errL_at = -1.0, pre_errR_at = -1.0;
  double dbg_t = -1e9;

  for (int i = 0; i < steps; i++) {
    agent.ActiveTask()->Transition(model, data);
    // 목표와 장애물을 명시적으로 유지한다 (mocap 0/1 은 FSM 이 손 타겟으로 쓴다).
    for (int k = 0; k < 3; k++) data->mocap_pos[6 + k] = goal[k];
    for (int k = 0; k < 4; k++) data->mocap_quat[8 + k] = goal_q[k];
    if (obs_mid >= 0) mju_copy3(data->mocap_pos + 3 * obs_mid, obs);
    agent.state.Set(model, data);
    agent.ActivePlanner().ActionFromPolicy(
        data->ctrl, agent.state.state().data(), agent.state.time(), false);
    mj_step(model, data);
    // 결정성 진단: 첫 몇 번의 계획 직후 제어값을 그대로 찍는다. 두 실행에서 이 값이
    // 같으면 노이즈는 결정적이고 갈라짐은 그 뒤(물리/FSM)에서 오는 것이고, 다르면
    // 시드가 노이즈를 고정하지 못하는 것이다. 후자면 어떤 비교 실험도 성립하지 않는다.
    if (std::getenv("MJPC_CC_CTRLDUMP") && i <= 4 * steps_per_plan &&
        i % steps_per_plan == 0) {
      std::fprintf(stderr, "[CTRL] i=%5d t=%.3f", i, data->time);
      for (int k = 0; k < model->nu; k++)
        std::fprintf(stderr, " %.9f", data->ctrl[k]);
      std::fprintf(stderr, "\n");
    }
    if (i % steps_per_plan == 0) {
      auto t0 = std::chrono::steady_clock::now();
      agent.PlanIteration(&pool);
      plan_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0).count();
      plan_n++;
    }

    const int phase = (g_task->parameters.size() >= 8)
                          ? static_cast<int>(g_task->parameters[7] + 0.5) : 0;
    if (phase > phase_max) phase_max = phase;
    // 재시도: 파지 실패 시 FSM 이 phase 1 로 되돌린다 (`fr3.cc:562` FsmReset).
    if (phase < prev_phase && prev_phase >= 2) retries++;
    prev_phase = phase;
    if (phase >= 4 && t_done < 0.0) t_done = data->time;

    // 폐사슬 내부 편차 -- 운반 중(phase >= 3)에만 기준이 latch 되어 있다.
    double int_mm = 0.0, int_deg = 0.0;
    if (phase >= 3 && g_task->parameters.size() >= 7) {
      double dp[3], rel[3], ql[4], qr[4], qlc[4], relq[4];
      for (int k = 0; k < 3; k++)
        dp[k] = data->site_xpos[3 * gs[1] + k] - data->site_xpos[3 * gs[0] + k];
      mju_mulMatTVec(rel, data->site_xmat + 9 * gs[0], dp, 3, 3);
      mju_mat2Quat(ql, data->site_xmat + 9 * gs[0]);
      mju_mat2Quat(qr, data->site_xmat + 9 * gs[1]);
      mju_negQuat(qlc, ql);
      mju_mulQuat(relq, qlc, qr);
      const double* rp = g_task->parameters.data();
      double e3[3] = {rel[0] - rp[0], rel[1] - rp[1], rel[2] - rp[2]};
      int_mm = mju_norm3(e3) * 1000.0;
      double rq0c[4], dq[4], v[3];
      mju_negQuat(rq0c, rp + 3);
      mju_mulQuat(dq, rq0c, relq);
      mju_quat2Vel(v, dq, 1.0);
      int_deg = mju_norm3(v) * 180.0 / mjPI;
      if (int_mm > int_max_mm) int_max_mm = int_mm;
      if (int_deg > int_max_deg) int_max_deg = int_deg;
    }
    const double wf[2] = {WeldForce(model, data, weld[0]),
                          WeldForce(model, data, weld[1])};
    const double wfm = mju_max(wf[0], wf[1]);
    if (wfm > weldF_max) weldF_max = wfm;

    if (phase <= 1 && ts[0] >= 0 && ts[1] >= 0) {
      double e[2];
      for (int a = 0; a < 2; a++) {
        const double* R = data->site_xmat + 9 * ts[a];
        const double ax[3] = {R[2], R[5], R[8]};          // 접근축 = 사이트 z
        const double* tp = data->site_xpos + 3 * ts[a];
        double pp[3];
        for (int k = 0; k < 3; k++) pp[k] = tp[k] - pre_off * ax[k];
        e[a] = mju_dist3(data->site_xpos + 3 * gs[a], pp) * 1000.0;
      }
      const double mx = mju_max(e[0], e[1]);
      if (mx < pre_minmax) { pre_minmax = mx; pre_errL_at = e[0]; pre_errR_at = e[1]; }
    }
    const double* po = data->xpos + 3 * pot_body;
    const double d_now = mju_dist3(po, goal) * 1000.0;
    const double ang_now = OriErrDeg(data->xquat + 4 * pot_body, goal_q);
    if (d_now < d_min) { d_min = d_now; ang_at_dmin = ang_now; }

    if (const char* e = std::getenv("MJPC_CC_DBG"); e && e[0]) {
      const double every = mju_max(0.05, std::atof(e));
      if (data->time - dbg_t >= every) {
        dbg_t = data->time;
        std::fprintf(stderr,
                     "[CC] t=%6.2f phase %d  pot %.3f %.3f %.3f  d %.1f mm  "
                     "ang %.2f deg  int %.2f mm/%.2f deg  weldF %.1f/%.1f N\n",
                     data->time, phase, po[0], po[1], po[2], d_now, ang_now,
                     int_mm, int_deg, wf[0], wf[1]);
      }
    }
    if (log && i % 20 == 0)
      std::fprintf(log, "%.4f,%d,%.4f,%.4f,%.4f,%.2f,%.3f,%.3f,%.3f,%.2f,%.2f\n",
                   data->time, phase, po[0], po[1], po[2], d_now, ang_now,
                   int_mm, int_deg, wf[0], wf[1]);
  }

  const double* po = data->xpos + 3 * pot_body;
  const double d_end = mju_dist3(po, goal) * 1000.0;
  const double ang_end = OriErrDeg(data->xquat + 4 * pot_body, goal_q);
  const int success = (phase_max >= 4) ? 1 : 0;
  // 어디서 멈췄는가. phase 이름은 fr3.h:40 의 주석과 같다.
  const char* stuck = (phase_max >= 4) ? "delivered"
                    : (phase_max == 3) ? "transport"
                    : (phase_max == 2) ? "approach_or_close"
                                       : "pre_grasp";

  std::printf("RESULT tag=%s seed=%d success=%d d_end_mm=%.1f ang_end_deg=%.2f"
              " d_min_mm=%.1f ang_at_dmin_deg=%.2f t_done=%.2f phase_max=%d"
              " stuck=%s retries=%d internal_max_mm=%.2f internal_max_deg=%.2f"
              " weldF_max=%.1f pre_minmax_mm=%.1f pre_L=%.1f pre_R=%.1f"
              " d0_mm=%.1f ang0_deg=%.2f plan_ms=%.2f\n",
              tag, seed, success, d_end, ang_end, d_min, ang_at_dmin, t_done,
              phase_max, stuck, retries, int_max_mm, int_max_deg, weldF_max,
              (pre_minmax > 1e8 ? -1.0 : pre_minmax), pre_errL_at, pre_errR_at,
              d0, ang0, plan_n ? plan_ms / plan_n : 0.0);

  if (log) std::fclose(log);
  mj_deleteData(data);
  return 0;
}
