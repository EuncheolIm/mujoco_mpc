// FR3 + H-gripper DUAL-arm COOPERATIVE-GRASP task: two arms (l_/r_) grasp a long
// bar at its two ends and carry it to a draggable target. Vanilla MPPI (planner
// 14 with the FM prior OFF). Cost: each hand -> its bar end, both grippers grasp,
// bar -> target. Orientation-free grasp (position only), like the single-arm task.

#ifndef MJPC_MJPC_TASKS_FR3HGRIPPERPOTDUAL_FR3_H_
#define MJPC_MJPC_TASKS_FR3HGRIPPERPOTDUAL_FR3_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class FR3HGripperPotDual : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const FR3HGripperPotDual* task)
        : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };
  FR3HGripperPotDual() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  bool goal_init_ = false;
  // ---- phase machine (쌍 단위). 실제 상태로 매 스텝 판정하고 mocap 타겟만 옮긴다. ----
  int phase_ = 1;                  // 1 pre-grasp, 2 approach, 2.5=close(squeeze_), 3 transport, 4 delivered
  bool squeeze_ = false;
  double t_near_ = 0.0, t_arrive_ = 0.0, t_conf_ = 0.0, t_squeeze_ = 0.0, t_done_ = 0.0;
  double s_app_ = 0.0;             // [0,1] 두 팔 공통 진행률
  double line_a_[2][3] = {};       // 팔별 하강 직선 시작/끝, phase 2 진입 시 latch
  double line_b_[2][3] = {};
  double line_q_[2][4] = {{1,0,0,0},{1,0,0,0}};
  bool frozen_ = false;
  double freeze_p_[2][3] = {};     // 파지 시점 손 위치
  double freeze_q_[2][4] = {{1,0,0,0},{1,0,0,0}};
  double pot_at_grasp_[3] = {0,0,0};
  double pot_q_at_grasp_[4] = {1,0,0,0};   // 파지 시점 냄비 자세
  double rel_p_[2][3] = {};                // 냄비 프레임 기준 손 지령 포즈
  double rel_q_[2][4] = {{1,0,0,0},{1,0,0,0}};
  double rot_ = 0.0;                       // [0,1] 회전 지령 진행률
  double delta_[3] = {0,0,0};      // 두 손이 공유하는 지령 변위
  double arc_ = 0.0;
  // 레퍼런스 트림 = 적분 작용. 냄비 1 kg 은 중력 보상되지 않고(실기와 동일) MPPI 에는
  // 적분항이 없으므로, 정지한 레퍼런스에 대해서도 부하만큼 편향이 남는다(측정 45~65 mm).
  // 지령을 그 편향만큼 밀어주면 냄비가 실제로 목표에 도달한다. 질량은 건드리지 않는다.
  double trim_[3] = {0,0,0};
  bool w_base_ok_ = false;
  int w_pos_idx_[2] = {-1,-1};     // L_pos, R_pos
  double w_pos_base_[2] = {0,0};
  int w_impact_idx_ = -1;          // Impact 항 weight 인덱스
  int w_place_idx_ = -1;           // Place 항 weight 인덱스
  int w_approach_idx_ = -1;        // Approach 항 weight 인덱스
  bool place_hold_ = false;        // 레퍼런스 정지 구간(마지막 구간은 MPPI 소유)
  bool latched_contact_ = false;   // 접촉 순간 지령을 도달 자세로 재래치했는지
  double dbg_t_ = -1e9;
  void FsmReset() {
    phase_ = 1; squeeze_ = false; frozen_ = false;
    t_near_ = t_arrive_ = t_conf_ = t_squeeze_ = t_done_ = 0.0;
    s_app_ = 0.0; arc_ = 0.0; place_hold_ = false; latched_contact_ = false;
    delta_[0] = delta_[1] = delta_[2] = 0.0;
    trim_[0] = trim_[1] = trim_[2] = 0.0;
  }
};
}  // namespace mjpc

#endif  // MJPC_MJPC_TASKS_FR3HGRIPPERPOTDUAL_FR3_H_
