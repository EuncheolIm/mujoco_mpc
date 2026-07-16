// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MJPC_TASKS_G1_STAND_H_
#define MJPC_TASKS_G1_STAND_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace g1 {

class Stand : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Stand* task) : mjpc::BaseResidualFn(task) {}

    // ----- Residuals (DIAL-MPC H1 jog cost structure) -----------------------
    //   Foot Track  (2)  : foot z vs gait-phase clock target
    //   Velocity    (2)  : pelvis vx - target_vx ;  pelvis vy
    //   Yaw         (1)  : pelvis yaw (kept at 0)
    //   Height      (1)  : pelvis z - stand height
    //   Upright     (3)  : pelvis z-axis vs world z
    //   Ctrl        (12) : leg actuator effort
    //   Base AngVel (3)  : pelvis angular velocity
    //   Joint Vel   (12) : leg joint velocities
    //   Limit       (12) : soft joint-limit violations (range-normalised)
    //   Parameters  : Target Vx, Duty, Foot Amp, Cadence
    // ------------------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  Stand() : residual_(this) {}

  std::string Name() const override;
  std::string XmlPath() const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace g1
}  // namespace mjpc

#endif  // MJPC_TASKS_G1_STAND_H_
