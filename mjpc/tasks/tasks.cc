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

#include "mjpc/tasks/tasks.h"

#include <memory>
#include <vector>

#include "mjpc/task.h"
#include "mjpc/tasks/acrobot/acrobot.h"
#include "mjpc/tasks/allegro/allegro.h"
#include "mjpc/tasks/bimanual/handover/handover.h"
#include "mjpc/tasks/bimanual/insert/insert.h"
#include "mjpc/tasks/bimanual/reorient/reorient.h"
#include "mjpc/tasks/cartpole/cartpole.h"
#include "mjpc/tasks/fingers/fingers.h"
#include "mjpc/tasks/humanoid/interact/interact.h"
#include "mjpc/tasks/humanoid/stand/stand.h"
#include "mjpc/tasks/humanoid/tracking/tracking.h"
#include "mjpc/tasks/humanoid/walk/walk.h"
#include "mjpc/tasks/manipulation/manipulation.h"
// DEEPMIND INTERNAL IMPORT
#include "mjpc/tasks/op3/stand.h"
#include "mjpc/tasks/g1/stand.h"
#include "mjpc/tasks/panda/panda.h"
#include "mjpc/tasks/particle/particle.h"
#include "mjpc/tasks/quadrotor/quadrotor.h"
#include "mjpc/tasks/quadruped/quadruped.h"
#include "mjpc/tasks/rubik/solve.h"
#include "mjpc/tasks/shadow_reorient/hand.h"
#include "mjpc/tasks/swimmer/swimmer.h"
#include "mjpc/tasks/walker/walker.h"

#include "mjpc/tasks/Fr3/fr3.h"
#include "mjpc/tasks/Fr3Reach/fr3.h"
#include "mjpc/tasks/Fr3OodSim2Real/fr3.h"
#include "mjpc/tasks/Fr3HGripper/fr3.h"
#include "mjpc/tasks/Fr3HGripperReach/fr3.h"
#include "mjpc/tasks/Fr3HGripperCarry/fr3.h"
#include "mjpc/tasks/Fr3HGripperDual/fr3.h"
#include "mjpc/tasks/Fr3HGripperPot/fr3.h"
#include "mjpc/tasks/Fr3Grasp/fr3.h"
#include "mjpc/tasks/Fr3Pick/fr3.h"
#include "mjpc/tasks/Fr3MazeForce/fr3_maze_force.h"
#include "mjpc/tasks/Fr3Obstacle/fr3_obstacle.h"
#include "mjpc/tasks/Fr3ObstacleQ/fr3_obstacle.h"

namespace mjpc {

std::vector<std::shared_ptr<Task>> GetTasks() {
  return {
      std::make_shared<FR3>(),
      std::make_shared<FR3Reach>(),
      std::make_shared<FR3OodSim2Real>(),
      std::make_shared<FR3HGripper>(),
      std::make_shared<FR3HGripperReach>(),
      std::make_shared<FR3HGripperCarry>(),
      std::make_shared<FR3HGripperDual>(),
      std::make_shared<FR3HGripperPot>(),
      std::make_shared<FR3Grasp>(),
      std::make_shared<FR3Pick>(),
      std::make_shared<FR3ObstacleQ>(),
      std::make_shared<FR3MazeForce>(),
      std::make_shared<FR3Obstacle>(),
      std::make_shared<g1::Stand>(),
      // std::make_shared<Acrobot>(),
      // std::make_shared<Allegro>(),
      // std::make_shared<aloha::Handover>(),
      // std::make_shared<aloha::Insert>(),
      // std::make_shared<aloha::Reorient>(),
      // std::make_shared<Cartpole>(),
      // std::make_shared<Fingers>(),
      // std::make_shared<humanoid::Interact>(),
      std::make_shared<humanoid::Stand>(),
      // std::make_shared<humanoid::Tracking>(),
      // std::make_shared<humanoid::Walk>(),
      std::make_shared<Panda>(),
      std::make_shared<manipulation::Bring>(),
      // DEEPMIND INTERNAL TASKS
      // std::make_shared<OP3>(),
      // std::make_shared<Particle>(),
      // std::make_shared<ParticleFixed>(),
      // std::make_shared<Rubik>(),
      // std::make_shared<ShadowReorient>(),
      // std::make_shared<Quadrotor>(),
      std::make_shared<QuadrupedFlat>(),
      // std::make_shared<QuadrupedHill>(),
      // std::make_shared<Swimmer>(),
      // std::make_shared<Walker>(),
  };
}
}  // namespace mjpc
