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
// DEEPMIND INTERNAL IMPORT
#include "mjpc/tasks/g1/stand.h"

#include "mjpc/tasks/G1Arm/g1_arm.h"

namespace mjpc {

std::vector<std::shared_ptr<Task>> GetTasks() {
  return {
      std::make_shared<g1::Stand>(),
      
      // ===== g1 upperbody task! ====== //
      std::make_shared<G1Reach>(),
      // ==================================

      // DEEPMIND INTERNAL TASKS
  };
}
}  // namespace mjpc
