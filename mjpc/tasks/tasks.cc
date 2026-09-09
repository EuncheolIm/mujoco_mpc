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

#include "mjpc/tasks/Fr3HGripperPotDual/fr3.h"
#include "mjpc/tasks/Fr3HGripperCoCarry/fr3.h"
#include "mjpc/tasks/Fr3HGripperDual/fr3.h"

namespace mjpc {

std::vector<std::shared_ptr<Task>> GetTasks() {
  return {
      std::make_shared<FR3HGripperPotDual>(),
      std::make_shared<FR3HGripperCoCarry>(),
      std::make_shared<FR3HGripperDual>(),
      // DEEPMIND INTERNAL TASKS
  };
}
}  // namespace mjpc
