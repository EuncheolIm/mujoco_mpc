# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copy the model files to the binary directory to make them available to the
# built binary.

add_custom_target(
  copy_model_resources ALL
  # G1 stand 와 G1Arm 은 모델과 assets 가 소스 트리에 있어서
  # copy_resources 가 처리한다. 외부(dm_control / menagerie) 모델을 쓰는
  # 태스크가 전부 삭제되어 여기서 복사할 것이 없다.
  COMMAND ${CMAKE_COMMAND} -E true
  COMMENT "no external assets to copy")

add_custom_target(copy_resources ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Copying tasks into binary directory")

add_dependencies(copy_model_resources copy_resources)

add_dependencies(libmjpc copy_model_resources)
