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
#
# 이 브랜치는 pot 협조 운반만 남긴다: FR3_H_Gripper_Dual / PotDual / CoCarry.
# 세 태스크 모두 자기 디렉터리의 `assets` 심볼릭 링크로 `../Fr3/assets` 를 보므로
# Fr3 의 mesh 만 준비하면 되고, dm_control / Menagerie 데모 태스크의 복사·patch
# 명령은 모두 지웠다(그 태스크 디렉터리 자체가 없다).

add_custom_target(
  copy_model_resources ALL
  # 1. panda.xml (Fr3 경로 생성)
  COMMAND ${CMAKE_COMMAND} -E copy
  ${menagerie_SOURCE_DIR}/franka_emika_panda/panda.xml
  ${CMAKE_CURRENT_BINARY_DIR}/Fr3/panda.xml

  # 2. panda assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${menagerie_SOURCE_DIR}/franka_emika_panda/assets
  ${CMAKE_CURRENT_BINARY_DIR}/Fr3/assets

  # 3. panda_modified.xml 생성
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/Fr3/panda_modified.xml
  ${CMAKE_CURRENT_BINARY_DIR}/Fr3/panda.xml
  <${CMAKE_CURRENT_SOURCE_DIR}/Fr3/fr3.xml.patch

  # 4. fr3_yeshand.xml (FM 학습 기구학과 일치)
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/Fr3/fr3_yeshand.xml
  ${CMAKE_CURRENT_BINARY_DIR}/Fr3/fr3_yeshand.xml

  COMMENT "Copying FR3 assets into binary directory")

add_custom_target(copy_resources ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Copying tasks into binary directory")

add_dependencies(copy_model_resources copy_resources)

add_dependencies(libmjpc copy_model_resources)
