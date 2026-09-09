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

add_library(threadpool STATIC)
target_sources(
  threadpool
  PUBLIC threadpool.h
  PRIVATE threadpool.cc
)
target_link_libraries(
  threadpool
  absl::base
)
target_include_directories(threadpool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(
  libmjpc STATIC
  states/state.cc
  states/state.h
  agent.cc
  agent.h
  trajectory.cc
  trajectory.h
  utilities.cc
  utilities.h
  tasks/tasks.cc
  tasks/tasks.h

  # ===== EC ===== #
  tasks/Fr3HGripperDual/fr3.cc
  tasks/Fr3HGripperDual/fr3.h
  tasks/Fr3HGripperPotDual/fr3.cc
  tasks/Fr3HGripperPotDual/fr3.h
  tasks/Fr3HGripperCoCarry/fr3.cc
  tasks/Fr3HGripperCoCarry/fr3.h

  # MPPI
  planners/MPPI/planner.cc
  planners/MPPI/planner.h

  planners/MPPI/policy.cc
  planners/MPPI/policy.h

  # PriorMPPI (MPPI subclass: ACT-prior warm-start / cost-residual injection)
  planners/PriorMPPI/planner.cc
  planners/PriorMPPI/planner.h

  # FlowMPPI (MPPI with FM warm-start)
  planners/FlowMPPI/planner.cc
  planners/FlowMPPI/planner.h

  planners/FlowMPPI/policy.cc
  planners/FlowMPPI/policy.h

  # FlowMPPIRpy (FlowMPPI clone with FM goal = pos+rpy, 6-D)
  planners/FlowMPPIRpy/planner.cc
  planners/FlowMPPIRpy/planner.h

  planners/FlowMPPIRpy/policy.cc
  planners/FlowMPPIRpy/policy.h

  # FMOnly (pure FM PD, no MPPI sampling)
  planners/FMOnly/planner.cc
  planners/FMOnly/planner.h

  planners/FMOnly/policy.cc
  planners/FMOnly/policy.h

  # RLMPPI (MPPI sampling + RL policy cost bias, G1 29-DoF)
  planners/RLMPPI/planner.cc
  planners/RLMPPI/planner.h

  # RLOnly (pure RL policy as controller, no sampling)
  planners/RLOnly/planner.cc
  planners/RLOnly/planner.h

  # MPOPI
  planners/MPOPI/planner.cc
  planners/MPOPI/planner.h

  planners/MPOPI/policy.cc
  planners/MPOPI/policy.h
  # ===== EC ===== #
  
  planners/planner.cc
  planners/planner.h
  planners/policy.h
  planners/include.cc
  planners/include.h
  planners/cost_derivatives.cc
  planners/cost_derivatives.h
  planners/model_derivatives.cc
  planners/model_derivatives.h
  planners/cross_entropy/planner.cc
  planners/cross_entropy/planner.h
  planners/robust/robust_planner.cc
  planners/robust/robust_planner.h
  planners/sample_gradient/planner.cc
  planners/sample_gradient/planner.h
  planners/sampling/planner.cc
  planners/sampling/planner.h
  planners/sampling/policy.cc
  planners/sampling/policy.h
  planners/gradient/gradient.cc
  planners/gradient/gradient.h
  planners/gradient/planner.cc
  planners/gradient/planner.h
  planners/gradient/policy.cc
  planners/gradient/policy.h
  planners/gradient/settings.h
  planners/gradient/spline_mapping.cc
  planners/gradient/spline_mapping.h
  planners/ilqg/backward_pass.cc
  planners/ilqg/backward_pass.h
  planners/ilqg/boxqp.h
  planners/ilqg/planner.cc
  planners/ilqg/planner.h
  planners/ilqg/policy.cc
  planners/ilqg/policy.h
  planners/ilqg/settings.h
  planners/ilqs/planner.cc
  planners/ilqs/planner.h
  estimators/batch.cc
  estimators/batch.h
  estimators/estimator.h
  estimators/include.cc
  estimators/include.h
  estimators/kalman.cc
  estimators/kalman.h
  estimators/unscented.cc
  estimators/unscented.h
  direct/direct.cc
  direct/direct.h
  direct/trajectory.h
  direct/model_parameters.cc
  direct/model_parameters.h
  spline/spline.cc
  spline/spline.h
  app.cc
  app.h
  norm.cc
  norm.h
  policies/onnx_policy.cc
  policies/onnx_policy.h
  policies/mlp_policy.cc
  policies/mlp_policy.h
  policies/clik_policy.cc
  policies/clik_policy.h
  policies/rl_policy.cc
  policies/rl_policy.h
  simulate.cc
  simulate.h
  task.cc
  task.h
  $<TARGET_OBJECTS:mujoco::platform_ui_adapter>
)
set_target_properties(libmjpc PROPERTIES OUTPUT_NAME mjpc)
target_compile_options(libmjpc PUBLIC ${MJPC_COMPILE_OPTIONS})
target_compile_definitions(libmjpc PRIVATE MJSIMULATE_STATIC)

# ------------------------------------------------------------------
# ONNX Runtime (optional, for FM policy warm-start). Looks for an
# install at $HOME/.local/onnxruntime first.
# ------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "$ENV{HOME}/.local/onnxruntime"
    CACHE PATH "Path to ONNX Runtime install (containing include/ and lib/)")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}/lib)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  message(STATUS "MJPC: ONNX Runtime found at ${ONNXRUNTIME_LIBRARY}")
  set(MJPC_HAS_ONNXRUNTIME ON)
  target_compile_definitions(libmjpc PUBLIC MJPC_USE_ONNXRUNTIME=1)
  target_include_directories(libmjpc PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
else()
  message(STATUS "MJPC: ONNX Runtime not found (FM warm-start disabled). "
                  "Set ONNXRUNTIME_ROOT to enable.")
  set(MJPC_HAS_ONNXRUNTIME OFF)
endif()

target_link_libraries(
  libmjpc
  absl::any_invocable
  absl::check
  absl::flat_hash_map
  absl::log
  absl::random_random
  absl::span
  glfw
  lodepng
  mujoco::mujoco
  mujoco::platform_ui_adapter
  threadpool
  Threads::Threads
  nlohmann_json::nlohmann_json
)

if(MJPC_HAS_ONNXRUNTIME)
  target_link_libraries(libmjpc ${ONNXRUNTIME_LIBRARY})
  # Eigen3 is a header-only dependency required by the FM policy wrapper.
  find_path(EIGEN3_INCLUDE_DIR Eigen/Dense
      HINTS /usr/include/eigen3 /usr/local/include/eigen3)
  if(EIGEN3_INCLUDE_DIR)
    message(STATUS "MJPC: Eigen3 at ${EIGEN3_INCLUDE_DIR}")
    target_include_directories(libmjpc PUBLIC ${EIGEN3_INCLUDE_DIR})
  else()
    message(WARNING "MJPC: Eigen3 not found — FM policy build will fail")
  endif()

  # cnpy for reading .npy stats files used by FM policy normalization.
  find_path(CNPY_INCLUDE_DIR cnpy.h
      HINTS $ENV{HOME}/.local/include /usr/local/include /usr/include)
  find_library(CNPY_LIBRARY cnpy
      HINTS $ENV{HOME}/.local/lib /usr/local/lib /usr/lib)
  if(CNPY_INCLUDE_DIR AND CNPY_LIBRARY)
    message(STATUS "MJPC: cnpy at ${CNPY_LIBRARY}")
    target_include_directories(libmjpc PUBLIC ${CNPY_INCLUDE_DIR})
    target_link_libraries(libmjpc ${CNPY_LIBRARY})
    target_compile_definitions(libmjpc PUBLIC MJPC_HAS_CNPY=1)
  else()
    message(WARNING "MJPC: cnpy not found — FM policy build will fail")
  endif()
endif()

target_include_directories(libmjpc
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
)

add_executable(
  mjpc
  main.cc
)
target_link_libraries(
  mjpc
  absl::flags
  absl::flags_parse
  absl::random_random
  absl::strings
  libmjpc
  mujoco::mujoco
  threadpool
  Threads::Threads
)
target_include_directories(mjpc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(mjpc PUBLIC ${MJPC_COMPILE_OPTIONS})
target_link_options(mjpc PRIVATE ${MJPC_LINK_OPTIONS})
target_compile_definitions(mjpc PRIVATE MJSIMULATE_STATIC)
if(APPLE)
  target_sources(mjpc PRIVATE macos_gui.mm)
  target_link_libraries(mjpc "-framework Cocoa")
endif()

add_library(
  libtestspeed STATIC
  testspeed_app.cc
  testspeed.h
  testspeed.cc
)
target_link_libraries(
  libtestspeed
  absl::random_random
  absl::strings
  libmjpc
  mujoco::mujoco
  threadpool
  Threads::Threads
)
target_include_directories(libtestspeed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(
  testspeed
  testspeed_app.cc
)
target_link_libraries(
  testspeed
  libtestspeed
  absl::flags
  absl::flags_parse
)
target_include_directories(testspeed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Headless FR3 closed-loop evaluator (q-vs-torque / mode comparison logging).
add_executable(cocarry_eval cocarry_eval.cc)
target_link_libraries(
  cocarry_eval
  libmjpc
  mujoco::mujoco
  threadpool
  Threads::Threads
)
target_include_directories(cocarry_eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Headless evaluator for FR3_H_Gripper_Handover (Gate 0 static-grasp test + plan run).
add_executable(impedance_wipe_test policies/impedance_wipe_test.cc)
target_link_libraries(impedance_wipe_test libmjpc mujoco::mujoco glfw Threads::Threads)
target_include_directories(impedance_wipe_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
# Standalone TimeSpline copy semantics test (FlowMPPI leak diagnosis)
add_executable(spline_copy_test spline/spline_copy_test.cc)
target_link_libraries(spline_copy_test libmjpc)
target_include_directories(spline_copy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Headless G1 walking-config tester (closed-loop MPPI, no viewer, CSV out).
target_compile_options(testspeed PUBLIC ${MJPC_COMPILE_OPTIONS})
target_link_options(testspeed PRIVATE ${MJPC_LINK_OPTIONS})
target_compile_definitions(testspeed PRIVATE MJSIMULATE_STATIC)

add_subdirectory(tasks)

if(BUILD_TESTING AND MJPC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(MJPC_BUILD_GRPC_SERVICE)
  add_subdirectory(grpc)
endif()

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
