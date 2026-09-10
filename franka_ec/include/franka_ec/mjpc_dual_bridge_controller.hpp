// Dual-arm torque bridge between mujoco-mpc's FR3_H_Gripper_Dual task and two real
// FR3s. Owns the POSIX shared-memory region /mjpc_bridge_dual (see
// franka_ec/mjpc_bridge_dual.h for the layout and the ownership rule).
//
// This is NOT a variant of MultiJointMPPIController. That controller receives joint
// POSITION/VELOCITY references from an external MPPI over ZMQ and closes a PD loop on
// them locally. This one receives FEEDFORWARD TORQUE over shared memory and passes it
// through, filtered and limited. The two are independent; neither replaces the other.
//
// Command interfaces are EFFORT. mjpc's `action` is N*m, so pairing this protocol with
// a position-command controller would send newton-metres where radians are expected --
// +-87 rad of commanded angle, with nothing to warn you.

#pragma once

#include <array>
#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>

#include <franka/model.h>
#include "franka_semantic_components/franka_robot_model.hpp"

#include "franka_ec/mjpc_bridge_dual.h"

using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_ec {

class MjpcDualBridgeController : public controller_interface::ControllerInterface {
 public:
  controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  static constexpr int kNumArms = MJPC_DUAL_NARM;      // 2
  static constexpr int kJointsPerArm = MJPC_DUAL_NJOINT;  // 7
  static constexpr int kNumJoints = MJPC_DUAL_NDOF;    // 14, left then right

  // Retention coefficient of a one-pole IIR: tau_f = a*tau_f + (1-a)*tau_new.
  // At the 1 kHz robot rate a = 0.2 puts the -3 dB point near 350 Hz, i.e. this
  // BARELY FILTERS -- it clips the sharpest corner off a replan step and nothing
  // more. Do not size the planner's jitter budget against it.
  static constexpr double kLpfAlpha = 0.2;

  // Torque slew cap, per 1 ms cycle -> 1000 N*m/s. Bounds the jerk a joint sees when
  // MPPI's action jumps between replans. Because a large step therefore arrives over
  // several cycles, the ceiling below is checked on the LIMITED value, not the raw one.
  static constexpr double kMaxDelta = 1.0;

  // Cycles without a fresh action_seq before the planner is declared gone. 100 at
  // 1 kHz = 100 ms.
  static constexpr int kStaleLimit = 100;

  // FR3-specific, per arm. Wrong values here either never trip or trip constantly.
  static constexpr std::array<double, kNumJoints> kTauMax = {
      87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0,   // left
      87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0};  // right

  void updateJointStates();

  std::string arm_id1_{"left"};   // -> bridge indices 0..6
  std::string arm_id2_{"right"};  // -> bridge indices 7..13

  // DIAGNOSTIC ONLY. The bridge exchanges joint angles and torque; it does not need
  // the robot's own kinematics. These exist so the robot's TCP can be printed next to
  // mjpc's, which is the only way to tell a frame-definition difference from a
  // kinematic one. They add the `<arm_id>/robot_model` state interfaces, so the
  // model broadcaster must be available -- the dual bringup spawns it.
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> model_1_, model_2_;
  const std::string k_robot_model_interface_name{"robot_model"};

  std::vector<double> q_, dq_;
  std::vector<double> tau_mjpc_, tau_filtered_, tau_cmd_prev_;

  MjpcBridgeDual* bridge_ = nullptr;
  int32_t last_action_seq_ = -1;
  bool action_ready_ = false;   // a first action has arrived and not been invalidated
  int stale_count_ = 0;

  // Diagnostic counters for the 1 Hz log: how many FRESH actions arrived, and how
  // many cycles fell back. Without these, "the arm does not reach the target" cannot
  // be told apart from "the torque never gets through".
  long fresh_actions_ = 0;
  long fallback_cycles_ = 0;
  long cycles_ = 0;

};

}  // namespace franka_ec
