// Minimal ros2_control controller that OWNS /mjpc_bridge and speaks torque to mjpc.
//
// TEMPLATE. This is franka_ec::MPPITrackController reduced to the bridge protocol and
// the safety envelope, with the vendor-specific parts removed so it builds against
// plain ros2_control on any 7-DOF arm that exposes effort command interfaces.
//
// What was dropped, and why it is safe to drop:
//   * franka_semantic_components::FrankaRobotModel. The real controller claims franka's
//     robot_state / robot_model interfaces, but ONLY to have FK available. It never
//     writes MjpcBridge::ee_pos -- verified: no mjpc task reads that field either, so
//     the 12 bytes are dead space kept for layout compatibility. Nothing needs FK.
//   * the unused k_gains_ / d_gains_ / initial_q_ members.
//
// What must NOT be dropped: the ownership rule, the counter-last write order, and all
// three fallbacks. See MJPC_ROS2_BRIDGE_GUIDE.md sections 2 and 3.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>

// Must stay byte-identical to every other copy in the workspace -- they all map the
// same region by name, so a struct that disagrees misaligns everything silently.
// templates/ros2_bridge/check_headers.sh checks this.
#include "example_mjpc_controller/mjpc_bridge.h"

namespace example_mjpc_controller {

class ExampleMjpcController : public controller_interface::ControllerInterface {
 public:
  [[nodiscard]] controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;

  [[nodiscard]] controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;

  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(
      const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn on_activate(
      const rclcpp_lifecycle::State& previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
      const rclcpp_lifecycle::State& previous_state) override;

 private:
  static constexpr int kNumJoints = 7;

  // Per-joint torque ceiling. THESE ARE FR3 VALUES -- change them for another arm.
  // Exceeding any one of them drops the whole controller to gravity compensation.
  static constexpr double kTauMax[kNumJoints] = {87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0};
  // Retention coefficient of the first-order filter on the planner's torque.
  // 0.2 at 1 kHz is a ~350 Hz pole, i.e. this barely filters. franka_ec's source
  // comment claims "160 Hz" and is wrong; do not size a jitter budget against it.
  static constexpr double kLpfAlpha = 0.2;
  static constexpr double kMaxDelta = 1.0;    // N*m per step -> 1000 N*m/s at 1 kHz
  static constexpr int kStaleLimit = 100;     // cycles without a new action_seq -> 100 ms

  void updateJointStates();

  MjpcBridge* bridge_ = nullptr;   // owned: created on activate, unlinked on deactivate
  int32_t last_action_seq_ = -1;
  bool action_ready_ = false;
  int stale_count_ = 0;

  std::string arm_id_;
  std::vector<double> q_;
  std::vector<double> dq_;
  std::vector<double> tau_mjpc_;
  std::vector<double> tau_filtered_;
  std::vector<double> tau_cmd_prev_;
};

}  // namespace example_mjpc_controller
