#include "example_mjpc_controller/example_mjpc_controller.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <pluginlib/class_list_macros.hpp>

namespace example_mjpc_controller {

// ─── Interfaces ─────────────────────────────────────────────────────────
// Commands EFFORT. mjpc's `action` is N*m, so a position-command controller paired
// with a torque task would read newton-metres as radians -- +-87 rad of commanded
// angle. That mismatch is silent, so the interface type is load-bearing.

controller_interface::InterfaceConfiguration
ExampleMjpcController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
ExampleMjpcController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

void ExampleMjpcController::updateJointStates() {
  for (int i = 0; i < kNumJoints; ++i) {
    q_[i] = state_interfaces_[2 * i + 0].get_value();
    dq_[i] = state_interfaces_[2 * i + 1].get_value();
  }
}

// ─── Update loop (robot rate, 1 kHz on FR3) ─────────────────────────────

controller_interface::return_type ExampleMjpcController::update(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  updateJointStates();

  // 1. Publish state. PAYLOAD FIRST, COUNTER LAST: mjpc only reads when state_seq
  //    moves, so bumping it first would hand over a half-written sample.
  //    ee_pos is deliberately left alone -- nothing writes or reads it (see the hpp).
  if (bridge_) {
    for (int i = 0; i < kNumJoints; ++i) {
      bridge_->q[i] = static_cast<float>(q_[i]);
      bridge_->dq[i] = static_cast<float>(dq_[i]);
    }
    bridge_->state_seq++;
  }

  // 2. Read the planner's action, but only when it is NEW. Reusing a stale action is
  //    how a dead planner turns into a runaway arm; counting staleness instead is what
  //    makes the timeout below possible.
  if (bridge_ && bridge_->action_seq > last_action_seq_) {
    last_action_seq_ = bridge_->action_seq;
    stale_count_ = 0;
    for (int i = 0; i < kNumJoints; ++i) {
      tau_mjpc_[i] = static_cast<double>(bridge_->action[i]);
    }
    action_ready_ = true;
  } else if (action_ready_) {
    stale_count_++;
  }

  // 3. First-order filter on the sampled torque.
  if (action_ready_) {
    for (int i = 0; i < kNumJoints; ++i) {
      tau_filtered_[i] = kLpfAlpha * tau_filtered_[i] + (1.0 - kLpfAlpha) * tau_mjpc_[i];
    }
  }

  // 4. Rate limit. MPPI's action can jump between replans; this bounds the jerk seen
  //    by the joint. It also means a large step arrives over several cycles, so the
  //    ceiling in 6 is checked on the LIMITED value, not the raw one.
  for (int i = 0; i < kNumJoints; ++i) {
    double delta = tau_filtered_[i] - tau_cmd_prev_[i];
    delta = std::clamp(delta, -kMaxDelta, kMaxDelta);
    tau_cmd_prev_[i] += delta;
  }

  // 5. Timeout. No fresh action for kStaleLimit cycles -> the planner is gone.
  const bool timed_out = (stale_count_ > kStaleLimit);

  // 6. Torque ceiling.
  bool torque_valid = true;
  if (action_ready_ && !timed_out) {
    for (int i = 0; i < kNumJoints; ++i) {
      if (std::abs(tau_cmd_prev_[i]) > kTauMax[i]) { torque_valid = false; break; }
    }
  }

  // 7. Command, or fall back. Zero effort IS gravity compensation on this hardware
  //    interface: the robot's own controller holds g(q). On an arm where that is not
  //    true, zero effort means the arm falls -- check before reusing this.
  //    Both fallbacks LATCH action_ready_ = false, so recovery needs a fresh
  //    action_seq rather than merely the offending condition going away.
  if (action_ready_ && !timed_out && torque_valid) {
    for (int i = 0; i < kNumJoints; ++i) {
      command_interfaces_[i].set_value(tau_cmd_prev_[i]);
    }
  } else {
    for (int i = 0; i < kNumJoints; ++i) {
      command_interfaces_[i].set_value(0.0);
    }
    if (timed_out && action_ready_) {
      action_ready_ = false;
      RCLCPP_WARN(get_node()->get_logger(), "action timeout -> gravity compensation");
    }
    if (!torque_valid) {
      action_ready_ = false;
      RCLCPP_ERROR(get_node()->get_logger(), "torque limit exceeded -> gravity compensation");
    }
  }

  return controller_interface::return_type::OK;
}

// ─── Lifecycle ──────────────────────────────────────────────────────────

controller_interface::CallbackReturn ExampleMjpcController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "fr3");
  } catch (const std::exception& e) {
    fprintf(stderr, "on_init: %s\n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  q_.assign(kNumJoints, 0.0);
  dq_.assign(kNumJoints, 0.0);
  tau_mjpc_.assign(kNumJoints, 0.0);
  tau_filtered_.assign(kNumJoints, 0.0);
  tau_cmd_prev_.assign(kNumJoints, 0.0);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ExampleMjpcController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ExampleMjpcController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  std::fill(tau_mjpc_.begin(), tau_mjpc_.end(), 0.0);
  std::fill(tau_filtered_.begin(), tau_filtered_.end(), 0.0);
  std::fill(tau_cmd_prev_.begin(), tau_cmd_prev_.end(), 0.0);
  last_action_seq_ = -1;
  action_ready_ = false;
  stale_count_ = 0;

  // OWNER: creates the region and zeroes it. mjpc always attaches as a non-owner, so
  // the controller may be restarted under a running mjpc -- its 2 s retry reattaches.
  bridge_ = mjpc_bridge_open(true);
  if (bridge_) {
    RCLCPP_INFO(get_node()->get_logger(), "activated -- shared memory %s created (%zu B)",
                MJPC_SHM_NAME, sizeof(MjpcBridge));
  } else {
    // Not fatal on purpose: the arm still holds gravity compensation, which is the
    // right behaviour for a controller that cannot reach its planner.
    RCLCPP_ERROR(get_node()->get_logger(), "failed to create %s", MJPC_SHM_NAME);
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ExampleMjpcController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  if (bridge_) {
    mjpc_bridge_close(bridge_);
    mjpc_bridge_unlink();   // owner unlinks; a leftover region would be read as live
    bridge_ = nullptr;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

}  // namespace example_mjpc_controller

PLUGINLIB_EXPORT_CLASS(example_mjpc_controller::ExampleMjpcController,
                       controller_interface::ControllerInterface)
