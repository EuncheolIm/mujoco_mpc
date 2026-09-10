#include "franka_ec/mjpc_dual_bridge_controller.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <pluginlib/class_list_macros.hpp>

namespace franka_ec {

// ─── Interfaces ─────────────────────────────────────────────────────────────
// Bridge index order is FIXED by mjpc_bridge_dual.h: 0..6 = arm_1 (left),
// 7..13 = arm_2 (right). Both sides must agree, so the loop order here is part of
// the protocol, not a style choice.

controller_interface::InterfaceConfiguration
MjpcDualBridgeController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto& arm : {arm_id1_, arm_id2_}) {
    for (int i = 1; i <= kJointsPerArm; ++i) {
      config.names.push_back(arm + "_joint" + std::to_string(i) + "/effort");
    }
  }
  return config;
}

controller_interface::InterfaceConfiguration
MjpcDualBridgeController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  // Per arm: all 7 positions, then all 7 velocities. updateJointStates() indexes
  // against exactly this order.
  for (const auto& arm : {arm_id1_, arm_id2_}) {
    for (int i = 1; i <= kJointsPerArm; ++i) {
      config.names.push_back(arm + "_joint" + std::to_string(i) + "/position");
    }
    for (int i = 1; i <= kJointsPerArm; ++i) {
      config.names.push_back(arm + "_joint" + std::to_string(i) + "/velocity");
    }
  }
  // robot_model interfaces, for the TCP diagnostic only (see the hpp).
  for (const auto& n : model_1_->get_state_interface_names()) config.names.push_back(n);
  for (const auto& n : model_2_->get_state_interface_names()) config.names.push_back(n);
  return config;
}

void MjpcDualBridgeController::updateJointStates() {
  for (int a = 0; a < kNumArms; ++a) {
    const int base = a * 2 * kJointsPerArm;  // 0 for arm_1, 14 for arm_2
    for (int j = 0; j < kJointsPerArm; ++j) {
      const int k = a * kJointsPerArm + j;
      q_[k] = state_interfaces_[base + j].get_value();
      dq_[k] = state_interfaces_[base + kJointsPerArm + j].get_value();
    }
  }
}

// ─── Update loop (1 kHz) ────────────────────────────────────────────────────

controller_interface::return_type MjpcDualBridgeController::update(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  updateJointStates();

  // 1. Publish both arms' state. PAYLOAD FIRST, COUNTER LAST: mjpc reads only when
  //    state_seq moves, so bumping it first would hand over a half-written sample --
  //    and here that sample spans both arms, so a torn read desynchronises them.
  if (bridge_) {
    for (int k = 0; k < kNumJoints; ++k) {
      bridge_->q[k] = static_cast<float>(q_[k]);
      bridge_->dq[k] = static_cast<float>(dq_[k]);
    }
    bridge_->state_seq++;
  }

  // 2. Read the planner's action, but ONLY when it is new. Reusing a stale action is
  //    how a dead planner turns into a moving arm; counting staleness instead is what
  //    makes the timeout in 5 possible at all.
  if (bridge_ && bridge_->action_seq > last_action_seq_) {
    last_action_seq_ = bridge_->action_seq;
    stale_count_ = 0;
    for (int k = 0; k < kNumJoints; ++k) {
      tau_mjpc_[k] = static_cast<double>(bridge_->action[k]);
    }
    action_ready_ = true;
    fresh_actions_++;
  } else if (action_ready_) {
    stale_count_++;
  }
  cycles_++;

  // 3. One-pole filter on the sampled torque (see kLpfAlpha: ~350 Hz, barely filters).
  if (action_ready_) {
    for (int k = 0; k < kNumJoints; ++k) {
      tau_filtered_[k] = kLpfAlpha * tau_filtered_[k] + (1.0 - kLpfAlpha) * tau_mjpc_[k];
    }
  }

  // 4. Slew limit.
  for (int k = 0; k < kNumJoints; ++k) {
    double delta = tau_filtered_[k] - tau_cmd_prev_[k];
    delta = std::clamp(delta, -kMaxDelta, kMaxDelta);
    tau_cmd_prev_[k] += delta;
  }

  // 5. Timeout: no fresh action for kStaleLimit cycles -> the planner is gone.
  const bool timed_out = (stale_count_ > kStaleLimit);

  // 6. Torque ceiling, checked on the slew-limited value.
  bool torque_valid = true;
  int bad_joint = -1;
  if (action_ready_ && !timed_out) {
    for (int k = 0; k < kNumJoints; ++k) {
      if (std::abs(tau_cmd_prev_[k]) > kTauMax[k]) {
        torque_valid = false;
        bad_joint = k;
        break;
      }
    }
  }

  // 7. Command, or fall back. Zero effort IS gravity compensation on this hardware
  //    interface, because the robot's own controller holds g(q). On an arm where zero
  //    effort means the arm falls, this fallback would have to become an explicit
  //    gravity torque before it is safe.
  //
  //    IMPORTANT: the fallback commands zero to BOTH arms, never one. A dual task
  //    whose arms share a load must not have one arm keep driving while the other
  //    goes slack.
  //
  //    Both fallbacks LATCH action_ready_ = false, so recovery needs a fresh
  //    action_seq rather than merely the offending condition going away.
  if (action_ready_ && !timed_out && torque_valid) {
    for (int k = 0; k < kNumJoints; ++k) {
      command_interfaces_[k].set_value(tau_cmd_prev_[k]);
    }
  } else {
    fallback_cycles_++;
    for (int k = 0; k < kNumJoints; ++k) {
      command_interfaces_[k].set_value(0.0);
    }
    if (timed_out && action_ready_) {
      action_ready_ = false;
      RCLCPP_WARN(get_node()->get_logger(),
                  "action timeout (%d ms without a new action_seq) -> gravity "
                  "compensation on BOTH arms",
                  kStaleLimit);
    }
    if (!torque_valid) {
      action_ready_ = false;
      const int arm = bad_joint / kJointsPerArm;
      const int joint = bad_joint % kJointsPerArm + 1;
      RCLCPP_ERROR(get_node()->get_logger(),
                   "torque limit exceeded on %s_joint%d (%.1f > %.1f Nm) -> gravity "
                   "compensation on BOTH arms",
                   (arm == 0 ? arm_id1_ : arm_id2_).c_str(), joint,
                   tau_cmd_prev_[bad_joint], kTauMax[bad_joint]);
    }
  }

  // ─── TCP diagnostic, 1 Hz ───────────────────────────────────────────────
  // ONE log call, not one per frame: RCLCPP_*_THROTTLE keeps its timer per CALL SITE,
  // so six calls inside a loop share a single timer and only the first gets through
  // each second. That is why an earlier version printed just "left J7".
  //
  // Three frames per arm so the comparison against mjpc is unambiguous: if kJoint7
  // matches mjpc's fr3_link7 but kEndEffector does not match mjpc's hand_site, the
  // difference is the configured tool frame, not the arm kinematics.
  // O_T_EE is column-major, so the translation is elements 12..14.
  {
    const franka::Frame frames[3] = {franka::Frame::kJoint7, franka::Frame::kFlange,
                                     franka::Frame::kEndEffector};
    static const char* const fname[3] = {"J7", "FLG", "EE"};
    char buf[1400];
    int n = 0;
    for (int a = 0; a < kNumArms && n < static_cast<int>(sizeof(buf)) - 1; ++a) {
      auto& fm = (a == 0) ? model_1_ : model_2_;
      const std::string& id = (a == 0) ? arm_id1_ : arm_id2_;
      n += snprintf(buf + n, sizeof(buf) - n, "\n  %-6s", id.c_str());
      for (int f = 0; f < 3 && n < static_cast<int>(sizeof(buf)) - 1; ++f) {
        const std::array<double, 16> T = fm->getPoseMatrix(frames[f]);
        n += snprintf(buf + n, sizeof(buf) - n, "  %s(%+.4f %+.4f %+.4f)", fname[f],
                      T[12], T[13], T[14]);
      }
    }
    // What actually reaches the robot, next to what mjpc published. If tau_cmd
    // tracks tau_mjpc then the torque path is fine and a missed target is the
    // planner's; if it is zeroed or clipped, it is this controller's.
    int m = n;
    m += snprintf(buf + m, sizeof(buf) - m,
                  "\n  fresh=%ld/%ld cycles  fallback=%ld  stale=%d  ready=%d",
                  fresh_actions_, cycles_, fallback_cycles_, stale_count_,
                  action_ready_ ? 1 : 0);
    for (int a = 0; a < kNumArms && m < static_cast<int>(sizeof(buf)) - 1; ++a) {
      m += snprintf(buf + m, sizeof(buf) - m, "\n  %-6s tau_mjpc[",
                    (a == 0 ? arm_id1_ : arm_id2_).c_str());
      for (int j = 0; j < kJointsPerArm; ++j)
        m += snprintf(buf + m, sizeof(buf) - m, "%6.2f", tau_mjpc_[a * kJointsPerArm + j]);
      m += snprintf(buf + m, sizeof(buf) - m, "]  tau_cmd[");
      for (int j = 0; j < kJointsPerArm; ++j)
        m += snprintf(buf + m, sizeof(buf) - m, "%6.2f",
                      tau_cmd_prev_[a * kJointsPerArm + j]);
      m += snprintf(buf + m, sizeof(buf) - m, "]");
    }
    fresh_actions_ = cycles_ = fallback_cycles_ = 0;   // per-second rates

    RCLCPP_INFO_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 1000,
                         "TCP, each in its own arm base frame:%s", buf);
  }

  return controller_interface::return_type::OK;
}

// ─── Lifecycle ──────────────────────────────────────────────────────────────

CallbackReturn MjpcDualBridgeController::on_init() {
  try {
    // Declared with defaults so the controller also loads from a minimal yaml. The
    // names follow this package's existing dual convention (arm_1 / arm_2).
    auto_declare<int>("arm_count", kNumArms);
    auto_declare<std::string>("arm_1.arm_id", "left");
    auto_declare<std::string>("arm_2.arm_id", "right");
  } catch (const std::exception& e) {
    fprintf(stderr, "MjpcDualBridgeController::on_init: %s\n", e.what());
    return CallbackReturn::ERROR;
  }

  q_.assign(kNumJoints, 0.0);
  dq_.assign(kNumJoints, 0.0);
  tau_mjpc_.assign(kNumJoints, 0.0);
  tau_filtered_.assign(kNumJoints, 0.0);
  tau_cmd_prev_.assign(kNumJoints, 0.0);
  return CallbackReturn::SUCCESS;
}

CallbackReturn MjpcDualBridgeController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  const int arm_count = static_cast<int>(get_node()->get_parameter("arm_count").as_int());
  if (arm_count != kNumArms) {
    // The shared-memory layout is fixed at 14 DOF, so this controller is genuinely
    // two-arm only; failing here is clearer than silently driving half a region.
    RCLCPP_ERROR(get_node()->get_logger(),
                 "arm_count is %d, but %s carries exactly %d arms", arm_count,
                 MJPC_DUAL_SHM_NAME, kNumArms);
    return CallbackReturn::FAILURE;
  }

  arm_id1_ = get_node()->get_parameter("arm_1.arm_id").as_string();
  arm_id2_ = get_node()->get_parameter("arm_2.arm_id").as_string();
  if (arm_id1_ == arm_id2_) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "arm_1.arm_id and arm_2.arm_id are both '%s'; the two arms would map "
                 "to the same interfaces", arm_id1_.c_str());
    return CallbackReturn::FAILURE;
  }

  model_1_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      arm_id1_ + "/" + k_robot_model_interface_name, arm_id1_);
  model_2_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      arm_id2_ + "/" + k_robot_model_interface_name, arm_id2_);

  RCLCPP_INFO(get_node()->get_logger(),
              "configured: bridge index 0-6 = '%s', 7-13 = '%s' (effort command)",
              arm_id1_.c_str(), arm_id2_.c_str());
  return CallbackReturn::SUCCESS;
}

CallbackReturn MjpcDualBridgeController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  std::fill(tau_mjpc_.begin(), tau_mjpc_.end(), 0.0);
  std::fill(tau_filtered_.begin(), tau_filtered_.end(), 0.0);
  std::fill(tau_cmd_prev_.begin(), tau_cmd_prev_.end(), 0.0);
  last_action_seq_ = -1;
  action_ready_ = false;
  stale_count_ = 0;

  // OWNER: creates the region, zeroes it, stamps the magic/size preamble. mjpc only
  // ever attaches, so this controller may be restarted under a running mjpc -- mjpc's
  // 2 s retry reattaches to the new region.
  model_1_->assign_loaned_state_interfaces(state_interfaces_);
  model_2_->assign_loaned_state_interfaces(state_interfaces_);

  bridge_ = mjpc_bridge_dual_open(true);
  if (bridge_) {
    RCLCPP_INFO(get_node()->get_logger(), "activated -- %s created (%zu B, %d DOF)",
                MJPC_DUAL_SHM_NAME, sizeof(MjpcBridgeDual), kNumJoints);
  } else {
    // Deliberately not fatal: with no region the arms simply hold gravity
    // compensation, which is the right behaviour for a controller that cannot reach
    // its planner.
    RCLCPP_ERROR(get_node()->get_logger(),
                 "failed to create %s -- both arms will hold gravity compensation",
                 MJPC_DUAL_SHM_NAME);
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn MjpcDualBridgeController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  model_1_->release_interfaces();
  model_2_->release_interfaces();

  if (bridge_) {
    mjpc_bridge_dual_close(bridge_);
    // The OWNER unlinks. A leftover region would be attached by the next mjpc run and
    // read as live, with a frozen state_seq.
    mjpc_bridge_dual_unlink();
    bridge_ = nullptr;
  }
  return CallbackReturn::SUCCESS;
}

}  // namespace franka_ec

PLUGINLIB_EXPORT_CLASS(franka_ec::MjpcDualBridgeController,
                       controller_interface::ControllerInterface)
