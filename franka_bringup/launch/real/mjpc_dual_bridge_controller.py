#  Copyright (c) 2024 Franka Robotics GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Brings up both FR3s and spawns franka_ec/MjpcDualBridgeController, which creates the
# /mjpc_bridge_dual shared-memory region. Start this BEFORE mjpc -- though it does not
# strictly matter, because the mjpc task retries the attach every 2 s.
#
#   ros2 launch franka_bringup mjpc_dual_bridge_controller.py \
#       robot_ip_1:=172.16.0.2 robot_ip_2:=172.16.0.3
#
# Both arms sit on the SAME /24 (this PC holds 172.16.0.1 on the robot NIC), so the
# second arm is .3 -- not another subnet.
#
# Then, in the mujoco-mpc tree, DRY RUN FIRST -- the arms must not move and |tau| must
# be ~0 at rest:
#
#   MJPC_TASKS_DIR=$PWD/mjpc/tasks MJPC_BRIDGE_DRYRUN=1 \
#       ./build/bin/mjpc --task FR3_H_Gripper_Dual

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_ip_1_parameter_name = 'robot_ip_1'
    robot_ip_2_parameter_name = 'robot_ip_2'

    load_gripper_1_parameter_name = 'load_gripper_1'
    load_gripper_2_parameter_name = 'load_gripper_2'

    arm_id_1_parameter_name = 'arm_id_1'
    arm_id_2_parameter_name = 'arm_id_2'

    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'

    use_rviz_parameter_name = 'use_rviz'

    robot_ip_1 = LaunchConfiguration(robot_ip_1_parameter_name)
    robot_ip_2 = LaunchConfiguration(robot_ip_2_parameter_name)

    arm_id_1 = LaunchConfiguration(arm_id_1_parameter_name)
    arm_id_2 = LaunchConfiguration(arm_id_2_parameter_name)

    load_gripper_1 = LaunchConfiguration(load_gripper_1_parameter_name)
    load_gripper_2 = LaunchConfiguration(load_gripper_2_parameter_name)

    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    use_rviz = LaunchConfiguration(use_rviz_parameter_name)

    return LaunchDescription([
        DeclareLaunchArgument(
            robot_ip_1_parameter_name,
            description='Hostname or IP address of robot 1 (the LEFT arm, bridge '
                        'indices 0-6).'),
        DeclareLaunchArgument(
            robot_ip_2_parameter_name,
            description='Hostname or IP address of robot 2 (the RIGHT arm, bridge '
                        'indices 7-13). Same /24 as robot 1, e.g. 172.16.0.3.'),
        # Defaults match mjpc_dual_bridge_controller's arm_1/arm_2 arm_id in
        # dual_controllers.yaml. Override BOTH together or the interface names and the
        # bridge index order stop agreeing.
        DeclareLaunchArgument(
            arm_id_1_parameter_name,
            default_value='left',
            description='Unique arm ID of robot 1. Must equal arm_1.arm_id in '
                        'dual_controllers.yaml.'),
        DeclareLaunchArgument(
            arm_id_2_parameter_name,
            default_value='right',
            description='Unique arm ID of robot 2. Must equal arm_2.arm_id in '
                        'dual_controllers.yaml.'),
        DeclareLaunchArgument(
            use_rviz_parameter_name,
            default_value='false',
            description='Visualize the robot in Rviz'),
        DeclareLaunchArgument(
            use_fake_hardware_parameter_name,
            default_value='false',
            description='Use fake hardware'),
        DeclareLaunchArgument(
            fake_sensor_commands_parameter_name,
            default_value='false',
            description="Fake sensor commands. Only valid when '{}' is true".format(
                use_fake_hardware_parameter_name)),
        # DEFAULT false, unlike the sibling launch files. Two reasons, both real:
        #
        # 1. `dual_fr3_arm_3finger.urdf.xacro`'s H-gripper macro is NOT namespaced, so
        #    instantiating it for both arms duplicates every one of its ~60 link names
        #    -- including its own `base_link`, giving three of them. robot_state_publisher
        #    then dies with "link 'base_link' is not unique" and TF never comes up.
        #    Measured: hand:=true -> 139 links, 61 names duplicated; hand:=false -> 19
        #    links, none duplicated.
        # 2. These are Hyundai H grippers on their own EtherCAT stack (art_gripper), not
        #    Franka Hands, so franka_gripper_node has nothing to talk to and just throws.
        #
        # The bridge is arm-only and commands effort, so the hand's absence from the URDF
        # costs it nothing: the robot holds g(q) from the load configured on the robot
        # itself, not from this URDF. Set true only once that xacro takes a prefix.
        DeclareLaunchArgument(
            load_gripper_1_parameter_name,
            default_value='false',
            description='Use Franka Gripper as an end-effector on robot 1. Keep false '
                        'for the H-gripper setup -- see the note above.'),
        DeclareLaunchArgument(
            load_gripper_2_parameter_name,
            default_value='false',
            description='Use Franka Gripper as an end-effector on robot 2. Keep false '
                        'for the H-gripper setup -- see the note above.'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([PathJoinSubstitution(
                [FindPackageShare('franka_bringup'), 'launch/real',
                 'dual_franka.launch.py'])]),
            launch_arguments={robot_ip_1_parameter_name: robot_ip_1,
                              robot_ip_2_parameter_name: robot_ip_2,
                              arm_id_1_parameter_name: arm_id_1,
                              arm_id_2_parameter_name: arm_id_2,
                              load_gripper_1_parameter_name: load_gripper_1,
                              load_gripper_2_parameter_name: load_gripper_2,
                              use_fake_hardware_parameter_name: use_fake_hardware,
                              fake_sensor_commands_parameter_name: fake_sensor_commands,
                              use_rviz_parameter_name: use_rviz
                              }.items(),
        ),

        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['mjpc_dual_bridge_controller'],
            output='screen',
        ),
    ])
