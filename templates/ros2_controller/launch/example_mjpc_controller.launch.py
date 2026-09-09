"""Spawn the template controller on top of an already-running controller_manager.

This deliberately does NOT bring up the robot: hardware bringup is vendor specific
(for franka_ros2 it is franka.launch.py, which needs robot_ip). Include that first,
then this, exactly as franka_bringup/launch/mppi_track_controller.launch.py does.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['example_mjpc_controller'],
            output='screen',
        ),
    ])
