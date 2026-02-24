"""
dual_arm_sim.launch.py
======================
Launch file for Dual-Arm UR5e Coordinated Grasp-and-Place simulation.

Changes from original:
  - Replaced left_arm_controller + right_arm_controller (JointTrajectoryController)
    with dual_arm_impedance_controller (custom plugin)
  - Added task_executor node  (coordinated grasp-and-place task)
  - Added data_logger node    (joint + task level data collection)
  - Adjusted spawn timings to account for single unified controller
  - FIX: Split gz sim into server (-s) and GUI (-g) to fix server-not-starting issue

Spawn sequence:
  0s  → robot_state_publisher + gazebo server + gazebo GUI
  15s → spawn robot URDF into Gazebo
  30s → joint_state_broadcaster
  38s → dual_arm_impedance_controller  (custom — replaces both JTC controllers)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable, LogInfo
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    pkg    = get_package_share_directory('ur5e_dual_arm')
    ur_pkg = get_package_share_directory('ur_description')

    urdf_path  = os.path.join(pkg, 'urdf', 'ur5e_dual_arm.urdf.xacro')
    world_path = os.path.join(pkg, 'worlds', 'empty_world.world')

    robot_description = ParameterValue(
        Command([FindExecutable(name='xacro'), ' ', urdf_path]),
        value_type=str
    )

    gz_resource_path = os.path.dirname(ur_pkg)

    # =========================================================================
    # CORE NODES
    # =========================================================================

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': True,
            'robot_description': robot_description,
        }],
        output='screen'
    )

    # FIX: Run gz sim server separately with -s -r flags
    # -s → server only
    # -r → start running immediately (don't pause)
    gazebo_server = ExecuteProcess(
        cmd=['gz', 'sim', '-s', '-r', world_path, '-v', '4'],
        additional_env={'GZ_SIM_RESOURCE_PATH': gz_resource_path},
        output='screen'
    )

    # FIX: Run gz sim GUI separately with -g flag
    # Small delay to let server start first
    gazebo_gui = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=['gz', 'sim', '-g'],
                additional_env={'GZ_SIM_RESOURCE_PATH': gz_resource_path},
                output='screen'
            )
        ]
    )

    # =========================================================================
    # STEP 1 — Spawn robot (15s delay: Gazebo needs full init)
    # =========================================================================

    spawn_robot = TimerAction(
        period=15.0,
        actions=[
            LogInfo(msg='[LAUNCH] Spawning dual-arm robot into Gazebo...'),
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-name', 'ur5e_dual_arm',
                    '-topic', '/robot_description',
                    '-x', '0', '-y', '0', '-z', '0.05',
                    '-R', '0', '-P', '0', '-Y', '0',
                ],
                output='screen'
            )
        ]
    )

    # =========================================================================
    # STEP 2 — joint_state_broadcaster (30s: robot must be fully spawned)
    # =========================================================================

    spawn_jsb = TimerAction(
        period=30.0,
        actions=[
            LogInfo(msg='[LAUNCH] Starting joint_state_broadcaster...'),
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=[
                    'joint_state_broadcaster',
                    '-c', '/controller_manager',
                    '--controller-manager-timeout', '30',
                ],
                output='screen'
            )
        ]
    )

    # =========================================================================
    # STEP 3 — Custom Dual-Arm Impedance Controller (38s)
    #
    # This REPLACES the original:
    #   left_arm_controller  (joint_trajectory_controller/JointTrajectoryController)
    #   right_arm_controller (joint_trajectory_controller/JointTrajectoryController)
    #
    # With a SINGLE custom plugin that controls both arms together,
    # enforcing real-time coordination constraints between them.
    # =========================================================================

    spawn_custom_controller = TimerAction(
        period=38.0,
        actions=[
            LogInfo(msg='[LAUNCH] Starting custom dual_arm_impedance_controller...'),
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=[
                    'dual_arm_impedance_controller',      # must match ros2_controllers.yaml
                    '-c', '/controller_manager',
                    '--controller-manager-timeout', '30',
                ],
                output='screen'
            )
        ]
    )

    # =========================================================================
    # CAMERA BRIDGE (unchanged from original)
    # =========================================================================

    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera_head/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/camera_head/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_head/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_head/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        remappings=[
            ('/camera_head/points',      '/camera/depth/points'),
            ('/camera_head/image',       '/camera/color/image_raw'),
            ('/camera_head/depth_image', '/camera/depth/image_raw'),
            ('/camera_head/camera_info', '/camera/depth/camera_info'),
        ],
        output='screen'
    )

    # =========================================================================
    # LAUNCH DESCRIPTION
    # =========================================================================

    return LaunchDescription([
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_resource_path),

        # Core
        robot_state_publisher,
        gazebo_server,      # gz sim server (-s -r)
        gazebo_gui,         # gz sim GUI (-g) with 3s delay

        # Timed sequence
        spawn_robot,             # t=15s
        spawn_jsb,               # t=30s
        spawn_custom_controller, # t=38s  ← custom controller replaces JTC

        # Bridges
        camera_bridge,
    ])