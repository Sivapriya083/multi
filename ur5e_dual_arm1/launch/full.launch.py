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

Spawn sequence:
  0s  → robot_state_publisher + gazebo
  8s  → spawn robot URDF into Gazebo
  20s → joint_state_broadcaster
  25s → dual_arm_impedance_controller  (custom — replaces both JTC controllers)
  32s → task_executor                   (starts autonomous task)
  33s → data_logger                     (starts collecting metrics)
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
    world_path = os.path.join(pkg, 'worlds', 'grasp_place_world.world')

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

    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', world_path, '-v', '4'],
        additional_env={'GZ_SIM_RESOURCE_PATH': gz_resource_path},
        output='screen'
    )

    # =========================================================================
    # STEP 1 — Spawn robot (8s delay: Gazebo needs full init)
    # =========================================================================

    spawn_robot = TimerAction(
        period=8.0,
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
    # STEP 2 — joint_state_broadcaster (20s: robot must be fully spawned)
    # =========================================================================

    spawn_jsb = TimerAction(
        period=20.0,
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
    # STEP 3 — Custom Dual-Arm Impedance Controller (25s)
    #
    # This REPLACES the original:
    #   left_arm_controller  (joint_trajectory_controller/JointTrajectoryController)
    #   right_arm_controller (joint_trajectory_controller/JointTrajectoryController)
    #
    # With a SINGLE custom plugin that controls both arms together,
    # enforcing real-time coordination constraints between them.
    # =========================================================================

    spawn_custom_controller = TimerAction(
        period=25.0,
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
    # STEP 4 — Task Executor (32s: controller must be active first)
    #
    # Autonomous coordinated grasp-and-place node.
    # Publishes joint targets to /left_arm/joint_target and /right_arm/joint_target
    # The custom controller reads these and enforces coordination constraints.
    # =========================================================================

    task_executor = TimerAction(
        period=32.0,
        actions=[
            LogInfo(msg='[LAUNCH] Starting autonomous grasp-and-place task executor...'),
            Node(
                package='ur5e_dual_arm',
                executable='task_executor',
                name='task_executor',
                parameters=[{
                    'use_sim_time': True,
                    # Object pose: placed at center (x=0, y=0, z=0.05 on table)
                    'object_x': 0.0,
                    'object_y': 0.0,
                    'object_z': 0.35,
                    # Place target: move object to right side
                    'place_x': 0.0,
                    'place_y': -0.5,
                    'place_z': 0.35,
                    # Task timing (seconds per phase)
                    'approach_duration': 5.0,
                    'grasp_duration':    3.0,
                    'lift_duration':     4.0,
                    'transfer_duration': 6.0,
                    'place_duration':    4.0,
                    'retract_duration':  4.0,
                }],
                output='screen'
            )
        ]
    )

    # =========================================================================
    # STEP 5 — Data Logger (33s: starts just after task begins)
    #
    # Subscribes to:
    #   /dual_arm/controller_data  → joint errors, coordination errors
    #   /joint_states              → positions, velocities
    #   /dual_arm/task_data        → EE pose, task phase, coordination error norm
    # Saves CSV + generates plots on shutdown (Ctrl+C)
    # =========================================================================

    data_logger = TimerAction(
        period=33.0,
        actions=[
            LogInfo(msg='[LAUNCH] Starting data logger...'),
            Node(
                package='ur5e_dual_arm',
                executable='data_logger',
                name='data_logger',
                parameters=[{
                    'use_sim_time': True,
                    'log_dir': '/tmp/dual_arm_logs',      # output directory for CSV + plots
                    'log_prefix': 'grasp_place_run',
                }],
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
        gazebo,

        # Timed sequence
        spawn_robot,            # t=8s
        spawn_jsb,              # t=20s
        spawn_custom_controller,# t=25s  ← KEY CHANGE: custom controller replaces JTC
        task_executor,          # t=32s  ← autonomous task
        data_logger,            # t=33s  ← performance logging

        # Bridges
        camera_bridge,
    ])