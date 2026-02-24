import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess, TimerAction,
    SetEnvironmentVariable, RegisterEventHandler
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    pkg = get_package_share_directory('ur5e_dual_arm')
    ur_pkg = get_package_share_directory('ur_description')

    urdf_path = os.path.join(pkg, 'urdf', 'ur5e_dual_arm.urdf.xacro')
    world_path = os.path.join(pkg, 'worlds', 'empty_world.world')
    controllers_yaml = os.path.join(pkg, 'config', 'ros2_controllers.yaml')

    # Process xacro - WRAP AS STRING
    robot_description_content = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", urdf_path]),
        value_type=str  # THIS IS CRITICAL
    )

    gz_resource_path = os.path.dirname(ur_pkg)

    # ========================================
    # robot_state_publisher
    # ========================================
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': True
        }]
    )

    # ========================================
    # Clock bridge
    # ========================================
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ],
        output='screen'
    )

    # ========================================
    # Spawn robot
    # ========================================
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'ur5e_dual_arm',
            '-z', '0.05'
        ],
        output='screen'
    )

    # ========================================
    # controller_manager
    # ========================================
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        output='screen',
        parameters=[
            {'robot_description': robot_description_content,
             'use_sim_time': True},
            controllers_yaml
        ]
    )

    # ========================================
    # Controller spawners
    # ========================================
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '-c', '/controller_manager'
        ],
        output='screen'
    )

    left_arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'left_arm_controller',
            '-c', '/controller_manager'
        ],
        output='screen'
    )

    right_arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'right_arm_controller',
            '-c', '/controller_manager'
        ],
        output='screen'
    )

    # ========================================
    # Camera bridge
    # ========================================
    camera_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera_head/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
        ],
        remappings=[
            ('/camera_head/points', '/camera/depth/points'),
        ],
        output='screen'
    )

    # ========================================
    # Sequencing
    # ========================================
    delay_controller_manager = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[controller_manager]
        )
    )

    delay_broadcaster = RegisterEventHandler(
        OnProcessExit(
            target_action=controller_manager,
            on_exit=[joint_state_broadcaster_spawner]
        )
    )

    delay_left = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[left_arm_controller_spawner]
        )
    )

    delay_right = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[right_arm_controller_spawner]
        )
    )

    return LaunchDescription([
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_resource_path),

        ExecuteProcess(
            cmd=['gz', 'sim', world_path, '-v', '4'],
            output='screen'
        ),

        robot_state_publisher,
        clock_bridge,

        TimerAction(period=3.0, actions=[spawn_robot]),

        delay_controller_manager,
        delay_broadcaster,
        delay_left,
        delay_right,

        camera_bridge,
    ])