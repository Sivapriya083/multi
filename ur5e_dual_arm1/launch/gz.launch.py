import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
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
        output='screen'
    )

    # Spawn robot after 8s — Gazebo needs time to fully initialize
    spawn_robot = TimerAction(
        period=8.0,
        actions=[
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

    # joint_state_broadcaster after 20s — gz_ros2_control needs robot fully spawned
    spawn_jsb = TimerAction(
        period=20.0,
        actions=[
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

    # Arm controllers after 25s
    spawn_left = TimerAction(
        period=25.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=[
                    'left_arm_controller',
                    '-c', '/controller_manager',
                    '--controller-manager-timeout', '30',
                ],
                output='screen'
            )
        ]
    )

    spawn_right = TimerAction(
        period=25.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=[
                    'right_arm_controller',
                    '-c', '/controller_manager',
                    '--controller-manager-timeout', '30',
                ],
                output='screen'
            )
        ]
    )
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
            ('/camera_head/points', '/camera/depth/points'),
            ('/camera_head/image', '/camera/color/image_raw'),
            ('/camera_head/depth_image', '/camera/depth/image_raw'),
            ('/camera_head/camera_info', '/camera/depth/camera_info'),
        ],
        output='screen'
    )

    return LaunchDescription([
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', gz_resource_path),
        robot_state_publisher,
        gazebo,
        spawn_robot,
        spawn_jsb,
        spawn_left,
        spawn_right,
        camera_bridge,
    ])
