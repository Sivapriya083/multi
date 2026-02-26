"""
ur5e_impulse_control.launch.py  v5
===================================
ROS 2 Jazzy | Gazebo Harmonic

FIXED based on actual ur.urdf.xacro args found on target system:
  <xacro:arg name="name"          default="ur"/>
  <xacro:arg name="ur_type"       default="ur5x"/>   <- must set to ur5e
  <xacro:arg name="tf_prefix"     default=""/>
  <xacro:arg name="joint_limit_params"  ... />
  <xacro:arg name="kinematics_params"   ... />
  <xacro:arg name="physical_params"     ... />
  <xacro:arg name="visual_params"       ... />
  <xacro:arg name="safety_limits"       default="false"/>
  <xacro:arg name="force_abs_paths"     default="false"/>

  NOTE: sim_gz and use_fake_hardware do NOT exist in this xacro version.
  ros2_control is added via ur_macro.xacro which is included automatically.

ROBOT DESCRIPTION STRATEGY:
  Reads pre-generated config/ur5e.urdf (static file).
  Generate it once before launching:

    ros2 run xacro xacro \
      $(ros2 pkg prefix ur_description)/share/ur_description/urdf/ur.urdf.xacro \
      name:=ur5e ur_type:=ur5e \
      > ~/arm_ws/src/ur5e_impulse_control/config/ur5e.urdf

    colcon build --packages-select ur5e_impulse_control
"""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    pkg              = get_package_share_directory('ur5e_impulse_control')
    world_file       = os.path.join(pkg, 'worlds', 'ur5e_world.sdf')
    aic_params_file  = os.path.join(pkg, 'config',  'aic_params.yaml')
    urdf_file        = os.path.join(pkg, 'config',  'ur5e.urdf')

    # ── Load pre-generated URDF ────────────────────────────────────────────
    # We read the static file rather than running xacro at launch time.
    # This avoids all xacro argument name mismatches across driver versions.
    if not os.path.exists(urdf_file):
        raise FileNotFoundError(
            '\n' + '='*60 + '\n'
            'URDF file missing: ' + urdf_file + '\n'
            '\nGenerate it by running these commands:\n'
            '\n  cd ~/arm_ws\n'
            '  ros2 run xacro xacro \\\n'
            '    $(ros2 pkg prefix ur_description)/share/ur_description/urdf/ur.urdf.xacro \\\n'
            '    name:=ur5e ur_type:=ur5e \\\n'
            '    > src/ur5e_impulse_control/config/ur5e.urdf\n'
            '\n  colcon build --packages-select ur5e_impulse_control\n'
            + '='*60
        )

    with open(urdf_file, 'r') as f:
        robot_description_xml = f.read()

    # ── Launch arguments ───────────────────────────────────────────────────
    arg_use_gui  = DeclareLaunchArgument(
        'use_gui', default_value='true',
        description='Open Gazebo GUI. Set false for headless/SSH.')

    arg_use_rviz = DeclareLaunchArgument(
        'use_rviz', default_value='false',
        description='Open RViz2. Requires hardware OpenGL. Default: false.')

    use_gui  = LaunchConfiguration('use_gui')
    use_rviz = LaunchConfiguration('use_rviz')

    # Software rendering — safe fallback for Optimus/VM/SSH systems.
    # Remove if you have a working dedicated GPU for better Gazebo speed.
    set_sw_render = SetEnvironmentVariable(
        name='LIBGL_ALWAYS_SOFTWARE', value='1')

    # ── 1. Gazebo Harmonic ─────────────────────────────────────────────────
    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args':          '-r -v2 ' + world_file,
            'on_exit_shutdown': 'true',
        }.items(),
    )

    # ── 2. Robot state publisher ───────────────────────────────────────────
    # Reads robot_description XML string and publishes /tf + /tf_static
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_xml,
            'publish_frequency': 500.0,
            'use_sim_time'     : True,
        }],
    )

    # ── 3. Spawn robot into Gazebo ─────────────────────────────────────────
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_ur5e',
        arguments=[
            '-topic', '/robot_description',
            '-name',  'ur5e',
            '-x', '0.0', '-y', '0.0', '-z', '0.0',
        ],
        output='screen',
    )

    # ── 4. ROS-Gazebo bridge ───────────────────────────────────────────────
    # Bridges Gazebo topics to ROS 2 topics:
    #   /clock                   gz->ros  (sim time)
    #   /joint_states            gz->ros  (robot joint positions/velocities)
    #   /ur5e/ft_sensor/wrench   gz->ros  (force-torque sensor)
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_ros2_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/world/ur5e_world/model/ur5e/joint_state'
            '@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/ur5e/ft_sensor/wrench'
            '@geometry_msgs/msg/WrenchStamped[gz.msgs.Wrench',
        ],
        remappings=[
            ('/world/ur5e_world/model/ur5e/joint_state', '/joint_states'),
        ],
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # ── 5. Joint state broadcaster ─────────────────────────────────────────
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/controller_manager',
        ],
        output='screen',
    )

    # ── 6. Forward effort controller ───────────────────────────────────────
    effort_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'forward_effort_controller',
            '--controller-manager', '/controller_manager',
        ],
        output='screen',
    )

    # ── 7. Adaptive Impedance Controller  (starts after 5s) ───────────────
    aic_node = TimerAction(
        period=5.0,
        actions=[Node(
            package='ur5e_impulse_control',
            executable='controller_node.py',
            name='adaptive_impedance_controller',
            output='screen',
            parameters=[aic_params_file, {'use_sim_time': True}],
        )],
    )

    # ── 8. Impulse Injector  (starts after 6s, then waits 3s internally) ──
    injector_node = TimerAction(
        period=6.0,
        actions=[Node(
            package='ur5e_impulse_control',
            executable='injector_node.py',
            name='impulse_injector',
            output='screen',
            parameters=[{
                'startup_delay_s': 3.0,
                'world_name'     : 'ur5e_world',
                'ee_link'        : 'ur5e/wrist_3_link',
                'use_sim_time'   : True,
            }],
        )],
    )

    # ── 9. Performance Monitor  (starts after 5s) ─────────────────────────
    monitor_node = TimerAction(
        period=5.0,
        actions=[Node(
            package='ur5e_impulse_control',
            executable='monitor_node.py',
            name='performance_monitor',
            output='screen',
            parameters=[{'use_sim_time': True}],
        )],
    )

  
    rviz_node = TimerAction(
        period=4.0,
        actions=[Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            condition=IfCondition(use_rviz),
            parameters=[{'use_sim_time': True}],
        )],
    )

    return LaunchDescription([
        set_sw_render,
        arg_use_gui,
        arg_use_rviz,
        gz_sim_launch,
        robot_state_publisher,
        spawn_robot,
        gz_bridge,
        joint_state_broadcaster_spawner,
        effort_controller_spawner,
        aic_node,
        injector_node,
        monitor_node,
        rviz_node,
    ])