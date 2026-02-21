import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    moveit_config_pkg = get_package_share_directory('ur5e_dual_arm_moveit_config')

    srdf_path = os.path.join(moveit_config_pkg, 'config', 'ur5e_dual_arm.srdf')
    with open(srdf_path, 'r') as f:
        robot_description_semantic = f.read()

    kinematics_path = os.path.join(moveit_config_pkg, 'config', 'kinematics.yaml')
    with open(kinematics_path, 'r') as f:
        kinematics_config = yaml.safe_load(f)

    controllers_path = os.path.join(moveit_config_pkg, 'config', 'moveit_controllers.yaml')
    with open(controllers_path, 'r') as f:
        controllers_config = yaml.safe_load(f)

    pilz_path = os.path.join(moveit_config_pkg, 'config', 'pilz_cartesian_limits.yaml')
    with open(pilz_path, 'r') as f:
        pilz_config = yaml.safe_load(f)

    joint_limits_path = os.path.join(moveit_config_pkg, 'config', 'joint_limits.yaml')
    with open(joint_limits_path, 'r') as f:
        joint_limits_config = yaml.safe_load(f)

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            {'robot_description_semantic': robot_description_semantic},
            kinematics_config,
            controllers_config,
            pilz_config,
            joint_limits_config,
            {
                'planning_pipelines': ['ompl'],
                'default_planning_pipeline': 'ompl',
                'ompl.planning_plugins': ['ompl_interface/OMPLPlanner'],
                'ompl.request_adapters': [
                    'default_planning_request_adapters/ResolveConstraintFrames',
                    'default_planning_request_adapters/ValidateWorkspaceBounds',
                    'default_planning_request_adapters/CheckStartStateBounds',
                    'default_planning_request_adapters/CheckStartStateCollision',
                ],
                'ompl.response_adapters': [
                    'default_planning_response_adapters/AddTimeOptimalParameterization',
                    'default_planning_response_adapters/ValidateSolution',
                    'default_planning_response_adapters/DisplayMotionPath',
                ],
                'use_sim_time': True,
                'publish_planning_scene': True,
                'publish_geometry_updates': True,
                'publish_state_updates': True,
                'publish_transforms_updates': True,
                'monitor_dynamics': False,
                'planning_scene_monitor_options': {
                    'robot_description': 'robot_description',
                    'joint_state_topic': '/joint_states',
                },
            },
        ]
    )

    return LaunchDescription([move_group])