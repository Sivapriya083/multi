"""
STATIC VIRTUAL JOINT TF PUBLISHER

Purpose: Publish TF transform for virtual joint
- Virtual joint connects robot base to world frame
- Defined in SRDF as: world → robot base
- Type: fixed (doesn't move)

Why needed?
- MoveIt planning happens in world frame
- Without this TF, MoveIt doesn't know where robot is
- static_transform_publisher publishes fixed TF

TF Tree will be:
world
  └─→ left_base_link (via virtual joint)
  └─→ right_base_link (via virtual joint)
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # Publish world → left_base_link (virtual joint)
    # This is a FIXED transform - never changes
    static_tf_left = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_left',
        output='log',
        arguments=[
            '--frame-id', 'world',
            '--child-frame-id', 'world',  # virtual joint connects here
            '--x', '0.0',
            '--y', '0.0', 
            '--z', '0.0',
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', '0.0'
        ]
    )

    return LaunchDescription([
        static_tf_left,
    ])
# ```

# **Key concept:**
# ```
# Without this TF:
# MoveIt: "Where is the robot?"
# ROS: "I don't know, no TF from world to robot"
# MoveIt: *fails*

# With this TF:
# MoveIt: "Where is the robot?"
# ROS: "At position (0,0,0) in world frame"
# MoveIt: *plans successfully*