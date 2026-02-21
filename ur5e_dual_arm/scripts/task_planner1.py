#!/usr/bin/env python3
import rclpy
import threading
import math
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseArray
from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')
        self.cb_group = ReentrantCallbackGroup()

        self.left_client = ActionClient(
            self, FollowJointTrajectory,
            '/left_arm_controller/follow_joint_trajectory',
            callback_group=self.cb_group)
        self.right_client = ActionClient(
            self, FollowJointTrajectory,
            '/right_arm_controller/follow_joint_trajectory',
            callback_group=self.cb_group)

        self.sub = self.create_subscription(
            PoseArray, '/detected_objects',
            self.objects_callback, 10,
            callback_group=self.cb_group)
        self.status_pub = self.create_publisher(String, '/task_status', 10)

        self.get_logger().info('Waiting for controllers...')
        self.left_client.wait_for_server()
        self.right_client.wait_for_server()
        self.get_logger().info('Task planner ready!')
        self.busy = False

        # Predefined joint positions (radians)
        self.left_joints = [
            'left_shoulder_pan_joint',
            'left_shoulder_lift_joint',
            'left_elbow_joint',
            'left_wrist_1_joint',
            'left_wrist_2_joint',
            'left_wrist_3_joint',
        ]
        self.right_joints = [
            'right_shoulder_pan_joint',
            'right_shoulder_lift_joint',
            'right_elbow_joint',
            'right_wrist_1_joint',
            'right_wrist_2_joint',
            'right_wrist_3_joint',
        ]

        # Home position
        self.left_home  = [0.0, -1.5707, 1.5707, 0.0, 0.0, 0.0]
        self.right_home = [0.0, -1.5707, 1.5707, 0.0, 0.0, 0.0]

        # Approach position — arm stretched toward object
        # Tune these angles to point toward your objects
        self.left_approach  = [0.5, -1.2, 1.2, -0.5, -1.5707, 0.0]
        self.right_approach = [-0.5, -1.2, 1.2, -0.5, 1.5707, 0.0]

    def objects_callback(self, msg):
        if self.busy or len(msg.poses) == 0:
            return
        self.busy = True
        t = threading.Thread(
            target=self.execute_task, args=(msg.poses,), daemon=True)
        t.start()

    def execute_task(self, poses):
        try:
            left_targets  = [p for p in poses if p.position.y > 0]
            right_targets = [p for p in poses if p.position.y <= 0]
            self.publish_status(
                f'Found {len(left_targets)} left, {len(right_targets)} right')

            threads = []
            if left_targets:
                threads.append(threading.Thread(
                    target=self.move_to,
                    args=(self.left_client, self.left_joints,
                          self.left_approach, 'left', 4.0), daemon=True))
            if right_targets:
                threads.append(threading.Thread(
                    target=self.move_to,
                    args=(self.right_client, self.right_joints,
                          self.right_approach, 'right', 4.0), daemon=True))

            # Move both arms simultaneously
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.publish_status('Approach done. Going home.')

            # Return home simultaneously
            home_threads = [
                threading.Thread(target=self.move_to,
                    args=(self.left_client, self.left_joints,
                          self.left_home, 'left', 3.0), daemon=True),
                threading.Thread(target=self.move_to,
                    args=(self.right_client, self.right_joints,
                          self.right_home, 'right', 3.0), daemon=True),
            ]
            for t in home_threads:
                t.start()
            for t in home_threads:
                t.join()

            self.publish_status('Cycle complete.')

        except Exception as e:
            self.get_logger().error(f'Task failed: {str(e)}')
        finally:
            self.busy = False

    def wait_for_future(self, future, timeout=30.0):
        event = threading.Event()
        future.add_done_callback(lambda f: event.set())
        event.wait(timeout=timeout)
        return future.result()

    def move_to(self, client, joint_names, positions, side, duration=3.0):
        traj = JointTrajectory()
        traj.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * len(positions)
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1) * 1e9)
        traj.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.publish_status(f'{side} arm moving...')
        gh = self.wait_for_future(client.send_goal_async(goal))
        if not gh or not gh.accepted:
            self.publish_status(f'{side} arm goal rejected')
            return

        result = self.wait_for_future(gh.get_result_async(), timeout=duration + 5.0)
        if result:
            self.publish_status(f'{side} arm done')
        else:
            self.publish_status(f'{side} arm timeout')

    def publish_status(self, msg):
        self.get_logger().info(msg)
        s = String()
        s.data = msg
        self.status_pub.publish(s)


def main():
    rclpy.init()
    node = TaskPlannerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
