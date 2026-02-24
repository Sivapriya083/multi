# scripts/task_executor.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np

class TaskExecutor(Node):
    def __init__(self):
        super().__init__('task_executor')
        self.left_pub  = self.create_publisher(Float64MultiArray, '/left_arm/joint_target', 10)
        self.right_pub = self.create_publisher(Float64MultiArray, '/right_arm/joint_target', 10)
        self.timer = self.create_timer(0.1, self.execute_task)
        self.t = 0.0
        self.phase = 'approach'
        self.get_logger().info('Task executor started')

    def execute_task(self):
        self.t += 0.1

        # Phase 1: Approach — both arms move to grasp position
        if self.phase == 'approach' and self.t < 5.0:
            left_target  = [0.0, -1.2, 1.4, -0.2, 0.0, 0.0]
            right_target = [0.0, -1.2, 1.4, -0.2, 0.0, 0.0]

        # Phase 2: Coordinated lift — synchronized upward motion
        elif self.phase == 'approach' and self.t >= 5.0:
            self.phase = 'lift'
            self.t = 0.0
            return
        elif self.phase == 'lift' and self.t < 5.0:
            progress = self.t / 5.0
            left_target  = [0.0, -1.2 - 0.3 * progress, 1.4, -0.2, 0.0, 0.0]
            right_target = [0.0, -1.2 - 0.3 * progress, 1.4, -0.2, 0.0, 0.0]

        # Phase 3: Hold
        else:
            left_target  = [0.0, -1.5, 1.4, -0.2, 0.0, 0.0]
            right_target = [0.0, -1.5, 1.4, -0.2, 0.0, 0.0]

        lmsg, rmsg = Float64MultiArray(), Float64MultiArray()
        lmsg.data = left_target
        rmsg.data = right_target
        self.left_pub.publish(lmsg)
        self.right_pub.publish(rmsg)

def main():
    rclpy.init()
    node = TaskExecutor()
    rclpy.spin(node)

if __name__ == '__main__':
    main()