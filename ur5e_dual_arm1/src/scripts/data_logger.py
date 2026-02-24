# scripts/data_logger.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import csv, time, matplotlib.pyplot as plt
import numpy as np

class DataLogger(Node):
    def __init__(self):
        super().__init__('data_logger')
        self.records = []
        self.create_subscription(Float64MultiArray, '/dual_arm/controller_data',
                                 self.ctrl_cb, 10)
        self.create_subscription(JointState, '/joint_states',
                                 self.js_cb, 10)
        self.js_data = None

    def ctrl_cb(self, msg):
        t = self.get_clock().now().nanoseconds * 1e-9
        d = msg.data  # [e_left x6, e_right x6, coord_error x6]
        self.records.append({'t': t,
            'left_errors': d[0:6], 'right_errors': d[6:12],
            'coord_errors': d[12:18]})

    def js_cb(self, msg):
        self.js_data = msg

    def save_and_plot(self):
        if not self.records:
            return
        times = [r['t'] - self.records[0]['t'] for r in self.records]
        le = np.array([r['left_errors'] for r in self.records])
        re = np.array([r['right_errors'] for r in self.records])
        ce = np.array([r['coord_errors'] for r in self.records])

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        axes[0].plot(times, le); axes[0].set_title('Left Arm Joint Errors'); axes[0].set_ylabel('rad')
        axes[1].plot(times, re); axes[1].set_title('Right Arm Joint Errors'); axes[1].set_ylabel('rad')
        axes[2].plot(times, ce); axes[2].set_title('Coordination Errors'); axes[2].set_ylabel('rad')
        for ax in axes: ax.set_xlabel('Time (s)'); ax.legend([f'J{i+1}' for i in range(6)])
        plt.tight_layout()
        plt.savefig('dual_arm_performance.png', dpi=150)
        plt.show()
        self.get_logger().info('Plots saved.')

def main():
    rclpy.init()
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_and_plot()

if __name__ == '__main__':
    main()