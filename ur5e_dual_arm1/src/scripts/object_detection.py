#!/usr/bin/env python3
"""
Object Detection Node — Day 4 Update
-------------------------------------
Additions vs Day 3:
  - Publishes /detected_objects_categories (std_msgs/String CSV)
    e.g. "large,small,medium" matching order of /detected_objects poses
  - Logs LEFT/RIGHT side per object
  - Reports total left/right count per frame

World objects (ur5e_dual_arm.sdf):
  cylinder : pose x=0.5  y=-0.2  z=0.5  -> right arm (y < 0)
  sphere   : pose x=0.6  y=0.4   z=0.5  -> left  arm (y >= 0)

Key fixes vs original:
  1. z PassThrough 0.02-2.0  (was 0.02-0.35, objects at z~0.5 were cut out)
  2. np.any(size > 0.03)     (was np.all, sphere axes slightly under threshold)
  3. min_size = 5             (was 15, depth cameras give sparse returns)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import String
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from scipy.spatial import cKDTree
import tf2_ros
import tf2_geometry_msgs


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        self.sub = self.create_subscription(
            PointCloud2, '/camera/depth/points', self.cloud_callback, 10)

        self.marker_pub   = self.create_publisher(
            MarkerArray, '/detected_objects_markers', 10)
        self.pose_pub     = self.create_publisher(
            PoseArray, '/detected_objects', 10)
        self.category_pub = self.create_publisher(
            String, '/detected_objects_categories', 10)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.target_frame = 'world'
        self.get_logger().info('Object detection node started (Day 4)')
        self.get_logger().info(
            'Expected: cylinder@(0.5,-0.2,0.5) sphere@(0.6,0.4,0.5)')

    def cloud_callback(self, msg):
        # ── read ──────────────────────────────────────────────────────────────
        points = np.array([
            [p[0], p[1], p[2]]
            for p in pc2.read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True)
        ])
        if len(points) == 0:
            return

        # ── PassThrough: wide z range ─────────────────────────────────────────
        mask     = (points[:, 2] > 0.02) & (points[:, 2] < 2.0)
        filtered = points[mask]
        if len(filtered) < 20:
            return

        # ── VoxelGrid downsample ──────────────────────────────────────────────
        voxel_size    = 0.008
        keys          = np.floor(filtered / voxel_size).astype(int)
        _, unique_idx = np.unique(keys, axis=0, return_index=True)
        downsampled   = filtered[unique_idx]
        if len(downsampled) < 10:
            return

        # ── Euclidean clustering ──────────────────────────────────────────────
        clusters = self.fast_cluster(downsampled, tolerance=0.02, min_size=5)

        # ── noise filter: at least one axis > 3 cm ────────────────────────────
        valid = []
        for c in clusters:
            size = np.max(c, axis=0) - np.min(c, axis=0)
            if np.any(size > 0.03):
                valid.append(c)

        if not valid:
            return

        # ── build output messages ─────────────────────────────────────────────
        marker_array = MarkerArray()
        pose_array   = PoseArray()
        pose_array.header.frame_id = self.target_frame
        pose_array.header.stamp    = msg.header.stamp

        clear        = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        categories = []

        for i, cluster in enumerate(valid):
            centroid = np.mean(cluster, axis=0)
            size     = np.max(cluster, axis=0) - np.min(cluster, axis=0)
            max_dim  = float(np.max(size))

            if max_dim > 0.15:
                category, color = 'large',  (1.0, 0.0, 0.0)
            elif max_dim > 0.08:
                category, color = 'medium', (1.0, 0.5, 0.0)
            else:
                category, color = 'small',  (0.0, 1.0, 0.0)

            categories.append(category)

            # ── TF transform ─────────────────────────────────────────────────
            try:
                ps                    = PoseStamped()
                ps.header             = msg.header
                ps.pose.position.x    = float(centroid[0])
                ps.pose.position.y    = float(centroid[1])
                ps.pose.position.z    = float(centroid[2])
                ps.pose.orientation.w = 1.0
                transformed = self.tf_buffer.transform(
                    ps, self.target_frame,
                    timeout=rclpy.duration.Duration(seconds=0.5))
                wx = transformed.pose.position.x
                wy = transformed.pose.position.y
                wz = transformed.pose.position.z
            except Exception as e:
                self.get_logger().warn(f'TF failed: {e} -- raw coords used')
                wx = float(centroid[0])
                wy = float(centroid[1])
                wz = float(centroid[2])

            side = 'LEFT ' if wy >= 0 else 'RIGHT'
            self.get_logger().info(
                f'Object {i} [{category}][{side}]: '
                f'pos=({wx:.3f},{wy:.3f},{wz:.3f}) '
                f'size=({size[0]:.3f},{size[1]:.3f},{size[2]:.3f})')

            # ── sphere marker ─────────────────────────────────────────────────
            m                    = Marker()
            m.header.frame_id    = self.target_frame
            m.header.stamp       = msg.header.stamp
            m.ns                 = 'objects'
            m.id                 = i + 1
            m.type               = Marker.SPHERE
            m.action             = Marker.ADD
            m.pose.position.x    = wx
            m.pose.position.y    = wy
            m.pose.position.z    = wz
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.06
            m.color.a            = 1.0
            m.color.r, m.color.g, m.color.b = color
            marker_array.markers.append(m)

            # ── text label ────────────────────────────────────────────────────
            t                    = Marker()
            t.header.frame_id    = self.target_frame
            t.header.stamp       = msg.header.stamp
            t.ns                 = 'labels'
            t.id                 = i + 100
            t.type               = Marker.TEXT_VIEW_FACING
            t.action             = Marker.ADD
            t.pose.position.x    = wx
            t.pose.position.y    = wy
            t.pose.position.z    = wz + 0.12
            t.pose.orientation.w = 1.0
            t.scale.z            = 0.05
            t.color.a            = 1.0
            t.color.r = t.color.g = t.color.b = 1.0
            t.text               = f'{category} [{side.strip()}]\n({wx:.2f},{wy:.2f},{wz:.2f})'
            marker_array.markers.append(t)

            # ── pose ──────────────────────────────────────────────────────────
            p                 = Pose()
            p.position.x      = wx
            p.position.y      = wy
            p.position.z      = wz
            p.orientation.w   = 1.0
            pose_array.poses.append(p)

        left_count  = sum(1 for p in pose_array.poses if p.position.y >= 0)
        right_count = sum(1 for p in pose_array.poses if p.position.y <  0)
        self.get_logger().info(
            f'Publishing {len(valid)} objects -- '
            f'LEFT: {left_count}  RIGHT: {right_count}  '
            f'categories: {categories}')

        # ── publish ───────────────────────────────────────────────────────────
        self.marker_pub.publish(marker_array)
        self.pose_pub.publish(pose_array)

        cat_msg      = String()
        cat_msg.data = ','.join(categories)
        self.category_pub.publish(cat_msg)

    def fast_cluster(self, points, tolerance=0.02, min_size=5, max_size=5000):
        tree    = cKDTree(points)
        visited = np.zeros(len(points), dtype=bool)
        clusters = []

        for i in range(len(points)):
            if visited[i]:
                continue
            indices = tree.query_ball_point(points[i], tolerance)
            if len(indices) < 2:
                continue
            cluster = set(indices)
            queue   = list(indices)
            visited[i] = True

            while queue:
                idx = queue.pop()
                if visited[idx]:
                    continue
                visited[idx] = True
                neighbors = tree.query_ball_point(points[idx], tolerance)
                for n in neighbors:
                    if not visited[n]:
                        cluster.add(n)
                        queue.append(n)

            if min_size <= len(cluster) <= max_size:
                clusters.append(points[list(cluster)])

        return clusters


def main():
    rclpy.init()
    node = ObjectDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()