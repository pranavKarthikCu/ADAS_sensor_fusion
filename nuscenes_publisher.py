"""
nuScenes Publisher Node — ROS2 Jazzy
=====================================
Replays nuScenes mini data as live ROS2 topics, simulating
real sensor streams. The fusion_node subscribes to these.

Topics published:
  /camera/image        (sensor_msgs/Image)
  /lidar/points        (sensor_msgs/PointCloud2)

Run AFTER fusion_node is running:
  ros2 run adas_fusion nuscenes_publisher
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header

import numpy as np
import cv2
from cv_bridge import CvBridge
import time


class NuScenesPublisher(Node):
    """
    Reads nuScenes samples from disk and publishes them as
    ROS2 Image and PointCloud2 messages at ~10Hz.

    This simulates what actual sensor drivers do in a real vehicle —
    they continuously publish to topics and any node can subscribe.
    """

    def __init__(self, dataroot: str = '/mnt/e/ADAS/v1.0-mini',
                 scene_index: int = 0):
        super().__init__('nuscenes_publisher')
        self.get_logger().info(f'Loading nuScenes from {dataroot}...')

        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.data_classes import LidarPointCloud

        self.nusc    = NuScenes(version='v1.0-mini',
                                dataroot=dataroot, verbose=False)
        self.LPC     = LidarPointCloud
        self.bridge  = CvBridge()

        # Build list of sample tokens for the chosen scene
        scene = self.nusc.scene[scene_index]
        self.get_logger().info(f"Publishing scene: {scene['name']}")

        self.samples = []
        token = scene['first_sample_token']
        while token:
            sample = self.nusc.get('sample', token)
            self.samples.append(sample)
            token = sample['next']

        self.idx = 0

        # Publishers
        self.cam_pub   = self.create_publisher(Image,       '/camera/image',  10)
        self.lidar_pub = self.create_publisher(PointCloud2, '/lidar/points',  10)

        # Publish at 10Hz — matching nuScenes keyframe rate
        self.timer = self.create_timer(0.1, self.publish_next_sample)
        self.get_logger().info(
            f'Publishing {len(self.samples)} samples at 10Hz...'
        )

    def publish_next_sample(self):
        if self.idx >= len(self.samples):
            self.get_logger().info('All samples published. Looping...')
            self.idx = 0

        sample = self.samples[self.idx]
        stamp  = self.get_clock().now().to_msg()

        # ── Camera ──────────────────────────────────────────────────
        cam_token = sample['data']['CAM_FRONT']
        img_path  = self.nusc.get_sample_data_path(cam_token)
        image     = cv2.imread(img_path)

        cam_msg         = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        cam_msg.header.stamp    = stamp
        cam_msg.header.frame_id = 'cam_front'
        self.cam_pub.publish(cam_msg)

        # ── LiDAR ───────────────────────────────────────────────────
        lid_token      = sample['data']['LIDAR_TOP']
        lid_path, _, _ = self.nusc.get_sample_data(lid_token)
        pc             = self.LPC.from_file(lid_path)
        points         = pc.points[:3].T.astype(np.float32)  # (N, 3)

        lidar_msg = self._xyz_to_pointcloud2(points, stamp)
        self.lidar_pub.publish(lidar_msg)

        self.get_logger().info(
            f'Published sample {self.idx:03d} | '
            f'LiDAR pts: {len(points)}'
        )
        self.idx += 1

    def _xyz_to_pointcloud2(self, points: np.ndarray,
                             stamp) -> PointCloud2:
        """
        Convert (N, 3) numpy array to a PointCloud2 message.

        PointCloud2 layout: each point stored as 3× float32
        (12 bytes per point). Fields declare the byte offset
        for x, y, z so any subscriber knows how to parse it.
        """
        msg = PointCloud2()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'lidar_top'

        msg.height    = 1
        msg.width     = len(points)
        msg.is_dense  = True
        msg.is_bigendian = False
        msg.point_step   = 12        # 3 floats × 4 bytes
        msg.row_step     = msg.point_step * msg.width

        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = points.tobytes()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = NuScenesPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
