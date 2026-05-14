import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import message_filters
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseArray, Pose, Point

import numpy as np
import cv2
from cv_bridge import CvBridge

# Import our existing fusion logic — no rewriting needed
from adas_fusion.fusion_core import (
    FusionTracker,
    Detection,
    project_lidar_to_image,
    get_3d_position_from_bbox,
    draw_fused_tracks,
)


# ─────────────────────────────────────────────
# POINTCLOUD2 HELPER
# ─────────────────────────────────────────────

def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """
    Convert a ROS2 PointCloud2 message to a (N, 3) numpy array.

    PointCloud2 stores data as raw bytes with a field layout defined
    in msg.fields. For a standard Velodyne/nuScenes LiDAR the fields
    are [x, y, z, intensity] each as float32 (4 bytes).

    In production you would use the ros2_numpy or sensor_msgs_py
    package for this, but doing it manually here so you understand
    exactly what's happening inside the message.
    """
    # Each point is a row of bytes; point_step is bytes per point
    point_step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8)

    n_points = len(data) // point_step
    # Reshape to (N, point_step) then extract x, y, z as float32
    # Assumes x=offset 0, y=offset 4, z=offset 8 (standard layout)
    points = np.zeros((n_points, 3), dtype=np.float32)
    for i, axis_offset in enumerate([0, 4, 8]):   # x, y, z byte offsets
        col_bytes = data.reshape(n_points, point_step)[:, axis_offset:axis_offset+4]
        points[:, i] = col_bytes.view(np.float32).reshape(-1)

    return points


# ─────────────────────────────────────────────
# FUSION NODE
# ─────────────────────────────────────────────

class FusionNode(Node):

    def __init__(self):
        super().__init__('adas_fusion_node')
        self.get_logger().info('ADAS Fusion Node starting...')

        # ── Parameters (tunable without recompiling) ──────────────────
        # In production you'd set these via a YAML config file
        self.declare_parameter('iou_threshold', 0.25)
        self.declare_parameter('max_misses', 3)
        self.declare_parameter('min_hits', 2)
        self.declare_parameter('show_lidar_points', True)

        iou_thresh  = self.get_parameter('iou_threshold').value
        max_misses  = self.get_parameter('max_misses').value
        min_hits    = self.get_parameter('min_hits').value
        self.show_lidar = self.get_parameter('show_lidar_points').value

        # ── Core components ───────────────────────────────────────────
        self.tracker = FusionTracker(
            max_misses=max_misses,
            min_hits=min_hits,
            iou_threshold=iou_thresh,
        )
        self.bridge = CvBridge()   # converts ROS Image ↔ OpenCV numpy array

        # Lazy-load YOLO to avoid slow import at node startup
        self._yolo = None

        # Camera intrinsic matrix K  Hard-coded
        # here to match nuScenes CAM_FRONT calibration.
        self.K = np.array([
            [1266.417,    0.0,    816.267],
            [   0.0,  1266.417,  491.507],
            [   0.0,     0.0,      1.0  ],
        ])

        # LiDAR → camera extrinsic 
        self.lidar2cam = np.eye(4)

        # LiDAR → ego transform 
        self.lid2ego = np.eye(4)

        # ── QoS profile ───────────────────────────────────────────────
       
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE

        # ── Subscribers (time-synchronized) ──────────────────────────
        
        cam_sub  = message_filters.Subscriber(self, Image,       '/camera/image',   qos_profile=qos)
        lidar_sub = message_filters.Subscriber(self, PointCloud2, '/lidar/points',   qos_profile=qos)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [cam_sub, lidar_sub],
            queue_size=10,
            slop=0.1,          # allow 100ms time difference
        )
        self.sync.registerCallback(self.fusion_callback)

        # ── Publishers ────────────────────────────────────────────────
        self.tracks_pub = self.create_publisher(
            PoseArray, '/adas/tracks', 10
        )
        self.vis_pub = self.create_publisher(
            Image, '/adas/visualization', 10
        )

        # ── Stats timer (logs every 5 seconds) ────────────────────────
        self.frame_count = 0
        self.create_timer(5.0, self.log_stats)

        self.get_logger().info('ADAS Fusion Node ready.')
        self.get_logger().info('Subscribed to: /camera/image, /lidar/points')
        self.get_logger().info('Publishing to: /adas/tracks, /adas/visualization')

    # ── Lazy YOLO loader ──────────────────────────────────────────────
    @property
    def yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO('yolov8n.pt')
            self.get_logger().info('YOLOv8 loaded.')
        return self._yolo

    # ── Main callback — runs on every synced camera+LiDAR pair ────────
    def fusion_callback(self, cam_msg: Image, lidar_msg: PointCloud2):
        """
        This is the heart of the node. Called at ~10-15Hz.

        Steps:
          1. Convert ROS messages to numpy arrays
          2. Run YOLO detection on image
          3. Project LiDAR points onto image
          4. Fuse detections with LiDAR depth
          5. Update Kalman tracker
          6. Publish results
        """
        # ── Step 1: Convert messages ──────────────────────────────────
        image = self.bridge.imgmsg_to_cv2(cam_msg, desired_encoding='bgr8')
        points_lidar = pointcloud2_to_xyz(lidar_msg)  # (N, 3)

        # ── Step 2: YOLO detection ────────────────────────────────────
        yolo_results = self.yolo(image, verbose=False)[0]
        detections = []

        # ── Step 3: LiDAR projection ──────────────────────────────────
        pixels, depths, valid_idx = project_lidar_to_image(
            points_lidar, self.lidar2cam, self.K
        )

        # ── Step 4: Fuse detections with LiDAR ────────────────────────
        for box in yolo_results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = self.yolo.names[cls_id]
            if cls_name not in ('car', 'truck', 'bus',
                                'motorcycle', 'bicycle', 'person'):
                continue

            bbox = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            pos_3d = get_3d_position_from_bbox(
                bbox, pixels, depths, points_lidar, valid_idx, self.lid2ego
            )
            if pos_3d is None:
                pos_3d = np.zeros(3)

            detections.append(Detection(
                bbox_2d=bbox,
                position_3d=pos_3d,
                confidence=conf,
                class_name=cls_name,
            ))

        # ── Step 5: Kalman tracker update ─────────────────────────────
        active_tracks = self.tracker.update(detections)

        # ── Step 6: Publish tracks as PoseArray ───────────────────────
       
        pose_array = PoseArray()
        pose_array.header.stamp    = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'ego_vehicle'   # coordinate frame

        for track in active_tracks:
            pose = Pose()
            # Use 3D position from LiDAR fusion
            pose.position.x = float(track.state[0])  # x in ego frame
            pose.position.y = float(track.state[1])  # y in ego frame
            pose.position.z = 0.0
            pose_array.poses.append(pose)

        self.tracks_pub.publish(pose_array)

        # ── Step 7: Publish visualization image ───────────────────────
        lidar_px = pixels if self.show_lidar else np.array([])
        lidar_dp = depths if self.show_lidar else np.array([])

        vis_image = draw_fused_tracks(
            image, active_tracks, detections, lidar_px, lidar_dp
        )
        vis_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
        vis_msg.header = pose_array.header
        self.vis_pub.publish(vis_msg)

        self.frame_count += 1

    def log_stats(self):
        self.get_logger().info(
            f'Processed {self.frame_count} frames | '
            f'Active tracks: {len(self.tracker._active_tracks())}'
        )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)        # keeps node alive, processing callbacks
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
