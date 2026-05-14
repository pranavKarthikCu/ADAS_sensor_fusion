import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────
# 1. DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Detection:
    """One fused detection from camera + LiDAR for a single frame."""
    bbox_2d: np.ndarray          # [x1, y1, x2, y2] in pixels
    position_3d: np.ndarray      # [x, y, z] in ego-vehicle frame (metres)
    confidence: float
    class_name: str


@dataclass
class Track:
    """A tracked object with Kalman state across frames."""
    track_id: int
    class_name: str

    # Kalman state: [x, y, vx, vy]  (2D for simplicity; extend to 3D easily)
    state: np.ndarray = field(default_factory=lambda: np.zeros(4))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(4) * 10.0)

    hits: int = 1
    misses: int = 0
    last_bbox: Optional[np.ndarray] = None


# ─────────────────────────────────────────────
# 2. KALMAN FILTER
# ─────────────────────────────────────────────

class KalmanFilter2D:
    """
    Constant-velocity Kalman filter for 2D position tracking.

    State vector: [x, y, vx, vy]
    Measurement:  [x, y]

    For ADAS interviews you'll also be asked about:
      - EKF (Extended KF)  — for non-linear motion (e.g. turning vehicles)
      - UKF (Unscented KF) — better for highly non-linear systems
      - Particle filter     — when the distribution is non-Gaussian
    """

    def __init__(self, dt: float = 0.1):
        # State transition: x_new = F @ x
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        # Measurement matrix: z = H @ x  (we observe [x, y] only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise covariance (tune these for your scenario)
        self.Q = np.diag([0.5, 0.5, 1.0, 1.0])

        # Measurement noise covariance (tune to sensor accuracy)
        self.R = np.diag([2.0, 2.0])

    def predict(self, state: np.ndarray, P: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step — move forward in time."""
        x_pred = self.F @ state
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update step — correct prediction with measurement."""
        z = measurement                             # observed [x, y]
        y = z - self.H @ x_pred                    # innovation
        S = self.H @ P_pred @ self.H.T + self.R    # innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        x_new = x_pred + K @ y
        P_new = (np.eye(4) - K @ self.H) @ P_pred
        return x_new, P_new


# ─────────────────────────────────────────────
# 3. LIDAR → IMAGE PROJECTION
# ─────────────────────────────────────────────

def project_lidar_to_image(
    points_3d: np.ndarray,       # (N, 3) in LiDAR frame
    lidar_to_cam: np.ndarray,    # (4, 4) extrinsic transform
    camera_intrinsic: np.ndarray # (3, 3) K matrix
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D LiDAR points into the camera image plane.

    Returns:
        pixels    — (M, 2) [u, v] for points in front of camera
        depths    — (M,)   depth in metres
        valid_idx — (M,)   original indices of valid points
    """
    N = points_3d.shape[0]

    # Homogeneous coordinates in LiDAR frame
    pts_hom = np.hstack([points_3d, np.ones((N, 1))])  # (N, 4)

    # Transform to camera frame
    pts_cam = (lidar_to_cam @ pts_hom.T).T              # (N, 4)

    # Keep only points in front of camera (positive Z)
    valid = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid]
    depths = pts_cam[:, 2]
    valid_idx = np.where(valid)[0]

    # Project with intrinsic matrix K
    # u = (fx * X + cx * Z) / Z  ->  K @ [X, Y, Z]
    uvw = (camera_intrinsic @ pts_cam[:, :3].T).T       # (M, 3)
    pixels = uvw[:, :2] / uvw[:, 2:3]                   # (M, 2)

    return pixels.astype(int), depths, valid_idx


# ─────────────────────────────────────────────
# 4. LIDAR–BBOX ASSOCIATION
# ─────────────────────────────────────────────

def get_3d_position_from_bbox(
    bbox: np.ndarray,
    pixels: np.ndarray,
    depths: np.ndarray,
    points_3d: np.ndarray,
    valid_idx: np.ndarray,
    lidar_to_ego: np.ndarray
) -> Optional[np.ndarray]:
    """
    For a 2D detection bbox, find all LiDAR points that fall inside it
    and return the median 3D position (in ego-vehicle frame).

    Median is used over mean because it's robust to noise and ground points.
    """
    x1, y1, x2, y2 = bbox.astype(int)

    # Find projected LiDAR pixels that lie inside this bounding box
    inside = (
        (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) &
        (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2)
    )

    if inside.sum() < 3:   # too few points — skip
        return None

    # Corresponding 3D points in LiDAR frame
    pts_inside = points_3d[valid_idx[inside]]

    # Convert to ego-vehicle frame
    pts_hom = np.hstack([pts_inside, np.ones((pts_inside.shape[0], 1))])
    pts_ego = (lidar_to_ego @ pts_hom.T).T

    # Median position (robust to outliers / ground plane leakage)
    return np.median(pts_ego[:, :3], axis=0)


# ─────────────────────────────────────────────
# 5. TRACKER (Hungarian matching)
# ─────────────────────────────────────────────

class FusionTracker:
    """
    Multi-object tracker using:
      - Kalman filter for state prediction
      - Hungarian algorithm for detection-to-track assignment
      - IoU distance as cost metric

    This is conceptually similar to SORT (Simple Online and Realtime Tracking)
    which is a popular baseline in autonomous driving.
    """

    def __init__(self, max_misses: int = 3, min_hits: int = 2,
                 iou_threshold: float = 0.3, dt: float = 0.1):
        self.kf = KalmanFilter2D(dt=dt)
        self.tracks: List[Track] = []
        self.next_id = 0
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Run one tracking cycle:
          1. Predict all existing tracks forward
          2. Match detections to tracks via Hungarian algorithm on IoU cost
          3. Update matched tracks, create new ones, delete stale ones
        """
        # ── Predict ──
        for t in self.tracks:
            t.state, t.covariance = self.kf.predict(t.state, t.covariance)

        if not detections:
            for t in self.tracks:
                t.misses += 1
            self._prune()
            return self._active_tracks()

        det_boxes = np.array([d.bbox_2d for d in detections])

        if not self.tracks:
            # No existing tracks — create one per detection
            for d in detections:
                self._init_track(d)
            return self._active_tracks()

        # ── Cost matrix (1 - IoU) ──
        track_boxes = np.array([t.last_bbox for t in self.tracks
                                 if t.last_bbox is not None])
        if len(track_boxes) == 0:
            for d in detections:
                self._init_track(d)
            return self._active_tracks()

        cost = 1.0 - iou_matrix(track_boxes, det_boxes)

        # ── Hungarian assignment ──
        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > (1.0 - self.iou_threshold):
                continue  # low IoU — not a real match
            t = self.tracks[r]
            d = detections[c]

            # Update Kalman state with centroid measurement
            cx = (d.bbox_2d[0] + d.bbox_2d[2]) / 2
            cy = (d.bbox_2d[1] + d.bbox_2d[3]) / 2
            t.state, t.covariance = self.kf.update(
                t.state, t.covariance, np.array([cx, cy])
            )
            t.last_bbox = d.bbox_2d
            t.hits += 1
            t.misses = 0
            matched_tracks.add(r)
            matched_dets.add(c)

        # ── Unmatched tracks → miss ──
        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.misses += 1

        # ── Unmatched detections → new tracks ──
        for j, d in enumerate(detections):
            if j not in matched_dets:
                self._init_track(d)

        self._prune()
        return self._active_tracks()

    def _init_track(self, d: Detection):
        cx = (d.bbox_2d[0] + d.bbox_2d[2]) / 2
        cy = (d.bbox_2d[1] + d.bbox_2d[3]) / 2
        state = np.array([cx, cy, 0.0, 0.0])
        t = Track(
            track_id=self.next_id,
            class_name=d.class_name,
            state=state,
            last_bbox=d.bbox_2d,
        )
        self.tracks.append(t)
        self.next_id += 1

    def _prune(self):
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

    def _active_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.hits >= self.min_hits]


# ─────────────────────────────────────────────
# 6. IoU UTILITY
# ─────────────────────────────────────────────

def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of [x1,y1,x2,y2] boxes."""
    iou = np.zeros((len(boxes_a), len(boxes_b)))
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
            xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = area_a + area_b - inter
            iou[i, j] = inter / union if union > 0 else 0
    return iou


# ─────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────

TRACK_COLORS = [
    (220,  80,  80), (80, 200, 120), ( 80, 130, 220),
    (220, 160,  50), (160,  80, 220), (50, 200, 200),
]

def draw_fused_tracks(image: np.ndarray, tracks: List[Track],
                      detections: List[Detection],
                      lidar_pixels: np.ndarray,
                      lidar_depths: np.ndarray) -> np.ndarray:
    vis = image.copy()

    # Draw projected LiDAR points (depth-coloured)
    if len(lidar_pixels) > 0:
        max_d = lidar_depths.max() if lidar_depths.max() > 0 else 1
        for (u, v), d in zip(lidar_pixels, lidar_depths):
            h, w = vis.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                t = d / max_d                    # 0 = close (warm), 1 = far (cool)
                r = int(255 * (1 - t))
                b = int(255 * t)
                cv2.circle(vis, (u, v), 2, (b, 100, r), -1)

    # Draw detection boxes (thin, white)
    for d in detections:
        x1, y1, x2, y2 = d.bbox_2d.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # Draw Kalman-tracked boxes (bold, coloured by ID)
    for t in tracks:
        if t.last_bbox is None:
            continue
        color = TRACK_COLORS[t.track_id % len(TRACK_COLORS)]
        x1, y1, x2, y2 = t.last_bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Velocity arrow from centroid
        cx, cy = int(t.state[0]), int(t.state[1])
        vx, vy = int(t.state[2] * 5), int(t.state[3] * 5)  # scale for visibility
        cv2.arrowedLine(vis, (cx, cy), (cx + vx, cy + vy), color, 2, tipLength=0.4)

        label = f"ID:{t.track_id} {t.class_name}"
        cv2.putText(vis, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return vis


# ─────────────────────────────────────────────
# 8. NUSCENES RUNNER
# ─────────────────────────────────────────────

def run_on_nuscenes(dataroot: str = "E:\\ADAS\v1.0-mini",
                    version: str = "v1.0-mini",
                    scene_index: int = 0,
                    output_dir: str = "./fusion_output"):
    """
    Full pipeline on a nuScenes scene.

    Args:
        dataroot:    path to extracted nuScenes mini dataset
        version:     "v1.0-mini" for the free mini split
        scene_index: which scene to process (0–9 for mini)
        output_dir:  where to save per-frame visualizations
    """
    import os
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    from ultralytics import YOLO

    os.makedirs(output_dir, exist_ok=True)

    print("Loading nuScenes...")
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    yolo = YOLO("yolov8n.pt")   # nano model — fast; swap for yolov8m for accuracy

    tracker = FusionTracker(max_misses=3, min_hits=2, iou_threshold=0.25)

    scene = nusc.scene[scene_index]
    print(f"Processing scene: {scene['name']}  "
          f"({scene['nbr_samples']} samples)")

    sample_token = scene["first_sample_token"]
    frame_idx = 0

    while sample_token:
        sample = nusc.get("sample", sample_token)

        # ── Camera ──
        cam_token = sample["data"]["CAM_FRONT"]
        cam_data  = nusc.get("sample_data", cam_token)
        img_path  = nusc.get_sample_data_path(cam_token)
        image     = cv2.imread(img_path)

        cam_cs    = nusc.get("calibrated_sensor",
                              cam_data["calibrated_sensor_token"])
        K         = np.array(cam_cs["camera_intrinsic"])   # 3×3
        cam_ego   = nusc.get("ego_pose", cam_data["ego_pose_token"])

        # ── LiDAR ──
        lid_token = sample["data"]["LIDAR_TOP"]
        lid_data  = nusc.get("sample_data", lid_token)
        lid_path, _, _ = nusc.get_sample_data(lid_token)

        lid_cs    = nusc.get("calibrated_sensor",
                              lid_data["calibrated_sensor_token"])
        lid_ego   = nusc.get("ego_pose", lid_data["ego_pose_token"])

        pc = LidarPointCloud.from_file(lid_path)
        points_lidar = pc.points[:3].T          # (N, 3)

        # ── Build transforms ──
        # LiDAR sensor → ego
        lid2ego = transform_matrix(
            lid_cs["translation"],
            Quaternion(lid_cs["rotation"])
        )
        # ego → camera sensor (inverse path: ego → cam_ego → cam_sensor)
        ego2cam = transform_matrix(
            cam_cs["translation"],
            Quaternion(cam_cs["rotation"]),
            inverse=True
        )
        lidar2cam = ego2cam @ lid2ego            # 4×4\

        # ── Height filter ──────────────────────────────────────────
        # In ego frame, Z points up. Keep only points between
        # -2.0m (below ground tolerance) and +2.5m (roof level).
        # This removes sky reflections, tree canopy, and building hits.
        pts_hom = np.hstack([points_lidar, np.ones((points_lidar.shape[0], 1))])
        pts_ego_all = (lid2ego @ pts_hom.T).T
        height_mask = (pts_ego_all[:, 2] > -2.0) & (pts_ego_all[:, 2] < 2.5)
        points_lidar = points_lidar[height_mask]

        # ── Project LiDAR → image ──
        pixels, depths, valid_idx = project_lidar_to_image(
            points_lidar, lidar2cam, K
        )

        # ── YOLOv8 detection ──
        yolo_results = yolo(image, verbose=False)[0]
        detections: List[Detection] = []

        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            cls_name = yolo.names[cls_id]
            if cls_name not in ("car", "truck", "bus", "motorcycle",
                                "bicycle", "person"):
                continue

            bbox = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            pos_3d = get_3d_position_from_bbox(
                bbox, pixels, depths, points_lidar, valid_idx, lid2ego
            )
            if pos_3d is None:
                pos_3d = np.zeros(3)     # fallback: camera-only detection

            detections.append(Detection(
                bbox_2d=bbox,
                position_3d=pos_3d,
                confidence=conf,
                class_name=cls_name,
            ))

        # ── Track ──
        active_tracks = tracker.update(detections)

        # ── Visualize ──
        vis = draw_fused_tracks(image, active_tracks, detections,
                                pixels, depths)

        out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path, vis)
        print(f"  Frame {frame_idx:03d} | "
              f"detections: {len(detections):2d} | "
              f"active tracks: {len(active_tracks):2d}")

        sample_token = sample["next"]
        frame_idx += 1

    print(f"\nDone. {frame_idx} frames saved to {output_dir}/")


# ─────────────────────────────────────────────
# 9. QUICK SMOKE TEST (no nuScenes needed)
# ─────────────────────────────────────────────

def smoke_test():
    """
    Runs the Kalman tracker on synthetic detections.
    Use this to verify the pipeline works before downloading nuScenes.
    """
    print("=== Smoke test: Kalman tracker on synthetic data ===\n")
    tracker = FusionTracker(max_misses=2, min_hits=1)

    # Simulate a car moving left-to-right across 10 frames
    for frame in range(10):
        x = 100 + frame * 30   # true x position
        noise = np.random.randn() * 5
        bbox = np.array([x + noise, 200, x + noise + 80, 280])
        det = Detection(
            bbox_2d=bbox,
            position_3d=np.array([x / 10.0, 0.0, 20.0]),
            confidence=0.9,
            class_name="car",
        )
        tracks = tracker.update([det])
        for t in tracks:
            print(f"  Frame {frame:02d} | Track {t.track_id} "
                  f"| x={t.state[0]:.1f}  vx={t.state[2]:.2f} px/frame"
                  f"| hits={t.hits}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "nuscenes":
        dataroot = sys.argv[2] if len(sys.argv) > 2 else "E:\\ADAS\v1.0-mini"
        run_on_nuscenes(dataroot=dataroot)
    else:
        smoke_test()
        print("\nTo run on nuScenes:")
        print("  python sensor_fusion.py nuscenes E:\\ADAS\v1.0-mini")
