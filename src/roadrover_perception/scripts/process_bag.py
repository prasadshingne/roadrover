#!/usr/bin/env python3
"""
Offline perception pipeline for roadrover.
Reads the ORIGINAL bag, applies rotation + lane detection + object detection
in a single pass and saves a new bag containing:
  - all original topics with image topics rotated 180°
  - /perception/image_annotated  (CompressedImage: YOLO boxes + lane overlay + speed)

Lane detection and object detection both run on the clean rotated frame
so neither interferes with the other.

Usage:
  python3 process_bag.py <original_bag_path>
  python3 process_bag.py <original_bag_path> --output <output_bag_path>
"""

import argparse
import math
import pickle
import warnings
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from foxglove_msgs.msg import GeoJSON
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image, Imu, NavSatFix
from std_msgs.msg import Header, String

COMPRESSED_TOPIC = '/usb_cam/image_raw/compressed'
RAW_TOPIC        = '/usb_cam/image_raw'
ANNOTATED_TOPIC  = '/perception/image_annotated'
VEL_TOPIC        = '/vel'
GPS_TOPIC             = '/fix'
EGO_ODOM_TOPIC        = '/ego/odometry'
EGO_IMU_TOPIC         = '/ego/imu'
EGO_MATCHED_FIX_TOPIC = '/ego/matched_fix'
EGO_LANE_INFO_TOPIC   = '/ego/lane_info'
MAP_ROADS_TOPIC       = '/map/roads'


# ── Image rotation helpers ────────────────────────────────────────────────────

def rotate_raw_msg(data: bytes, msg_type) -> bytes:
    msg = deserialize_message(data, msg_type)
    channels = msg.step // msg.width
    img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
        msg.height, msg.width, channels
    )
    img = cv2.rotate(img, cv2.ROTATE_180)
    msg.data = img.tobytes()
    return serialize_message(msg)


def rotate_compressed_msg(data: bytes, msg_type):
    """Deserialize, rotate, re-encode. Returns (rotated_bytes, bgr_img)."""
    msg = deserialize_message(data, msg_type)
    np_arr = np.frombuffer(bytes(msg.data), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return data, None
    img = cv2.rotate(img, cv2.ROTATE_180)
    ok, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ok:
        msg.data = encoded.tobytes()
    return serialize_message(msg), img


# ── Map matching ──────────────────────────────────────────────────────────────

class MapMatcher:
    """
    Matches GPS fixes to the nearest OSM road edge and estimates lane number.
    Uses an osmnx graph pickled by make_map.py.
    Lane estimation assumes right-hand traffic, ~3.5 m lane width.
    """
    LANE_WIDTH = 3.5   # metres

    def __init__(self, graph_path: str):
        import osmnx as ox
        with open(graph_path, 'rb') as f:
            self._G = pickle.load(f)
        self._ox = ox

    def match(self, lat: float, lon: float, timestamp_ns: int):
        """
        Returns (matched_NavSatFix, lane_info_String) for the given GPS position.
        """
        ox = self._ox
        G  = self._G

        # ── Nearest edge ────────────────────────────────────────────────────
        (u, v, key), dist_deg = ox.nearest_edges(
            G, X=lon, Y=lat, return_dist=True)
        edge = G[u][v][key]

        # ── Road name ───────────────────────────────────────────────────────
        name = edge.get('name', '')
        if isinstance(name, list):
            name = name[0]
        name = name or edge.get('highway', 'unknown road')
        if isinstance(name, list):
            name = name[0]

        # ── Snap GPS point onto edge geometry ───────────────────────────────
        from shapely.geometry import Point
        geom  = edge.get('geometry')
        point = Point(lon, lat)

        if geom is not None:
            proj_dist  = geom.project(point)
            nearest_pt = geom.interpolate(proj_dist)

            # Signed lateral offset: positive = left of road direction
            delta     = 1e-5
            p_before  = geom.interpolate(max(0.0, proj_dist - delta))
            p_after   = geom.interpolate(min(geom.length, proj_dist + delta))
            tx, ty    = p_after.x - p_before.x, p_after.y - p_before.y  # tangent
            ex, ey    = lon - nearest_pt.x,      lat - nearest_pt.y      # ego vec
            cross     = tx * ey - ty * ex   # +ve = left of travel direction

            # Convert degrees → metres (equirectangular approx)
            m_per_deg_lat = 111_320.0
            m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
            lateral_m     = math.sqrt((ex * m_per_deg_lon) ** 2 +
                                      (ey * m_per_deg_lat) ** 2)
            lateral_m    *= 1.0 if cross >= 0 else -1.0

            snap_lat, snap_lon = nearest_pt.y, nearest_pt.x
        else:
            # Edge has no geometry: use node positions as snap point
            node_data  = G.nodes[u]
            snap_lat   = node_data['y']
            snap_lon   = node_data['x']
            lateral_m  = 0.0

        # ── Lane estimate ───────────────────────────────────────────────────
        lanes_raw = edge.get('lanes', 2)
        try:
            total_lanes = int(lanes_raw[0] if isinstance(lanes_raw, list)
                              else lanes_raw)
        except (ValueError, TypeError):
            total_lanes = 2
        # Right-hand traffic: lane 1 is rightmost (most negative lateral offset)
        # Road centerline sits at total_lanes/2 * LANE_WIDTH to the left of lane 1
        lane_offset = total_lanes / 2.0 * self.LANE_WIDTH + lateral_m
        lane_num    = max(1, min(total_lanes, int(lane_offset / self.LANE_WIDTH) + 1))

        # ── Build ROS messages ──────────────────────────────────────────────
        h = Header()
        h.frame_id      = 'map'
        h.stamp.sec     = int(timestamp_ns // 1_000_000_000)
        h.stamp.nanosec = int(timestamp_ns %  1_000_000_000)

        fix               = NavSatFix()
        fix.header        = h
        fix.latitude      = snap_lat
        fix.longitude     = snap_lon
        fix.altitude      = 0.0
        fix.status.status = 0   # STATUS_FIX

        info      = String()
        info.data = (f'{name} | lane {lane_num}/{total_lanes} | '
                     f'offset {lateral_m:+.1f} m')

        return fix, info


# ── Ego state estimation ──────────────────────────────────────────────────────

class EgoStateEstimator:
    """
    Derives heading, yaw rate, and longitudinal/lateral acceleration from /vel.
    /vel is geometry_msgs/TwistWithCovarianceStamped published by nmea_navsat_driver:
      linear.x = east velocity (m/s), linear.y = north velocity (m/s).
    No IMU on this rover — everything is computed from GPS velocity.
    """
    _ALPHA     = 0.15   # EMA smoothing for derivatives (lower = smoother)
    _MIN_SPEED = 0.5    # m/s — below this heading is unreliable; hold last value

    def __init__(self):
        self._t_prev       = None
        self._speed_prev   = 0.0
        self._heading_prev = None   # ENU yaw: atan2(vy_north, vx_east)
        self.yaw_rate      = 0.0    # rad/s, EMA-smoothed
        self.lon_accel     = 0.0    # m/s², along direction of travel
        self.lat_accel     = 0.0    # m/s², centripetal (speed × yaw_rate)

    def _make_header(self, timestamp_ns: int, frame_id: str) -> Header:
        h = Header()
        h.frame_id     = frame_id
        h.stamp.sec    = int(timestamp_ns // 1_000_000_000)
        h.stamp.nanosec = int(timestamp_ns % 1_000_000_000)
        return h

    def update(self, vel_msg, timestamp_ns: int):
        """
        Process one /vel message. Returns (Odometry, Imu) ready to serialize.
        """
        vx_e  = vel_msg.twist.linear.x   # east  (m/s)
        vy_n  = vel_msg.twist.linear.y   # north (m/s)
        speed = math.sqrt(vx_e ** 2 + vy_n ** 2)
        t_s   = timestamp_ns * 1e-9

        heading = None
        if speed >= self._MIN_SPEED:
            heading = math.atan2(vy_n, vx_e)   # ENU yaw (rad)

        if self._t_prev is not None:
            dt = t_s - self._t_prev
            if dt > 0.0:
                # Longitudinal acceleration: d(speed)/dt
                a_lon_raw  = (speed - self._speed_prev) / dt
                self.lon_accel = (self._ALPHA * a_lon_raw +
                                  (1 - self._ALPHA) * self.lon_accel)

                # Yaw rate: d(heading)/dt with angle unwrapping
                if heading is not None and self._heading_prev is not None:
                    d_yaw = heading - self._heading_prev
                    if d_yaw >  math.pi: d_yaw -= 2 * math.pi
                    if d_yaw < -math.pi: d_yaw += 2 * math.pi
                    yaw_rate_raw = d_yaw / dt
                    self.yaw_rate = (self._ALPHA * yaw_rate_raw +
                                     (1 - self._ALPHA) * self.yaw_rate)

                # Lateral acceleration (centripetal): v × ω
                self.lat_accel = speed * self.yaw_rate

        self._t_prev     = t_s
        self._speed_prev = speed
        if heading is not None:
            self._heading_prev = heading

        yaw = self._heading_prev if self._heading_prev is not None else 0.0

        # ── nav_msgs/Odometry ────────────────────────────────────────────
        odom = Odometry()
        odom.header          = self._make_header(timestamp_ns, 'odom')
        odom.child_frame_id  = 'base_link'
        half_yaw = yaw / 2.0
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = math.sin(half_yaw)
        odom.pose.pose.orientation.w = math.cos(half_yaw)
        odom.twist.twist.linear.x   = speed
        odom.twist.twist.angular.z  = self.yaw_rate

        # ── sensor_msgs/Imu ──────────────────────────────────────────────
        imu = Imu()
        imu.header = self._make_header(timestamp_ns, 'base_link')
        imu.orientation_covariance[0] = -1.0   # orientation not provided
        imu.angular_velocity.z        = self.yaw_rate
        imu.linear_acceleration.x     = self.lon_accel
        imu.linear_acceleration.y     = self.lat_accel

        return odom, imu


# ── Lane detection ────────────────────────────────────────────────────────────

HOOD_CUTOFF = 0.63   # below this fraction of height = car hood

# Bird's-eye view (BEV / IPM) constants
BEV_W          = 400
BEV_H          = 300
BEV_MARGIN     = 60   # left/right padding inside BEV canvas
BEV_DRAW_EXTRA = 15   # extra BEV rows to draw below fit region (pulls line toward hood)


def _bev_transforms(h: int, w: int):
    """Return (M_bev, M_inv) perspective transforms for a frame of size h×w."""
    y_top    = int(h * 0.45)
    y_bottom = int(h * HOOD_CUTOFF)
    # Source: same trapezoid as the ROI mask
    src = np.float32([
        [w * 0.05, y_bottom],
        [w * 0.28, y_top],
        [w * 0.65, y_top],
        [w * 0.95, y_bottom],
    ])
    # Destination: rectangle in BEV space (bottom of image = near, top = far)
    dst = np.float32([
        [BEV_MARGIN,         BEV_H],
        [BEV_MARGIN,         0],
        [BEV_W - BEV_MARGIN, 0],
        [BEV_W - BEV_MARGIN, BEV_H],
    ])
    M_bev = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M_bev, M_inv


class LaneTracker:
    """EMA-smoothed degree-2 polynomial lane tracker (operates in BEV space)."""

    STALE_LIMIT = 15

    def __init__(self, alpha: float = 0.20):
        self.alpha = alpha
        self.left_poly    = None
        self.right_poly   = None
        self._left_stale  = 0
        self._right_stale = 0

    def update(self, left_pts, right_pts):
        lp = self._fit(left_pts,  side='left')
        rp = self._fit(right_pts, side='right')

        if lp is not None:
            self.left_poly   = lp if self.left_poly  is None else \
                               self.alpha * lp + (1 - self.alpha) * self.left_poly
            self._left_stale = 0
        else:
            self._left_stale += 1
            if self._left_stale > self.STALE_LIMIT:
                self.left_poly = None

        if rp is not None:
            self.right_poly   = rp if self.right_poly is None else \
                                self.alpha * rp + (1 - self.alpha) * self.right_poly
            self._right_stale = 0
        else:
            self._right_stale += 1
            if self._right_stale > self.STALE_LIMIT:
                self.right_poly = None

    def _fit(self, pts, side: str):
        if len(pts) < 6:
            return None
        xs, ys = zip(*pts)
        xs, ys = np.array(xs, dtype=float), np.array(ys, dtype=float)
        # Require vertical spread ≥ 20% of BEV height
        if (ys.max() - ys.min()) < BEV_H * 0.20:
            return None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=np.RankWarning)
                poly = np.polyfit(ys, xs, 2)
        except (np.linalg.LinAlgError, np.RankWarning):
            return None
        # Curvature sanity in BEV space (pixels roughly uniform → higher limit)
        if abs(poly[0]) > 0.01:
            return None
        # Base-of-frame x must land inside expected half of the image
        x_base = np.polyval(poly, BEV_H)
        if side == 'left'  and not (BEV_MARGIN * 0.5 <= x_base <= BEV_W * 0.55):
            return None
        if side == 'right' and not (BEV_W * 0.45 <= x_base <= BEV_W - BEV_MARGIN * 0.5):
            return None
        return poly


def _bev_sliding_window(bev_binary: np.ndarray, tracker: LaneTracker,
                         n_windows: int = 9, margin: int = 30, minpix: int = 10):
    """
    Sliding window lane-pixel search in BEV binary edge image.
    Seed from EMA polynomial when available; cold-start via histogram.
    Returns (left_pts, right_pts) as lists of (x, y) in BEV space.
    """
    mid = BEV_W // 2

    # ── Seed x positions ────────────────────────────────────────────────────
    if tracker.left_poly is not None:
        leftx_cur = int(np.clip(np.polyval(tracker.left_poly, BEV_H),
                                BEV_MARGIN, mid))
    else:
        hist = np.sum(bev_binary[BEV_H // 2:, :mid], axis=0).astype(np.float32)
        leftx_cur = int(np.argmax(hist)) if hist.max() >= minpix else None

    if tracker.right_poly is not None:
        rightx_cur = int(np.clip(np.polyval(tracker.right_poly, BEV_H),
                                 mid, BEV_W - BEV_MARGIN))
    else:
        hist = np.sum(bev_binary[BEV_H // 2:, mid:], axis=0).astype(np.float32)
        peak = int(np.argmax(hist))
        rightx_cur = peak + mid if hist.max() >= minpix else None

    nz  = bev_binary.nonzero()
    nzy = np.array(nz[0])
    nzx = np.array(nz[1])

    left_pts, right_pts = [], []
    win_h = BEV_H // n_windows

    for win in range(n_windows):
        y_lo = BEV_H - (win + 1) * win_h
        y_hi = BEV_H - win * win_h

        if leftx_cur is not None:
            inds = ((nzy >= y_lo) & (nzy < y_hi) &
                    (nzx >= leftx_cur - margin) & (nzx < leftx_cur + margin)).nonzero()[0]
            if len(inds) >= minpix:
                xs = nzx[inds]
                left_pts.extend(zip(xs.tolist(), nzy[inds].tolist()))
                leftx_cur = int(np.clip(int(np.mean(xs)), BEV_MARGIN, mid))

        if rightx_cur is not None:
            inds = ((nzy >= y_lo) & (nzy < y_hi) &
                    (nzx >= rightx_cur - margin) & (nzx < rightx_cur + margin)).nonzero()[0]
            if len(inds) >= minpix:
                xs = nzx[inds]
                right_pts.extend(zip(xs.tolist(), nzy[inds].tolist()))
                rightx_cur = int(np.clip(int(np.mean(xs)), mid, BEV_W - BEV_MARGIN))

    return left_pts, right_pts


def _bev_poly_to_image(poly: np.ndarray, M_inv: np.ndarray, n: int = 50):
    """Warp a BEV-space polynomial back to image-space point array for drawing."""
    ys = np.linspace(0, BEV_H - 1 + BEV_DRAW_EXTRA, n)
    xs = np.clip(np.polyval(poly, ys), 0, BEV_W - 1)
    bev_pts = np.column_stack([xs, ys]).astype(np.float32).reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(bev_pts, M_inv)
    return img_pts.astype(np.int32)


def detect_lanes(img: np.ndarray, tracker: LaneTracker,
                  det_boxes=None) -> np.ndarray:
    """
    Detect lane lines via BEV perspective warp + sliding window search.
    det_boxes: YOLO Boxes object (optional) — detected vehicle regions are
               zeroed out of the edge image so they don't pollute lane pixels.
    Returns a black overlay with smooth curved lane lines and filled lane area.
    """
    h, w = img.shape[:2]
    overlay = np.zeros_like(img)

    # ── Edge detection ───────────────────────────────────────────────────────
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur     = cv2.GaussianBlur(enhanced, (7, 7), 0)
    edges    = cv2.Canny(blur, 40, 120)

    # ── Trapezoid ROI mask ───────────────────────────────────────────────────
    # y_top=45%: below horizon (noisy far-field); y_bottom=63%: above hood.
    # Top corners shifted right to account for camera ~5–6" left of centreline.
    y_top    = int(h * 0.45)
    y_bottom = int(h * HOOD_CUTOFF)
    roi_pts = np.array([
        [int(w * 0.05), y_bottom],
        [int(w * 0.28), y_top],
        [int(w * 0.65), y_top],
        [int(w * 0.95), y_bottom],
    ], np.int32)
    roi_mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(roi_mask, [roi_pts], 255)
    masked = cv2.bitwise_and(edges, roi_mask)

    # ── Blank out detected vehicles so their edges don't fool the lane fit ───
    if det_boxes is not None and len(det_boxes):
        pad = 10
        for x1, y1, x2, y2 in det_boxes.xyxy.cpu().numpy().astype(int)[:, :4]:
            masked[max(0, y1 - pad):min(h, y2 + pad),
                   max(0, x1 - pad):min(w, x2 + pad)] = 0

    # ── Perspective warp → BEV ───────────────────────────────────────────────
    M_bev, M_inv = _bev_transforms(h, w)
    bev = cv2.warpPerspective(masked, M_bev, (BEV_W, BEV_H))
    # Thicken 1-px Canny edges so windows catch them reliably
    bev = cv2.dilate(bev, np.ones((3, 3), np.uint8), iterations=1)

    # ── Sliding window in BEV space ──────────────────────────────────────────
    left_pts, right_pts = _bev_sliding_window(bev, tracker)
    tracker.update(left_pts, right_pts)

    lp = tracker.left_poly
    rp = tracker.right_poly

    # ── Draw back in image space via inverse warp ────────────────────────────
    if lp is not None and rp is not None:
        left_img  = _bev_poly_to_image(lp, M_inv)
        right_img = _bev_poly_to_image(rp, M_inv)
        fill_pts  = np.vstack([left_img, right_img[::-1]])
        cv2.fillPoly(overlay, [fill_pts], (0, 180, 0))
        cv2.polylines(overlay, [left_img],  False, (255, 80,   0), 3)
        cv2.polylines(overlay, [right_img], False, (0,   80, 255), 3)
    elif lp is not None:
        cv2.polylines(overlay, [_bev_poly_to_image(lp, M_inv)], False, (255, 80, 0), 3)
    elif rp is not None:
        cv2.polylines(overlay, [_bev_poly_to_image(rp, M_inv)], False, (0, 80, 255), 3)

    return overlay


# ── Speed overlay ─────────────────────────────────────────────────────────────

def draw_speed(img: np.ndarray, speed_ms: float) -> None:
    speed_mph = speed_ms * 2.237
    h = img.shape[0]
    x, y = 15, h - 40
    cv2.rectangle(img, (x - 5, y - 28), (x + 165, y + 12), (0, 0, 0), -1)
    cv2.putText(img, f'{speed_mph:.1f} mph', (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f'{speed_ms:.1f} m/s', (x, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def make_compressed_msg(img: np.ndarray, header) -> CompressedImage:
    ok, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    msg = CompressedImage()
    msg.header = header
    msg.format = 'jpeg'
    msg.data = encoded.tobytes()
    return msg


def main():
    ap = argparse.ArgumentParser(
        description='Rotate + lane detection + object detection on a rosbag2')
    ap.add_argument('bag_path', help='Path to original rosbag2 directory')
    ap.add_argument('--output', default=None,
                    help='Output bag path (default: <input>_processed)')
    ap.add_argument('--map-graph', default=None,
                    help='Path to map_graph.pkl from make_map.py (enables map matching)')
    args = ap.parse_args()

    bag_path    = str(Path(args.bag_path).resolve())
    output_path = args.output or bag_path.rstrip('/') + '_processed'

    print(f'Input:  {bag_path}')
    print(f'Output: {output_path}')

    out = Path(output_path)
    if out.exists():
        import shutil
        shutil.rmtree(out)
        print(f'Removed existing output bag.')

    print('Loading YOLOv8s on GPU...')
    from ultralytics import YOLO
    model = YOLO('yolov8s.pt')
    model.to('cuda')
    print('Model ready.')

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''),
    )
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=output_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''),
    )
    for t in topic_types:
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=t.name, type=t.type, serialization_format='cdr',
        ))
    writer.create_topic(rosbag2_py.TopicMetadata(
        name=ANNOTATED_TOPIC,
        type='sensor_msgs/msg/CompressedImage',
        serialization_format='cdr',
    ))
    writer.create_topic(rosbag2_py.TopicMetadata(
        name=EGO_ODOM_TOPIC,
        type='nav_msgs/msg/Odometry',
        serialization_format='cdr',
    ))
    writer.create_topic(rosbag2_py.TopicMetadata(
        name=EGO_IMU_TOPIC,
        type='sensor_msgs/msg/Imu',
        serialization_format='cdr',
    ))

    map_matcher  = None
    map_geojson  = None
    if args.map_graph:
        print(f'Loading map graph: {args.map_graph}')
        map_matcher = MapMatcher(args.map_graph)
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=EGO_MATCHED_FIX_TOPIC,
            type='sensor_msgs/msg/NavSatFix',
            serialization_format='cdr',
        ))
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=EGO_LANE_INFO_TOPIC,
            type='std_msgs/msg/String',
            serialization_format='cdr',
        ))
        # GeoJSON road overlay — same directory as the graph pickle
        geojson_path = Path(args.map_graph).parent / 'map.geojson'
        if geojson_path.exists():
            map_geojson = GeoJSON()
            map_geojson.geojson = geojson_path.read_text()
            writer.create_topic(rosbag2_py.TopicMetadata(
                name=MAP_ROADS_TOPIC,
                type='foxglove_msgs/msg/GeoJSON',
                serialization_format='cdr',
            ))
            print(f'Road GeoJSON loaded: {geojson_path}')
        else:
            print(f'Warning: {geojson_path} not found — road overlay disabled.')
        print('Map matching enabled.')

    tracker     = LaneTracker(alpha=0.25)
    ego_est     = EgoStateEstimator()
    speed_ms    = 0.0
    frame_count = 0
    total       = 0

    geojson_written = False
    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        # Write static road GeoJSON once at the first message timestamp
        if not geojson_written and map_geojson is not None:
            writer.write(MAP_ROADS_TOPIC, serialize_message(map_geojson), timestamp)
            geojson_written = True

        if topic == VEL_TOPIC:
            msg = deserialize_message(data, get_message(type_map[topic]))
            speed_ms = math.sqrt(
                msg.twist.linear.x ** 2 + msg.twist.linear.y ** 2)
            writer.write(topic, data, timestamp)
            odom, imu = ego_est.update(msg, timestamp)
            writer.write(EGO_ODOM_TOPIC, serialize_message(odom), timestamp)
            writer.write(EGO_IMU_TOPIC,  serialize_message(imu),  timestamp)

        elif topic == GPS_TOPIC:
            writer.write(topic, data, timestamp)
            if map_matcher is not None:
                gps_msg = deserialize_message(data, get_message(type_map[topic]))
                if gps_msg.status.status >= 0:
                    matched_fix, lane_info = map_matcher.match(
                        gps_msg.latitude, gps_msg.longitude, timestamp)
                    writer.write(EGO_MATCHED_FIX_TOPIC,
                                 serialize_message(matched_fix), timestamp)
                    writer.write(EGO_LANE_INFO_TOPIC,
                                 serialize_message(lane_info), timestamp)

        elif topic == RAW_TOPIC:
            # rotate raw image and write to same topic
            data = rotate_raw_msg(data, get_message(type_map[topic]))
            writer.write(topic, data, timestamp)

        elif topic == COMPRESSED_TOPIC:
            # rotate, then run perception on the corrected frame
            msg_orig = deserialize_message(data, get_message(type_map[topic]))
            rotated_data, img = rotate_compressed_msg(data, get_message(type_map[topic]))
            writer.write(topic, rotated_data, timestamp)

            if img is not None:
                h_img = img.shape[0]
                hood_y = int(h_img * HOOD_CUTOFF)

                # YOLO first — boxes are used to mask vehicles out of lane detection
                det_results = model(img, verbose=False)[0]

                # filter out detections whose centre is in the car hood region
                if det_results.boxes is not None and len(det_results.boxes):
                    centres_y = det_results.boxes.xyxy[:, 1] + \
                                (det_results.boxes.xyxy[:, 3] - det_results.boxes.xyxy[:, 1]) / 2
                    keep = centres_y < hood_y
                    det_results.boxes = det_results.boxes[keep]

                lane_overlay = detect_lanes(img, tracker, det_results.boxes)

                # compose: YOLO boxes on clean frame, then blend lane overlay
                annotated = det_results.plot()
                annotated = cv2.addWeighted(annotated, 1.0, lane_overlay, 0.35, 0)
                draw_speed(annotated, speed_ms)

                ann_msg = make_compressed_msg(annotated, msg_orig.header)
                writer.write(ANNOTATED_TOPIC, serialize_message(ann_msg), timestamp)

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f'  {frame_count} frames processed...')

        else:
            writer.write(topic, data, timestamp)

        total += 1

    del writer
    print(f'Done — {frame_count} frames processed, {total} messages → {output_path}')


if __name__ == '__main__':
    main()
