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

import json

import cv2
import numpy as np
import rosbag2_py
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CompressedImage, Image, Imu, NavSatFix
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray

COMPRESSED_TOPIC = '/usb_cam/image_raw/compressed'
RAW_TOPIC        = '/usb_cam/image_raw'
ANNOTATED_TOPIC  = '/perception/image_annotated'
VEL_TOPIC        = '/vel'
GPS_TOPIC             = '/fix'
EGO_ODOM_TOPIC        = '/ego/odometry'
EGO_IMU_TOPIC         = '/ego/imu'
EGO_MATCHED_FIX_TOPIC = '/ego/matched_fix'
EGO_LANE_INFO_TOPIC   = '/ego/lane_info'
MAP_LANES_TOPIC       = '/map/lanes'
EGO_MARKER_TOPIC      = '/ego/marker'
EGO_POSE_TOPIC        = '/ego/pose'
ACTORS_TOPIC          = '/perception/actors'

# ── Actor detection constants ────────────────────────────────────────────────
_ACT_F_PX          = 500.0
_ACT_CX            = 320.0
_ACT_VEHICLE_H     = 1.5    # assumed vehicle height (metres)
_ACT_MAX_DIST      = 80.0   # discard projections beyond this (metres)
_ACT_VEHICLE_CLS   = {2, 3, 5, 7}   # COCO: car, motorcycle, bus, truck
_ACT_IOU_THRESH    = 0.30
_ACT_MAX_GAP       = 15             # frames a track survives without a detection
_ACT_CONF_THRESH   = 0.40           # minimum YOLO confidence to START a new track
_ACT_CONF_TRACK    = 0.25           # minimum confidence to UPDATE an existing track
_ACT_DRAW_GAP_2D   = 2              # camera-view boxes: only when recently detected
_ACT_DRAW_GAP_3D   = 8              # 3D markers: hold last ENU position through short gaps
_ACT_BOX_ALPHA     = 0.40           # image-space box EMA weight
_ACT_MAX_LAT       = 12.25          # max lateral offset (m) — drops oncoming traffic
_ACT_EMA_ALPHA     = 0.40           # ENU position EMA weight (global frame — see below)
_ACT_HEAD_ALPHA    = 0.05           # heading EMA weight — slow to prevent spin
_ACT_DIMS = {                       # YOLO class → (length, width, height) metres
    2: (4.5,  2.0, 1.5),            # car — same footprint as ego
    3: (2.2,  0.8, 1.2),            # motorcycle
    5: (12.0, 2.5, 3.5),            # bus
    7: (8.0,  2.5, 3.0),            # truck
}
_COCO_VEHICLE_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}


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
    Lane estimation assumes right-hand traffic; uses module-level LANE_WIDTH.
    """

    def __init__(self, graph_path: str):
        import osmnx as ox
        with open(graph_path, 'rb') as f:
            self._G = pickle.load(f)
        self._ox = ox
        lats = [d['y'] for _, d in self._G.nodes(data=True)]
        lons = [d['x'] for _, d in self._G.nodes(data=True)]
        self.lat0 = float(np.mean(lats))
        self.lon0 = float(np.mean(lons))
        self._lane_ema           = 1.0   # EMA of continuous lane index; init = rightmost (1)
        self._lane_alpha         = 0.10  # EMA weight for new estimates (slow)
        self._lateral_m_ema      = None  # used only for lane_info string
        self._enu_ema            = None  # asymmetric-smoothed GPS ENU position
        self._lane_num           = 1     # confirmed lane (hysteresis-filtered)
        self._lane_change_count  = 0     # consecutive fixes proposing a lane change
        self._LANE_CHANGE_THRESH = 4     # fixes needed to confirm a lane change

        # Pre-build edge geometry list for heading-consistent snap (used in match())
        from shapely.geometry import LineString as _SHLine
        self._edge_geoms = []
        for eu, ev, ek, edata in self._G.edges(keys=True, data=True):
            g = edata.get('geometry')
            if g is None:
                nu, nv = self._G.nodes[eu], self._G.nodes[ev]
                g = _SHLine([(nu['x'], nu['y']), (nv['x'], nv['y'])])
            self._edge_geoms.append((g, eu, ev, ek))

    def match(self, lat: float, lon: float, timestamp_ns: int,
              bev_d_left_m: float = None, ego_heading: float = 0.0,
              heading_valid: bool = True):
        """
        Returns (matched_NavSatFix, lane_info_String, (enu_x, enu_y)).
        bev_d_left_m: ego's distance in metres from the detected left lane boundary
                      (from LaneTracker.bev_lateral()); used to refine lane number.
        ego_heading:  ENU yaw (rad) from /vel; used to resolve edge-direction ambiguity.
        """
        ox = self._ox
        G  = self._G

        # ── Nearest edge ────────────────────────────────────────────────────
        (u, v, key), dist_deg = ox.nearest_edges(
            G, X=lon, Y=lat, return_dist=True)

        # On divided highways ox.nearest_edges may return the oncoming carriageway.
        # If the nearest edge strongly opposes ego heading, search within 30 m for
        # the nearest edge that is heading-consistent (dot > 0.3).
        from shapely.geometry import Point as _Pt
        def _geom_of(eu, ev, ek):
            g = G[eu][ev][ek].get('geometry')
            if g is None:
                from shapely.geometry import LineString as _L
                nu, nv = G.nodes[eu], G.nodes[ev]
                return _L([(nu['x'], nu['y']), (nv['x'], nv['y'])])
            return g
        def _heading_dot(g):
            pd = g.project(_Pt(lon, lat))
            pb = g.interpolate(max(0.0, pd - 1e-5))
            pa = g.interpolate(min(g.length, pd + 1e-5))
            clat = math.cos(math.radians(lat))
            tx_ = (pa.x - pb.x) * clat * 111_320.0
            ty_ = (pa.y - pb.y) * 111_320.0
            n = math.sqrt(tx_*tx_ + ty_*ty_)
            return (tx_*math.cos(ego_heading) + ty_*math.sin(ego_heading)) / n if n > 1e-9 else 1.0
        _nearest_g = _geom_of(u, v, key)
        if _heading_dot(_nearest_g) < -0.3:
            _SEARCH_DEG = 30.0 / 111_320.0
            _pt_q = _Pt(lon, lat)
            _best_d = _nearest_g.distance(_pt_q)
            for _eg, _eu, _ev, _ek in self._edge_geoms:
                _b = _eg.bounds
                if (_b[2] < lon - _SEARCH_DEG or _b[0] > lon + _SEARCH_DEG or
                        _b[3] < lat - _SEARCH_DEG or _b[1] > lat + _SEARCH_DEG):
                    continue
                _d = _eg.distance(_pt_q)
                if _d < _best_d and _heading_dot(_eg) > 0.3:
                    _best_d, u, v, key = _d, _eu, _ev, _ek

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

            # If the OSM edge opposes ego heading, flip tangent and lateral sign.
            # This handles opposite-direction carriageways being snapped as nearest.
            cos_lat_h = math.cos(math.radians(lat))
            tang_enu_x = tx * cos_lat_h * 111_320.0
            tang_enu_y = ty * 111_320.0
            dot = tang_enu_x * math.cos(ego_heading) + tang_enu_y * math.sin(ego_heading)
            if dot < 0:
                tx, ty     = -tx, -ty          # flip tangent so perp is also correct
                lateral_m  = -lateral_m
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

        if (bev_d_left_m is not None
                and -LANE_WIDTH < bev_d_left_m < 2.0 * LANE_WIDTH):
            # make_map.py centres lanes on the OSM edge (road centreline convention).
            # lateral_m is measured from the OSM edge = road centre, so we subtract
            # N/2 to convert from "lanes from road centre" to "lane index from right".
            i_float = (lateral_m + bev_d_left_m) / LANE_WIDTH - total_lanes / 2.0
            method  = 'bev'
        else:
            i_float = lateral_m / LANE_WIDTH - total_lanes / 2.0
            method  = 'gps'

        # EMA smoothing + hysteresis:
        #   - initialised at lane 1 (rightmost) to match typical highway driving
        #   - slow alpha damps GPS/BEV noise
        #   - hysteresis counter prevents transient jumps (lane merges, GPS glitches)
        self._lane_ema  = min(self._lane_ema, float(total_lanes))  # clamp on merge
        i_clamped       = max(1.0, min(float(total_lanes), i_float))
        self._lane_ema  = self._lane_alpha * i_clamped + (1.0 - self._lane_alpha) * self._lane_ema
        candidate       = max(1, min(total_lanes, round(self._lane_ema)))
        if candidate != self._lane_num:
            self._lane_change_count += 1
            if self._lane_change_count >= self._LANE_CHANGE_THRESH:
                self._lane_num = candidate
                self._lane_change_count = 0
        else:
            self._lane_change_count = 0
        # If lane_num shrank due to merge, force valid
        self._lane_num = max(1, min(total_lanes, self._lane_num))
        lane_num = self._lane_num

        # ── ENU position: snap + lane-centre offset ─────────────────────────────
        #
        # !! DO NOT REPLACE THIS WITH RAW GPS !!
        #
        # GPS lateral accuracy on a motorway is typically 5–15 m — far worse
        # than a lane width.  In our bags the GPS antenna reports a position
        # that is ~10 m west of the ego's actual lane, placing it in the median.
        # Using raw GPS coordinates for the ego ENU position therefore always
        # shows the ego marker in the wrong lane (or off-road entirely).
        #
        # make_map.py centres lanes symmetrically on the OSM edge (road centreline
        # convention — matching actual OSM road geometry).  The snap point therefore
        # lands at the CENTRE of the carriageway.  Offset formula:
        #   offset = (N/2 – lane_num + 0.5) × LANE_WIDTH  (right-of-travel)
        #   lane 1 (rightmost) → +0.5 × LANE_WIDTH  (for N=2)
        #   lane N (leftmost)  → −0.5 × LANE_WIDTH  (for N=2)
        #
        # !! DO NOT REVERT TO (N – lane_num + 0.5) × LANE_WIDTH !!
        # That formula assumed snap = left road boundary (old make_map.py behaviour).
        # It would place the ego ~N/2 × LANE_WIDTH too far to the right.
        #
        # The snap point reliably tracks the road position regardless of GPS
        # lateral error, making it the correct base for lane-level positioning.
        cos_lat0 = math.cos(math.radians(self.lat0))
        snap_x = (snap_lon - self.lon0) * cos_lat0 * 111_320.0
        snap_y = (snap_lat - self.lat0) * 111_320.0

        # EMA-smooth the snap point to absorb GPS-fix jitter along the road.
        if self._enu_ema is None:
            self._enu_ema = [snap_x, snap_y]
        else:
            _a = 0.6
            self._enu_ema[0] = _a * snap_x + (1.0 - _a) * self._enu_ema[0]
            self._enu_ema[1] = _a * snap_y + (1.0 - _a) * self._enu_ema[1]
        sx, sy = self._enu_ema

        # Right-of-travel vector in ENU: (sin h, -cos h).
        # Do not apply the offset if heading is invalid (speed < MIN_SPEED);
        # a zero heading would project the offset southward instead of rightward.
        if heading_valid:
            offset = (total_lanes / 2.0 - lane_num + 0.5) * LANE_WIDTH
            lane_enu_x = sx + offset * math.sin(ego_heading)
            lane_enu_y = sy - offset * math.cos(ego_heading)
        else:
            lane_enu_x, lane_enu_y = sx, sy

        lane_lat = snap_lat
        lane_lon = snap_lon

        # ── Build ROS messages ──────────────────────────────────────────────
        h = Header()
        h.frame_id      = 'map'
        h.stamp.sec     = int(timestamp_ns // 1_000_000_000)
        h.stamp.nanosec = int(timestamp_ns %  1_000_000_000)

        fix               = NavSatFix()
        fix.header        = h
        fix.latitude      = lane_lat
        fix.longitude     = lane_lon
        fix.altitude      = 0.0
        fix.status.status = 0   # STATUS_FIX

        bev_str   = (f' | bev_d_left {bev_d_left_m:.2f} m'
                     if bev_d_left_m is not None else '')
        info      = String()
        info.data = (f'{name} | lane {lane_num}/{total_lanes} [{method}] | '
                     f'gps_offset {lateral_m:+.1f} m{bev_str}')

        return fix, info, (lane_enu_x, lane_enu_y)


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
LANE_WIDTH  = 3.5    # metres, standard lane width (used for BEV scale and map matching)

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

    def bev_lateral(self):
        """
        Returns (d_left_m, valid).
        d_left_m: ego's distance in metres from the detected left lane boundary,
                  computed from the BEV polynomial at y=BEV_H (closest to car).
                  Self-calibrating: uses detected lane width to set the m/px scale.
        valid: False when either polynomial is missing or detected width is implausible.
        """
        if self.left_poly is None or self.right_poly is None:
            return 0.0, False
        left_x  = float(np.polyval(self.left_poly,  BEV_H))
        right_x = float(np.polyval(self.right_poly, BEV_H))
        ego_x   = BEV_W / 2.0
        lane_px = right_x - left_x
        # Sanity: detected lane width must be 10–70% of BEV canvas width
        if not (BEV_W * 0.10 <= lane_px <= BEV_W * 0.70):
            return 0.0, False
        scale  = LANE_WIDTH / lane_px   # metres per BEV pixel (self-calibrating)
        d_left = (ego_x - left_x) * scale
        return float(d_left), True


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


# ── Lane marker helpers ───────────────────────────────────────────────────────

_BOUNDARY_COLORS = {
    'lane':   (0.2, 0.8, 1.0),   # bright cyan  — interior lane boundary
    'edge':   (1.0, 1.0, 1.0),   # white        — road outer edge
    'center': (1.0, 0.9, 0.0),   # yellow       — road centreline / carriageway divider
}


def lanes_geojson_to_markers(geojson_path: str, lat0: float, lon0: float) -> MarkerArray:
    """Convert lane boundary GeoJSON (from make_map.py) to a MarkerArray in ENU frame."""
    with open(geojson_path) as f:
        fc = json.load(f)

    ma = MarkerArray()
    cos_lat = math.cos(math.radians(lat0))

    for idx, feature in enumerate(fc.get('features', [])):
        geom = feature.get('geometry', {})
        if geom.get('type') != 'LineString':
            continue
        coords = geom['coordinates']   # [[lon, lat], ...]
        if len(coords) < 2:
            continue
        props = feature.get('properties', {})
        btype = props.get('boundary', 'lane')
        r, g, b = _BOUNDARY_COLORS.get(btype, (0.6, 0.6, 0.6))

        m = Marker()
        m.header.frame_id = 'map'
        m.ns    = 'lanes'
        m.id    = idx
        m.type  = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.5           # line width in metres
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 0.9
        m.pose.orientation.w = 1.0

        for lon, lat in coords:
            p = Point()
            p.x = (lon - lon0) * cos_lat * 111_320.0
            p.y = (lat - lat0) * 111_320.0
            p.z = 0.0
            m.points.append(p)

        ma.markers.append(m)

    return ma


def make_ego_marker(x: float, y: float, heading: float,
                    timestamp_ns: int) -> Marker:
    """Car-box marker in ENU 'map' frame representing the ego vehicle."""
    m = Marker()
    m.header.frame_id      = 'map'
    m.header.stamp.sec     = int(timestamp_ns // 1_000_000_000)
    m.header.stamp.nanosec = int(timestamp_ns %  1_000_000_000)
    m.ns    = 'ego'
    m.id    = 0
    m.type  = Marker.CUBE
    m.action = Marker.ADD
    m.scale.x = 4.5   # car length (m)
    m.scale.y = 2.0   # car width  (m)
    m.scale.z = 1.5   # car height (m)
    m.color.r = 0.0
    m.color.g = 1.0
    m.color.b = 0.4
    m.color.a = 0.9
    half = heading / 2.0
    m.pose.position.x    = x
    m.pose.position.y    = y
    m.pose.position.z    = 0.75   # half height so base sits on road plane
    m.pose.orientation.x = 0.0
    m.pose.orientation.y = 0.0
    m.pose.orientation.z = math.sin(half)
    m.pose.orientation.w = math.cos(half)
    return m


def make_ego_pose(x: float, y: float, heading: float,
                  timestamp_ns: int) -> PoseStamped:
    """PoseStamped in ENU 'map' frame — used by Foxglove Follow topic."""
    ps = PoseStamped()
    ps.header.frame_id      = 'map'
    ps.header.stamp.sec     = int(timestamp_ns // 1_000_000_000)
    ps.header.stamp.nanosec = int(timestamp_ns %  1_000_000_000)
    half = heading / 2.0
    ps.pose.position.x    = x
    ps.pose.position.y    = y
    ps.pose.position.z    = 0.0
    ps.pose.orientation.z = math.sin(half)
    ps.pose.orientation.w = math.cos(half)
    return ps


# ── Actor tracking ───────────────────────────────────────────────────────────

def _act_box_to_rel(box):
    """Convert YOLO box to ego-relative (d_fwd, d_lat) in metres, or None."""
    x1, y1, x2, y2 = box
    box_h = y2 - y1
    if box_h < 5:
        return None
    d_fwd = _ACT_F_PX * _ACT_VEHICLE_H / box_h
    if d_fwd > _ACT_MAX_DIST:
        return None
    d_lat = (_ACT_CX - (x1 + x2) / 2.0) * d_fwd / _ACT_F_PX
    if abs(d_lat) > _ACT_MAX_LAT:
        return None
    return d_fwd, d_lat


def _act_rel_to_enu(d_fwd: float, d_lat: float,
                    ego_x: float, ego_y: float, ego_heading: float):
    sh, ch = math.sin(ego_heading), math.cos(ego_heading)
    return ego_x + d_fwd * ch - d_lat * sh, ego_y + d_fwd * sh + d_lat * ch


def _act_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


class _ActorTrack:
    # ENU position is smoothed in the GLOBAL frame, not ego-relative.
    # Smoothing ego-relative d_rel causes "back-and-forth" oscillation: as ego
    # moves, d_fwd changes every frame, so a stale d_smooth and a fresh YOLO
    # measurement fight each other after every gap.  In ENU space a stationary
    # or slow actor barely moves, so the EMA is well-behaved and holds position
    # correctly through short YOLO gaps without any ego-motion coupling.
    _next_id = 0
    def __init__(self, box, cls):
        self.id          = _ActorTrack._next_id; _ActorTrack._next_id += 1
        self.box         = list(box)
        self.box_smooth  = list(box)
        self.cls         = cls
        self.cls_votes   = {cls: 1}
        self.enu         = None   # EMA-smoothed ENU position (global frame)
        self.heading_ema = None
        self.gap         = 0


class ActorTracker:
    def __init__(self):
        self._active: list[_ActorTrack] = []

    def update(self, detections, ego_x: float, ego_y: float, ego_heading: float):
        """detections: list of (box, cls, conf) tuples above _ACT_CONF_TRACK threshold."""
        boxes = [d[0] for d in detections]
        clss  = [d[1] for d in detections]
        confs = [d[2] for d in detections]
        matched_t: set = set(); matched_d: set = set()
        iou_m = np.zeros((len(self._active), len(boxes)))
        for ti, tr in enumerate(self._active):
            for di, b in enumerate(boxes):
                iou_m[ti, di] = _act_iou(tr.box, b)
        while iou_m.size > 0 and iou_m.max() >= _ACT_IOU_THRESH:
            ti, di = np.unravel_index(iou_m.argmax(), iou_m.shape)
            tr = self._active[ti]
            b  = boxes[di]
            a  = _ACT_BOX_ALPHA
            tr.box_smooth = [a * b[k] + (1 - a) * tr.box_smooth[k] for k in range(4)]
            tr.box = b
            d_rel = _act_box_to_rel(tr.box_smooth)
            if d_rel is not None:
                enu_raw = _act_rel_to_enu(d_rel[0], d_rel[1], ego_x, ego_y, ego_heading)
                if tr.enu is None:
                    tr.enu = enu_raw
                else:
                    ea = _ACT_EMA_ALPHA
                    tr.enu = (ea * enu_raw[0] + (1 - ea) * tr.enu[0],
                              ea * enu_raw[1] + (1 - ea) * tr.enu[1])
            tr.cls_votes[clss[di]] = tr.cls_votes.get(clss[di], 0) + 1
            tr.cls = max(tr.cls_votes, key=tr.cls_votes.get)
            tr.gap = 0
            matched_t.add(ti); matched_d.add(di)
            iou_m[ti, :] = 0.0; iou_m[:, di] = 0.0
        for di, b in enumerate(boxes):
            # New tracks require high confidence; existing tracks already updated above.
            if di not in matched_d and confs[di] >= _ACT_CONF_THRESH:
                tr = _ActorTrack(b, clss[di])
                d_rel = _act_box_to_rel(b)
                if d_rel is not None:
                    tr.enu = _act_rel_to_enu(d_rel[0], d_rel[1], ego_x, ego_y, ego_heading)
                self._active.append(tr)
        still = []
        for ti, tr in enumerate(self._active):
            if ti not in matched_t:
                tr.gap += 1
            if tr.gap <= _ACT_MAX_GAP:
                still.append(tr)
        self._active = still

    def draw_on(self, img: np.ndarray) -> None:
        """Draw smoothed bounding boxes — only when recently detected (no ghost boxes)."""
        for tr in self._active:
            if tr.gap > _ACT_DRAW_GAP_2D:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in tr.box_smooth]
            name = _COCO_VEHICLE_NAMES.get(tr.cls, 'vehicle')
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(img, name, (x1, max(y1 - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

    def active_markers(self, timestamp_ns: int, ego_heading: float = 0.0) -> MarkerArray:
        ma = MarkerArray()
        for tr in self._active:
            if tr.enu is None or tr.gap > _ACT_DRAW_GAP_3D:
                continue
            x, y = tr.enu
            heading = tr.heading_ema if tr.heading_ema is not None else ego_heading
            dims = _ACT_DIMS.get(tr.cls, (4.5, 2.0, 1.5))
            m = Marker()
            m.header.frame_id      = 'map'
            m.header.stamp.sec     = int(timestamp_ns // 1_000_000_000)
            m.header.stamp.nanosec = int(timestamp_ns %  1_000_000_000)
            m.ns     = 'actors'
            m.id     = tr.id % 10_000
            m.type   = Marker.CUBE
            m.action = Marker.ADD
            m.scale.x = dims[0]; m.scale.y = dims[1]; m.scale.z = dims[2]
            m.color.r = 1.0; m.color.g = 0.5; m.color.b = 0.0; m.color.a = 0.85
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = dims[2] / 2.0
            half = heading / 2.0
            m.pose.orientation.x = 0.0
            m.pose.orientation.y = 0.0
            m.pose.orientation.z = math.sin(half)
            m.pose.orientation.w = math.cos(half)
            m.lifetime.sec = 1
            ma.markers.append(m)
        return ma


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

    map_matcher   = None
    lane_markers  = None
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
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=EGO_MARKER_TOPIC,
            type='visualization_msgs/msg/Marker',
            serialization_format='cdr',
        ))
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=EGO_POSE_TOPIC,
            type='geometry_msgs/msg/PoseStamped',
            serialization_format='cdr',
        ))
        writer.create_topic(rosbag2_py.TopicMetadata(
            name='/tf',
            type='tf2_msgs/msg/TFMessage',
            serialization_format='cdr',
        ))
        # Lane boundary markers — prefer lanes.geojson, fall back to map.geojson
        base = Path(args.map_graph).parent
        geojson_path = base / 'lanes.geojson'
        if not geojson_path.exists():
            geojson_path = base / 'map.geojson'
        if geojson_path.exists():
            lane_markers = lanes_geojson_to_markers(
                str(geojson_path), map_matcher.lat0, map_matcher.lon0)
            writer.create_topic(rosbag2_py.TopicMetadata(
                name=MAP_LANES_TOPIC,
                type='visualization_msgs/msg/MarkerArray',
                serialization_format='cdr',
            ))
            print(f'Lane markers built: {len(lane_markers.markers)} segments '
                  f'from {geojson_path}')
        else:
            print(f'Warning: {geojson_path} not found — lane overlay disabled.')
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=ACTORS_TOPIC,
            type='visualization_msgs/msg/MarkerArray',
            serialization_format='cdr',
        ))
        print('Map matching enabled.')

    tracker        = LaneTracker(alpha=0.25)
    actor_tracker  = ActorTracker() if map_matcher else None
    ego_est        = EgoStateEstimator()
    speed_ms       = 0.0
    frame_count    = 0
    total          = 0
    ego_heading    = 0.0   # tracks latest heading for ego marker (from /vel)
    gps_heading    = None  # heading from consecutive GPS positions (fallback)
    prev_gps_lat   = None
    prev_gps_lon   = None
    bev_d_left     = None  # metres from detected left lane boundary (updated per frame)
    latest_ego_x   = 0.0   # most recent ENU position from GPS (for actor projection)
    latest_ego_y   = 0.0
    dr_ego_x       = 0.0   # dead-reckoned ENU x (East): integrates velocity between 1 Hz GPS fixes
    dr_ego_y       = 0.0   # dead-reckoned ENU y (North): keeps actor ENU smooth at camera frame rate
    prev_cam_ts    = None  # nanosecond timestamp of previous camera frame

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if topic == VEL_TOPIC:
            msg = deserialize_message(data, get_message(type_map[topic]))
            speed_ms = math.sqrt(
                msg.twist.linear.x ** 2 + msg.twist.linear.y ** 2)
            if speed_ms >= EgoStateEstimator._MIN_SPEED:
                ego_heading = math.atan2(msg.twist.linear.y, msg.twist.linear.x)
            writer.write(topic, data, timestamp)
            odom, imu = ego_est.update(msg, timestamp)
            writer.write(EGO_ODOM_TOPIC, serialize_message(odom), timestamp)
            writer.write(EGO_IMU_TOPIC,  serialize_message(imu),  timestamp)

        elif topic == GPS_TOPIC:
            writer.write(topic, data, timestamp)
            if map_matcher is not None:
                gps_msg = deserialize_message(data, get_message(type_map[topic]))
                if gps_msg.status.status >= 0:
                    # GPS-track heading from consecutive fixes (robust fallback)
                    if prev_gps_lat is not None:
                        dlat = gps_msg.latitude  - prev_gps_lat
                        dlon = gps_msg.longitude - prev_gps_lon
                        if abs(dlat) + abs(dlon) > 1e-6:
                            gps_heading = math.atan2(
                                dlat,
                                dlon * math.cos(math.radians(gps_msg.latitude)))
                    prev_gps_lat = gps_msg.latitude
                    prev_gps_lon = gps_msg.longitude
                    heading_for_match = (gps_heading if gps_heading is not None
                                         else ego_heading)
                    heading_valid = speed_ms >= EgoStateEstimator._MIN_SPEED
                    matched_fix, lane_info, (lx, ly) = map_matcher.match(
                        gps_msg.latitude, gps_msg.longitude, timestamp,
                        bev_d_left_m=bev_d_left,
                        ego_heading=heading_for_match,
                        heading_valid=heading_valid)
                    latest_ego_x, latest_ego_y = lx, ly
                    dr_ego_x, dr_ego_y = lx, ly   # anchor dead-reckoning to map-matched fix
                    writer.write(EGO_MATCHED_FIX_TOPIC,
                                 serialize_message(matched_fix), timestamp)
                    writer.write(EGO_LANE_INFO_TOPIC,
                                 serialize_message(lane_info), timestamp)
                    # Ego arrow + pose snapped to lane centre in ENU 'map' frame
                    ego_m = make_ego_marker(lx, ly, ego_heading, timestamp)
                    writer.write(EGO_MARKER_TOPIC,
                                 serialize_message(ego_m), timestamp)
                    ego_ps = make_ego_pose(lx, ly, ego_heading, timestamp)
                    writer.write(EGO_POSE_TOPIC,
                                 serialize_message(ego_ps), timestamp)
                    # TF: map → base_link at lane centre
                    ts = TransformStamped()
                    ts.header.frame_id      = 'map'
                    ts.header.stamp.sec     = int(timestamp // 1_000_000_000)
                    ts.header.stamp.nanosec = int(timestamp %  1_000_000_000)
                    ts.child_frame_id       = 'base_link'
                    ts.transform.translation.x = lx
                    ts.transform.translation.y = ly
                    ts.transform.translation.z = 0.0
                    half = ego_heading / 2.0
                    ts.transform.rotation.z = math.sin(half)
                    ts.transform.rotation.w = math.cos(half)
                    tf_msg = TFMessage(transforms=[ts])
                    writer.write('/tf', serialize_message(tf_msg), timestamp)
                    # Re-stamp and re-publish lane MarkerArray so Foxglove sees it live
                    if lane_markers is not None:
                        h_stamp = Header()
                        h_stamp.frame_id      = 'map'
                        h_stamp.stamp.sec     = int(timestamp // 1_000_000_000)
                        h_stamp.stamp.nanosec = int(timestamp %  1_000_000_000)
                        for lm in lane_markers.markers:
                            lm.header = h_stamp
                        writer.write(MAP_LANES_TOPIC,
                                     serialize_message(lane_markers), timestamp)

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
                d, bev_ok = tracker.bev_lateral()
                bev_d_left = d if bev_ok else None

                # actor tracking + markers (only when map-matching is active)
                if actor_tracker is not None:
                    # Dead-reckon ego between 1 Hz GPS fixes so actor ENU stays smooth.
                    # Without this, ego_pos is frozen for ~30 frames → actor drifts 27 m
                    # then snaps on each GPS fix, producing the back-and-forth oscillation.
                    if prev_cam_ts is not None and speed_ms > 0.1:
                        dt = (timestamp - prev_cam_ts) / 1e9
                        dr_ego_x += speed_ms * math.cos(ego_heading) * dt
                        dr_ego_y += speed_ms * math.sin(ego_heading) * dt
                    prev_cam_ts = timestamp
                    vehicle_dets = []
                    if det_results.boxes is not None:
                        for i in range(len(det_results.boxes)):
                            ci = int(det_results.boxes.cls[i].item())
                            fi = float(det_results.boxes.conf[i].item())
                            # Collect at low threshold; update() only starts NEW tracks
                            # at _ACT_CONF_THRESH (hysteresis prevents flicker near border).
                            if ci in _ACT_VEHICLE_CLS and fi >= _ACT_CONF_TRACK:
                                vehicle_dets.append((det_results.boxes.xyxy[i].tolist(), ci, fi))
                    actor_tracker.update(vehicle_dets, dr_ego_x, dr_ego_y, ego_heading)
                    actor_ma = actor_tracker.active_markers(timestamp, ego_heading)
                    writer.write(ACTORS_TOPIC, serialize_message(actor_ma), timestamp)

                # compose: smoothed vehicle boxes + non-vehicle detections + lane overlay
                annotated = img.copy()
                if det_results.boxes is not None:
                    for i in range(len(det_results.boxes)):
                        ci = int(det_results.boxes.cls[i].item())
                        fi = float(det_results.boxes.conf[i].item())
                        if ci not in _ACT_VEHICLE_CLS and fi >= _ACT_CONF_THRESH:
                            x1, y1, x2, y2 = [int(v) for v in det_results.boxes.xyxy[i].tolist()]
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 200, 200), 1)
                            cv2.putText(annotated, f'{det_results.names[ci]} {fi:.2f}',
                                        (x1, max(y1 - 4, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
                                        cv2.LINE_AA)
                if actor_tracker is not None:
                    actor_tracker.draw_on(annotated)
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
