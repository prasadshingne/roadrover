#!/usr/bin/env python3
"""
Extract an OpenSCENARIO 1.x (.xosc) scenario from a *processed* roadrover bag.

Reads:
  /ego/matched_fix   (NavSatFix)   — snapped GPS position per fix
  /ego/odometry      (Odometry)    — heading (quaternion) + speed per fix
  /perception/image_annotated (CompressedImage) — frames with YOLO boxes

Outputs:
  scenario.xosc      — OpenSCENARIO file with ego + detected actors
  map.xodr           — (auto-generated) OpenDRIVE road network (calls make_xodr.py)

Usage:
  python3 make_scenario.py <processed_bag> --map-graph <map_graph.pkl>

  # Specify output directory:
  python3 make_scenario.py <processed_bag> --map-graph <map_graph.pkl> \\
      --out-dir /path/to/output

The processed bag must have been produced by process_bag.py (with --map-graph).

Actor detection pipeline:
  1. Re-run YOLOv8s on each annotated frame (or read existing YOLO detections).
     Vehicle classes: 2=car, 3=motorcycle, 5=bus, 7=truck.
  2. Track actors across frames using IoU matching (greedy Hungarian-style).
  3. Require >= MIN_TRACK_FRAMES consecutive frames to export a track.
  4. Project each bounding-box bottom-centre to ENU via pinhole road-plane model.
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CompressedImage, NavSatFix

# ── Camera intrinsics (uncalibrated approximation) ────────────────────────────
F_PX          = 500.0    # focal length in pixels
CX            = 320.0    # principal point x
VEHICLE_HEIGHT = 1.5     # metres — assumed generic vehicle height for box projection

# ── Tracking parameters ───────────────────────────────────────────────────────
VEHICLE_CLASSES    = {2, 3, 5, 7}   # car, motorcycle, bus, truck
IOU_THRESHOLD      = 0.30           # min IoU to continue a track
MIN_TRACK_FRAMES   = 10             # discard tracks shorter than this
MAX_TRACK_GAP      = 5              # max missed frames before track is closed
MAX_PROJ_DIST      = 80.0           # metres — discard projections beyond this (noisy)

# ── Lane / road width (matches process_bag.py) ───────────────────────────────
LANE_WIDTH = 3.5   # metres


# ── ENU helpers ──────────────────────────────────────────────────────────────

def ll_to_enu(lat: float, lon: float, lat0: float, lon0: float):
    """Approximate lat/lon → local ENU East, North (metres)."""
    x = (lon - lon0) * math.cos(math.radians(lat0)) * 111_320.0
    y = (lat - lat0) * 111_320.0
    return x, y


def quat_to_yaw(qz: float, qw: float) -> float:
    """Extract ENU yaw (rad) from a quaternion with qx=qy=0."""
    return 2.0 * math.atan2(qz, qw)


# ── Box-height projection ─────────────────────────────────────────────────────

def box_to_enu(box, ego_x: float, ego_y: float,
               ego_heading: float) -> tuple[float, float] | None:
    """
    Project a bounding box to ENU by estimating distance from box height.
    d_fwd = F_PX * VEHICLE_HEIGHT / box_height_px.
    Works for any image position — no horizon dependency.
    Returns None when box is too small or actor is beyond MAX_PROJ_DIST.
    """
    x1, y1, x2, y2 = box
    box_h_px = y2 - y1
    if box_h_px < 5:
        return None
    d_fwd = F_PX * VEHICLE_HEIGHT / box_h_px
    if d_fwd > MAX_PROJ_DIST:
        return None
    u_c   = (x1 + x2) / 2.0
    d_lat = (u_c - CX) * d_fwd / F_PX   # positive = right in image
    sin_h, cos_h = math.sin(ego_heading), math.cos(ego_heading)
    dx = d_fwd * cos_h - d_lat * sin_h
    dy = d_fwd * sin_h + d_lat * cos_h
    return ego_x + dx, ego_y + dy


# ── IoU tracker ──────────────────────────────────────────────────────────────

def _iou(a, b) -> float:
    """IoU of two boxes [x1, y1, x2, y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    _next_id = 0

    def __init__(self, box, frame_idx: int, enu_pos):
        self.id        = Track._next_id
        Track._next_id += 1
        self.box       = box
        self.last_frame = frame_idx
        self.gap        = 0
        self.waypoints  = []    # list of (t_s, x, y, heading, speed)
        if enu_pos is not None:
            self.waypoints.append(enu_pos)

    def update(self, box, frame_idx: int, enu_pos):
        self.box        = box
        self.last_frame = frame_idx
        self.gap        = 0
        if enu_pos is not None:
            self.waypoints.append(enu_pos)


class IoUTracker:
    def __init__(self):
        self._active:  list[Track] = []
        self._closed:  list[Track] = []

    def update(self, detections, frame_idx: int,
               ego_x: float, ego_y: float, ego_heading: float, t_s: float):
        """
        detections: list of [x1, y1, x2, y2] boxes (already filtered by class).
        Returns nothing — accumulates in self._active / self._closed.
        """
        matched_track_ids = set()
        matched_det_ids   = set()

        # Greedy best-IoU matching (good enough for sparse road traffic)
        iou_matrix = np.zeros((len(self._active), len(detections)))
        for ti, track in enumerate(self._active):
            for di, det in enumerate(detections):
                iou_matrix[ti, di] = _iou(track.box, det)

        while iou_matrix.size > 0 and iou_matrix.max() >= IOU_THRESHOLD:
            ti, di = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            track = self._active[ti]
            box   = detections[di]
            enu   = box_to_enu(box, ego_x, ego_y, ego_heading)
            wp    = (t_s, enu[0], enu[1], ego_heading, 0.0) if enu else None
            track.update(box, frame_idx, wp)
            matched_track_ids.add(ti)
            matched_det_ids.add(di)
            iou_matrix[ti, :] = 0.0
            iou_matrix[:, di] = 0.0

        # Unmatched detections → new tracks
        for di, det in enumerate(detections):
            if di in matched_det_ids:
                continue
            enu = box_to_enu(det, ego_x, ego_y, ego_heading)
            wp  = (t_s, enu[0], enu[1], ego_heading, 0.0) if enu else None
            self._active.append(Track(det, frame_idx, wp))

        # Age unmatched tracks; close stale ones
        still_active = []
        for ti, track in enumerate(self._active):
            if ti not in matched_track_ids:
                track.gap += 1
            if track.gap > MAX_TRACK_GAP:
                self._closed.append(track)
            else:
                still_active.append(track)
        self._active = still_active

    def finish(self):
        """Close all remaining active tracks."""
        self._closed.extend(self._active)
        self._active = []

    def good_tracks(self) -> list[Track]:
        return [t for t in self._closed if len(t.waypoints) >= MIN_TRACK_FRAMES]


# ── Bag reading helpers ───────────────────────────────────────────────────────

def open_bag(bag_path: str):
    storage = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3')
    converter = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)
    return reader


def topic_type_map(reader) -> dict[str, str]:
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


# ── OpenSCENARIO generation ──────────────────────────────────────────────────

def build_xosc(ego_waypoints: list, actor_tracks: list[Track],
               xodr_path: str, lat0: float, lon0: float,
               author: str = 'roadrover') -> 'Scenario':
    from scenariogeneration import xosc

    t_scenario_start = ego_waypoints[0][0]

    def wp_to_world(wp) -> 'xosc.WorldPosition':
        _t_s, x, y, heading, _speed = wp
        return xosc.WorldPosition(x=x, y=y, h=heading)

    def smooth_positions(waypoints: list, window: int = 5) -> list:
        """Replace (x, y) with a centred moving average to remove outlier jumps."""
        if len(waypoints) < window:
            return waypoints
        wps = list(waypoints)
        xs = np.array([w[1] for w in wps])
        ys = np.array([w[2] for w in wps])
        half = window // 2
        xs_s = np.convolve(xs, np.ones(window) / window, mode='same')
        ys_s = np.convolve(ys, np.ones(window) / window, mode='same')
        # edges: keep originals where convolution is padded (first/last half)
        xs_s[:half]  = xs[:half]
        xs_s[-half:] = xs[-half:]
        ys_s[:half]  = ys[:half]
        ys_s[-half:] = ys[-half:]
        return [(wps[i][0], float(xs_s[i]), float(ys_s[i]), wps[i][3], wps[i][4])
                for i in range(len(wps))]

    def with_headings(waypoints: list) -> list:
        """Return a copy of waypoints with heading derived from motion direction."""
        wps = list(waypoints)
        for i in range(len(wps) - 1):
            t0, x0, y0, h0, s0 = wps[i]
            _t1, x1, y1, _h1, _s1 = wps[i + 1]
            dx, dy = x1 - x0, y1 - y0
            if dx * dx + dy * dy > 0.01:   # >0.1 m movement
                h0 = math.atan2(dy, dx)
            wps[i] = (t0, x0, y0, h0, s0)
        if len(wps) > 1:
            # last waypoint keeps the heading of the second-to-last
            t, x, y, _, s = wps[-1]
            wps[-1] = (t, x, y, wps[-2][3], s)
        return wps

    def make_trajectory(name: str, waypoints: list, t_start: float) -> 'xosc.Trajectory':
        # times relative to scenario start so actors move at the right moment
        times     = [wp[0] - t_start for wp in waypoints]
        positions = [wp_to_world(wp) for wp in waypoints]
        poly      = xosc.Polyline(time=times, positions=positions)
        traj      = xosc.Trajectory(name=name, closed=False)
        traj.add_shape(poly)
        return traj

    def follow_action(traj) -> 'xosc.FollowTrajectoryAction':
        # position mode: entity teleported to interpolated trajectory position
        # each timestep — correct for replay; no dynamics/speed required.
        return xosc.FollowTrajectoryAction(
            trajectory=traj,
            following_mode=xosc.FollowingMode.position,
            reference_domain='relative',
            scale=1.0,
            offset=0.0,
        )

    def _vehicle(vname: str) -> 'xosc.Vehicle':
        # BoundingBox(width, length, height, x_center, y_center, z_center)
        bb = xosc.BoundingBox(2.0, 4.5, 1.8, 1.5, 0.0, 0.9)
        # Axle(maxsteer, wheeldia, track_width, xpos, zpos)
        front = xosc.Axle(0.5, 0.6, 1.68, 2.98, 0.3)
        rear  = xosc.Axle(0.0, 0.6, 1.68, 0.0,  0.3)
        return xosc.Vehicle(
            name=vname,
            vehicle_type=xosc.VehicleCategory.car,
            boundingbox=bb,
            frontaxle=front,
            rearaxle=rear,
            max_speed=60.0,
            max_acceleration=10.0,
            max_deceleration=10.0,
        )

    def _step_dynamics() -> 'xosc.TransitionDynamics':
        return xosc.TransitionDynamics(
            shape=xosc.DynamicsShapes.step,
            dimension=xosc.DynamicsDimension.time,
            value=0.0,
        )

    def _empty_trigger() -> 'xosc.EmptyTrigger':
        return xosc.EmptyTrigger('start')

    # ── Road network ─────────────────────────────────────────────────────────
    road_network = xosc.RoadNetwork(roadfile=str(xodr_path))
    catalog      = xosc.Catalog()

    # ── Entities ─────────────────────────────────────────────────────────────
    entities = xosc.Entities()
    entities.add_scenario_object(name='Ego', entityobject=_vehicle('Ego'))
    for track in actor_tracks:
        entities.add_scenario_object(name=f'Actor_{track.id}',
                                     entityobject=_vehicle(f'Actor_{track.id}'))

    # ── Init ─────────────────────────────────────────────────────────────────
    init = xosc.Init()

    init.add_init_action('Ego', xosc.TeleportAction(wp_to_world(ego_waypoints[0])))
    init.add_init_action('Ego', xosc.AbsoluteSpeedAction(
        speed=ego_waypoints[0][4], transition_dynamics=_step_dynamics()))

    for track in actor_tracks:
        init.add_init_action(f'Actor_{track.id}',
                             xosc.TeleportAction(wp_to_world(track.waypoints[0])))
        init.add_init_action(f'Actor_{track.id}',
                             xosc.AbsoluteSpeedAction(speed=0.0,
                                                       transition_dynamics=_step_dynamics()))

    # ── Story / StoryBoard ────────────────────────────────────────────────────
    story = xosc.Story(name='RoadRoverStory')

    def _make_act(act_name: str, entity_name: str, traj) -> 'xosc.Act':
        event = xosc.Event(name=f'{act_name}Event', priority=xosc.Priority.overwrite)
        event.add_action(actionname=f'{act_name}Action', action=follow_action(traj))
        event.add_trigger(_empty_trigger())

        maneuver = xosc.Maneuver(name=f'{act_name}Maneuver')
        maneuver.add_event(event)

        mg = xosc.ManeuverGroup(name=f'{act_name}MG')
        mg.add_actor(entity_name)
        mg.add_maneuver(maneuver)

        act = xosc.Act(name=act_name, starttrigger=_empty_trigger())
        act.add_maneuver_group(mg)
        return act

    story.add_act(_make_act('EgoAct', 'Ego',
                            make_trajectory('EgoTrajectory', ego_waypoints,
                                            t_scenario_start)))
    for track in actor_tracks:
        story.add_act(_make_act(
            f'ActorAct_{track.id}',
            f'Actor_{track.id}',
            make_trajectory(f'ActorTraj_{track.id}',
                            with_headings(smooth_positions(track.waypoints)),
                            t_scenario_start),
        ))

    storyboard = xosc.StoryBoard(init=init, stoptrigger=xosc.EmptyTrigger('stop'))
    storyboard.add_story(story)

    scenario = xosc.Scenario(
        name='RoadRoverScenario',
        author=author,
        parameters=xosc.ParameterDeclarations(),
        entities=entities,
        storyboard=storyboard,
        roadnetwork=road_network,
        catalog=catalog,
    )
    return scenario


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Extract OpenSCENARIO from a processed roadrover bag.')
    ap.add_argument('bag', help='Path to the processed bag (from process_bag.py)')
    ap.add_argument('--map-graph', required=True,
                    help='Path to map_graph.pkl produced by make_map.py')
    ap.add_argument('--out-dir', default=None,
                    help='Output directory (default: same as bag parent)')
    ap.add_argument('--yolo-weights', default='yolov8s.pt',
                    help='YOLOv8 weights file (default: yolov8s.pt)')
    ap.add_argument('--min-track-frames', type=int, default=MIN_TRACK_FRAMES,
                    help=f'Minimum consecutive frames to export a track (default: {MIN_TRACK_FRAMES})')
    args = ap.parse_args()

    bag_path   = Path(args.bag)
    graph_path = Path(args.map_graph)
    out_dir    = Path(args.out_dir) if args.out_dir else bag_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    xodr_path  = out_dir / 'map.xodr'
    xosc_path  = out_dir / 'scenario.xosc'

    # ── ENU origin from graph ─────────────────────────────────────────────────
    import pickle
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    lats = [d['y'] for _, d in G.nodes(data=True)]
    lons = [d['x'] for _, d in G.nodes(data=True)]
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    print(f'ENU origin: lat0={lat0:.6f}  lon0={lon0:.6f}')

    # ── Load YOLOv8 ──────────────────────────────────────────────────────────
    print(f'Loading YOLOv8 weights: {args.yolo_weights}')
    from ultralytics import YOLO
    model = YOLO(args.yolo_weights)

    # ── First pass: read ego trajectory from bag ──────────────────────────────
    print('Pass 1: reading ego trajectory ...')

    matched_fixes: dict[int, tuple[float, float]] = {}   # ts_ns → (lat, lon)
    ego_odoms:     dict[int, tuple[float, float]] = {}   # ts_ns → (heading, speed)

    reader = open_bag(str(bag_path))
    types  = topic_type_map(reader)

    pose_type = get_message(types['/ego/pose'])
    odom_type = get_message(types['/ego/odometry'])
    img_type  = get_message(types['/perception/image_annotated'])

    # ego_poses: ts_ns → (x, y, heading)  — direct ENU, same source as Foxglove
    ego_poses: dict[int, tuple[float, float, float]] = {}

    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic == '/ego/pose':
            msg = deserialize_message(data, pose_type)
            x   = msg.pose.position.x
            y   = msg.pose.position.y
            heading = quat_to_yaw(msg.pose.orientation.z,
                                  msg.pose.orientation.w)
            ego_poses[ts] = (x, y, heading)
        elif topic == '/ego/odometry':
            msg   = deserialize_message(data, odom_type)
            speed = msg.twist.twist.linear.x
            qz    = msg.pose.pose.orientation.z
            qw    = msg.pose.pose.orientation.w
            ego_odoms[ts] = (quat_to_yaw(qz, qw), speed)

    # Join ego/pose and ego/odometry on nearest timestamp for speed
    odom_times = sorted(ego_odoms.keys())
    odom_arr_pass1 = np.array(odom_times, dtype=np.int64)
    ego_waypoints = []
    for pose_ts, (x, y, heading) in sorted(ego_poses.items()):
        idx = int(np.searchsorted(odom_arr_pass1, pose_ts))
        idx = min(idx, len(odom_times) - 1)
        if idx > 0 and abs(odom_times[idx-1] - pose_ts) < abs(odom_times[idx] - pose_ts):
            idx -= 1
        _, speed = ego_odoms[odom_times[idx]]
        ego_waypoints.append((pose_ts * 1e-9, x, y, heading, speed))

    print(f'  Ego waypoints: {len(ego_waypoints)}')
    if not ego_waypoints:
        sys.exit('ERROR: no /ego/pose messages found — '
                 'was the bag processed with --map-graph?')

    # ── Generate map.xodr from full graph ────────────────────────────────────
    print(f'Generating {xodr_path} ...')
    make_xodr_script = Path(__file__).parent / 'make_xodr.py'
    subprocess.run(
        [sys.executable, str(make_xodr_script),
         str(graph_path), '--out', str(xodr_path)],
        check=True,
    )

    # ── Second pass: detect + track actors in annotated frames ───────────────
    print('Pass 2: detecting and tracking actors ...')

    # Rebuild odom array for Pass 2 nearest-timestamp lookups
    odom_arr = np.array(odom_times, dtype=np.int64)

    # Build ego/pose lookup for use during actor projection
    pose_times = sorted(ego_poses.keys())
    pose_arr   = np.array(pose_times, dtype=np.int64)

    def nearest_ego_state(ts: int):
        # Linearly interpolate ego pose between the two surrounding GPS fixes
        # so actor projections don't jump when a new fix arrives (~1 Hz).
        pi = int(np.searchsorted(pose_arr, ts))
        if pi == 0:
            ex, ey, heading = ego_poses[pose_times[0]]
        elif pi >= len(pose_times):
            ex, ey, heading = ego_poses[pose_times[-1]]
        else:
            t0, t1 = pose_times[pi - 1], pose_times[pi]
            alpha = (ts - t0) / (t1 - t0)
            x0, y0, h0 = ego_poses[t0]
            x1, y1, h1 = ego_poses[t1]
            ex = x0 + alpha * (x1 - x0)
            ey = y0 + alpha * (y1 - y0)
            dh = math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))
            heading = h0 + alpha * dh

        oi = int(np.searchsorted(odom_arr, ts))
        oi = min(oi, len(odom_arr) - 1)
        if oi > 0 and abs(odom_arr[oi-1] - ts) < abs(odom_arr[oi] - ts):
            oi -= 1
        _, speed = ego_odoms[odom_times[oi]]
        return ex, ey, heading, speed

    tracker = IoUTracker()
    frame_idx = 0

    reader2 = open_bag(str(bag_path))
    while reader2.has_next():
        topic, data, ts = reader2.read_next()
        if topic != '/perception/image_annotated':
            continue

        msg    = deserialize_message(data, img_type)
        np_arr = np.frombuffer(bytes(msg.data), np.uint8)
        img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            frame_idx += 1
            continue

        results = model(img, verbose=False)[0]
        boxes   = []
        if results.boxes is not None:
            for i in range(len(results.boxes)):
                cls_id = int(results.boxes.cls[i].item())
                if cls_id not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = results.boxes.xyxy[i].tolist()
                boxes.append([x1, y1, x2, y2])

        ex, ey, heading, speed = nearest_ego_state(ts)
        t_s = ts * 1e-9
        tracker.update(boxes, frame_idx, ex, ey, heading, t_s)
        frame_idx += 1

    tracker.finish()
    good_tracks = [t for t in tracker.good_tracks()
                   if len(t.waypoints) >= args.min_track_frames]
    print(f'  Total tracks: {len(tracker._closed)}  '
          f'  Qualifying (>= {args.min_track_frames} frames): {len(good_tracks)}')

    # ── Build and write OpenSCENARIO ──────────────────────────────────────────
    print('Building OpenSCENARIO ...')
    scenario = build_xosc(
        ego_waypoints=ego_waypoints,
        actor_tracks=good_tracks,
        xodr_path=xodr_path,
        lat0=lat0,
        lon0=lon0,
    )

    print(f'Writing {xosc_path}')
    scenario.write_xml(str(xosc_path))

    print('\nDone.')
    print(f'  Ego waypoints : {len(ego_waypoints)}')
    print(f'  Actor tracks  : {len(good_tracks)}')
    print(f'  Output        : {xosc_path}')
    print(f'  Road network  : {xodr_path}')
    print(f'\nVerify in esmini:')
    print(f'  esmini --osc {xosc_path.name} --window 60 60 1200 800')


if __name__ == '__main__':
    main()
