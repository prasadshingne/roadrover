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
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CompressedImage, Image

COMPRESSED_TOPIC = '/usb_cam/image_raw/compressed'
RAW_TOPIC        = '/usb_cam/image_raw'
ANNOTATED_TOPIC  = '/perception/image_annotated'
VEL_TOPIC        = '/vel'


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


# ── Lane detection ────────────────────────────────────────────────────────────

HOOD_CUTOFF = 0.70   # below this fraction of height = car hood


class LaneTracker:
    """EMA-smoothed degree-2 polynomial lane tracker."""

    STALE_LIMIT = 15

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.left_poly   = None
        self.right_poly  = None
        self._left_stale  = 0
        self._right_stale = 0

    def update(self, left_pts, right_pts, h, w):
        y_top    = int(h * 0.38)
        y_bottom = int(h * HOOD_CUTOFF)

        lp = self._fit(left_pts,  y_top, y_bottom, w)
        rp = self._fit(right_pts, y_top, y_bottom, w)

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

    def _fit(self, pts, y_top, y_bottom, w):
        if len(pts) < 4:
            return None
        xs, ys = zip(*pts)
        try:
            poly = np.polyfit(ys, xs, 2)
        except np.linalg.LinAlgError:
            return None
        # Only reject wildly out-of-bounds fits
        margin = w * 0.30
        x_bot = np.polyval(poly, y_bottom)
        x_top = np.polyval(poly, y_top)
        if not (-margin <= x_bot <= w + margin):
            return None
        if not (-margin <= x_top <= w + margin):
            return None
        return poly


def _poly_curve(poly, y_bottom, y_top, w, n=40):
    """Evaluate a polynomial along y, return Nx1x2 array for cv2.polylines."""
    ys = np.linspace(y_bottom, y_top, n)
    xs = np.clip(np.polyval(poly, ys), 0, w - 1)
    return np.column_stack([xs, ys]).astype(np.int32).reshape(-1, 1, 2)


def detect_lanes(img: np.ndarray, tracker: LaneTracker) -> np.ndarray:
    """
    Detect lane lines, update the tracker with EMA smoothing, and return
    a black overlay with smooth curved lane lines and filled lane area.
    """
    h, w = img.shape[:2]
    overlay = np.zeros_like(img)

    # CLAHE + grayscale edges — robust in overcast/wet conditions
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur     = cv2.GaussianBlur(enhanced, (7, 7), 0)
    edges    = cv2.Canny(blur, 40, 120)

    # Trapezoid ROI: road surface — below horizon, above car hood
    y_top    = int(h * 0.38)
    y_bottom = int(h * HOOD_CUTOFF)
    roi_pts = np.array([
        [int(w * 0.05), y_bottom],
        [int(w * 0.38), y_top],
        [int(w * 0.62), y_top],
        [int(w * 0.95), y_bottom],
    ], np.int32)
    roi_mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(roi_mask, [roi_pts], 255)
    masked = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(masked, 1, np.pi / 180, 15,
                            minLineLength=10, maxLineGap=300)

    left_pts, right_pts = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            mid_x = (x1 + x2) / 2
            if slope < -0.3 and mid_x < w * 0.60:
                left_pts  += [(x1, y1), (x2, y2)]
            elif slope > 0.3 and mid_x > w * 0.40:
                right_pts += [(x1, y1), (x2, y2)]

    tracker.update(left_pts, right_pts, h, w)

    lp = tracker.left_poly
    rp = tracker.right_poly

    if lp is not None and rp is not None:
        left_curve  = _poly_curve(lp, y_bottom, y_top, w)
        right_curve = _poly_curve(rp, y_bottom, y_top, w)
        # filled polygon between the two curves
        fill_pts = np.vstack([left_curve, right_curve[::-1]])
        cv2.fillPoly(overlay, [fill_pts], (0, 180, 0))
        cv2.polylines(overlay, [left_curve],  False, (255, 80,   0), 3)
        cv2.polylines(overlay, [right_curve], False, (0,   80, 255), 3)
    elif lp is not None:
        cv2.polylines(overlay, [_poly_curve(lp, y_bottom, y_top, w)],
                      False, (255, 80, 0), 3)
    elif rp is not None:
        cv2.polylines(overlay, [_poly_curve(rp, y_bottom, y_top, w)],
                      False, (0, 80, 255), 3)

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

    tracker     = LaneTracker(alpha=0.25)
    speed_ms    = 0.0
    frame_count = 0
    total       = 0

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if topic == VEL_TOPIC:
            msg = deserialize_message(data, get_message(type_map[topic]))
            speed_ms = math.sqrt(
                msg.twist.linear.x ** 2 + msg.twist.linear.y ** 2)
            writer.write(topic, data, timestamp)

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

                # both run on the clean rotated frame
                lane_overlay = detect_lanes(img, tracker)
                det_results  = model(img, verbose=False)[0]

                # filter out detections whose centre is in the car hood region
                if det_results.boxes is not None and len(det_results.boxes):
                    centres_y = det_results.boxes.xyxy[:, 1] + \
                                (det_results.boxes.xyxy[:, 3] - det_results.boxes.xyxy[:, 1]) / 2
                    keep = centres_y < hood_y
                    det_results.boxes = det_results.boxes[keep]

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
