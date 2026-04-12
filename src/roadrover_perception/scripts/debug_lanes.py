#!/usr/bin/env python3
"""
Extracts a single frame from a bag and saves lane detection debug images
so you can see exactly where the pipeline is failing.

Usage:
  python3 debug_lanes.py <bag_path> [--frame 50] [--out-dir /tmp/lane_debug]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

COMPRESSED_TOPIC = '/usb_cam/image_raw/compressed'


def save(path, img, label):
    cv2.imwrite(str(path), img)
    print(f'  saved {label}: {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag_path')
    ap.add_argument('--frame', type=int, default=50,
                    help='Which image frame to extract (0-indexed)')
    ap.add_argument('--out-dir', default='/tmp/lane_debug')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(Path(args.bag_path).resolve()),
                                  storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    frame_idx = 0
    img = None
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != COMPRESSED_TOPIC:
            continue
        if frame_idx == args.frame:
            msg = deserialize_message(data, get_message(type_map[topic]))
            np_arr = np.frombuffer(bytes(msg.data), np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            break
        frame_idx += 1

    if img is None:
        print(f'Frame {args.frame} not found')
        return

    img = cv2.rotate(img, cv2.ROTATE_180)
    h, w = img.shape[:2]
    save(out / '0_original.jpg', img, 'original')

    # ── Step 1: grayscale + CLAHE ────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    save(out / '1_clahe.jpg', enhanced, 'CLAHE enhanced')

    # ── Step 2: blur + Canny ─────────────────────────────────────────────
    blur  = cv2.GaussianBlur(enhanced, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 120)
    save(out / '2_edges.jpg', edges, 'Canny edges')


    # ── Step 3: ROI mask ─────────────────────────────────────────────────
    y_top    = int(h * 0.38)
    y_bottom = int(h * 0.70)
    roi_pts = np.array([
        [0,          y_bottom],
        [int(w * 0.25), y_top],
        [int(w * 0.75), y_top],
        [w - 1,      y_bottom],
    ], np.int32)
    roi_vis = img.copy()
    cv2.polylines(roi_vis, [roi_pts], True, (0, 255, 0), 2)
    save(out / '3_roi.jpg', roi_vis, 'ROI boundary')

    roi_mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(roi_mask, [roi_pts], 255)
    masked = cv2.bitwise_and(edges, roi_mask)
    save(out / '4_masked_edges.jpg', masked, 'masked edges')

    # ── Step 4: Hough lines (raw, before slope filter) ───────────────────
    lines = cv2.HoughLinesP(masked, 1, np.pi / 180, 15,
                            minLineLength=10, maxLineGap=300)
    hough_vis = img.copy()
    if lines is not None:
        print(f'  Hough found {len(lines)} raw lines')
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
    else:
        print('  Hough found NO lines — edges are too sparse')
    save(out / '5_hough_raw.jpg', hough_vis, 'raw Hough lines')

    # ── Step 5: after slope filter ───────────────────────────────────────
    slope_vis = img.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.2 and max(x1, x2) < w * 0.85:
                cv2.line(slope_vis, (x1, y1), (x2, y2), (255, 80, 0), 2)   # left
            elif slope > 0.2 and min(x1, x2) > w * 0.15:
                cv2.line(slope_vis, (x1, y1), (x2, y2), (0, 80, 255), 2)   # right
    save(out / '6_slope_filtered.jpg', slope_vis, 'slope-filtered lines')

    print(f'\nDebug images saved to {out}/')
    print('Check 4_masked_edges.jpg — if it looks empty, edges are not being detected.')
    print('Check 5_hough_raw.jpg   — if empty, Hough threshold is too high.')
    print('Check 6_slope_filtered.jpg — if empty, slope filter is too strict.')


if __name__ == '__main__':
    main()
