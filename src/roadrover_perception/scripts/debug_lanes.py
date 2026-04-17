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
    y_top    = int(h * 0.45)
    y_bottom = int(h * 0.63)
    roi_pts = np.array([
        [int(w * 0.05), y_bottom],
        [int(w * 0.28), y_top],
        [int(w * 0.65), y_top],
        [int(w * 0.95), y_bottom],
    ], np.int32)
    roi_vis = img.copy()
    cv2.polylines(roi_vis, [roi_pts], True, (0, 255, 0), 2)
    save(out / '3_roi.jpg', roi_vis, 'ROI boundary')

    roi_mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(roi_mask, [roi_pts], 255)
    masked = cv2.bitwise_and(edges, roi_mask)
    save(out / '4_masked_edges.jpg', masked, 'masked edges')

    # ── Step 4: BEV warp ─────────────────────────────────────────────────
    BEV_W, BEV_H, BEV_MARGIN = 400, 300, 60
    src = np.float32([
        [w * 0.05, y_bottom], [w * 0.28, y_top],
        [w * 0.65, y_top],    [w * 0.95, y_bottom],
    ])
    dst = np.float32([
        [BEV_MARGIN,         BEV_H],
        [BEV_MARGIN,         0],
        [BEV_W - BEV_MARGIN, 0],
        [BEV_W - BEV_MARGIN, BEV_H],
    ])
    M_bev = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    bev_edges = cv2.warpPerspective(masked, M_bev, (BEV_W, BEV_H))
    bev_dilated = cv2.dilate(bev_edges, np.ones((3, 3), np.uint8), iterations=1)
    save(out / '5_bev_edges.jpg', bev_dilated, 'BEV edges (dilated)')

    # BEV colour image for overlay
    bev_color = cv2.warpPerspective(img, M_bev, (BEV_W, BEV_H))

    # ── Step 5: sliding windows in BEV ───────────────────────────────────
    n_windows, margin_sw, minpix = 9, 30, 10
    mid = BEV_W // 2
    win_h = BEV_H // n_windows

    hist_l = np.sum(bev_dilated[BEV_H // 2:, :mid],  axis=0).astype(np.float32)
    hist_r = np.sum(bev_dilated[BEV_H // 2:, mid:],  axis=0).astype(np.float32)
    leftx_cur  = int(np.argmax(hist_l)) if hist_l.max() >= minpix else None
    rightx_cur = int(np.argmax(hist_r)) + mid if hist_r.max() >= minpix else None
    print(f'  Histogram seeds — left: {leftx_cur}, right: {rightx_cur}')

    nz  = bev_dilated.nonzero()
    nzy = np.array(nz[0])
    nzx = np.array(nz[1])

    sw_vis = cv2.cvtColor(bev_dilated, cv2.COLOR_GRAY2BGR)
    left_pts, right_pts = [], []

    for win in range(n_windows):
        y_lo = BEV_H - (win + 1) * win_h
        y_hi = BEV_H - win * win_h

        if leftx_cur is not None:
            cv2.rectangle(sw_vis, (leftx_cur - margin_sw, y_lo),
                          (leftx_cur + margin_sw, y_hi), (255, 80, 0), 1)
            inds = ((nzy >= y_lo) & (nzy < y_hi) &
                    (nzx >= leftx_cur - margin_sw) &
                    (nzx < leftx_cur + margin_sw)).nonzero()[0]
            if len(inds) >= minpix:
                xs = nzx[inds]
                left_pts.extend(zip(xs.tolist(), nzy[inds].tolist()))
                leftx_cur = int(np.clip(int(np.mean(xs)), BEV_MARGIN, mid))

        if rightx_cur is not None:
            cv2.rectangle(sw_vis, (rightx_cur - margin_sw, y_lo),
                          (rightx_cur + margin_sw, y_hi), (0, 80, 255), 1)
            inds = ((nzy >= y_lo) & (nzy < y_hi) &
                    (nzx >= rightx_cur - margin_sw) &
                    (nzx < rightx_cur + margin_sw)).nonzero()[0]
            if len(inds) >= minpix:
                xs = nzx[inds]
                right_pts.extend(zip(xs.tolist(), nzy[inds].tolist()))
                rightx_cur = int(np.clip(int(np.mean(xs)), mid, BEV_W - BEV_MARGIN))

    print(f'  BEV sliding window — left pixels: {len(left_pts)}, right pixels: {len(right_pts)}')
    save(out / '6_bev_windows.jpg', sw_vis, 'BEV sliding windows')

    # ── Step 6: polynomial fit + warp back ───────────────────────────────
    overlay = img.copy()

    def fit_and_draw(pts, side, color):
        if len(pts) < 6:
            print(f'  {side}: too few points ({len(pts)}), skipping fit')
            return
        xs, ys = zip(*pts)
        xs, ys = np.array(xs, float), np.array(ys, float)
        if (ys.max() - ys.min()) < BEV_H * 0.20:
            print(f'  {side}: insufficient y-spread, skipping fit')
            return
        poly = np.polyfit(ys, xs, 2)
        print(f'  {side} poly: {poly}  (a={poly[0]:.5f})')
        # draw in BEV
        bev_ys = np.linspace(0, BEV_H - 1, 50)
        bev_xs = np.clip(np.polyval(poly, bev_ys), 0, BEV_W - 1)
        bev_curve = np.column_stack([bev_xs, bev_ys]).astype(np.float32).reshape(-1, 1, 2)
        cv2.polylines(sw_vis, [bev_curve.astype(np.int32)], False, color, 2)
        # warp back to image space
        img_pts = cv2.perspectiveTransform(bev_curve, M_inv).astype(np.int32)
        cv2.polylines(overlay, [img_pts], False, color, 3)

    fit_and_draw(left_pts,  'left',  (255, 80,   0))
    fit_and_draw(right_pts, 'right', (0,   80, 255))
    save(out / '7_bev_fit.jpg',    sw_vis,  'BEV fit + windows')
    save(out / '8_lane_overlay.jpg', overlay, 'final lane overlay')

    print(f'\nDebug images saved to {out}/')
    print('Check 5_bev_edges.jpg    — edges in bird\'s-eye view.')
    print('Check 6_bev_windows.jpg  — sliding windows (blue=left, orange=right).')
    print('Check 7_bev_fit.jpg      — polynomial fit in BEV.')
    print('Check 8_lane_overlay.jpg — lanes warped back to image.')


if __name__ == '__main__':
    main()
