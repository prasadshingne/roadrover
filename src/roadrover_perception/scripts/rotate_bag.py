#!/usr/bin/env python3
"""
Rotates all camera image frames 180° in a rosbag2 and saves a new bag.
All other topics (GPS, velocity, etc.) are copied byte-for-byte unchanged.

Usage:
  python3 rotate_bag.py <input_bag_path>
  python3 rotate_bag.py <input_bag_path> --output <output_bag_path>
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message

IMAGE_TOPICS = {'/usb_cam/image_raw', '/usb_cam/image_raw/compressed'}


def rotate_image_msg(data: bytes, msg_type_str: str) -> bytes:
    msg_type = get_message(msg_type_str)
    msg = deserialize_message(data, msg_type)

    if msg_type_str == 'sensor_msgs/msg/Image':
        channels = msg.step // msg.width
        img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
            msg.height, msg.width, channels
        )
        img = cv2.rotate(img, cv2.ROTATE_180)
        msg.data = img.tobytes()

    elif msg_type_str == 'sensor_msgs/msg/CompressedImage':
        np_arr = np.frombuffer(bytes(msg.data), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.rotate(img, cv2.ROTATE_180)
            ok, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                msg.data = encoded.tobytes()

    return serialize_message(msg)


def main():
    ap = argparse.ArgumentParser(description='Rotate camera images 180° in a rosbag2')
    ap.add_argument('input_bag', help='Path to input rosbag2 directory')
    ap.add_argument('--output', default=None,
                    help='Output bag path (default: <input>_rotated)')
    args = ap.parse_args()

    input_path = str(Path(args.input_bag).resolve())
    output_path = args.output or input_path.rstrip('/') + '_rotated'

    print(f'Input:  {input_path}')
    print(f'Output: {output_path}')

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_path, storage_id='sqlite3'),
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
            name=t.name,
            type=t.type,
            serialization_format='cdr',
        ))

    total = 0
    frame_count = 0
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic in IMAGE_TOPICS:
            data = rotate_image_msg(data, type_map[topic])
            frame_count += 1
        writer.write(topic, data, timestamp)
        total += 1
        if total % 500 == 0:
            print(f'  {total} messages, {frame_count} frames rotated...')

    del writer  # flushes and closes
    print(f'Done — {total} messages written, {frame_count} frames rotated → {output_path}')


if __name__ == '__main__':
    main()
