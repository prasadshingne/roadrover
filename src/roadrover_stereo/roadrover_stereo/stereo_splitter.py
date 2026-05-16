import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Header


class StereoSplitter(Node):
    def __init__(self):
        super().__init__('stereo_splitter')

        self.declare_parameter('device', '/dev/video4')
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)

        device = self.get_parameter('device').value
        width  = self.get_parameter('width').value
        height = self.get_parameter('height').value
        fps    = self.get_parameter('fps').value

        self._eye_w = width // 2
        self._eye_h = height

        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, float(fps))
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            self.get_logger().error(f'Cannot open camera {device}')
            raise RuntimeError(f'Cannot open {device}')

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f'Opened {device} — requested {width}x{height}, got {actual_w}x{actual_h} @ {fps} fps')

        qos = 10
        self._pub_l_comp = self.create_publisher(CompressedImage, '/stereo/left/image_raw/compressed',  qos)
        self._pub_l_info = self.create_publisher(CameraInfo,      '/stereo/left/camera_info',           qos)
        self._pub_r_comp = self.create_publisher(CompressedImage, '/stereo/right/image_raw/compressed', qos)
        self._pub_r_info = self.create_publisher(CameraInfo,      '/stereo/right/camera_info',          qos)

        self._left_info  = self._make_camera_info('stereo_left')
        self._right_info = self._make_camera_info('stereo_right')

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self.get_logger().info('Stereo splitter ready.')

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_camera_info(self, frame_id: str) -> CameraInfo:
        info = CameraInfo()
        info.header.frame_id  = frame_id
        info.width            = self._eye_w
        info.height           = self._eye_h
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = [0.0] * 9
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.p = [0.0] * 12
        return info

    def _stamp(self) -> Header:
        ns = self.get_clock().now().nanoseconds
        h = Header()
        h.stamp.sec     = ns // 1_000_000_000
        h.stamp.nanosec = ns %  1_000_000_000
        return h

    def _compressed_msg(self, img: np.ndarray, header: Header) -> CompressedImage:
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        msg        = CompressedImage()
        msg.header = header
        msg.format = 'jpeg'
        msg.data   = buf.tobytes()
        return msg

    # ── capture loop ──────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self.get_logger().warn('Frame capture failed', throttle_duration_sec=5.0)
                continue

            left  = frame[:, :self._eye_w]
            right = frame[:, self._eye_w:]

            h_left           = self._stamp()
            h_left.frame_id  = 'stereo_left'
            h_right          = self._stamp()
            h_right.frame_id = 'stereo_right'

            self._pub_l_comp.publish(self._compressed_msg(left, h_left))
            self._left_info.header = h_left
            self._pub_l_info.publish(self._left_info)

            self._pub_r_comp.publish(self._compressed_msg(right, h_right))
            self._right_info.header = h_right
            self._pub_r_info.publish(self._right_info)

    def destroy_node(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StereoSplitter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
