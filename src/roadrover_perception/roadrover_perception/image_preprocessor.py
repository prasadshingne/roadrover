import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Depth=1 so we always process the latest frame and never build a backlog
_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
)


class ImagePreprocessor(Node):
    def __init__(self):
        super().__init__('image_preprocessor')
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/perception/image', 1)
        self.sub = self.create_subscription(
            Image,
            '/usb_cam/image_raw',
            self.callback,
            _QOS,
        )
        self.get_logger().info('Image preprocessor ready — rotating 180°')

    def callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = cv2.rotate(img, cv2.ROTATE_180)
        out = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ImagePreprocessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
