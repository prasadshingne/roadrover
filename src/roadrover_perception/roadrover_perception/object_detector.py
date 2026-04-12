import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
)


class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.bridge = CvBridge()

        self.get_logger().info('Loading YOLOv8s model on GPU...')
        # Import here so ROS node starts and logs before the potentially slow model load
        from ultralytics import YOLO
        self.model = YOLO('yolov8s.pt')
        self.model.to('cuda')
        self.get_logger().info('YOLOv8s ready')

        self.pub = self.create_publisher(Image, '/perception/image_annotated', 1)
        self.sub = self.create_subscription(
            Image,
            '/perception/image',
            self.callback,
            _QOS,
        )

    def callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(img, verbose=False)[0]
        annotated = results.plot()
        out = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
