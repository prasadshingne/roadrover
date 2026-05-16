#include <atomic>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

class StereoSplitter : public rclcpp::Node
{
public:
    StereoSplitter() : Node("stereo_splitter"), running_(false)
    {
        declare_parameter("device",       "/dev/video4");
        declare_parameter("width",        1280);
        declare_parameter("height",       480);
        declare_parameter("fps",          30);
        // jpeg_quality: 80 balances file size and image quality at 30/60 fps;
        // lower values (e.g. 60) reduce encode time if CPU becomes the bottleneck.
        declare_parameter("jpeg_quality", 80);

        auto device       = get_parameter("device").as_string();
        int  width        = get_parameter("width").as_int();
        int  height       = get_parameter("height").as_int();
        int  fps          = get_parameter("fps").as_int();
        jpeg_quality_     = get_parameter("jpeg_quality").as_int();

        eye_w_ = width / 2;
        eye_h_ = height;

        cap_.open(device, cv::CAP_V4L2);
        cap_.set(cv::CAP_PROP_FOURCC,    cv::VideoWriter::fourcc('M','J','P','G'));
        cap_.set(cv::CAP_PROP_FRAME_WIDTH,  width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap_.set(cv::CAP_PROP_FPS,          static_cast<double>(fps));
        // 2 buffers: camera fills the next buffer while we encode the current one.
        // BUFFERSIZE=1 limits delivery to ~15 fps because the camera stalls while
        // we hold the only buffer; 2 restores full 30/60 fps pipelining.
        cap_.set(cv::CAP_PROP_BUFFERSIZE,   2);

        if (!cap_.isOpened()) {
            RCLCPP_ERROR(get_logger(), "Cannot open camera %s", device.c_str());
            throw std::runtime_error("Cannot open " + device);
        }

        int actual_w = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        int actual_h = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        RCLCPP_INFO(get_logger(),
            "Opened %s — requested %dx%d, got %dx%d @ %d fps",
            device.c_str(), width, height, actual_w, actual_h, fps);

        auto qos = rclcpp::QoS(10);
        pub_l_comp_ = create_publisher<sensor_msgs::msg::CompressedImage>(
            "/stereo/left/image_raw/compressed",  qos);
        pub_l_info_ = create_publisher<sensor_msgs::msg::CameraInfo>(
            "/stereo/left/camera_info",            qos);
        pub_r_comp_ = create_publisher<sensor_msgs::msg::CompressedImage>(
            "/stereo/right/image_raw/compressed", qos);
        pub_r_info_ = create_publisher<sensor_msgs::msg::CameraInfo>(
            "/stereo/right/camera_info",           qos);

        left_info_  = make_camera_info("stereo_left");
        right_info_ = make_camera_info("stereo_right");

        running_ = true;
        thread_  = std::thread(&StereoSplitter::capture_loop, this);
        RCLCPP_INFO(get_logger(), "Stereo splitter ready.");
    }

    ~StereoSplitter()
    {
        running_ = false;
        if (thread_.joinable()) thread_.join();
        cap_.release();
    }

private:
    sensor_msgs::msg::CameraInfo make_camera_info(const std::string & frame_id)
    {
        sensor_msgs::msg::CameraInfo info;
        info.header.frame_id  = frame_id;
        info.width            = static_cast<uint32_t>(eye_w_);
        info.height           = static_cast<uint32_t>(eye_h_);
        info.distortion_model = "plumb_bob";
        info.d = {0.0, 0.0, 0.0, 0.0, 0.0};
        info.k = {0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0};
        info.r = {1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0};
        info.p = {0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0};
        return info;
    }

    sensor_msgs::msg::CompressedImage encode_jpeg(
        const cv::Mat & img, const std_msgs::msg::Header & header)
    {
        std::vector<uchar> buf;
        cv::imencode(".jpg", img, buf, {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_});

        sensor_msgs::msg::CompressedImage msg;
        msg.header = header;
        msg.format = "jpeg";
        msg.data   = std::move(buf);
        return msg;
    }

    void capture_loop()
    {
        cv::Mat frame;
        while (running_) {
            if (!cap_.read(frame) || frame.empty()) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Frame capture failed");
                continue;
            }

            auto stamp = now();

            std_msgs::msg::Header h_left, h_right;
            h_left.stamp    = stamp;  h_left.frame_id  = "stereo_left";
            h_right.stamp   = stamp;  h_right.frame_id = "stereo_right";

            cv::Mat left  = frame(cv::Rect(0,      0, eye_w_, eye_h_));
            cv::Mat right = frame(cv::Rect(eye_w_, 0, eye_w_, eye_h_));

            pub_l_comp_->publish(encode_jpeg(left,  h_left));
            left_info_.header = h_left;
            pub_l_info_->publish(left_info_);

            pub_r_comp_->publish(encode_jpeg(right, h_right));
            right_info_.header = h_right;
            pub_r_info_->publish(right_info_);
        }
    }

    int  eye_w_, eye_h_, jpeg_quality_;
    cv::VideoCapture cap_;
    std::atomic<bool> running_;
    std::thread thread_;

    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_l_comp_, pub_r_comp_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr       pub_l_info_, pub_r_info_;

    sensor_msgs::msg::CameraInfo left_info_, right_info_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StereoSplitter>());
    rclcpp::shutdown();
    return 0;
}
