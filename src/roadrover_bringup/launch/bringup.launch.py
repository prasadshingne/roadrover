from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    video_device_arg = DeclareLaunchArgument(
        'video_device', default_value='/dev/video0',
        description='Video device for usb_cam')

    gps_port_arg = DeclareLaunchArgument(
        'gps_port', default_value='/dev/ttyUSB0',
        description='Serial port for USB GPS')

    gps_baud_arg = DeclareLaunchArgument(
        'gps_baud', default_value='4800',
        description='Baud rate for USB GPS')

    berry_gps_port_arg = DeclareLaunchArgument(
        'berry_gps_port', default_value='/dev/ttyAMA0',
        description='Serial port for BerryGPS')

    berry_gps_baud_arg = DeclareLaunchArgument(
        'berry_gps_baud', default_value='9600',
        description='Baud rate for BerryGPS')

    return LaunchDescription([
        video_device_arg,
        gps_port_arg,
        gps_baud_arg,
        berry_gps_port_arg,
        berry_gps_baud_arg,

        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            namespace='usb_cam',
            parameters=[{
                'video_device': LaunchConfiguration('video_device'),
                'image_width': 640,
                'image_height': 480,
                'framerate': 30.0,
                'pixel_format': 'mjpeg2rgb',
                'camera_name': 'usb_cam',
            }],
            output='screen',
        ),

        Node(
            package='nmea_navsat_driver',
            executable='nmea_serial_driver',
            name='nmea_navsat_driver',
            parameters=[{
                'port': LaunchConfiguration('gps_port'),
                'baud': LaunchConfiguration('gps_baud'),
                'frame_id': 'gps',
                'use_GNSS_time': False,
            }],
            output='screen',
        ),

        Node(
            package='nmea_navsat_driver',
            executable='nmea_serial_driver',
            name='nmea_navsat_driver_berry',
            namespace='berry',
            parameters=[{
                'port': LaunchConfiguration('berry_gps_port'),
                'baud': LaunchConfiguration('berry_gps_baud'),
                'frame_id': 'gps_berry',
                'use_GNSS_time': False,
            }],
            # Relative source name required: node is in 'berry' namespace so its
            # topic is /berry/fix; an absolute '/fix' remapping would never match.
            remappings=[('fix', '/fix_berry')],
            output='screen',
        ),

        Node(
            package='roadrover_imu',
            executable='berry_imu_node',
            name='berry_imu_node',
            parameters=[{
                'i2c_bus': 1,
                'imu_rate': 100.0,
                'baro_rate': 10.0,
                'frame_id': 'imu',
            }],
            output='screen',
        ),

        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            parameters=[{'port': 8765}],
            output='screen',
        ),
    ])
