from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='roadrover_stereo',
            executable='stereo_splitter',
            name='stereo_splitter',
            parameters=[{
                'device': '/dev/video4',
                'width':  1280,
                'height': 480,
                'fps':    30,
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