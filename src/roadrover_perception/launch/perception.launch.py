from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='roadrover_perception',
            executable='image_preprocessor',
            name='image_preprocessor',
            output='screen',
        ),
        Node(
            package='roadrover_perception',
            executable='object_detector',
            name='object_detector',
            output='screen',
        ),
    ])
