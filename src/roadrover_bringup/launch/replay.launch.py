from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        description='Absolute path to the rosbag directory to replay',
    )

    return LaunchDescription([
        bag_path_arg,
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            parameters=[{'port': 8765}],
            output='screen',
        ),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_path'), '--clock'],
            output='screen',
        ),
    ])
