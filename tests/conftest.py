"""
Stub out all ROS 2 runtime dependencies so perception scripts can be
imported in a plain Python / pytest environment without a ROS install.
This runs at collection time, before any test module is imported.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

_ROS_MODULES = [
    'rosbag2_py',
    'rclpy', 'rclpy.serialization',
    'geometry_msgs', 'geometry_msgs.msg',
    'tf2_msgs', 'tf2_msgs.msg',
    'nav_msgs', 'nav_msgs.msg',
    'rosidl_runtime_py', 'rosidl_runtime_py.utilities',
    'sensor_msgs', 'sensor_msgs.msg',
    'std_msgs', 'std_msgs.msg',
    'visualization_msgs', 'visualization_msgs.msg',
]
for _mod in _ROS_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Make perception scripts importable by module name.
_SCRIPTS = Path(__file__).parent.parent / 'src' / 'roadrover_perception' / 'scripts'
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
