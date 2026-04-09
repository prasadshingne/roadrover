# roadrover

A ROS 2 Humble project for data collection, perception, lane detection, and localization, intended to run on a Raspberry Pi.

## Hardware

| Device | Default path |
|--------|-------------|
| USB camera | `/dev/video4` |
| GPS receiver (NMEA serial) | `/dev/ttyUSB0` |

## Prerequisites

- Ubuntu 22.04 with ROS 2 Humble installed
- The following ROS 2 packages available in your workspace or installed via apt:

```bash
sudo apt install ros-humble-usb-cam \
                 ros-humble-nmea-navsat-driver \
                 ros-humble-foxglove-bridge
```

> **Raspberry Pi note:** run the same apt command after installing ROS 2 Humble on the Pi.

## Setup

### 1. Clone the repository

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone git@github.com:prasadshingne/roadrover.git
```

### 2. Build

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

### 3. Source the workspace

```bash
source ~/ros2_ws/install/setup.bash
```

Add this line to `~/.bashrc` to source automatically on every terminal:

```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Usage

### Launch everything

```bash
ros2 launch roadrover_bringup bringup.launch.py
```

This starts three nodes:

| Node | Package | Topic(s) |
|------|---------|----------|
| `usb_cam` | `usb_cam` | `/usb_cam/image_raw`, `/usb_cam/camera_info` |
| `nmea_navsat_driver` | `nmea_navsat_driver` | `/fix`, `/vel`, `/time_reference` |
| `foxglove_bridge` | `foxglove_bridge` | WebSocket on port **8765** |

### Visualize with Foxglove Studio

1. Open [Foxglove Studio](https://app.foxglove.dev) in a browser or the desktop app.
2. Click **Open connection** → **Foxglove WebSocket**.
3. Enter `ws://<device-ip>:8765` (use `ws://localhost:8765` if running locally).
4. Subscribe to `/usb_cam/image_raw` for camera feed and `/fix` for GPS.

## Recording and replay

### Record a session

```bash
ros2 launch roadrover_bringup record.launch.py
```

Starts all sensors and records the following topics to `~/roadrover_bags/session_<timestamp>/`:

| Topic | Content |
|-------|---------|
| `/usb_cam/image_raw` | Raw camera frames |
| `/usb_cam/image_raw/compressed` | Compressed camera frames |
| `/usb_cam/camera_info` | Camera calibration |
| `/fix` | GPS position (NavSatFix) |
| `/vel` | GPS velocity (TwistStamped) |
| `/time_reference` | GPS time |

Stop recording with **Ctrl-C**. The bag is fully written before the process exits.

### Replay a session

```bash
ros2 launch roadrover_bringup replay.launch.py bag_path:=/path/to/session_<timestamp>
```

Plays the bag back and starts foxglove_bridge on port 8765 so you can inspect it in Foxglove Studio. The `--clock` flag is passed automatically so nodes that use `/clock` stay in sync with the recorded timeline.

## Changing device paths

If your camera or GPS receiver is on a different device node, edit the parameters in
[src/roadrover_bringup/launch/bringup.launch.py](src/roadrover_bringup/launch/bringup.launch.py):

```python
# Camera
'video_device': '/dev/video4',   # change to your device, e.g. /dev/video0

# GPS
'port': '/dev/ttyUSB0',          # change to your device, e.g. /dev/ttyUSB1
'baud': 4800,                    # change if your receiver uses a different baud rate
```

Rebuild after any changes:

```bash
colcon build --symlink-install --packages-select roadrover_bringup
```

## Troubleshooting

**Camera not found**
```bash
v4l2-ctl --list-devices
```
Find your camera and update `video_device` accordingly.

**GPS not found**
```bash
ls /dev/ttyUSB*
```
Update `port` accordingly. You may also need to add yourself to the `dialout` group:
```bash
sudo usermod -aG dialout $USER   # log out and back in after this
```

**Cannot open Foxglove connection**
- Confirm the bridge is running: `ros2 node list | grep foxglove`
- Check that port 8765 is not blocked by a firewall.

## License

Apache-2.0
