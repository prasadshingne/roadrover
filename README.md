# roadrover

A ROS 2 Humble project for data collection, offline perception, lane detection, object detection, and ego state estimation, intended to run on a Raspberry Pi.

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

## Offline perception pipeline

`src/roadrover_perception/scripts/process_bag.py` reads an original recorded bag, runs the full perception stack, and writes a new bag:

```bash
# From the repo root (YOLOv8s weights must be present as yolov8s.pt)
python3 src/roadrover_perception/scripts/process_bag.py ~/roadrover_bags/session_<timestamp>

# Optionally specify the output path
python3 src/roadrover_perception/scripts/process_bag.py <bag> --output <out_bag>
```

### What it does

| Step | Detail |
|------|--------|
| Image rotation | All camera frames rotated 180° in-place |
| Object detection | YOLOv8s on GPU; detections in the car hood region are filtered out |
| Lane detection | Bird's-eye view (BEV) perspective warp + sliding window search; EMA-smoothed degree-2 polynomial per lane |
| Ego state estimation | Heading, yaw rate, longitudinal and lateral acceleration derived from GPS velocity |

### Output topics

| Topic | Type | Content |
|-------|------|---------|
| `/perception/image_annotated` | `CompressedImage` | YOLO boxes + lane overlay + speed |
| `/ego/odometry` | `nav_msgs/Odometry` | Heading (quaternion), speed, yaw rate |
| `/ego/imu` | `sensor_msgs/Imu` | Yaw rate, longitudinal accel, lateral accel |

### Ego state signals

All signals are derived from the GPS `/vel` topic (no IMU on this rover):

| Signal | Source | Method |
|--------|--------|--------|
| Heading | `/vel` east/north components | `atan2(vy_north, vx_east)` |
| Yaw rate | Heading | Finite diff + EMA smoothing |
| Longitudinal accel | Speed | `d(speed)/dt` + EMA |
| Lateral accel | Speed + yaw rate | `speed × yaw_rate` (centripetal) |

### Viewing in Foxglove

Open the processed bag in Foxglove Studio (File → Open local file). Useful panel configurations:

- **Image** panel → `/perception/image_annotated` — annotated video with lanes and YOLO boxes
- **Map** panel → `/fix` — GPS track on a satellite map
- **Plot** panel — add series for time-series signals:

| Signal | Topic | Field path |
|--------|-------|------------|
| Speed (m/s) | `/ego/odometry` | `twist.twist.linear.x` |
| Yaw rate (rad/s) | `/ego/odometry` | `twist.twist.angular.z` |
| Longitudinal accel | `/ego/imu` | `linear_acceleration.x` |
| Lateral accel | `/ego/imu` | `linear_acceleration.y` |

### Lane detection debug tool

Inspect the BEV pipeline on a single frame without running the full bag:

```bash
python3 src/roadrover_perception/scripts/debug_lanes.py <bag_path> --frame 50 --out-dir /tmp/lane_debug
```

Output images in `/tmp/lane_debug/`:

| File | Content |
|------|---------|
| `0_original.jpg` | Raw rotated frame |
| `1_clahe.jpg` | CLAHE-enhanced grayscale |
| `2_edges.jpg` | Canny edges |
| `3_roi.jpg` | Trapezoid ROI boundary |
| `4_masked_edges.jpg` | Edges inside ROI |
| `5_bev_edges.jpg` | Edges warped to bird's-eye view |
| `6_bev_windows.jpg` | Sliding windows in BEV |
| `7_bev_fit.jpg` | Polynomial fit in BEV |
| `8_lane_overlay.jpg` | Lanes warped back to image space |

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
