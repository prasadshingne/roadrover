# RoadRover — Raspberry Pi 4 Setup Guide

← [Back to main README](../README.md)

End-to-end setup for running RoadRover data recording on a Raspberry Pi 4 with a USB camera and GPS receiver. Covers OS installation, ROS 2, device configuration, WiFi hotspot, and common issues found during bring-up.

## Contents

1. [Hardware](#hardware)
2. [Flash Ubuntu 22.04 Server](#1-flash-ubuntu-2204-server-arm64)
3. [First boot](#2-first-boot)
4. [Enable SSH password authentication](#3-enable-ssh-password-authentication)
5. [System update](#4-system-update)
6. [Install ROS 2 Humble](#5-install-ros-2-humble)
7. [Install ROS 2 packages](#6-install-ros-2-packages)
8. [Install RoadRover](#7-install-roadrover)
9. [Fix camera device path](#8-fix-camera-device-path)
10. [Fix device permissions](#9-fix-device-permissions)
11. [WiFi hotspot setup](#10-wifi-hotspot-setup-iphone--mobile-hotspot)
12. [Running RoadRover](#11-running-roadrover)
13. [Replaying bags on a laptop](#12-replaying-bags-on-a-laptop)
14. [Troubleshooting](#troubleshooting)

---

## Hardware

| Component | Spec used |
|-----------|-----------|
| Raspberry Pi 4 Model B | 8 GB RAM |
| Micro SD card | **64 GB minimum** — see [SD card sizing](#sd-card-sizing) |
| USB camera | Logitech C920 (appears as `/dev/video0`) |
| GPS receiver | NMEA serial over USB (`/dev/ttyUSB0`, 4800 baud) |

### SD card sizing

Recording compressed video + GPS at 30 fps uses ~2 MB/s. A 64 GB card gives roughly 8+ hours of recording capacity after the OS and ROS install (~8 GB). Do not use a 32 GB card — the OS + ROS alone takes ~8 GB, leaving almost no headroom.

> **Raspbian is not supported.** ROS 2 Humble requires Ubuntu 22.04. Raspbian Buster (Debian 10) and Bullseye (Debian 11) do not have `ros-humble-*` packages in apt. Do not attempt to install ROS 2 on Raspbian.

---

## 1. Flash Ubuntu 22.04 Server (arm64)

Download the Ubuntu 22.04 Server image for Raspberry Pi:

```
ubuntu-22.04.5-preinstalled-server-arm64+raspi.img.xz
```

Flash from a Linux laptop (replace `/dev/sdc` with your SD card device — verify with `lsblk`):

```bash
xz -dc ubuntu-22.04.5-preinstalled-server-arm64+raspi.img.xz | \
  sudo dd of=/dev/sdc bs=4M status=progress conv=fsync
```

On **Windows/macOS** use [Raspberry Pi Imager](https://www.raspberrypi.com/software/) → Other general-purpose OS → Ubuntu → Ubuntu Server 22.04 LTS (64-bit).

---

## 2. First boot

Default credentials: username `ubuntu`, password `ubuntu`. You are forced to change the password on first login.

Connect a monitor and keyboard for first boot, or connect over serial. Once you have an IP address you can switch to SSH.

---

## 3. Enable SSH password authentication

Ubuntu Server 22.04 on the Pi ships with password SSH disabled via cloud-init override files. Both files must be changed:

```bash
sudo sed -i 's/PasswordAuthentication no/PasswordAuthentication yes/' \
  /etc/ssh/sshd_config.d/50-cloud-init.conf

sudo sed -i 's/PasswordAuthentication no/PasswordAuthentication yes/' \
  /etc/ssh/sshd_config.d/60-cloudimg-settings.conf

sudo systemctl restart ssh
```

You can now SSH from your laptop:

```bash
ssh ubuntu@<pi-ip>
```

---

## 4. System update

```bash
sudo apt update && sudo apt upgrade -y
```

---

## 5. Install ROS 2 Humble

```bash
sudo apt install -y software-properties-common curl

curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  | sudo apt-key add -

sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu jammy main" \
  > /etc/apt/sources.list.d/ros2.list'

sudo apt update
sudo apt install -y ros-humble-ros-base
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-argcomplete

sudo rosdep init
rosdep update

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 6. Install ROS 2 packages

```bash
sudo apt install -y \
  ros-humble-usb-cam \
  ros-humble-nmea-navsat-driver \
  ros-humble-foxglove-bridge \
  ros-humble-rosbag2 \
  ros-humble-rosbag2-storage-default-plugins
```

---

## 7. Install RoadRover

```bash
cd ~
git clone https://github.com/prasadshingne/roadrover.git
cd ~/roadrover
source /opt/ros/humble/setup.bash
colcon build --symlink-install
echo "source ~/roadrover/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 8. Fix camera device path

The default launch files reference `/dev/video4`. On a Pi 4 with a single USB camera, the device is almost always `/dev/video0`. Update both launch files:

```bash
sed -i "s|/dev/video4|/dev/video0|g" \
  ~/roadrover/src/roadrover_bringup/launch/bringup.launch.py

sed -i "s|/dev/video4|/dev/video0|g" \
  ~/roadrover/src/roadrover_bringup/launch/record.launch.py

cd ~/roadrover
colcon build --symlink-install
```

Verify the camera is visible and supports MJPEG before starting:

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext | grep -A2 MJPEG
```

---

## 9. Fix device permissions

Add your user to the `dialout` group (GPS serial) and `video` group (camera). Log out and back in after:

```bash
sudo usermod -aG dialout $USER
sudo usermod -aG video $USER
```

---

## 10. WiFi hotspot setup (iPhone / mobile hotspot)

Create the netplan config:

```bash
sudo nano /etc/netplan/99-wifi.yaml
```

```yaml
network:
  version: 2
  wifis:
    wlan0:
      dhcp4: true
      access-points:
        "YourHotspotName":
          password: "YourHotspotPassword"
```

Apply:

```bash
sudo chmod 600 /etc/netplan/99-wifi.yaml
sudo netplan apply
```

The Pi picks up an IP in the `172.20.10.x` range on an iPhone hotspot. Find it from another device on the same hotspot:

```bash
# Linux/macOS
nmap -sn 172.20.10.0/28

# Windows
arp -a
```

---

## 11. Running RoadRover

**Live preview only (Foxglove, no recording):**

```bash
ros2 launch roadrover_bringup bringup.launch.py
```

**Record a session (camera + GPS + Foxglove + rosbag):**

```bash
ros2 launch roadrover_bringup record.launch.py
```

Bags are saved to `~/roadrover_bags/session_YYYYMMDD_HHMMSS/` and auto-split into 30-second files.

Stop with **Ctrl-C**.

**View live data in Foxglove Studio:**

1. Open [Foxglove Studio](https://app.foxglove.dev)
2. Open connection → Foxglove WebSocket → `ws://<pi-ip>:8765`

---

## 12. Replaying bags on a laptop

On the laptop (with ROS 2 Humble and foxglove_bridge installed):

```bash
# Terminal 1 — start bridge
ros2 run foxglove_bridge foxglove_bridge

# Terminal 2 — play bag
ros2 bag play ~/path/to/session_folder
```

Connect Foxglove Studio to `ws://localhost:8765`.

---

## Troubleshooting

### Recording stops unexpectedly / usb_cam crashes

The recorder was originally configured to record both `/usb_cam/image_raw` (uncompressed) and `/usb_cam/image_raw/compressed`. The raw stream at 640×480 RGB 30 fps requires ~27 MB/s sustained — at or above the Pi 4 SD card write limit — causing the rosbag2 write queue to overflow and crash the recorder or the camera node.

**Fix:** The current `record.launch.py` records only the compressed topic (~2 MB/s). If you see this issue with an older checkout, verify `/usb_cam/image_raw` is **not** in the topics list.

### Bag missing metadata.yaml (recorded session won't play)

If the recorder is killed hard (power cut, OOM), `metadata.yaml` may not be written. Recover with:

```bash
ros2 bag reindex ~/roadrover_bags/session_<timestamp> --storage sqlite3
```

Then replay as normal.

### ROS_DISTRO mixing warning at startup

If you have both ROS 1 Noetic and ROS 2 Humble sourced (e.g. both in `.bashrc`), you will see:

```
ROS_DISTRO was set to 'noetic' before...
```

This is harmless for ROS 2-only workloads but can cause subtle issues. Open a fresh terminal and source only the Humble workspace:

```bash
source /opt/ros/humble/setup.bash
source ~/roadrover/install/setup.bash
```

### GPS not producing velocity

The GPS receiver publishes `/vel` once it has a satellite fix and the vehicle is moving. The topic will be present but all-zero while stationary or before fix acquisition. Outdoor, cold-start fix typically takes 1–3 minutes. The `/fix` topic's `status.status` field will be `-1` (no fix) until this completes.

### Camera MJPEG check

The `pixel_format: mjpeg2rgb` setting in the launch file requires the camera to support MJPEG output. Verify:

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

If only `YUYV` is listed, change `pixel_format` to `yuyv2rgb` in the launch files and rebuild.
