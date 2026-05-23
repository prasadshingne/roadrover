#!/usr/bin/env python3
"""
BerryGPS-IMU-4 ROS 2 node.
Reads LSM9DS1 (accel/gyro at 0x6A) and LPS22 (barometer at 0x77)
via I2C and publishes sensor_msgs/Imu and sensor_msgs/FluidPressure.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, FluidPressure
from builtin_interfaces.msg import Time
import smbus2
import math
import time

# I2C addresses
LSM9DS1_AG_ADDR = 0x6A   # accelerometer + gyroscope
LPS22_ADDR      = 0x77   # barometer

# LSM9DS1 accel/gyro registers
AG_CTRL_REG1_G  = 0x10   # gyro control
AG_CTRL_REG6_XL = 0x20   # accel control
AG_OUT_X_L_G    = 0x18   # gyro output start
AG_OUT_X_L_XL   = 0x28   # accel output start

# LPS22 registers
LPS22_CTRL_REG1 = 0x10
LPS22_PRES_OUT_XL = 0x28

# Scale factors
GYRO_SCALE  = math.radians(245.0 / 32768.0)   # 245 dps full scale -> rad/s
ACCEL_SCALE = 2.0 * 9.81 / 32768.0             # 2g full scale -> m/s^2
BARO_SCALE  = 1.0 / 4096.0                     # hPa per LSB


def read_word_signed(bus, addr, reg):
    low  = bus.read_byte_data(addr, reg)
    high = bus.read_byte_data(addr, reg + 1)
    val  = (high << 8) | low
    if val >= 0x8000:
        val -= 0x10000
    return val


class BerryImuNode(Node):

    def __init__(self):
        super().__init__('berry_imu_node')

        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('imu_rate', 100.0)   # Hz
        self.declare_parameter('baro_rate', 10.0)   # Hz
        self.declare_parameter('frame_id', 'imu')

        bus_num  = self.get_parameter('i2c_bus').value
        imu_rate = self.get_parameter('imu_rate').value
        baro_rate = self.get_parameter('baro_rate').value
        self.frame_id = self.get_parameter('frame_id').value

        self.bus = smbus2.SMBus(bus_num)
        self._init_imu()
        self._init_baro()

        self.imu_pub  = self.create_publisher(Imu, '/imu/raw', 10)
        self.baro_pub = self.create_publisher(FluidPressure, '/imu/baro', 10)

        self.create_timer(1.0 / imu_rate,  self._read_imu)
        self.create_timer(1.0 / baro_rate, self._read_baro)

        self.get_logger().info('BerryGPS-IMU node started')

    def _init_imu(self):
        # Gyro: 119 Hz, 245 dps
        self.bus.write_byte_data(LSM9DS1_AG_ADDR, AG_CTRL_REG1_G, 0x60)
        # Accel: 119 Hz, 2g
        self.bus.write_byte_data(LSM9DS1_AG_ADDR, AG_CTRL_REG6_XL, 0x60)
        time.sleep(0.1)

    def _init_baro(self):
        # LPS22: 25 Hz output
        self.bus.write_byte_data(LPS22_ADDR, LPS22_CTRL_REG1, 0x30)
        time.sleep(0.1)

    def _read_imu(self):
        try:
            ax = read_word_signed(self.bus, LSM9DS1_AG_ADDR, AG_OUT_X_L_XL)     * ACCEL_SCALE
            ay = read_word_signed(self.bus, LSM9DS1_AG_ADDR, AG_OUT_X_L_XL + 2) * ACCEL_SCALE
            az = read_word_signed(self.bus, LSM9DS1_AG_ADDR, AG_OUT_X_L_XL + 4) * ACCEL_SCALE

            gx = read_word_signed(self.bus, LSM9DS1_AG_ADDR, AG_OUT_X_L_G)     * GYRO_SCALE
            gy = read_word_signed(self.bus, LSM9DS1_AG_ADDR, AG_OUT_X_L_G + 2) * GYRO_SCALE
            gz = read_word_signed(self.bus, LSM9DS1_AG_ADDR, AG_OUT_X_L_G + 4) * GYRO_SCALE

            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id

            msg.linear_acceleration.x = ax
            msg.linear_acceleration.y = ay
            msg.linear_acceleration.z = az

            msg.angular_velocity.x = gx
            msg.angular_velocity.y = gy
            msg.angular_velocity.z = gz

            # Orientation unknown, set covariance to -1
            msg.orientation_covariance[0] = -1.0

            # Diagonal covariance estimates
            msg.linear_acceleration_covariance[0] = 0.01
            msg.linear_acceleration_covariance[4] = 0.01
            msg.linear_acceleration_covariance[8] = 0.01

            msg.angular_velocity_covariance[0] = 0.001
            msg.angular_velocity_covariance[4] = 0.001
            msg.angular_velocity_covariance[8] = 0.001

            self.imu_pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f'IMU read error: {e}')

    def _read_baro(self):
        try:
            xl = self.bus.read_byte_data(LPS22_ADDR, LPS22_PRES_OUT_XL)
            lo = self.bus.read_byte_data(LPS22_ADDR, LPS22_PRES_OUT_XL + 1)
            hi = self.bus.read_byte_data(LPS22_ADDR, LPS22_PRES_OUT_XL + 2)
            raw = (hi << 16) | (lo << 8) | xl
            if raw >= 0x800000:
                raw -= 0x1000000
            pressure_hpa = raw * BARO_SCALE

            msg = FluidPressure()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.fluid_pressure = pressure_hpa * 100.0  # hPa -> Pa
            msg.variance = 0.0

            self.baro_pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f'Baro read error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = BerryImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
