#!/usr/bin/env python3
"""
BerryGPS-IMU-4 ROS 2 node.
Reads LSM6DSL (accel/gyro at 0x6A) and BMP388 (barometer at 0x77)
via I2C and publishes sensor_msgs/Imu and sensor_msgs/FluidPressure.
"""

import struct
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, FluidPressure
import smbus2

# ── LSM6DSL ──────────────────────────────────────────────────────────────────
LSM6DSL_ADDR = 0x6A
CTRL1_XL     = 0x10   # accel: ODR[7:4], FS[3:2]
CTRL2_G      = 0x11   # gyro:  ODR[7:4], FS[3:2]
CTRL3_C      = 0x12   # BDU[6], IF_INC[2]
OUTX_L_G     = 0x22   # gyro  output start (6 bytes, XYZ)
OUTX_L_XL    = 0x28   # accel output start (6 bytes, XYZ)

GYRO_SCALE  = math.radians(250.0 / 32768.0)  # 250 dps -> rad/s
ACCEL_SCALE = 2.0 * 9.81 / 32768.0           # ±2g -> m/s²

# ── BMP388 ───────────────────────────────────────────────────────────────────
BMP388_ADDR    = 0x77
BMP388_PWR_CTRL = 0x1B  # mode[5:4], temp_en[1], press_en[0]
BMP388_OSR     = 0x1C   # oversampling
BMP388_ODR     = 0x1D   # output data rate
BMP388_DATA    = 0x04   # 6-byte pressure + temperature raw data
BMP388_CAL_REG = 0x31   # start of 21-byte calibration block


def _s16(lo, hi):
    v = (hi << 8) | lo
    return v - 65536 if v >= 32768 else v


def _read_xyz(bus, addr, reg):
    d = bus.read_i2c_block_data(addr, reg, 6)
    return _s16(d[0], d[1]), _s16(d[2], d[3]), _s16(d[4], d[5])


class BerryImuNode(Node):

    def __init__(self):
        super().__init__('berry_imu_node')

        self.declare_parameter('i2c_bus',   1)
        self.declare_parameter('imu_rate',  100.0)
        self.declare_parameter('baro_rate', 10.0)
        self.declare_parameter('frame_id',  'imu')

        bus_num   = self.get_parameter('i2c_bus').value
        imu_rate  = self.get_parameter('imu_rate').value
        baro_rate = self.get_parameter('baro_rate').value
        self.frame_id = self.get_parameter('frame_id').value

        self.bus = smbus2.SMBus(bus_num)
        self._init_lsm6dsl()
        self._bmp_cal = self._init_bmp388()

        self.imu_pub  = self.create_publisher(Imu,           '/imu/raw',  10)
        self.baro_pub = self.create_publisher(FluidPressure, '/imu/baro', 10)

        self.create_timer(1.0 / imu_rate,  self._read_imu)
        self.create_timer(1.0 / baro_rate, self._read_baro)

        self.get_logger().info('BerryGPS-IMU node started (LSM6DSL + BMP388)')

    # ── initialisation ───────────────────────────────────────────────────────

    def _init_lsm6dsl(self):
        # Block data update + address auto-increment
        self.bus.write_byte_data(LSM6DSL_ADDR, CTRL3_C,  0x44)
        # Accel: 104 Hz ODR, ±2 g
        self.bus.write_byte_data(LSM6DSL_ADDR, CTRL1_XL, 0x40)
        # Gyro:  104 Hz ODR, 250 dps
        self.bus.write_byte_data(LSM6DSL_ADDR, CTRL2_G,  0x40)
        time.sleep(0.1)

    def _init_bmp388(self):
        # Normal mode, pressure + temperature enabled
        self.bus.write_byte_data(BMP388_ADDR, BMP388_PWR_CTRL, 0x33)
        # Oversampling: x1 temp, x4 pressure
        self.bus.write_byte_data(BMP388_ADDR, BMP388_OSR, 0x02)
        # ODR: 25 Hz (sufficient for a 10 Hz ROS timer)
        self.bus.write_byte_data(BMP388_ADDR, BMP388_ODR, 0x03)
        time.sleep(0.1)

        raw = self.bus.read_i2c_block_data(BMP388_ADDR, BMP388_CAL_REG, 21)
        # Struct layout matches BMP388 datasheet Table 11 (LE byte order):
        # PAR_T1(u16) PAR_T2(u16) PAR_T3(s8)
        # PAR_P1..P2(s16) PAR_P3..P4(s8) PAR_P5..P6(u16) PAR_P7..P8(s8)
        # PAR_P9(s16) PAR_P10..P11(s8)
        r = struct.unpack_from('<HHbhhbbHHbbhbb', bytes(raw))
        return {
            'T1':  r[0]  * 256.0,           # PAR_T1 / 2^-8
            'T2':  r[1]  / (1 << 30),
            'T3':  r[2]  / (1 << 48),
            'P1':  (r[3]  - (1 << 14)) / (1 << 20),
            'P2':  (r[4]  - (1 << 14)) / (1 << 29),
            'P3':  r[5]  / (1 << 32),
            'P4':  r[6]  / (1 << 37),
            'P5':  r[7]  * 8.0,             # PAR_P5 / 2^-3
            'P6':  r[8]  / (1 << 6),
            'P7':  r[9]  / (1 << 8),
            'P8':  r[10] / (1 << 15),
            'P9':  r[11] / (1 << 48),
            'P10': r[12] / (1 << 48),
            'P11': r[13] / (1 << 65),
        }

    # ── callbacks ────────────────────────────────────────────────────────────

    def _read_imu(self):
        try:
            gx, gy, gz = _read_xyz(self.bus, LSM6DSL_ADDR, OUTX_L_G)
            ax, ay, az = _read_xyz(self.bus, LSM6DSL_ADDR, OUTX_L_XL)

            msg = Imu()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id

            msg.angular_velocity.x = gx * GYRO_SCALE
            msg.angular_velocity.y = gy * GYRO_SCALE
            msg.angular_velocity.z = gz * GYRO_SCALE

            msg.linear_acceleration.x = ax * ACCEL_SCALE
            msg.linear_acceleration.y = ay * ACCEL_SCALE
            msg.linear_acceleration.z = az * ACCEL_SCALE

            msg.orientation_covariance[0] = -1.0

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
            d = self.bus.read_i2c_block_data(BMP388_ADDR, BMP388_DATA, 6)
            uncomp_press = (d[2] << 16) | (d[1] << 8) | d[0]
            uncomp_temp  = (d[5] << 16) | (d[4] << 8) | d[3]

            c = self._bmp_cal

            # Temperature compensation (Bosch BMP388 datasheet section 8.5)
            pd1   = uncomp_temp - c['T1']
            t_lin = pd1 * c['T2'] + (pd1 * pd1) * c['T3']

            # Pressure compensation
            po1 = c['P5'] + c['P6'] * t_lin + c['P7'] * t_lin**2 + c['P8'] * t_lin**3
            po2 = uncomp_press * (c['P1'] + c['P2'] * t_lin
                                  + c['P3'] * t_lin**2 + c['P4'] * t_lin**3)
            pd4 = (uncomp_press**2 * (c['P9'] + c['P10'] * t_lin)
                   + uncomp_press**3 * c['P11'])
            pressure_pa = po1 + po2 + pd4

            msg = FluidPressure()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.fluid_pressure  = pressure_pa  # already in Pa
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
