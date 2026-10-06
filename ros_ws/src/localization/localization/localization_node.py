"""Observe standard inertial measurements before adding a state estimator."""

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


class LocalizationNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("localization", **kwargs)
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("log_interval_sec", 2.0)
        self.log_interval_sec = self.get_parameter("log_interval_sec").value
        if not math.isfinite(self.log_interval_sec) or self.log_interval_sec <= 0:
            raise ValueError("log_interval_sec must be finite and positive")

        self.imu_count = 0
        self._last_imu_log = float("-inf")
        self.imu_subscription = self.create_subscription(
            Imu,
            self.get_parameter("imu_topic").value,
            self.imu_callback,
            qos_profile_sensor_data,
        )

    def imu_callback(self, msg: Imu) -> None:
        # This callback is the future measurement-conversion / prediction boundary.
        self.imu_count += 1
        now = time.monotonic()
        if now - self._last_imu_log >= self.log_interval_sec:
            gyro = msg.angular_velocity
            accel = msg.linear_acceleration
            self.get_logger().info(
                f"[IMU] count={self.imu_count} "
                f"gyro=({gyro.x:.3f}, {gyro.y:.3f}, {gyro.z:.3f}) "
                f"accel=({accel.x:.3f}, {accel.y:.3f}, {accel.z:.3f}) "
                f"frame={msg.header.frame_id}"
            )
            self._last_imu_log = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = LocalizationNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
