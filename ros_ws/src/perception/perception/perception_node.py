"""Observe standard camera and lidar messages, independently of their source."""

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2


class PerceptionNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__("perception", **kwargs)
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("lidar_topic", "/lidar/points")
        self.declare_parameter("log_interval_sec", 2.0)
        self.log_interval_sec = self.get_parameter("log_interval_sec").value
        if not math.isfinite(self.log_interval_sec) or self.log_interval_sec <= 0:
            raise ValueError("log_interval_sec must be finite and positive")

        self.camera_count = 0
        self.lidar_count = 0
        self._last_camera_log = float("-inf")
        self._last_lidar_log = float("-inf")
        self.camera_subscription = self.create_subscription(
            Image,
            self.get_parameter("camera_topic").value,
            self.camera_callback,
            qos_profile_sensor_data,
        )
        self.lidar_subscription = self.create_subscription(
            PointCloud2,
            self.get_parameter("lidar_topic").value,
            self.lidar_callback,
            qos_profile_sensor_data,
        )

    def camera_callback(self, msg: Image) -> None:
        self.camera_count += 1
        now = time.monotonic()
        if now - self._last_camera_log >= self.log_interval_sec:
            self.get_logger().info(
                f"[CAMERA] count={self.camera_count} size={msg.width}x{msg.height} "
                f"encoding={msg.encoding} frame={msg.header.frame_id}"
            )
            self._last_camera_log = now

    def lidar_callback(self, msg: PointCloud2) -> None:
        self.lidar_count += 1
        now = time.monotonic()
        if now - self._last_lidar_log >= self.log_interval_sec:
            self.get_logger().info(
                f"[LIDAR] count={self.lidar_count} width={msg.width} "
                f"height={msg.height} point_step={msg.point_step} "
                f"frame={msg.header.frame_id}"
            )
            self._last_lidar_log = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = PerceptionNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
