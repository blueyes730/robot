"""Integration checks using standard ROS messages and a synthetic ROS source.

Run after sourcing Jazzy and ros_ws/install/setup.bash.
"""

import time
import unittest
from unittest.mock import Mock, patch

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, PointCloud2

from localization.localization_node import LocalizationNode
from perception.perception_node import PerceptionNode


class AutonomyNodesTest(unittest.TestCase):
    def setUp(self):
        rclpy.init(args=[])
        self.perception = PerceptionNode()
        self.localization = LocalizationNode()
        self.source = Node("test_sensor_source")
        self.executor = SingleThreadedExecutor()
        for node in (self.perception, self.localization, self.source):
            self.executor.add_node(node)

    def tearDown(self):
        self.executor.shutdown()
        for node in (self.perception, self.localization, self.source):
            node.destroy_node()
        rclpy.shutdown()

    def test_standard_messages_reach_independent_nodes(self):
        topics = (
            (Image, "/camera/image_raw"),
            (PointCloud2, "/lidar/points"),
            (Imu, "/imu/data"),
        )
        publishers = [
            self.source.create_publisher(message_type, topic, qos_profile_sensor_data)
            for message_type, topic in topics
        ]
        image = Image(width=640, height=480, encoding="rgb8", step=1920)
        image.header.frame_id = "camera_sensor"
        cloud = PointCloud2(width=12, height=1, point_step=16, row_step=192)
        cloud.header.frame_id = "lidar_sensor"
        imu = Imu()
        imu.header.frame_id = "imu_sensor"
        imu.angular_velocity.x = 0.25
        imu.linear_acceleration.z = 9.81
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            for publisher, message in zip(publishers, (image, cloud, imu)):
                publisher.publish(message)
            self.executor.spin_once(timeout_sec=0.05)
            if (self.perception.camera_count and self.perception.lidar_count
                    and self.localization.imu_count):
                break
        self.assertGreater(self.perception.camera_count, 0)
        self.assertGreater(self.perception.lidar_count, 0)
        self.assertGreater(self.localization.imu_count, 0)
        for subscription in (
            self.perception.camera_subscription,
            self.perception.lidar_subscription,
            self.localization.imu_subscription,
        ):
            self.assertEqual(subscription.qos_profile.reliability, ReliabilityPolicy.BEST_EFFORT)

    def test_logging_is_throttled_per_stream_and_excludes_payload(self):
        logger = Mock()
        image = Image(width=1280, height=720, encoding="rgba8")
        image.header.frame_id = "optical_frame"
        image.data = b"private_pixel_payload"
        cloud = PointCloud2(width=32768, height=1, point_step=32)
        cloud.header.frame_id = "scan_frame"
        with patch.object(self.perception, "get_logger", return_value=logger), patch(
            "perception.perception_node.time.monotonic", return_value=10.0
        ):
            for _ in range(30):
                self.perception.camera_callback(image)
                self.perception.lidar_callback(cloud)
        self.assertEqual(logger.info.call_count, 2)
        self.assertIn("size=1280x720 encoding=rgba8 frame=optical_frame", logger.info.call_args_list[0].args[0])
        self.assertIn("width=32768 height=1 point_step=32 frame=scan_frame", logger.info.call_args_list[1].args[0])
        self.assertNotIn("private_pixel_payload", str(logger.info.call_args_list))
        with patch.object(self.perception, "get_logger", return_value=logger), patch(
            "perception.perception_node.time.monotonic", return_value=12.0
        ):
            self.perception.camera_callback(image)
        self.assertIn("count=31", logger.info.call_args.args[0])

        logger.reset_mock()
        imu = Imu()
        imu.header.frame_id = "inertial_frame"
        imu.angular_velocity.x = 0.5
        imu.linear_acceleration.z = 9.81
        with patch.object(self.localization, "get_logger", return_value=logger), patch(
            "localization.localization_node.time.monotonic", return_value=10.0
        ):
            for _ in range(100):
                self.localization.imu_callback(imu)
        logger.info.assert_called_once()
        self.assertIn("gyro=(0.500, 0.000, 0.000) accel=(0.000, 0.000, 9.810)", logger.info.call_args.args[0])
        self.assertIn("frame=inertial_frame", logger.info.call_args.args[0])

    def test_ros_remapping(self):
        from rclpy.parameter import Parameter

        configured = PerceptionNode(
            cli_args=["--ros-args", "-r", "__node:=test_perception", "-r", "/camera/image_raw:=/test/image"],
            parameter_overrides=[Parameter("lidar_topic", value="/test/points")],
        )
        inertial = LocalizationNode(
            cli_args=["--ros-args", "-r", "__node:=test_localization"],
            parameter_overrides=[Parameter("imu_topic", value="/test/imu")],
        )
        try:
            self.assertEqual(configured.camera_subscription.topic_name, "/test/image")
            self.assertEqual(configured.lidar_subscription.topic_name, "/test/points")
            self.assertEqual(inertial.imu_subscription.topic_name, "/test/imu")
        finally:
            configured.destroy_node()
            inertial.destroy_node()


if __name__ == "__main__":
    unittest.main()
