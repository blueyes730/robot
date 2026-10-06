"""Test adapter routing and cleanup without starting the GPU simulator."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class IsaacSimRosPublishersTest(unittest.TestCase):
    def setUp(self):
        names = [
            "isaacsim", "isaacsim.core", "isaacsim.core.experimental",
            "isaacsim.core.experimental.utils", "isaacsim.core.experimental.utils.stage",
            "isaacsim.ros2", "isaacsim.ros2.core", "omni", "omni.graph",
            "omni.graph.core", "omni.syntheticdata", "pxr",
            "robots.nova_carter.sensors",
        ]
        modules = {name: types.ModuleType(name) for name in names}
        for name, module in modules.items():
            parent, _, child = name.rpartition(".")
            if parent in modules:
                setattr(modules[parent], child, module)

        self.stage_utils = modules["isaacsim.core.experimental.utils.stage"]
        self.stage = Mock()
        self.existing_paths = set()
        self.stage.GetPrimAtPath.side_effect = lambda path: types.SimpleNamespace(
            IsValid=lambda: path in self.existing_paths,
        )
        self.stage_utils.get_current_stage = Mock(return_value=self.stage)
        self.stage_utils.delete_prim = Mock(side_effect=self.existing_paths.discard)
        self.og = modules["omni.graph.core"]
        self.og.Controller = Mock()
        self.og.Controller.Keys = types.SimpleNamespace(
            CREATE_NODES="create", CONNECT="connect", SET_VALUES="values",
        )
        self.og.GraphPipelineStage = types.SimpleNamespace(
            GRAPH_PIPELINE_STAGE_ONDEMAND="on_demand",
        )
        self.graph = object()

        def edit(settings, wiring):
            self.existing_paths.add(settings["graph_path"])
            return self.graph, None, None, None

        self.og.Controller.edit.side_effect = edit
        modules["omni.syntheticdata"].SyntheticData = Mock()
        modules["omni.syntheticdata"].SyntheticData.convert_sensor_type_to_rendervar.return_value = "Rgb"
        modules["pxr"].Sdf = types.SimpleNamespace(
            Path=lambda path: types.SimpleNamespace(name=path.rsplit("/", 1)[-1]),
        )
        modules["robots.nova_carter.sensors"].NovaCarterSensors = object
        self.info = types.SimpleNamespace(
            width=800, height=600, distortion_model="plumb_bob",
            k=[4.0] * 9, r=[5.0] * 9, p=[6.0] * 12, d=[0.0] * 5,
        )
        self.read_info = Mock(return_value=(self.info, object()))
        modules["isaacsim.ros2.core"].read_camera_info = self.read_info
        self.sensors = types.SimpleNamespace(camera=Mock(), lidar=Mock(), imu=Mock())
        for name in ("camera", "lidar", "imu"):
            getattr(self.sensors, name).authoring_object.paths = [f"/World/OtherRobot/{name}_frame"]
        self.sensors.camera.render_product.GetPath.return_value = "/Render/ExistingCameraProduct"

        filename = Path(__file__).resolve().parents[1] / "robots/nova_carter/isaacsim_ros_publishers.py"
        spec = importlib.util.spec_from_file_location("isaacsim_ros_publishers_under_test", filename)
        self.adapter = importlib.util.module_from_spec(spec)
        modules[spec.name] = self.adapter
        self.patcher = patch.dict(sys.modules, modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        spec.loader.exec_module(self.adapter)

    def test_existing_sensor_frames_calibration_and_sim_time_are_used(self):
        handles = self.adapter.create_isaacsim_ros_publishers(self.sensors)
        self.assertIs(handles.sensors, self.sensors)
        self.assertIs(handles.imu_writer, self.graph)
        self.read_info.assert_called_once_with(render_product_path="/Render/ExistingCameraProduct")
        image, info = self.sensors.camera.attach_writer.call_args_list
        self.assertEqual(image.kwargs["topicName"], "/camera/image_raw")
        self.assertEqual(image.kwargs["frameId"], "camera_frame")
        self.assertEqual(info.kwargs["topicName"], "/camera/camera_info")
        self.assertEqual(info.kwargs["k"], self.info.k)
        self.assertEqual(info.kwargs["width"], 800)
        cloud = self.sensors.lidar.attach_writer.call_args
        self.assertEqual(cloud.kwargs["topicName"], "/lidar/points")
        self.assertEqual(cloud.kwargs["frameId"], "lidar_frame")
        wiring = self.og.Controller.edit.call_args.args[1]
        self.assertIn(("ReadImu.outputs:sensorTime", "PublishImu.inputs:timeStamp"), wiring["connect"])
        self.assertIn(("ReadImu.outputs:execOut", "PublishImu.inputs:execIn"), wiring["connect"])
        handles.close()
        self.assertEqual(self.sensors.camera.detach_writer.call_count, 2)
        self.sensors.lidar.detach_writer.assert_called_once_with("RtxLidarROS2PublishPointCloud")
        self.assertFalse(self.existing_paths)

    def test_override_topics_and_frames(self):
        config = self.adapter.IsaacSimRosPublisherConfig(
            image_topic="/test/image", lidar_topic="/test/points", imu_topic="/test/imu",
            camera_frame="optical", lidar_frame="scan", imu_frame="inertial",
        )
        self.adapter.create_isaacsim_ros_publishers(self.sensors, config)
        self.assertEqual(self.sensors.camera.attach_writer.call_args_list[0].kwargs["frameId"], "optical")
        self.assertEqual(self.sensors.lidar.attach_writer.call_args.kwargs["topicName"], "/test/points")
        values = dict(self.og.Controller.edit.call_args.args[1]["values"])
        self.assertEqual(values["PublishImu.inputs:topicName"], "/test/imu")
        self.assertEqual(values["PublishImu.inputs:frameId"], "inertial")

    def test_partial_initialization_cleans_up(self):
        self.sensors.lidar.attach_writer.side_effect = RuntimeError("writer attachment failed")
        with self.assertRaisesRegex(RuntimeError, "attachment failed"):
            self.adapter.create_isaacsim_ros_publishers(self.sensors)
        self.assertEqual(self.sensors.camera.detach_writer.call_count, 2)
        self.assertFalse(self.existing_paths)

    def test_duplicate_graph_does_not_attach_again(self):
        self.existing_paths.add("/World/IsaacSimRosPublishers")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.adapter.create_isaacsim_ros_publishers(self.sensors)
        self.sensors.camera.attach_writer.assert_not_called()
        self.og.Controller.edit.assert_not_called()

    def test_camera_config_width_height_matches_runtime_height_width(self):
        # Exercise the real factory without loading the rest of the GPU extension.
        filename = Path(__file__).resolve().parents[1] / "robots/nova_carter/sensors.py"
        tree = ast.parse(filename.read_text())
        factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_camera")
        namespace = {"np": Mock(), "RtxCamera": Mock(), "CameraSensor": Mock()}
        code = "from __future__ import annotations\n" + ast.unparse(factory)
        exec(compile(code, str(filename), "exec"), namespace)
        config = types.SimpleNamespace(
            prim_name="camera_sensor", frequency_hz=30.0, resolution=(640, 480),
            translation=(0.1, 0.0, 0.0), orientation=(0.5, 0.5, -0.5, -0.5),
            annotators=("rgb",),
        )
        namespace["create_camera"]("/World/Robot/head", config)
        # Isaac Sim 6.1 CameraSensor converts runtime [1], [0] to USD width, height.
        height, width = namespace["CameraSensor"].call_args.kwargs["resolution"]
        self.assertEqual((width, height), config.resolution)


if __name__ == "__main__":
    unittest.main()
