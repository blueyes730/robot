"""Adapt the existing Isaac Sim sensors to the standard ROS sensor interface.

Import after SimulationApp starts and the ROS bridge and physics sensor nodes
extensions are enabled. Transport belongs to the Isaac ROS bridge.
"""

from __future__ import annotations

from dataclasses import dataclass

import isaacsim.core.experimental.utils.stage as stage_utils
import omni.graph.core as og
import omni.syntheticdata
from isaacsim.ros2.core import read_camera_info
from pxr import Sdf

from robots.nova_carter.sensors import NovaCarterSensors


@dataclass(frozen=True)
class IsaacSimRosPublisherConfig:
    image_topic: str = "/camera/image_raw"
    camera_info_topic: str = "/camera/camera_info"
    lidar_topic: str = "/lidar/points"
    imu_topic: str = "/imu/data"
    # Defaults use the existing sensor prim names. Override for an external TF tree.
    camera_frame: str | None = None
    lidar_frame: str | None = None
    imu_frame: str | None = None
    graph_path: str = "/World/IsaacSimRosPublishers"


@dataclass
class IsaacSimRosPublishers:
    """Keep the writers, IMU graph, and their source sensors alive."""

    rgb_writer: object
    lidar_writer: object
    imu_writer: object
    camera_info_writer: object
    sensors: NovaCarterSensors
    config: IsaacSimRosPublisherConfig
    rgb_writer_name: str

    def close(self) -> None:
        """Detach transport before destroying sensors or closing SimulationApp."""
        self.sensors.camera.detach_writer(self.rgb_writer_name)
        self.sensors.camera.detach_writer("ROS2PublishCameraInfo")
        self.sensors.lidar.detach_writer("RtxLidarROS2PublishPointCloud")
        if stage_utils.get_current_stage().GetPrimAtPath(self.config.graph_path).IsValid():
            stage_utils.delete_prim(self.config.graph_path)


def create_isaacsim_ros_publishers(
    sensors: NovaCarterSensors,
    config: IsaacSimRosPublisherConfig | None = None,
) -> IsaacSimRosPublishers:
    """Attach ROS writers and a physics-driven IMU graph to existing sensors.

    Image, calibration, and point cloud writers reuse their sensors' render
    products. The IMU graph reads the existing prim and publishes only valid
    readings. All headers use simulation time.
    """
    config = config or IsaacSimRosPublisherConfig()
    stage = stage_utils.get_current_stage()
    if stage.GetPrimAtPath(config.graph_path).IsValid():
        raise ValueError(f"ROS publisher graph already exists: {config.graph_path}")

    imu_path = sensors.imu.authoring_object.paths[0]
    camera_frame = config.camera_frame or Sdf.Path(sensors.camera.authoring_object.paths[0]).name
    lidar_frame = config.lidar_frame or Sdf.Path(sensors.lidar.authoring_object.paths[0]).name
    imu_frame = config.imu_frame or Sdf.Path(imu_path).name
    camera_info, _ = read_camera_info(
        render_product_path=str(sensors.camera.render_product.GetPath())
    )
    # Resolve the registry prefix exactly as the installed bridge does.
    rgb_var = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar("Rgb")
    rgb_writer_name = f"{rgb_var}ROS2PublishImage"
    attached = []
    try:
        keys = og.Controller.Keys
        imu_graph, _, _, _ = og.Controller.edit(
            {
                "graph_path": config.graph_path,
                "evaluator_name": "execution",
                # OnPhysicsStep evaluates this graph on each physics event.
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            },
            {
                keys.CREATE_NODES: [
                    ("PhysicsStep", "isaacsim.core.nodes.OnPhysicsStep"),
                    ("ReadImu", "isaacsim.sensors.physics.IsaacReadIMU"),
                    ("PublishImu", "isaacsim.ros2.bridge.ROS2PublishImu"),
                ],
                keys.CONNECT: [
                    ("PhysicsStep.outputs:step", "ReadImu.inputs:execIn"),
                    ("ReadImu.outputs:execOut", "PublishImu.inputs:execIn"),
                    ("ReadImu.outputs:orientation", "PublishImu.inputs:orientation"),
                    ("ReadImu.outputs:angVel", "PublishImu.inputs:angularVelocity"),
                    ("ReadImu.outputs:linAcc", "PublishImu.inputs:linearAcceleration"),
                    ("ReadImu.outputs:sensorTime", "PublishImu.inputs:timeStamp"),
                ],
                keys.SET_VALUES: [
                    ("ReadImu.inputs:imuPrim", [Sdf.Path(imu_path)]),
                    ("ReadImu.inputs:readGravity", True),
                    ("ReadImu.inputs:useLatestData", True),
                    ("PublishImu.inputs:topicName", config.imu_topic),
                    ("PublishImu.inputs:frameId", imu_frame),
                    ("PublishImu.inputs:queueSize", 5),
                    ("PublishImu.inputs:publishOrientation", True),
                    ("PublishImu.inputs:publishAngularVelocity", True),
                    ("PublishImu.inputs:publishLinearAcceleration", True),
                ],
            },
        )
        rgb_writer = sensors.camera.attach_writer(
            rgb_writer_name,
            topicName=config.image_topic,
            frameId=camera_frame,
            queueSize=5,
        )
        attached.append((sensors.camera, rgb_writer_name))
        camera_info_writer = sensors.camera.attach_writer(
            "ROS2PublishCameraInfo",
            topicName=config.camera_info_topic,
            frameId=camera_frame,
            queueSize=5,
            width=camera_info.width,
            height=camera_info.height,
            projectionType=camera_info.distortion_model,
            k=camera_info.k,
            r=camera_info.r,
            p=camera_info.p,
            physicalDistortionModel=camera_info.distortion_model,
            physicalDistortionCoefficients=camera_info.d,
        )
        attached.append((sensors.camera, "ROS2PublishCameraInfo"))
        lidar_writer = sensors.lidar.attach_writer(
            "RtxLidarROS2PublishPointCloud",
            topicName=config.lidar_topic,
            frameId=lidar_frame,
            queueSize=5,
        )
        attached.append((sensors.lidar, "RtxLidarROS2PublishPointCloud"))
    except Exception:
        for sensor, writer_name in reversed(attached):
            sensor.detach_writer(writer_name)
        if stage.GetPrimAtPath(config.graph_path).IsValid():
            stage_utils.delete_prim(config.graph_path)
        raise

    return IsaacSimRosPublishers(
        rgb_writer=rgb_writer,
        lidar_writer=lidar_writer,
        imu_writer=imu_graph,
        camera_info_writer=camera_info_writer,
        sensors=sensors,
        config=config,
        rgb_writer_name=rgb_writer_name,
    )
