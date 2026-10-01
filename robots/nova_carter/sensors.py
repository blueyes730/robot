"""Humanoid-height sensor frames and readers for Nova Carter."""

from dataclasses import dataclass

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.sensors.experimental.physics import IMU, IMUSensor
from isaacsim.sensors.experimental.rtx import CameraSensor, Lidar, LidarSensor, RtxCamera
from pxr import UsdPhysics


TORSO_MOUNT_TRANSLATION = (0.0, 0.0, 0.9)
HEAD_MOUNT_TRANSLATION = (0.0, 0.0, 1.5)
CAMERA_TRANSLATION = (0.1, 0.0, 0.0)
LIDAR_TRANSLATION = (0.0, 0.0, 0.12)
# Isaac camera's optical frame looks down local -Z; rotate it toward chassis +X.
CAMERA_ORIENTATION = (0.5, 0.5, -0.5, -0.5)
IDENTITY_ORIENTATION = (1.0, 0.0, 0.0, 0.0)
LIDAR_PROFILE = "Example_Rotary"


@dataclass(frozen=True)
class NovaCarterSensors:
    """Runtime handles for the three custom Nova Carter sensors."""

    imu: IMUSensor
    camera: CameraSensor
    lidar: LidarSensor


def _mount(path: str, translation: tuple[float, float, float]) -> XformPrim:
    """Create a transform-only local frame."""
    if stage_utils.get_current_stage().GetPrimAtPath(path).IsValid():
        raise ValueError(f"Mount prim already exists: {path}")
    stage_utils.define_prim(path, type_name="Xform")
    mount = XformPrim(path, reset_xform_op_properties=True)
    mount.set_local_poses(
        translations=np.asarray([translation]),
        orientations=np.asarray([IDENTITY_ORIENTATION]),
    )
    return mount


def create_nova_carter_sensors(chassis_prim_path: str) -> NovaCarterSensors:
    """Attach a separate sensor rig beneath Nova Carter's rigid chassis link.

    Mount offsets are local to the chassis; sensor offsets are local to mounts.
    Raises if the rig already exists instead of creating duplicate sensors.
    """
    stage = stage_utils.get_current_stage()
    chassis = stage.GetPrimAtPath(chassis_prim_path)
    if not chassis.IsValid():
        raise ValueError(f"Chassis prim does not exist: {chassis_prim_path}")
    if not chassis.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"Chassis prim is not a rigid body: {chassis_prim_path}")

    rig_path = f"{chassis_prim_path}/sensor_rig"
    if stage.GetPrimAtPath(rig_path).IsValid():
        raise ValueError(f"Sensor rig already exists: {rig_path}")

    _mount(rig_path, (0.0, 0.0, 0.0))
    torso_path = f"{rig_path}/torso_mount"
    head_path = f"{rig_path}/head_mount"
    _mount(torso_path, TORSO_MOUNT_TRANSLATION)
    _mount(head_path, HEAD_MOUNT_TRANSLATION)

    imu_prim = IMU.create(
        f"{torso_path}/imu_sensor",
        translations=np.asarray([(0.0, 0.0, 0.0)]),
        orientations=np.asarray([IDENTITY_ORIENTATION]),
        linear_acceleration_filter_size=5,
        angular_velocity_filter_size=5,
        orientation_filter_size=5,
    )
    camera_prim = RtxCamera(
        f"{head_path}/camera_sensor",
        tick_rate=30.0,
        translations=np.asarray([CAMERA_TRANSLATION]),
        orientations=np.asarray([CAMERA_ORIENTATION]),
    )
    lidar_prim = Lidar.create(
        f"{head_path}/lidar_sensor",
        config=LIDAR_PROFILE,
        tick_rate=10.0,
        translations=np.asarray([LIDAR_TRANSLATION]),
        orientations=np.asarray([IDENTITY_ORIENTATION]),
        attributes={"omni:sensor:Core:scanRateBaseHz": 10.0},
    )
    return NovaCarterSensors(
        imu=IMUSensor(imu_prim),
        camera=CameraSensor(camera_prim, resolution=(640, 480), annotators=["rgb"]),
        lidar=LidarSensor(lidar_prim, annotators=["generic-model-output"]),
    )
