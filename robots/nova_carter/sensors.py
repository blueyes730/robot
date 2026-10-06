"""Configuration, creation, and attachment helpers for Nova Carter sensors."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.sensors.experimental.physics import IMU, IMUSensor
from isaacsim.sensors.experimental.rtx import CameraSensor, Lidar, LidarSensor, RtxCamera
from pxr import UsdPhysics


@dataclass(frozen=True)
class ImuConfig:
    parent_prim: str
    prim_name: str
    translation: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    linear_acceleration_filter_size: int
    angular_velocity_filter_size: int
    orientation_filter_size: int


@dataclass(frozen=True)
class CameraConfig:
    parent_prim: str
    prim_name: str
    frequency_hz: float
    resolution: tuple[int, int]
    translation: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    annotators: tuple[str, ...]


@dataclass(frozen=True)
class LidarConfig:
    parent_prim: str
    prim_name: str
    config: str
    frequency_hz: float
    translation: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    annotators: tuple[str, ...]


@dataclass(frozen=True)
class MountConfig:
    prim_name: str
    translation: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


@dataclass(frozen=True)
class RigConfig:
    parent_link: str
    prim_name: str
    translation: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    mounts: tuple[MountConfig, ...]


@dataclass(frozen=True)
class NovaCarterSensorConfig:
    rig: RigConfig
    imu: ImuConfig
    camera: CameraConfig
    lidar: LidarConfig


@dataclass(frozen=True)
class NovaCarterSensors:
    """Runtime handles for the three custom Nova Carter sensors."""

    imu: IMUSensor
    camera: CameraSensor
    lidar: LidarSensor


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping/object")
    return value


def _string(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_number(data: Mapping[str, Any], field: str) -> float:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field} must be a positive number")
    return float(value)


def _positive_integer(data: Mapping[str, Any], field: str) -> int:
    value = data.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _float_tuple(
    data: Mapping[str, Any],
    field: str,
    length: int,
) -> tuple[float, ...]:
    value = data.get(field)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must contain {length} numbers")
    if len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} numbers")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        raise ValueError(f"{field} must contain only numbers")
    return tuple(float(item) for item in value)


def _orientation(
    data: Mapping[str, Any],
    field: str = "orientation",
) -> tuple[float, float, float, float]:
    value = _float_tuple(data, field, 4)
    if not np.isclose(np.linalg.norm(value), 1.0, atol=1e-6):
        raise ValueError(f"{field} must be a normalized wxyz quaternion")
    return value


def _resolution(data: Mapping[str, Any]) -> tuple[int, int]:
    value = data.get("resolution")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("resolution must contain width and height")
    if len(value) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        raise ValueError("resolution must contain two positive integers")
    return int(value[0]), int(value[1])


def _annotators(data: Mapping[str, Any]) -> tuple[str, ...]:
    value = data.get("annotators")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("annotators must be a list of strings")
    if not value or any(not isinstance(item, str) or not item for item in value):
        raise ValueError("annotators must contain at least one non-empty string")
    return tuple(value)


def _parse_sensor_config(data: Mapping[str, Any]) -> NovaCarterSensorConfig:
    sensors = _mapping(data.get("sensors"), "sensors")
    imu = _mapping(sensors.get("imu"), "sensors.imu")
    camera = _mapping(sensors.get("camera"), "sensors.camera")
    lidar = _mapping(sensors.get("lidar"), "sensors.lidar")

    rig = _mapping(data.get("rig"), "rig")
    raw_mounts = _mapping(rig.get("mounts"), "rig.mounts")
    mounts = tuple(
        MountConfig(
            prim_name=_string(_mapping(mount, f"rig.mounts.{name}"), "prim_name"),
            translation=_float_tuple(mount, "translation", 3),
            orientation=_orientation(mount),
        )
        for name, mount in raw_mounts.items()
    )
    mount_names = [mount.prim_name for mount in mounts]
    if len(set(mount_names)) != len(mount_names):
        raise ValueError("rig.mounts must have unique prim_name values")
    for name, sensor in (("imu", imu), ("camera", camera), ("lidar", lidar)):
        if _string(sensor, "parent_prim") not in mount_names:
            raise ValueError(f"sensors.{name}.parent_prim must name a configured mount")
    sensor_paths = [
        (_string(sensor, "parent_prim"), _string(sensor, "prim_name"))
        for sensor in (imu, camera, lidar)
    ]
    if len(set(sensor_paths)) != len(sensor_paths):
        raise ValueError("Sensors must have distinct prim paths")

    return NovaCarterSensorConfig(
        rig=RigConfig(
            parent_link=_string(rig, "parent_link"),
            prim_name=_string(rig, "prim_name"),
            translation=_float_tuple(rig, "translation", 3),
            orientation=_orientation(rig),
            mounts=mounts,
        ),
        imu=ImuConfig(
            parent_prim=_string(imu, "parent_prim"),
            prim_name=_string(imu, "prim_name"),
            translation=_float_tuple(imu, "translation", 3),
            orientation=_orientation(imu),
            linear_acceleration_filter_size=_positive_integer(
                imu, "linear_acceleration_filter_size"
            ),
            angular_velocity_filter_size=_positive_integer(
                imu, "angular_velocity_filter_size"
            ),
            orientation_filter_size=_positive_integer(
                imu, "orientation_filter_size"
            ),
        ),
        camera=CameraConfig(
            parent_prim=_string(camera, "parent_prim"),
            prim_name=_string(camera, "prim_name"),
            frequency_hz=_positive_number(camera, "frequency_hz"),
            resolution=_resolution(camera),
            translation=_float_tuple(camera, "translation", 3),
            orientation=_orientation(camera),
            annotators=_annotators(camera),
        ),
        lidar=LidarConfig(
            parent_prim=_string(lidar, "parent_prim"),
            prim_name=_string(lidar, "prim_name"),
            config=_string(lidar, "config"),
            frequency_hz=_positive_number(lidar, "frequency_hz"),
            translation=_float_tuple(lidar, "translation", 3),
            orientation=_orientation(lidar),
            annotators=_annotators(lidar),
        ),
    )


def load_sensor_config(config_path: str | Path) -> NovaCarterSensorConfig:
    """Load and validate a Nova Carter sensor configuration from YAML or JSON."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Sensor configuration does not exist: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        if path.suffix.lower() == ".json":
            raw_config = json.load(config_file)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "Loading YAML requires PyYAML. Install it into Isaac Sim's "
                    "Python environment or use the equivalent JSON config."
                ) from exc
            raw_config = yaml.safe_load(config_file)
        else:
            raise ValueError(
                f"Unsupported sensor config extension '{path.suffix}'; "
                "expected .yaml, .yml, or .json"
            )

    return _parse_sensor_config(_mapping(raw_config, "configuration root"))


def create_imu(parent_prim_path: str, config: ImuConfig) -> IMUSensor:
    """Author an IMU prim and create its runtime reader."""
    imu_prim = IMU.create(
        f"{parent_prim_path}/{config.prim_name}",
        translations=np.asarray([config.translation]),
        orientations=np.asarray([config.orientation]),
        linear_acceleration_filter_size=config.linear_acceleration_filter_size,
        angular_velocity_filter_size=config.angular_velocity_filter_size,
        orientation_filter_size=config.orientation_filter_size,
    )
    return IMUSensor(imu_prim)


def create_camera(
    parent_prim_path: str,
    config: CameraConfig,
) -> CameraSensor:
    """Create a configured forward-facing RGB camera."""
    camera_prim = RtxCamera(
        f"{parent_prim_path}/{config.prim_name}",
        tick_rate=config.frequency_hz,
        translations=np.asarray([config.translation]),
        orientations=np.asarray([config.orientation]),
    )
    return CameraSensor(
        camera_prim,
        # Config uses (width, height); the experimental runtime uses (height, width).
        resolution=(config.resolution[1], config.resolution[0]),
        annotators=list(config.annotators),
    )


def create_lidar(parent_prim_path: str, config: LidarConfig) -> LidarSensor:
    """Create a configured rotating RTX lidar."""
    lidar_prim = Lidar.create(
        f"{parent_prim_path}/{config.prim_name}",
        config=config.config,
        tick_rate=config.frequency_hz,
        translations=np.asarray([config.translation]),
        orientations=np.asarray([config.orientation]),
        # These rates must match for correct full-scan accumulation.
        attributes={"omni:sensor:Core:scanRateBaseHz": config.frequency_hz},
    )
    return LidarSensor(lidar_prim, annotators=list(config.annotators))


def _mount(
    path: str,
    translation: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> XformPrim:
    """Create a transform-only local frame."""
    if stage_utils.get_current_stage().GetPrimAtPath(path).IsValid():
        raise ValueError(f"Mount prim already exists: {path}")
    stage_utils.define_prim(path, type_name="Xform")
    mount = XformPrim(path, reset_xform_op_properties=True)
    mount.set_local_poses(
        translations=np.asarray([translation]),
        orientations=np.asarray([orientation]),
    )
    return mount


def create_nova_carter_sensors(
    chassis_prim_path: str,
    config: NovaCarterSensorConfig,
) -> NovaCarterSensors:
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

    rig_path = f"{chassis_prim_path}/{config.rig.prim_name}"
    if stage.GetPrimAtPath(rig_path).IsValid():
        raise ValueError(f"Sensor rig already exists: {rig_path}")

    _mount(rig_path, config.rig.translation, config.rig.orientation)
    for mount in config.rig.mounts:
        _mount(f"{rig_path}/{mount.prim_name}", mount.translation, mount.orientation)

    return NovaCarterSensors(
        imu=create_imu(f"{rig_path}/{config.imu.parent_prim}", config.imu),
        camera=create_camera(f"{rig_path}/{config.camera.parent_prim}", config.camera),
        lidar=create_lidar(f"{rig_path}/{config.lidar.parent_prim}", config.lidar),
    )
