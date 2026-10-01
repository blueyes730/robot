"""Configuration, creation, and attachment helpers for G1 sensors."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import omni.usd
from isaacsim.core.experimental.prims import Articulation
from isaacsim.sensors.experimental.physics import IMU, IMUSensor
from isaacsim.sensors.experimental.rtx import (
    CameraSensor,
    Lidar,
    LidarSensor,
    RtxCamera,
)
from pxr import Usd


@dataclass(frozen=True)
class ImuConfig:
    parent_link: str
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
class G1SensorConfig:
    imu: ImuConfig
    head_camera: CameraConfig
    lidar: LidarConfig


@dataclass(frozen=True)
class G1Sensors:
    """Runtime handles for all sensors mounted on a G1."""

    imu: IMUSensor
    head_camera: CameraSensor
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


def _parse_sensor_config(data: Mapping[str, Any]) -> G1SensorConfig:
    sensors = _mapping(data.get("sensors"), "sensors")
    imu = _mapping(sensors.get("imu"), "sensors.imu")
    camera = _mapping(sensors.get("head_camera"), "sensors.head_camera")
    lidar = _mapping(sensors.get("lidar"), "sensors.lidar")

    return G1SensorConfig(
        imu=ImuConfig(
            parent_link=_string(imu, "parent_link"),
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
        head_camera=CameraConfig(
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


def load_sensor_config(config_path: str | Path) -> G1SensorConfig:
    """Load and validate a G1 sensor configuration from YAML or JSON."""
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


def create_head_camera(
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
        resolution=config.resolution,
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


def find_articulation_link(robot: Articulation, link_name: str) -> str:
    """Return an exact articulation link path for a physics-based sensor."""
    link_paths = robot.link_paths
    if len(link_paths) != 1:
        raise ValueError("Sensor attachment expects exactly one G1 articulation.")

    paths_by_name = {
        name.lower(): path
        for name, path in zip(robot.link_names, link_paths[0], strict=True)
    }
    path = paths_by_name.get(link_name.lower())
    if path is None:
        available = ", ".join(robot.link_names)
        raise RuntimeError(
            f"G1 link '{link_name}' was not found.\n"
            f"Available links:\n{available}"
        )
    return path


def find_prim_under_robot(robot_root_path: str, prim_name: str) -> str:
    """Find one mounting prim, including non-articulation child Xforms."""
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")
    root = stage.GetPrimAtPath(robot_root_path)
    if not root.IsValid():
        raise RuntimeError(f"Robot root '{robot_root_path}' was not found.")

    matches = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim.GetName().lower() == prim_name.lower()
    ]
    if not matches:
        raise RuntimeError(
            f"Mounting prim '{prim_name}' was not found under '{robot_root_path}'."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Mounting prim '{prim_name}' is ambiguous: {matches}"
        )
    return matches[0]


def attach_sensors_to_robot(
    robot: Articulation,
    config: G1SensorConfig,
    robot_root_path: str,
) -> G1Sensors:
    """Mount the IMU to a rigid link and RTX sensors to USD frames."""
    imu_parent = find_articulation_link(robot, config.imu.parent_link)
    camera_parent = find_prim_under_robot(
        robot_root_path, config.head_camera.parent_prim
    )
    lidar_parent = find_prim_under_robot(robot_root_path, config.lidar.parent_prim)

    return G1Sensors(
        imu=create_imu(imu_parent, config.imu),
        head_camera=create_head_camera(camera_parent, config.head_camera),
        lidar=create_lidar(lidar_parent, config.lidar),
    )
