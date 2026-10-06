"""Standalone Isaac Sim launcher for the NVIDIA Nova Carter robot."""

import argparse
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SENSOR_CONFIG = (
    REPOSITORY_ROOT / "robots" / "nova_carter" / "config" / "sensors.yaml"
)
PHYSICS_HZ = 200.0
RENDER_HZ = 60.0


def parse_args() -> argparse.Namespace:
    """Parse launcher arguments before starting Isaac Sim."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the Isaac Sim window.",
    )
    parser.add_argument(
        "--sensor-config",
        type=Path,
        default=DEFAULT_SENSOR_CONFIG,
        help="Path to a .yaml, .yml, or .json sensor configuration.",
    )
    return parser.parse_args()


args = parse_args()

# Isaac Sim must be running before importing Omniverse or Core modules.
simulation_app = SimulationApp({"headless": args.headless})

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import omni.timeline

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path

# Register ROS writers and the IMU reader before importing the adapter.
app_utils.enable_extension("isaacsim.ros2.bridge")
app_utils.enable_extension("isaacsim.sensors.physics.nodes")
simulation_app.update()

from robots.nova_carter.isaacsim_ros_publishers import create_isaacsim_ros_publishers
from robots.nova_carter.spawn import (
    DEFAULT_NOVA_CARTER_PRIM_PATH,
    spawn_nova_carter,
)
from robots.nova_carter.sensors import create_nova_carter_sensors, load_sensor_config


def setup_stage() -> None:
    """Create a stage containing the standard Isaac Sim grid environment."""
    stage_utils.create_new_stage()
    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Isaac Sim's asset root could not be resolved.")

    stage_utils.add_reference_to_stage(
        usd_path=f"{assets_root}/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
        path="/World/Environment",
    )


def main() -> None:
    """Set up the stage, spawn Nova Carter, and run the simulation loop."""
    sensor_config = load_sensor_config(args.sensor_config)
    print("Setting up stage...")
    setup_stage()

    print(f"Spawning Nova Carter at {DEFAULT_NOVA_CARTER_PRIM_PATH}...")
    robot = spawn_nova_carter()

    # Resolve referenced child links before attaching sensors to the rigid chassis.
    simulation_app.update()
    simulation_app.update()
    chassis_paths = [
        path
        for name, path in zip(robot.link_names, robot.link_paths[0], strict=True)
        if name == sensor_config.rig.parent_link
    ]
    if len(chassis_paths) != 1:
        raise RuntimeError(
            f"Expected one Nova Carter link '{sensor_config.rig.parent_link}', "
            f"found {chassis_paths}"
        )
    chassis_path = chassis_paths[0]
    sensors = create_nova_carter_sensors(chassis_path, sensor_config)
    rig_path = f"{chassis_path}/{sensor_config.rig.prim_name}"
    sensor_paths = {
        name: f"{rig_path}/{config.parent_prim}/{config.prim_name}"
        for name, config in (
            ("imu", sensor_config.imu),
            ("camera", sensor_config.camera),
            ("lidar", sensor_config.lidar),
        )
    }
    stage = stage_utils.get_current_stage()
    for name, path in sensor_paths.items():
        if not stage.GetPrimAtPath(path).IsValid():
            raise RuntimeError(f"{name} prim was not created: {path}")
    print(f"Nova Carter spawned at {DEFAULT_NOVA_CARTER_PRIM_PATH}")
    print(f"Sensor rig:\n  chassis: {chassis_path}")
    for name, path in sensor_paths.items():
        print(f"  {name}: {path}")

    SimulationManager.setup_simulation(dt=1.0 / PHYSICS_HZ)
    RenderingManager.set_dt(1.0 / RENDER_HZ)

    simulation_app.update()
    ros_publishers = create_isaacsim_ros_publishers(sensors)
    simulation_app.update()

    timeline = omni.timeline.get_timeline_interface()
    try:
        timeline.play()
        simulation_app.update()
        print("Nova Carter spawned successfully.")
        print("Publishing /camera/image_raw, /camera/camera_info, /lidar/points, /imu/data")
        print("Starting simulation...")
        while simulation_app.is_running():
            SimulationManager.step()
            RenderingManager.render()
            simulation_app.update()
    except KeyboardInterrupt:
        pass
    finally:
        timeline.stop()
        ros_publishers.close()


try:
    main()
except KeyboardInterrupt:
    pass
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
