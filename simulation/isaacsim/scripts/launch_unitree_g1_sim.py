"""Standalone Isaac Sim launcher for a sensor-equipped Unitree G1."""

import argparse
import sys
from pathlib import Path

import traceback

from isaacsim import SimulationApp


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SENSOR_CONFIG = (
    REPOSITORY_ROOT / "robots" / "unitree_g1" / "config" / "sensors.yaml"
)
PHYSICS_HZ = 200.0
RENDER_HZ = 60.0


def parse_args() -> argparse.Namespace:
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

# Isaac Sim must be running before importing Omniverse, Core, or sensor modules.
simulation_app = SimulationApp({"headless": args.headless})

# Make project helpers importable when run directly through python.sh.
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import omni.timeline

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path

from simulation.isaacsim.scripts.sensors import (
    attach_sensors_to_robot,
    load_sensor_config,
)
from robots.unitree_g1.spawn import DEFAULT_G1_PRIM_PATH, spawn_g1


def setup_stage() -> None:
    """Create a stage containing the standard Isaac Sim environment."""
    stage_utils.create_new_stage()

    assets_root = get_assets_root_path()
    
    if not assets_root:
        raise RuntimeError("Isaac Sim's asset root could not be resolved.")

    stage_utils.add_reference_to_stage(
        usd_path=(
            f"{assets_root}/Isaac/Environments/Grid/default_environment.usd"
        ),
        path="/World/Environment",
    )


def main() -> None:
    sensor_config = load_sensor_config(args.sensor_config)
    print("-----------SETTING UP STAGE------------")
    setup_stage()

    print("----- SPAWNING ROBOT ------")
    robot = spawn_g1(prim_path=DEFAULT_G1_PRIM_PATH)

    SimulationManager.setup_simulation(dt=1.0 / PHYSICS_HZ)
    RenderingManager.set_dt(1.0 / RENDER_HZ)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    try:
        # Warm up physics before reading articulation link metadata.
        print("1")
        simulation_app.update() 

        print("2")
        sensors = attach_sensors_to_robot(
            robot, sensor_config, robot_root_path=DEFAULT_G1_PRIM_PATH
        )

        # Process the newly authored sensor prims and render products.
        print("3")
        simulation_app.update()
        print("4")

        print(
            f"Spawned Unitree G1 with {robot.num_dofs} DOFs and attached "
            f"{type(sensors.imu).__name__}, "
            f"{type(sensors.head_camera).__name__}, "
            f"and {type(sensors.lidar).__name__}."
        )

        # Render at 60 simulated Hz while physics advances at 200 simulated Hz.
        render_phase = 0.0

        print("------ RUNNING SIM ------", flush=True)
        print("is_running:", simulation_app.is_running(), flush=True)
        print("is_exiting:", simulation_app.is_exiting(), flush=True)
        while simulation_app.is_running():
            # SimulationManager.step()
            # render_phase += RENDER_HZ
            # if render_phase >= PHYSICS_HZ:
            #     RenderingManager.render()
            #     render_phase -= PHYSICS_HZ
            SimulationManager.step()
            RenderingManager.render()
            simulation_app.update()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        timeline.stop()


try:
    main()
except KeyboardInterrupt:
    pass
finally:
    simulation_app.close()
