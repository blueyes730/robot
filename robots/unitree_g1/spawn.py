"""Helpers for spawning the Unitree G1 Isaac Sim asset."""

from collections.abc import Sequence

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.prims import Articulation
from isaacsim.storage.native import get_assets_root_path


G1_ASSET_PATH = "/Isaac/Robots/Unitree/G1/g1.usd"
DEFAULT_G1_PRIM_PATH = "/World/G1"


def spawn_g1(
    prim_path: str = DEFAULT_G1_PRIM_PATH,
    position: Sequence[float] = (0.0, 0.0, 0.78),
) -> Articulation:
    """Spawn the Unitree G1 and return its articulation wrapper."""
    if len(position) != 3:
        raise ValueError("position must contain exactly three values")

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError(
            "Isaac Sim's asset root could not be resolved. Check the asset "
            "server configuration and network connection."
        )

    stage_utils.add_reference_to_stage(
        usd_path=f"{assets_root}{G1_ASSET_PATH}",
        path=prim_path,
    )

    robot = Articulation(prim_path)
    robot.set_world_poses(
        positions=np.asarray([position], dtype=np.float64),
    )

    return robot
