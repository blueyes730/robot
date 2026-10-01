"""Helpers for spawning the NVIDIA Nova Carter Isaac Sim asset."""

from collections.abc import Sequence

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.prims import Articulation
from isaacsim.storage.native import get_assets_root_path


NOVA_CARTER_ASSET_PATH = "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
DEFAULT_NOVA_CARTER_PRIM_PATH = "/World/NovaCarter"
DEFAULT_NOVA_CARTER_POSITION = (0.0, 0.0, 0.0)
DEFAULT_NOVA_CARTER_ORIENTATION = (1.0, 0.0, 0.0, 0.0)


def spawn_nova_carter(
    prim_path: str = DEFAULT_NOVA_CARTER_PRIM_PATH,
    position: Sequence[float] = DEFAULT_NOVA_CARTER_POSITION,
    orientation: Sequence[float] = DEFAULT_NOVA_CARTER_ORIENTATION,
) -> Articulation:
    """Spawn Nova Carter and return its articulation wrapper.

    ``orientation`` is a world-frame quaternion in ``wxyz`` order.
    """
    if len(position) != 3:
        raise ValueError("position must contain exactly three values")
    if len(orientation) != 4:
        raise ValueError("orientation must contain exactly four values (wxyz)")

    stage = stage_utils.get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at {prim_path!r}")

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError(
            "Isaac Sim's asset root could not be resolved. Check the asset "
            "server configuration and network connection."
        )

    stage_utils.add_reference_to_stage(
        usd_path=f"{assets_root}{NOVA_CARTER_ASSET_PATH}",
        path=prim_path,
    )

    robot = Articulation(prim_path)
    robot.set_world_poses(
        positions=np.asarray([position], dtype=np.float64),
        orientations=np.asarray([orientation], dtype=np.float64),
    )
    return robot
