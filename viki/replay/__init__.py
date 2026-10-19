"""
viki.replay
-----------
Pipeline stage 5: plan.h5 -> replay.h5 (paper §3.8).

Execute on the manipulator, log proprioception, screen feasibility.
UR10e arm-only RTDE execution is implemented; UR3 and the re-solve loop are stubs.
"""

from viki.replay.driver import DryRunDriver, RobotDriver, load_driver  # noqa: F401
from viki.replay.run import replay_episode  # noqa: F401
from viki.replay.screen import Verdict, screen  # noqa: F401
from viki.replay.plan import URReplayOptions, check_replay
from viki.replay.ur import UR10eDriver, inspect_ur10e
from viki.replay.manager import ReplayManager
from viki.replay.simulation import preview_mujoco
from viki.replay.live import LiveController
from viki.replay.benchmark import benchmark_live_ik
from viki.replay.hand_preview import DG5FHandPreview

__all__ = [
    "replay_episode",
    "RobotDriver",
    "DryRunDriver",
    "load_driver",
    "screen",
    "Verdict",
    "UR10eDriver",
    "URReplayOptions",
    "check_replay",
    "inspect_ur10e",
    "ReplayManager",
    "preview_mujoco",
    "LiveController",
    "benchmark_live_ik",
    "DG5FHandPreview",
]
