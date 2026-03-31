# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""
GeneSim RL task: junpu_place_workpiece

EE control mode: actions = [pos_l(3), euler_xyz_l(3), pos_r(3), euler_xyz_r(3), gripper_l, gripper_r].
IK is solved on the simulation side (mujoco_ros_node.py).
"""

from rlinf.envs.geniesim import register_geniesim_env
from rlinf.envs.geniesim.geniesim_env import GenieSimBaseEnv


@register_geniesim_env("junpu_place_workpiece")
class JunpuPlaceWorkpieceEnv(GenieSimBaseEnv):
    """
    Minimal subclass for junpu_place_workpiece task.
    All logic is inherited from GenieSimBaseEnv; ADER evaluation is disabled
    (enable_reward: false) — reward must be provided by an external supervisor.
    """
    pass
