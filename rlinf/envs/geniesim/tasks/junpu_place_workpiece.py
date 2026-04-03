# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""
GeneSim RL task: junpu_place_workpiece

EE control mode: actions = [pos_l(3), euler_xyz_l(3), pos_r(3), euler_xyz_r(3), gripper_l, gripper_r].
IK is solved on the simulation side (mujoco_ros_node.py).

Reward:
    Dense: negative L2 distance between workpiece_r and /World/workspace01.
    The agent is rewarded for bringing the workpiece closer to the workspace.
Termination:
    Placeholder: never terminates early (episodes end via truncation at
    max_episode_steps).  Replace this with e.g. success detection from
    task sensor data when available.
"""

import numpy as np
import torch

from rlinf.envs.geniesim import register_geniesim_env
from rlinf.envs.geniesim.geniesim_env import GenieSimBaseEnv


@register_geniesim_env("junpu_place_workpiece")
class JunpuPlaceWorkpieceEnv(GenieSimBaseEnv):
    """
    RLinf RL env for the Junpu pick-and-place task.

    State layout (dim=40):
        [0:7]   arm_l joint positions
        [7:14]  arm_r joint positions
        [14:21] arm_l joint velocities
        [21:28] arm_r joint velocities
        [28:31] left  EE position  (x, y, z)  in base_link frame
        [31:34] left  EE euler XYZ (roll, pitch, yaw)
        [34:37] right EE position  (x, y, z)  in base_link frame
        [37:40] right EE euler XYZ (roll, pitch, yaw)

    info["body_poses"]:
        "workpiece_r"       -> (num_envs, 7) world-frame [x,y,z,qw,qx,qy,qz]
        "/World/workspace01" -> (num_envs, 7) world-frame [x,y,z,qw,qx,qy,qz]
    """

    def step(self, actions, auto_reset: bool = True):
        obs, _rewards, terminated, truncated, infos = super().step(
            actions, auto_reset=auto_reset
        )
        rewards = self._compute_reward(infos)
        infos["reward_detail"] = {"distance": self._last_distance.clone()}
        return obs, rewards, terminated, truncated, infos

    def _compute_reward(self, infos) -> torch.Tensor:
        body_poses = infos.get("body_poses")
        if body_poses is None:
            n = infos["episode"]["episode_len"].shape[0]
            self._last_distance = torch.zeros(n)
            return torch.full((n,), -0.01, dtype=torch.float32)

        wp = body_poses.get("workpiece_r")
        ws = body_poses.get("/World/workspace01")
        if wp is None or ws is None:
            n = infos["episode"]["episode_len"].shape[0]
            self._last_distance = torch.zeros(n)
            return torch.full((n,), -0.01, dtype=torch.float32)

        wp_pos = torch.from_numpy(wp[:, :3].copy())
        ws_pos = torch.from_numpy(ws[:, :3].copy())
        dist = torch.linalg.norm(wp_pos - ws_pos, dim=-1)
        self._last_distance = dist
        return -dist
