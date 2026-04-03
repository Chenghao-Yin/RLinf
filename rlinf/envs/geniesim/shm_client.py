# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# GenieSimShmClient — RLinf-native client for the GeneSim simulation container.
#
# Replaces GenieSimVectorEnv (from the geniesim Python package) with a
# stdlib-only implementation that connects to the container via shared memory.
#
# All communication with the simulation happens through two SHM segments per env:
#   Frame SHM  — camera images written by Isaac Sim renderer
#   Ctrl SHM   — states/actions/reset flags written by MuJoCo node
#
# NO geniesim, rclpy, or ROS dependencies are required on the host.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from multiprocessing import resource_tracker as _resource_tracker
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rlinf.envs.geniesim.shm_layout import (
    BODY_POSE_DIM,
    CTRL_HEADER_BYTES,
    NUM_CAMS,
    RESET_DONE,
    RESET_IDLE,
    RESET_REQUESTED,
    SHM_HEADER_BYTES,
    ctrl_shm_name,
    ctrl_total_bytes,
    shm_total_bytes,
)


# ---------------------------------------------------------------------------- #
# Config dataclass (mirrors GenieSimVectorEnvConfig from the geniesim package)
# ---------------------------------------------------------------------------- #

@dataclass
class GenieSimVectorEnvConfig:
    """Configuration for GenieSimShmClient (or the legacy GenieSimVectorEnv).

    All fields match geniesim.rl.envs.geniesim_vec_env.GenieSimVectorEnvConfig
    so that SimContainerManager.pm_kwargs_from_vec_cfg() works unchanged.
    """
    # Task / assets
    mjcf_path: str = ""
    scene_usd: str = ""
    robot_usd: str = ""
    robot_prim: str = "/robot"
    task_file: str = ""
    task_name: str = ""
    task_description: str = ""
    robot_cfg: str = "G2_omnipicker"
    robot_type: str = "G2"
    task_instance_id: int = 0

    # Parallel environments
    num_envs: int = 1

    # Episode
    max_episode_steps: int = 300
    enable_reward: bool = False
    reward_coef: float = 1.0
    use_rel_reward: bool = False
    ignore_terminations: bool = False
    auto_reset: bool = True

    # Camera
    cam_width: int = 640
    cam_height: int = 480
    main_cam_prim: str = "/camera_main"
    wrist_cam_prim: str = ""

    # SHM
    shm_name: str = "geniesim_frames"
    shm_open_timeout_sec: int = 180

    # Sim
    physics_hz: float = 1000.0
    render_hz: float = 30.0
    headless: bool = True
    ros_domain_id: int = 0
    isaac_python: str = "/isaac-sim/python.sh"
    mujoco_python: str = ""

    # Robot state / control
    state_dim: int = 28
    action_dim: int = 14
    state_joint_offset: int = 0
    ctrl_offset: int = 0
    ctrl_offset_r: int = -1
    control_mode: str = "joint"
    gripper_ctrl_l: int = -1
    gripper_ctrl_r: int = -1
    ee_body_l: str = "arm_l_link7"
    ee_body_r: str = "arm_r_link7"
    ik_max_iter: int = 10
    ik_damp: float = 0.05

    # Randomisation / init
    randomization_cfg_json: str = ""
    init_qpos_json: str = ""
    reset_ee_r_json: str = ""
    seed: int = 42

    # Ground-truth info: body names whose world-frame poses are passed via SHM
    info_body_names: List[str] = field(default_factory=list)

    # Container mode
    attach_to_running: bool = False


# ---------------------------------------------------------------------------- #
# SHM client
# ---------------------------------------------------------------------------- #

class GenieSimShmClient:
    """
    Lightweight environment client that talks to a running GeneSim container
    exclusively via shared memory.

    This class reimplements the attach-mode subset of GenieSimVectorEnv
    without any dependency on the geniesim Python package, rclpy, or ROS.

    Interface (matches GenieSimVectorEnv)::

        reset(env_idx=None) -> (obs_dict, {})
        step(actions, auto_reset=False) -> (obs, rewards, terminated, truncated, infos)
        close()
    """

    def __init__(self, cfg: GenieSimVectorEnvConfig):
        self.cfg = cfg
        self.num_envs = cfg.num_envs

        self._elapsed_steps = np.zeros(cfg.num_envs, dtype=np.int32)
        self._episode_returns = np.zeros(cfg.num_envs, dtype=np.float32)
        self._success_once = np.zeros(cfg.num_envs, dtype=bool)

        # Frame SHM (camera images, written by Isaac Sim)
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._frames: Optional[np.ndarray] = None
        self._frame_counter: Optional[np.ndarray] = None
        self._open_shm(max_attempts=cfg.shm_open_timeout_sec)

        # Ctrl SHMs (states / actions / reset / info, one per env)
        self._ctrl_shms: List[shared_memory.SharedMemory] = []
        self._ctrl_counters: List[np.ndarray] = []
        self._ctrl_reset_flags: List[np.ndarray] = []
        self._ctrl_states_bufs: List[np.ndarray] = []
        self._ctrl_actions_bufs: List[np.ndarray] = []
        self._ctrl_info_bufs: List[Optional[np.ndarray]] = []
        self._info_body_names: List[str] = list(cfg.info_body_names)
        self._info_dim = len(self._info_body_names) * BODY_POSE_DIM
        self._open_ctrl_shms(max_attempts=cfg.shm_open_timeout_sec)

        print(
            f"[GenieSimShmClient] Initialised | num_envs={self.num_envs} "
            f"state_dim={cfg.state_dim} action_dim={cfg.action_dim} "
            f"info_dim={self._info_dim}"
        )

    # ---------------------------------------------------------------------- #
    # SHM attachment
    # ---------------------------------------------------------------------- #

    def _open_shm(self, max_attempts: int = 180):
        h, w = self.cfg.cam_height, self.cfg.cam_width
        shm_bytes = shm_total_bytes(self.num_envs, h, w)
        for _ in range(max_attempts):
            try:
                self._shm = shared_memory.SharedMemory(
                    name=self.cfg.shm_name, create=False, size=shm_bytes
                )
                # SHM is owned by the container process.  Unregister from
                # Python's resource tracker to suppress PermissionError on exit.
                _resource_tracker.unregister(f"/{self.cfg.shm_name}", "shared_memory")
                break
            except FileNotFoundError:
                time.sleep(1.0)
        else:
            raise RuntimeError(
                f"[GenieSimShmClient] Frame SHM '{self.cfg.shm_name}' "
                f"not available after {max_attempts}s"
            )

        self._frames = np.ndarray(
            (self.num_envs, NUM_CAMS, h, w, 3),
            dtype=np.uint8,
            buffer=self._shm.buf,
            offset=SHM_HEADER_BYTES,
        )
        self._frame_counter = np.ndarray(
            (1,), dtype=np.uint32, buffer=self._shm.buf, offset=0
        )

    def _open_ctrl_shms(self, max_attempts: int = 180):
        _S = self.cfg.state_dim * 4
        _A = self.cfg.action_dim * 4
        _total = ctrl_total_bytes(self.cfg.state_dim, self.cfg.action_dim, self._info_dim)
        for i in range(self.num_envs):
            name = ctrl_shm_name(self.cfg.shm_name, i)
            shm = None
            for _ in range(max_attempts):
                try:
                    shm = shared_memory.SharedMemory(name=name, create=False, size=_total)
                    break
                except FileNotFoundError:
                    time.sleep(1.0)
            if shm is None:
                raise RuntimeError(
                    f"[GenieSimShmClient] Ctrl SHM '{name}' "
                    f"not available after {max_attempts}s"
                )
            _resource_tracker.unregister(f"/{name}", "shared_memory")
            self._ctrl_shms.append(shm)
            self._ctrl_counters.append(
                np.ndarray((1,), dtype=np.uint32, buffer=shm.buf, offset=0)
            )
            self._ctrl_reset_flags.append(
                np.ndarray((1,), dtype=np.uint32, buffer=shm.buf, offset=4)
            )
            self._ctrl_states_bufs.append(
                np.ndarray(
                    (self.cfg.state_dim,), dtype=np.float32,
                    buffer=shm.buf, offset=CTRL_HEADER_BYTES,
                )
            )
            self._ctrl_actions_bufs.append(
                np.ndarray(
                    (self.cfg.action_dim,), dtype=np.float32,
                    buffer=shm.buf, offset=CTRL_HEADER_BYTES + _S,
                )
            )
            if self._info_dim > 0:
                self._ctrl_info_bufs.append(
                    np.ndarray(
                        (self._info_dim,), dtype=np.float32,
                        buffer=shm.buf, offset=CTRL_HEADER_BYTES + _S + _A,
                    )
                )
            else:
                self._ctrl_info_bufs.append(None)
        print(
            f"[GenieSimShmClient] Ctrl SHMs attached | "
            f"state_dim={self.cfg.state_dim} action_dim={self.cfg.action_dim} "
            f"info_dim={self._info_dim}"
        )

    # ---------------------------------------------------------------------- #
    # Observation helpers
    # ---------------------------------------------------------------------- #

    def _wait_new_frame(self, timeout: float = 2.0):
        current = int(self._frame_counter[0])
        deadline = time.time() + timeout
        while time.time() < deadline:
            if int(self._frame_counter[0]) != current:
                return
            time.sleep(0.001)

    def _get_obs(self) -> Dict[str, Any]:
        self._wait_new_frame()
        h, w = self.cfg.cam_height, self.cfg.cam_width
        main_images = np.copy(self._frames[:, 0])       # [N, H, W, 3]
        wrist_images = (
            np.copy(self._frames[:, 1]) if self.cfg.wrist_cam_prim else None
        )
        states = np.stack([np.copy(b) for b in self._ctrl_states_bufs], axis=0)
        return {
            "main_images": main_images,
            "wrist_images": wrist_images,
            "states": states,
            "task_descriptions": [self.cfg.task_description] * self.num_envs,
        }

    # ---------------------------------------------------------------------- #
    # Action dispatch
    # ---------------------------------------------------------------------- #

    def _send_actions(self, actions: np.ndarray):
        """Write actions into ctrl SHMs.  actions: [N, action_dim]."""
        for i, buf in enumerate(self._ctrl_actions_bufs):
            n = min(actions.shape[1], len(buf))
            np.copyto(buf[:n], actions[i, :n].astype(np.float32))

    # ---------------------------------------------------------------------- #
    # Reset
    # ---------------------------------------------------------------------- #

    def _reset_env(self, env_idx: int):
        flag = self._ctrl_reset_flags[env_idx]
        flag[0] = RESET_REQUESTED
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if int(flag[0]) == RESET_DONE:
                flag[0] = RESET_IDLE
                return
            time.sleep(0.001)
        print(f"[GenieSimShmClient] reset timeout for env_{env_idx}")

    def reset(self, env_idx=None) -> Tuple[Dict, Dict]:
        if env_idx is None:
            indices = list(range(self.num_envs))
        elif isinstance(env_idx, int):
            indices = [env_idx]
        else:
            indices = list(env_idx)

        for i in indices:
            self._reset_env(i)
            self._elapsed_steps[i] = 0
            self._episode_returns[i] = 0.0
            self._success_once[i] = False

        obs = self._get_obs()
        return obs, {}

    # ---------------------------------------------------------------------- #
    # Step
    # ---------------------------------------------------------------------- #

    def _build_infos(self, rewards, terminated, truncated) -> Dict:
        infos: Dict = {
            "episode": {
                "success_once": self._success_once.copy(),
                "return": self._episode_returns.copy(),
                "episode_len": self._elapsed_steps.copy(),
                "reward": np.where(
                    self._elapsed_steps > 0,
                    self._episode_returns / np.maximum(self._elapsed_steps, 1),
                    0.0,
                ),
            },
            "task_progress": [[] for _ in range(self.num_envs)],
        }
        if self._info_dim > 0:
            infos["body_poses"] = self._read_body_poses()
        return infos

    def _read_body_poses(self) -> Dict[str, np.ndarray]:
        n = len(self._info_body_names)
        poses = {}
        for bname_idx, bname in enumerate(self._info_body_names):
            arr = np.zeros((self.num_envs, BODY_POSE_DIM), dtype=np.float32)
            for env_i in range(self.num_envs):
                buf = self._ctrl_info_bufs[env_i]
                if buf is not None:
                    off = bname_idx * BODY_POSE_DIM
                    arr[env_i] = buf[off:off + BODY_POSE_DIM]
            poses[bname] = arr
        return poses

    def _handle_auto_reset(
        self, dones: np.ndarray, final_obs: Dict, infos: Dict
    ) -> Tuple[Dict, Dict]:
        import copy
        _final_obs = copy.deepcopy(final_obs)
        _final_info = copy.deepcopy(infos)
        done_indices = np.where(dones)[0].tolist()
        obs, _ = self.reset(env_idx=done_indices)
        infos["final_observation"] = _final_obs
        infos["final_info"] = _final_info
        infos["_final_observation"] = dones
        infos["_final_info"] = dones
        return obs, infos

    def step(
        self,
        actions: np.ndarray,
        auto_reset: bool = True,
    ) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, Dict]:
        self._send_actions(actions)

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)

        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        dones = terminated | truncated

        self._episode_returns += rewards
        self._success_once |= terminated

        obs = self._get_obs()
        infos = self._build_infos(rewards, terminated, truncated)

        if self.cfg.ignore_terminations:
            infos["episode"]["success_at_end"] = terminated.copy()
            terminated = np.zeros_like(terminated)

        if dones.any() and auto_reset and self.cfg.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return obs, rewards, terminated, truncated, infos

    # ---------------------------------------------------------------------- #
    # Cleanup
    # ---------------------------------------------------------------------- #

    def close(self):
        for shm in self._ctrl_shms:
            try:
                shm.close()
            except Exception:
                pass
        self._ctrl_shms.clear()
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None
