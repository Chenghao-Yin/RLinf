# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SpacemouseSimIntervention — wraps a vectorized GenieSimEnv and injects
# SpaceMouse teleoperation for env_0.
#
# Action layout assumed (ee mode, action_dim=14):
#   [0:3]  left  EE position  (base_link frame, metres)
#   [3:6]  left  EE euler XYZ (base_link frame, radians)
#   [6:9]  right EE position  (base_link frame, metres)
#   [9:12] right EE euler XYZ (base_link frame, radians)
#   [12]   left  gripper command
#   [13]   right gripper command
#
# State layout assumed (state_dim=40):
#   [0:7]   left  arm joint positions
#   [7:14]  right arm joint positions
#   [14:21] left  arm joint velocities
#   [21:28] right arm joint velocities
#   [28:31] left  EE position
#   [31:34] left  EE RPY
#   [34:37] right EE position
#   [37:40] right EE RPY
#
# SpaceMouse button semantics (see ``button_mode`` on ``SpacemouseSimIntervention``):
#
#   ``legacy`` (default):
#     Translation/rotation → right arm EEF delta (accumulated → absolute target)
#     Left  button → right arm gripper close
#     Right button → episode done + success (save demo when CollectEpisode.only_success)
#
#   ``junpu`` (Junpu collect_sim_data):
#     Translation/rotation → same as above
#     Left  button → episode done + success (save)
#     Right button → episode done + failure (discard when only_success=True)
#     No gripper mapping from buttons

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# SpaceMouse expert interfaces
# ---------------------------------------------------------------------------

class SpaceMouseExpertBase:
    """Abstract base for SpaceMouse-like experts."""

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Return (action[6], buttons[2]).

        action: [dx, dy, dz, droll, dpitch, dyaw] — unit-ish values in
                SpaceMouse sensor frame (already remapped to robot base frame).
        buttons: [left_button, right_button] — each 0 or 1.
        """
        raise NotImplementedError

    def on_episode_reset(self):
        """Called when the environment resets so the expert can update state."""


class FakeSpaceMouseExpert(SpaceMouseExpertBase):
    """Deterministic SpaceMouse simulator for integration testing.

    ``button_semantics="legacy"`` (default): each episode plays:

        Phase 0  (move_steps)    : translate right arm in +Y direction
        Phase 1  (gripper_steps) : hold position, press left button (close)
        Phase 2  (lift_steps)    : translate right arm in +Z direction
        Done                     : press right button (episode done + success)

    ``button_semantics="junpu"``: no gripper phase; sequence is move, lift,
    then **left** button once (episode done + success) — matches Junpu collect
    semantics (left = save).

    After the done signal the step counter resets automatically so the expert
    can drive as many episodes as needed without external intervention.
    """

    def __init__(
        self,
        move_steps: int = 40,
        gripper_steps: int = 15,
        lift_steps: int = 40,
        translation_speed: float = 1.0,
        button_semantics: str = "legacy",
    ):
        self.move_steps = move_steps
        self.gripper_steps = gripper_steps
        self.lift_steps = lift_steps
        self.translation_speed = translation_speed
        self.button_semantics = button_semantics

        self._step = 0
        self._done_returned = False

    # ------------------------------------------------------------------

    def get_action(self) -> Tuple[np.ndarray, list]:
        s = self._step

        if self.button_semantics == "junpu":
            total_motion = self.move_steps + self.lift_steps
            if s < self.move_steps:
                action = np.array([0.0, self.translation_speed, 0.0, 0.0, 0.0, 0.0])
                buttons = [0, 0]
            elif s < total_motion:
                action = np.array([0.0, 0.0, self.translation_speed, 0.0, 0.0, 0.0])
                buttons = [0, 0]
            else:
                if not self._done_returned:
                    self._done_returned = True
                    return np.zeros(6), [1, 0]  # left = save
                action = np.zeros(6)
                buttons = [0, 0]
            self._step += 1
            return action, buttons

        total_motion = self.move_steps + self.gripper_steps + self.lift_steps

        if s < self.move_steps:
            # Phase 0: move in +Y
            action = np.array([0.0, self.translation_speed, 0.0, 0.0, 0.0, 0.0])
            buttons = [0, 0]
        elif s < self.move_steps + self.gripper_steps:
            # Phase 1: close gripper
            action = np.zeros(6)
            buttons = [1, 0]
        elif s < total_motion:
            # Phase 2: move up (+Z)
            action = np.array([0.0, 0.0, self.translation_speed, 0.0, 0.0, 0.0])
            buttons = [0, 0]
        else:
            # Done: press right button once, then idle
            if not self._done_returned:
                self._done_returned = True
                return np.zeros(6), [0, 1]
            action = np.zeros(6)
            buttons = [0, 0]

        self._step += 1
        return action, buttons

    def on_episode_reset(self):
        self._step = 0
        self._done_returned = False


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class SpacemouseSimIntervention:
    """Wraps a vectorized GenieSimEnv to inject SpaceMouse teleoperation.

    Only env ``intervention_env_id`` (default: 0) is driven by the SpaceMouse;
    all other envs receive the unmodified policy actions.

    The SpaceMouse outputs a 6-DoF *delta* which is accumulated on top of the
    right arm's current end-effector target.  The underlying environment
    expects an *absolute* EEF target in ee-control mode, so this wrapper
    maintains an internal ``_current_target`` that starts at the EE pose
    read from the reset observation and is updated each step.

    Button semantics (``button_mode``)
    ----------------------------------
    ``legacy`` (default):
        Left  button : close the right arm gripper.
        Right button : episode done and successful (save when CollectEpisode
                       uses ``only_success=True``).

    ``junpu`` (Junpu demonstration collection):
        Left  button : episode done and successful (save).
        Right button : episode done and unsuccessful (discard when
                       ``only_success=True``).
        No gripper action from buttons.

    ``intervene_action`` / ``intervene_flag``
    -----------------------------------------
    When the SpaceMouse is active (non-zero motion or relevant buttons), the
    wrapper writes::

        info["intervene_action"] = actions[intervention_env_id]   # tensor
        info["intervene_flag"]   = torch.ones(1, dtype=torch.bool)

    CollectEpisode's LeRobot writer uses these fields to replace the
    policy action with the expert action when saving demos.

    Args:
        env: Vectorized GenieSimEnv (or compatible).
        expert: SpaceMouseExpertBase instance.
        action_scale: Multiplier applied to SpaceMouse translational output
            before adding to the EEF position target (metres per unit).
        rotation_scale: Multiplier applied to SpaceMouse rotational output
            before adding to the EEF orientation target (radians per unit).
        intervention_env_id: Index of the env instance driven by SpaceMouse.
        button_mode: ``"legacy"`` or ``"junpu"`` — see class docstring.
    """

    # Action vector indices (ee mode, action_dim=14)
    _L_POS = slice(0, 3)
    _L_ROT = slice(3, 6)
    _R_POS = slice(6, 9)
    _R_ROT = slice(9, 12)
    _L_GRIP = 12
    _R_GRIP = 13

    # State vector indices for EE poses (state_dim=40)
    _S_L_EE_POS = slice(28, 31)
    _S_L_EE_ROT = slice(31, 34)
    _S_R_EE_POS = slice(34, 37)
    _S_R_EE_ROT = slice(37, 40)

    def __init__(
        self,
        env,
        expert: SpaceMouseExpertBase,
        action_scale: float = 0.01,
        rotation_scale: float = 0.05,
        intervention_env_id: int = 0,
        button_mode: str = "legacy",
    ):
        self.env = env
        self.expert = expert
        self.action_scale = action_scale
        self.rotation_scale = rotation_scale
        self.intervention_env_id = intervention_env_id
        if button_mode not in ("legacy", "junpu"):
            raise ValueError(
                f"button_mode must be 'legacy' or 'junpu', got {button_mode!r}"
            )
        self.button_mode = button_mode

        self._current_target: Optional[np.ndarray] = None  # shape (14,)
        self._last_intervene_time: float = 0.0
        self._intervene_timeout: float = 0.5  # seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_target_from_obs(self, obs: Dict[str, Any]) -> None:
        """Initialise both-arm EEF targets from the current observation."""
        states = obs["states"]  # [N, 40] torch tensor
        s = states[self.intervention_env_id].cpu().numpy().astype(np.float32)

        target = np.zeros(14, dtype=np.float32)
        target[self._L_POS] = s[self._S_L_EE_POS]
        target[self._L_ROT] = s[self._S_L_EE_ROT]
        target[self._R_POS] = s[self._S_R_EE_POS]
        target[self._R_ROT] = s[self._S_R_EE_ROT]
        target[self._L_GRIP] = 1.0   # open
        target[self._R_GRIP] = 1.0   # open
        self._current_target = target

    # ------------------------------------------------------------------
    # gym-compatible interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        obs, info = self.env.reset(**kwargs)
        self._init_target_from_obs(obs)
        self._last_intervene_time = 0.0
        self.expert.on_episode_reset()
        return obs, info

    def step(
        self,
        actions,
        **kwargs,
    ) -> Tuple[Dict, Any, Any, Any, Dict]:
        """Inject SpaceMouse action for env_0, pass others through unchanged."""
        # Work with a mutable float32 tensor copy
        if isinstance(actions, torch.Tensor):
            actions = actions.clone().float()
        else:
            actions = torch.tensor(np.asarray(actions, dtype=np.float32))

        sm_delta, buttons = self.expert.get_action()
        left_btn = bool(buttons[0]) if len(buttons) > 0 else False
        right_btn = bool(buttons[1]) if len(buttons) > 1 else False

        has_motion = float(np.linalg.norm(sm_delta)) > 1e-4
        if self.button_mode == "junpu":
            intervened = has_motion or left_btn or right_btn
        else:
            intervened = has_motion or left_btn

        if intervened:
            self._last_intervene_time = time.time()

        # Update right arm EEF target
        if has_motion:
            sm_delta[3], sm_delta[4] = sm_delta[4], -sm_delta[3]
            self._current_target[self._R_POS] += sm_delta[:3] * self.action_scale
            self._current_target[self._R_ROT] += sm_delta[3:] * self.rotation_scale

        # Gripper: legacy only — left button closes right gripper
        if self.button_mode == "legacy" and left_btn:
            self._current_target[self._R_GRIP] = -1.0   # close

        # Always fill env_0 with the maintained target (holds position when idle)
        eid = self.intervention_env_id
        for i in range(14):
            actions[eid, i] = float(self._current_target[i])

        # Step the underlying env (auto_reset disabled; caller handles resets)
        obs, reward, terminated, truncated, info = self.env.step(
            actions, auto_reset=False, **kwargs
        )

        # Episode end buttons
        if self.button_mode == "junpu":
            if left_btn or right_btn:
                terminated = terminated.clone() if isinstance(terminated, torch.Tensor) \
                    else torch.tensor(np.array(terminated))
                terminated[eid] = True
                success = bool(left_btn)
                info["success"] = success
                if "episode" in info and isinstance(info["episode"], dict) and success:
                    sc = info["episode"].get("success_once")
                    if isinstance(sc, torch.Tensor):
                        sc = sc.clone()
                        sc[eid] = True
                        info["episode"]["success_once"] = sc
        elif right_btn:
            terminated = terminated.clone() if isinstance(terminated, torch.Tensor) \
                else torch.tensor(np.array(terminated))
            terminated[eid] = True
            info["success"] = True
            if "episode" in info and isinstance(info["episode"], dict):
                sc = info["episode"].get("success_once")
                if isinstance(sc, torch.Tensor):
                    sc = sc.clone()
                    sc[eid] = True
                    info["episode"]["success_once"] = sc

        # Record intervene fields for CollectEpisode / LeRobot writer
        active = intervened or (
            time.time() - self._last_intervene_time < self._intervene_timeout
        )
        if active:
            info["intervene_action"] = actions[eid]
            info["intervene_flag"] = torch.ones(1, dtype=torch.bool)

        return obs, reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    # Delegate attribute lookups so wrappers stacked on top work correctly
    def __getattr__(self, name: str):
        return getattr(self.env, name)
