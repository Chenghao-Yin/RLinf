# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Shared-memory layout constants for the GeneSim RL renderer pipeline.
#
# This file is an RLinf-local copy of
#   geniesim/rl/renderer/shm_layout.py
# so that the RLinf host process requires NO geniesim Python package.
#
# Both sides of the SHM boundary must agree on these constants.
# Update here if the container-side shm_layout.py changes.
#
# This file intentionally has NO external dependencies.

# ---------------------------------------------------------------------------
# Frame SHM  (name = shm_name, e.g. "geniesim_frames")
# Written by Isaac Sim renderer; read by GenieSimShmClient on the host.
# ---------------------------------------------------------------------------

NUM_CAMS: int = 2
SHM_HEADER_BYTES: int = 4


def shm_total_bytes(num_envs: int, height: int, width: int) -> int:
    """Total frame SHM size in bytes."""
    return SHM_HEADER_BYTES + num_envs * NUM_CAMS * height * width * 3


# ---------------------------------------------------------------------------
# Control SHM  (name = ctrl_shm_name(shm_name, env_id))
# Layout:
#   [0:4]      uint32  state_counter
#   [4:8]      uint32  reset_flag  (RESET_IDLE / RESET_REQUESTED / RESET_DONE)
#   [8:8+S]    float32 states      shape (state_dim,)
#   [8+S:8+S+A] float32 actions    shape (action_dim,)
#   [8+S+A:8+S+A+I] float32 info_buf  shape (info_dim,) ground-truth body poses
# where S = state_dim * 4, A = action_dim * 4, I = info_dim * 4
# ---------------------------------------------------------------------------

CTRL_HEADER_BYTES: int = 8
EE_STATE_DIM: int = 12
BODY_POSE_DIM: int = 7

RESET_IDLE:      int = 0
RESET_REQUESTED: int = 1
RESET_DONE:      int = 2


def ctrl_shm_name(shm_name: str, env_id: int) -> str:
    return f"{shm_name}_ctrl_{env_id}"


def ctrl_total_bytes(state_dim: int, action_dim: int, info_dim: int = 0) -> int:
    return CTRL_HEADER_BYTES + (state_dim + action_dim + info_dim) * 4
