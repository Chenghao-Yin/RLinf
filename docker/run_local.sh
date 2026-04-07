#!/usr/bin/env bash
# run_local.sh — convenience wrapper for Mode 2 (in-container) usage.
#
# Sets up the full environment needed to run RLinf scripts that use GenieSim
# in local mode (both training code and sim_server in the same container).
#
# Usage (inside the geniesim-rlinf-train container):
#
#   /workspace/repo/RLinf/docker/run_local.sh <command> [args...]
#
# Examples:
#
#   # Collect 20 demos
#   /workspace/repo/RLinf/docker/run_local.sh \
#       python examples/embodiment/collect_demos.py \
#           --config-path config/geniesim \
#           --config-name place_block_into_box \
#           num_envs=1 num_demos=20
#
#   # Convert demos to replay buffer
#   /workspace/repo/RLinf/docker/run_local.sh \
#       python examples/embodiment/convert_demos_to_buffer.py \
#           --config-path config/geniesim \
#           --config-name place_block_into_box
#
#   # SAC training
#   /workspace/repo/RLinf/docker/run_local.sh \
#       python examples/embodiment/train.py \
#           --config-path config/geniesim \
#           --config-name sac_place_block
#
# Environment variables set by this script:
#   GENIESIM_ROOT     — rlinf_open_source/ root inside container
#   SIM_REPO_ROOT     — geniesim main repo root inside container
#   PYTHONPATH        — includes RLinf source
#   ROS_DOMAIN_ID     — 0 (override with ROS_DOMAIN_ID=N ./run_local.sh ...)
#   ROS_LOCALHOST_ONLY — 1

set -eo pipefail

# ---------------------------------------------------------------------------
# Configurable paths — override via env vars if your mount differs
# ---------------------------------------------------------------------------
GENIESIM_ROOT="${GENIESIM_ROOT:-/geniesim/main}"
SIM_REPO_ROOT="${SIM_REPO_ROOT:-${GENIESIM_ROOT}/main}"
RLINF_ROOT="${RLINF_ROOT:-${GENIESIM_ROOT}/RLinf}"
ROS_WS_INSTALL="${ROS_WS_INSTALL:-/geniesim/ros_ws_build/install}"
RLINF_VENV="${RLINF_VENV:-/opt/rlinf_venv/rlinf}"

export GENIESIM_ROOT SIM_REPO_ROOT
export GENIESIM_CONTAINER="${GENIESIM_CONTAINER:-1}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY=1

# ---------------------------------------------------------------------------
# Activate RLinf venv (adds torch, ray, transformers, etc.)
# ---------------------------------------------------------------------------
if [ -f "${RLINF_VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${RLINF_VENV}/bin/activate"
else
    echo "[run_local.sh] WARNING: RLinf venv not found at ${RLINF_VENV}" >&2
fi

# ---------------------------------------------------------------------------
# Add RLinf source to PYTHONPATH
# ---------------------------------------------------------------------------
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"

# EMBODIED_PATH is used by Hydra searchpath in training configs.
export EMBODIED_PATH="${EMBODIED_PATH:-${RLINF_ROOT}/examples/embodiment}"

# ---------------------------------------------------------------------------
# Source ROS 2 Jazzy (needed by sim_server.py, not usually needed by training
# code directly, but harmless to have on PATH)
# ---------------------------------------------------------------------------
if [ -f /opt/ros/jazzy/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
fi

if [ -f "${ROS_WS_INSTALL}/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "${ROS_WS_INSTALL}/setup.bash"
fi

# ---------------------------------------------------------------------------
# Run command from RLinf root
# ---------------------------------------------------------------------------
cd "${RLINF_ROOT}"

if [ $# -eq 0 ]; then
    echo "[run_local.sh] No command given — dropping into bash."
    exec bash
else
    exec "$@"
fi
