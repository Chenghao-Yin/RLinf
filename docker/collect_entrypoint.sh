#!/usr/bin/env bash
set -eo pipefail

apt-get update -qq && apt-get install -y -qq libhidapi-hidraw0 > /dev/null 2>&1 || true

VENV="${RLINF_VENV:-/opt/rlinf_venv/rlinf}"
if [ -f "${VENV}/bin/pip" ]; then
    "${VENV}/bin/pip" install -q pyspacemouse==1.1.5 2>/dev/null || true
fi

GENIESIM_ROOT="${GENIESIM_ROOT:-/geniesim/main}"
echo "[collect] Cleaning stale sentinel files and SHM..."
rm -f "${GENIESIM_ROOT}/.geniesim_ready" \
      "${GENIESIM_ROOT}/.geniesim_progress" \
      "${GENIESIM_ROOT}/.geniesim_idle" \
      "${GENIESIM_ROOT}/.geniesim_start" \
      "${GENIESIM_ROOT}/.geniesim_stop" \
      "${GENIESIM_ROOT}/.geniesim_error" \
      "${GENIESIM_ROOT}/.sim_server_config.json" 2>/dev/null || true
rm -f /dev/shm/geniesim* 2>/dev/null || true

SIM_REPO_ROOT="${SIM_REPO_ROOT:-${GENIESIM_ROOT}/main}"
EXPECTED_SRC="${SIM_REPO_ROOT}/rl_ros_ws/src/geniesim_rl_interfaces"
FALLBACK_SRC="${GENIESIM_ROOT}/rl_ros_ws/src/geniesim_rl_interfaces"
if [ ! -d "${EXPECTED_SRC}" ] && [ -d "${FALLBACK_SRC}" ]; then
    echo "[collect] Symlinking rl_ros_ws source from ${FALLBACK_SRC}"
    mkdir -p "$(dirname "${EXPECTED_SRC}")"
    ln -sfn "${FALLBACK_SRC}" "${EXPECTED_SRC}"
fi

exec /entrypoint_geniesim_rlinf.sh "$@"
