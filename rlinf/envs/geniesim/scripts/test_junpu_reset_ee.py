#!/usr/bin/env python3
"""
Test: verify that junpu env resets right EE to the configured base_link-frame pose.

Expected pose (from junpu_place_workpiece.yaml reset_ee_r):
  pos  = [0.4953, 0.012, 1.231]   (base_link frame, metres)
  rpy  = [-3.12832732, 0.5282591, -3.11193856]  (intrinsic XYZ Euler, radians)

states layout (dim=40):
  [0:14]   joint pos (left_arm[0:7], right_arm[7:14])
  [14:28]  joint vel
  [28:31]  ee_l pos  (base_link frame)
  [31:34]  ee_l rpy
  [34:37]  ee_r pos  (base_link frame)   <-- check this
  [37:40]  ee_r rpy                       <-- and this

Run:
  cd /home/zy/code/rlinf_open_source/RLinf
  GENIESIM_ROOT=/home/zy/code/rlinf_open_source \
    .venv/bin/python3 rlinf/envs/geniesim/scripts/test_junpu_reset_ee.py
"""

import os, sys

# ROS bootstrap (same pattern as test_rlinf_integration.py)
def _ensure_ros_env():
    _ws_lib_marker = "geniesim_rl_interfaces"
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    _ros_marker = "/opt/ros/humble_311"
    if _ws_lib_marker in _ld and _ros_marker in _ld:
        return
    import subprocess
    _gs_root = os.environ.get(
        "GENIESIM_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")),
    )
    _ws_setup = os.path.join(_gs_root, "main", "rl_ros_ws", "install", "setup.bash")
    _ros_setup = "/opt/ros/humble_311/setup.bash"
    source_cmds = []
    if os.path.isfile(_ros_setup):
        source_cmds.append(f"source {_ros_setup}")
    if os.path.isfile(_ws_setup):
        source_cmds.append(f"source {_ws_setup}")
    env_fwd = " ".join(
        f'{k}={v}' for k, v in os.environ.items()
        if k in ("GENIESIM_ROOT", "PYTHONPATH", "ROS_DOMAIN_ID")
    )
    cmd = " && ".join(source_cmds) + f" && {env_fwd} {sys.executable} {sys.argv[0]}"
    print("[bootstrap] Re-executing with ROS env...", flush=True)
    sys.exit(subprocess.call(["bash", "-c", cmd]))

_ensure_ros_env()

import numpy as np
from omegaconf import OmegaConf

EXPECTED_POS = np.array([0.4953, 0.012, 1.231])
EXPECTED_RPY = np.array([-3.12832732, 0.5282591, -3.11193856])
POS_TOL = 0.02   # 2 cm
RPY_TOL = 0.05   # ~3 deg

CONFIG = os.path.join(
    os.path.dirname(__file__),
    "../configs/junpu_place_workpiece.yaml",
)

cfg = OmegaConf.load(CONFIG)
print(f"[test] Loaded config: state_dim={cfg.init_params.state_dim}  "
      f"reset_ee_r={list(cfg.init_params.reset_ee_r)}")

assert cfg.init_params.state_dim == 40, \
    f"state_dim should be 40 (28 joint + 12 EE), got {cfg.init_params.state_dim}"

# Import after ROS bootstrap
from rlinf.envs.geniesim import REGISTER_GENIESIM_ENVS
from rlinf.envs.geniesim.tasks import junpu_place_workpiece  # registers the env  # noqa: F401

EnvCls = REGISTER_GENIESIM_ENVS["junpu_place_workpiece"]
print(f"[test] EnvCls = {EnvCls.__name__}")

import atexit
from omegaconf import open_dict
with open_dict(cfg):
    cfg.init_params.num_envs = 1

env = EnvCls(cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info={})
atexit.register(env.close)

print("[test] Calling reset() ...")
obs, _ = env.reset()

states = obs["states"].numpy()  # [1, 40]
print(f"[test] states shape = {states.shape}")

ee_r_pos = states[0, 34:37]
ee_r_rpy = states[0, 37:40]
ee_l_pos = states[0, 28:31]
ee_l_rpy = states[0, 31:34]
joint_pos = states[0, 0:14]

print(f"\n--- Joint positions (14 DOF) ---")
print(f"  left  arm: {np.round(joint_pos[:7], 4).tolist()}")
print(f"  right arm: {np.round(joint_pos[7:14], 4).tolist()}")
print(f"\n--- EE positions (base_link frame) ---")
print(f"  left  pos: {np.round(ee_l_pos, 4).tolist()}  rpy: {np.round(ee_l_rpy, 4).tolist()}")
print(f"  right pos: {np.round(ee_r_pos, 4).tolist()}  rpy: {np.round(ee_r_rpy, 4).tolist()}")
print(f"\n--- Expected right EE ---")
print(f"  pos: {EXPECTED_POS.tolist()}")
print(f"  rpy: {EXPECTED_RPY.tolist()}")

pos_err = np.linalg.norm(ee_r_pos - EXPECTED_POS)
rpy_err = np.linalg.norm(ee_r_rpy - EXPECTED_RPY)
print(f"\n--- Errors ---")
print(f"  pos error: {pos_err:.4f} m  (tol={POS_TOL} m)")
print(f"  rpy error: {rpy_err:.4f} rad  (tol={RPY_TOL} rad)")

pos_ok = pos_err < POS_TOL
rpy_ok = rpy_err < RPY_TOL
if pos_ok and rpy_ok:
    print("\n[test] PASSED ✓")
else:
    print(f"\n[test] FAILED ✗  pos_ok={pos_ok}  rpy_ok={rpy_ok}")
    sys.exit(1)
