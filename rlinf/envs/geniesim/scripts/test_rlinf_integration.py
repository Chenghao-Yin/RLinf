#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Integration test: verifies that GenieSimEnv can be instantiated via the
# standard RLinf get_env_cls() factory and that the env interface is complete.
#
# Run from the RLinf repo root (ROS must be sourced beforehand):
#   source /opt/ros/humble_311/setup.zsh
#   GENIESIM_ROOT=$(pwd)/.. PYTHONPATH=".:${PYTHONPATH}" \
#       .venv/bin/python3 rlinf/envs/geniesim/scripts/test_rlinf_integration.py \
#       --config examples/embodiment/config/env/geniesim_place_block_into_box.yaml \
#       [--dry-run]   # skip actual sim launch, just test class resolution
#

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# ROS environment bootstrap (re-exec through bash if not already set up).
# glibc's ld.so caches LD_LIBRARY_PATH at process startup; we cannot patch it
# after the fact via os.environ.  Re-executing through bash that sources the
# ROS setup files ensures the dynamic linker finds all typesupport libraries
# before any Python code runs.
# ---------------------------------------------------------------------------
def _ensure_ros_env():
    _ws_lib_marker = "geniesim_rl_interfaces"
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    _ros_marker = "/opt/ros/humble_311"
    if _ws_lib_marker in _ld and _ros_marker in _ld:
        return  # already bootstrapped, nothing to do

    import subprocess
    # Resolve the geniesim workspace root from GENIESIM_ROOT or this file's location.
    _gs_root = os.environ.get(
        "GENIESIM_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")),
    )
    _ws_setup = os.path.join(_gs_root, "main", "rl_ros_ws", "install", "setup.bash")
    _ros_setup = "/opt/ros/humble_311/setup.bash"

    # Build a bash command that sources ROS setup then re-runs this script.
    source_cmds = []
    if os.path.isfile(_ros_setup):
        source_cmds.append(f"source {_ros_setup}")
    if os.path.isfile(_ws_setup):
        source_cmds.append(f"source {_ws_setup}")

    env_fwd = " ".join(
        f'{k}={v}' for k, v in os.environ.items()
        if k in ("GENIESIM_ROOT", "PYTHONPATH", "ROS_DOMAIN_ID")
    )
    args_fwd = " ".join(sys.argv[1:])
    cmd = " && ".join(source_cmds) + f" && {env_fwd} {sys.executable} {sys.argv[0]} {args_fwd}"
    print(f"[bootstrap] Re-executing with ROS environment...", flush=True)
    ret = subprocess.call(["bash", "-c", cmd])
    sys.exit(ret)


_ensure_ros_env()

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="examples/embodiment/config/env/geniesim_place_block_into_box.yaml")
parser.add_argument("--dry-run", action="store_true",
                    help="Only verify class resolution, don't launch simulation")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=5)
args = parser.parse_args()

# ---- 1. Verify env registration ----
from rlinf.envs import get_env_cls, SupportedEnvType
from rlinf.envs.geniesim.tasks import PlaceBlockIntoBoxEnv  # triggers registration
from rlinf.envs.geniesim import REGISTER_GENIESIM_ENVS

assert "place_block_into_box" in REGISTER_GENIESIM_ENVS, \
    "place_block_into_box not registered in REGISTER_GENIESIM_ENVS"
print(f"[test] Registered GeneSim tasks: {list(REGISTER_GENIESIM_ENVS.keys())}")

# ---- 2. Load config ----
from omegaconf import OmegaConf
cfg = OmegaConf.load(args.config)
print(f"[test] Loaded config: task_id={cfg.init_params.id}")

# ---- 3. Resolve env class ----
EnvCls = get_env_cls("geniesim", cfg)
assert EnvCls is PlaceBlockIntoBoxEnv, f"Unexpected class: {EnvCls}"
print(f"[test] get_env_cls() resolved to {EnvCls.__name__} ✓")

# ---- 4. Check interface completeness ----
required_methods = ["reset", "step", "chunk_step", "close",
                    "update_reset_state_ids", "elapsed_steps"]
for m in required_methods:
    assert hasattr(EnvCls, m), f"Missing method: {m}"
print(f"[test] Interface check passed ✓")

if args.dry_run:
    print("[test] Dry-run mode: skipping simulation launch.")
    print("[test] PASSED (dry-run) ✓")
    sys.exit(0)

# ---- 5. Instantiate and run ----
import numpy as np
from omegaconf import open_dict

with open_dict(cfg):
    cfg.init_params.num_envs = args.num_envs

import atexit
env = EnvCls(cfg, num_envs=args.num_envs, seed_offset=0,
             total_num_processes=1, worker_info={})
atexit.register(env.close)

print("[test] Calling reset()...")
obs, info = env.reset()
assert "main_images" in obs
assert "states" in obs
print(f"[test] reset() OK | images={obs['main_images'].shape} states={obs['states'].shape}")

print(f"[test] Running {args.steps} steps...")
import torch
action_dim = cfg.init_params.get("action_dim", 14)
for i in range(args.steps):
    actions = torch.zeros(args.num_envs, action_dim)
    obs, rewards, terminated, truncated, info = env.step(actions)
    assert "episode" in info
    print(f"  step {i}: rewards={rewards.tolist()} terminated={terminated.tolist()}")

env.close()
print("[test] PASSED ✓")
