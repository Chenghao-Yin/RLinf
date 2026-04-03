#!/usr/bin/env python3
"""Diagnostic: check EE stability after reset with zero actions (junpu EE mode)."""
import os, sys

def _ensure_ros_env():
    if "geniesim_rl_interfaces" in os.environ.get("LD_LIBRARY_PATH", ""):
        return
    import subprocess
    _gs = os.environ.get("GENIESIM_ROOT", "/home/zy/code/rlinf_open_source")
    _ws = os.path.join(_gs, "main", "rl_ros_ws", "install", "setup.bash")
    if os.path.isfile(_ws):
        cmd = f"source {_ws} && GENIESIM_ROOT={_gs} {sys.executable} {sys.argv[0]}"
        sys.exit(subprocess.call(["bash", "-c", cmd]))
_ensure_ros_env()

import numpy as np, torch
from omegaconf import OmegaConf, open_dict
from rlinf.envs.geniesim.tasks import junpu_place_workpiece  # noqa
from rlinf.envs.geniesim import REGISTER_GENIESIM_ENVS

_RLINF_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
cfg = OmegaConf.load(
    os.path.join(
        _RLINF_REPO,
        "examples",
        "embodiment",
        "config",
        "env",
        "geniesim_junpu_place_workpiece.yaml",
    )
)
EnvCls = REGISTER_GENIESIM_ENVS["junpu_place_workpiece"]
with open_dict(cfg):
    cfg.init_params.num_envs = 1

env = EnvCls(cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info={})
obs, _ = env.reset()
print("After reset:")
print(f"  arm_r[7:14] = {obs['states'][0,7:14].numpy().round(4).tolist()}")
print(f"  ee_r_pos    = {obs['states'][0,34:37].numpy().round(4).tolist()}")

print("\nZero-action steps (EE targets = zeros):")
zero_act = torch.zeros(1, 14)
for i in range(10):
    obs, _, _, _, _ = env.step(zero_act)
    s = obs['states'][0].numpy()
    print(f"  step {i+1:2d}: arm_r[7]={s[7]:.4f}  arm_r[11]={s[11]:.4f}  "
          f"ee_r_pos={s[34:37].round(4).tolist()}")

env.close()
print("\n[done]")
