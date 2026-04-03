#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Fake-policy benchmark for GenieSimEnv.
#
# Tests the full observation + control pipeline without a real model:
#   - Verifies image/state shapes and data ranges
#   - Runs zero / random / sinusoidal fake policies
#   - Reports per-step timing statistics
#   - Tests auto-reset behaviour on episode end
#
# Usage (ROS must be sourced; run from RLinf/ repo root):
#   GENIESIM_ROOT=.. PYTHONPATH=".:${PYTHONPATH}" \
#       .venv/bin/python3 rlinf/envs/geniesim/scripts/fake_policy_benchmark.py \
#       --config examples/embodiment/config/env/geniesim_place_block_into_box.yaml \
#       [--num-envs 1] [--steps 20] [--policy zero|random|sin] [--action-dim 14]

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# ROS environment bootstrap (re-exec through bash if not yet sourced).
# ---------------------------------------------------------------------------
def _ensure_ros_env() -> None:
    """
    Re-execute this script under a ROS-sourced environment if needed.

    In container mode (config has container_cfg), the host does not need ROS
    at all — skip setup entirely.  In host mode, ROS must be sourced.
    """
    # Check if container mode is configured (host doesn't need ROS in that case)
    _cfg_path = None
    for i, a in enumerate(sys.argv):
        if a == "--config" and i + 1 < len(sys.argv):
            _cfg_path = sys.argv[i + 1]
            break
    if _cfg_path:
        try:
            import yaml  # pyyaml, available in rlinf venv
            with open(_cfg_path) as _f:
                _raw = yaml.safe_load(_f)
            if _raw.get("container_cfg"):
                return  # container mode — no ROS needed on host
        except Exception:
            pass  # can't read config; fall through to ROS check

    # Host mode: ensure ROS environment is sourced
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    if "geniesim_rl_interfaces" in _ld and "/opt/ros" in _ld:
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

    if not source_cmds:
        return  # nothing to source, proceed anyway

    env_fwd = " ".join(
        f"{k}={v}" for k, v in os.environ.items()
        if k in ("GENIESIM_ROOT", "PYTHONPATH", "ROS_DOMAIN_ID")
    )
    args_fwd = " ".join(sys.argv[1:])
    cmd = " && ".join(source_cmds) + f" && {env_fwd} {sys.executable} {sys.argv[0]} {args_fwd}"
    print("[bootstrap] Re-executing with ROS environment...", flush=True)
    sys.exit(subprocess.call(["bash", "-c", cmd]))


_ensure_ros_env()

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs import get_env_cls
from rlinf.envs.geniesim.tasks import PlaceBlockIntoBoxEnv  # register task
from rlinf.envs.geniesim import REGISTER_GENIESIM_ENVS

# ---------------------------------------------------------------------------
# Fake policies
# ---------------------------------------------------------------------------

class ZeroPolicy:
    """All-zero actions — robot stays at neutral/hold pose."""
    name = "zero"

    def __init__(self, num_envs: int, action_dim: int):
        self._actions = torch.zeros(num_envs, action_dim)

    def act(self, obs: Dict, step: int) -> torch.Tensor:
        return self._actions


class RandomPolicy:
    """Uniform-random actions in [-0.05, 0.05] (small random perturbations)."""
    name = "random"

    def __init__(self, num_envs: int, action_dim: int, scale: float = 0.05):
        self._num_envs = num_envs
        self._action_dim = action_dim
        self._scale = scale

    def act(self, obs: Dict, step: int) -> torch.Tensor:
        return (torch.rand(self._num_envs, self._action_dim) * 2 - 1) * self._scale


class SinusoidalPolicy:
    """
    Sinusoidal joint-position targets.
    Each joint oscillates at a slightly different frequency so they move
    independently, making it easy to visually verify Isaac Sim renders
    the correct per-joint motion.
    """
    name = "sinusoidal"

    def __init__(self, num_envs: int, action_dim: int,
                 amplitude: float = 0.3, base_hz: float = 0.5):
        self._num_envs = num_envs
        self._action_dim = action_dim
        self._amplitude = amplitude
        # Slightly offset frequency per joint so motion looks natural
        self._freqs = np.array(
            [base_hz * (1 + 0.1 * i) for i in range(action_dim)], dtype=np.float32
        )

    def act(self, obs: Dict, step: int) -> torch.Tensor:
        t = step / 30.0  # assume ~30Hz
        phases = 2 * math.pi * self._freqs * t
        single_env = torch.from_numpy(self._amplitude * np.sin(phases))
        return single_env.unsqueeze(0).expand(self._num_envs, -1)


_POLICIES = {p.name: p for p in [ZeroPolicy, RandomPolicy, SinusoidalPolicy]}

# ---------------------------------------------------------------------------
# Observation validation
# ---------------------------------------------------------------------------

def validate_obs(obs: Dict, num_envs: int, step_label: str) -> List[str]:
    """
    Check observation dict structure and value ranges.
    Returns a list of warning strings (empty = all OK).
    """
    warnings: List[str] = []

    # main_images
    imgs = obs.get("main_images")
    if imgs is None:
        warnings.append(f"[{step_label}] main_images is None")
    else:
        if imgs.shape[0] != num_envs:
            warnings.append(f"[{step_label}] main_images batch dim {imgs.shape[0]} != {num_envs}")
        if imgs.dtype not in (torch.uint8,):
            warnings.append(f"[{step_label}] main_images dtype={imgs.dtype}, expected uint8")
        mn, mx = int(imgs.min()), int(imgs.max())
        if mx == 0:
            warnings.append(f"[{step_label}] main_images all-zero (black frame?)")
        if mx > 255 or mn < 0:
            warnings.append(f"[{step_label}] main_images value out of [0,255]: [{mn},{mx}]")

    # states
    states = obs.get("states")
    if states is None:
        warnings.append(f"[{step_label}] states is None")
    else:
        if states.shape[0] != num_envs:
            warnings.append(f"[{step_label}] states batch dim {states.shape[0]} != {num_envs}")
        if torch.isnan(states).any():
            warnings.append(f"[{step_label}] states contain NaN")
        if torch.isinf(states).any():
            warnings.append(f"[{step_label}] states contain Inf")

    # task_descriptions
    descs = obs.get("task_descriptions")
    if not descs or len(descs) != num_envs:
        warnings.append(f"[{step_label}] task_descriptions len={len(descs) if descs else None}")

    return warnings


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    env,
    policy,
    num_envs: int,
    steps: int,
    action_dim: int,
) -> None:
    sep = "─" * 60

    # ---- reset ----
    print(f"\n{sep}")
    print(f"Policy: {policy.name}  |  envs={num_envs}  |  steps={steps}")
    print(sep)

    t0 = time.perf_counter()
    obs, _ = env.reset()
    reset_ms = (time.perf_counter() - t0) * 1000

    print(f"reset()   {reset_ms:7.1f} ms")

    # Validate reset observation
    warnings = validate_obs(obs, num_envs, "reset")
    imgs = obs["main_images"]
    states = obs["states"]
    print(f"  main_images : {tuple(imgs.shape)}  dtype={imgs.dtype}  "
          f"range=[{int(imgs.min())},{int(imgs.max())}]")
    print(f"  states      : {tuple(states.shape)}  dtype={states.dtype}  "
          f"mean={states.float().mean():.4f}  std={states.float().std():.4f}")
    print(f"  task_desc   : {obs['task_descriptions'][0]!r}")
    if obs.get("wrist_images") is not None:
        wi = obs["wrist_images"]
        print(f"  wrist_images: {tuple(wi.shape)}")
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")

    # ---- step loop ----
    step_times: List[float] = []
    all_rewards: List[float] = []
    reset_counts = 0

    for i in range(steps):
        actions = policy.act(obs, step=i)

        t0 = time.perf_counter()
        obs, rewards, terminated, truncated, info = env.step(actions)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        step_times.append(elapsed_ms)
        all_rewards.extend(rewards.tolist())

        auto_resets = int((terminated | truncated).sum())
        if auto_resets:
            reset_counts += auto_resets

        # Spot-check every 5th step
        if i % 5 == 0:
            warnings = validate_obs(obs, num_envs, f"step{i}")
            reward_str = f"r={rewards.tolist()}"
            done_str = f"term={terminated.tolist()} trunc={truncated.tolist()}"
            warn_str = f"  ⚠ {warnings}" if warnings else ""
            print(f"  step {i:3d}: {elapsed_ms:6.1f}ms  {reward_str}  {done_str}{warn_str}")

    # ---- summary ----
    times = np.array(step_times)
    print(f"\n{sep}")
    print(f"Step timing  ({steps} steps)")
    print(f"  mean={times.mean():.1f}ms  p50={np.percentile(times,50):.1f}ms  "
          f"p90={np.percentile(times,90):.1f}ms  max={times.max():.1f}ms")
    print(f"  throughput: {1000/times.mean():.1f} steps/sec  "
          f"({1000*num_envs/times.mean():.1f} env-steps/sec)")
    print(f"Auto-resets: {reset_counts}")
    print(f"Reward stats: min={min(all_rewards):.3f}  max={max(all_rewards):.3f}  "
          f"mean={sum(all_rewards)/len(all_rewards):.3f}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GenieSimEnv fake-policy benchmark")
    parser.add_argument("--config",
                        default="examples/embodiment/config/env/geniesim_place_block_into_box.yaml")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20,
                        help="Steps to run per policy")
    parser.add_argument("--policy", choices=list(_POLICIES) + ["all"], default="all",
                        help="Which fake policy to run (default: all)")
    parser.add_argument("--action-dim", type=int, default=14,
                        help="Robot action dimension")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only verify registration, skip sim launch")
    args = parser.parse_args()

    # ---- verify registration ----
    assert "place_block_into_box" in REGISTER_GENIESIM_ENVS, \
        "Task not registered"
    print(f"[bench] Registered tasks: {list(REGISTER_GENIESIM_ENVS.keys())}")

    cfg = OmegaConf.load(args.config)
    print(f"[bench] Config loaded: task={cfg.init_params.id}")

    EnvCls = get_env_cls("geniesim", cfg)
    print(f"[bench] EnvCls = {EnvCls.__name__}")

    if args.dry_run:
        print("[bench] Dry-run: skipping sim launch. PASSED ✓")
        return

    # ---- build env ----
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.init_params.num_envs = args.num_envs

    env = EnvCls(cfg, num_envs=args.num_envs, seed_offset=0,
                 total_num_processes=1, worker_info={})
    print(f"[bench] Env created: {EnvCls.__name__}  num_envs={args.num_envs}")

    # ---- choose policies ----
    policy_names = list(_POLICIES.keys()) if args.policy == "all" else [args.policy]

    try:
        for name in policy_names:
            PolicyCls = _POLICIES[name]
            policy = PolicyCls(num_envs=args.num_envs, action_dim=args.action_dim)
            run_benchmark(env, policy, args.num_envs, args.steps, args.action_dim)
    except KeyboardInterrupt:
        print("\n[bench] Interrupted — shutting down...", flush=True)
        env.close()
        print("[bench] Shutdown complete.", flush=True)
        import sys; sys.exit(130)

    env.close()
    print("\n[bench] ALL POLICIES PASSED ✓")


if __name__ == "__main__":
    main()
