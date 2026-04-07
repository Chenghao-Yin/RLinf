#!/usr/bin/env python3
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
# Replay recorded demonstration actions back into GenieSimEnv.
#
# Given m parallel envs and n demo files:
#   - Start by assigning one demo to each of the first min(m, n) envs.
#   - As each env finishes replaying its trajectory, reset that specific env
#     and assign the next pending demo (individual-env reset is supported via
#     GenieSimBaseEnv.reset(env_ids=np.array([i]))).
#   - Idle envs (no pending demo) send a hold-position action (the last
#     replayed target) to keep the arm stable.
#   - Demos cycle repeatedly until the total number of completed episodes
#     reaches num_rounds * n (controlled by --num-rounds).
#
# Action source priority (per recorded step):
#   1. infos[step]["intervene_action"]  — actual EEF absolute target from
#      SpaceMouse (written by SpacemouseSimIntervention when intervening).
#   2. actions[step]                    — policy action (may be all-zeros if
#      the demo was collected via SpaceMouse with zero policy actions).
#
# Usage:
#   cd RLinf
#   GENIESIM_ROOT=.. python examples/embodiment/replay_sim_demos.py \
#       --config examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml \
#       --demo-dir /tmp/sim_demos \
#       --num-envs 2 \
#       --num-rounds 3 \
#       [--step-hz 10]

import argparse
import atexit
import glob
import os
import pickle
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict


# ---------------------------------------------------------------------------
# Resolve the RLinf repo root so that `rlinf.*` is importable when the script
# is run from any working directory (not just the repo root).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RLINF_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _RLINF_ROOT not in sys.path:
    sys.path.insert(0, _RLINF_ROOT)

# Ensure GENIESIM_ROOT is set so that GenieSimBaseEnv can find main/source.
if "GENIESIM_ROOT" not in os.environ:
    _default_gs_root = os.path.abspath(os.path.join(_RLINF_ROOT, ".."))
    os.environ["GENIESIM_ROOT"] = _default_gs_root
    print(f"[replay_sim_demos] GENIESIM_ROOT not set; defaulting to {_default_gs_root}")


# ---------------------------------------------------------------------------
# Demo loading
# ---------------------------------------------------------------------------

def _load_demos(demo_dir: str) -> List[dict]:
    """Load all pickle episode files from *demo_dir*, sorted by filename."""
    pattern = os.path.join(demo_dir, "*.pkl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"[replay_sim_demos] No .pkl files found in {demo_dir!r}")

    demos = []
    for p in paths:
        with open(p, "rb") as f:
            ep = pickle.load(f)
        demos.append(ep)
        print(f"[replay_sim_demos]   loaded {os.path.basename(p)}: "
              f"{len(ep['actions'])} steps, success={ep.get('success', '?')}")

    print(f"[replay_sim_demos] {len(demos)} demo(s) loaded from {demo_dir}")
    return demos


def _get_action_for_step(demo: dict, step: int) -> Optional[torch.Tensor]:
    """Extract the best available action tensor for *step* in *demo*.

    Prefers ``infos[step]["intervene_action"]`` (the actual EEF target written
    by SpacemouseSimIntervention) and falls back to ``actions[step]``.
    Returns None if *step* is out of range.
    """
    actions = demo["actions"]
    infos = demo["infos"]

    if step >= len(actions):
        return None

    # Prefer the actual expert / SpaceMouse action
    info = infos[step] if step < len(infos) else {}
    if isinstance(info, dict) and "intervene_action" in info:
        act = info["intervene_action"]
        if isinstance(act, torch.Tensor):
            return act.float()
        return torch.tensor(np.asarray(act, dtype=np.float32))

    # Fall back to recorded policy action
    act = actions[step]
    if isinstance(act, torch.Tensor):
        return act.float()
    return torch.tensor(np.asarray(act, dtype=np.float32))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay recorded sim demos back into GenieSimEnv."
    )
    p.add_argument(
        "--config",
        default="examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml",
        help="Path to the GenieSimEnv OmegaConf config YAML.",
    )
    p.add_argument(
        "--demo-dir",
        default="/tmp/sim_demos",
        help="Directory containing recorded .pkl episode files.",
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel simulation instances for replay.",
    )
    p.add_argument(
        "--step-hz",
        type=float,
        default=10.0,
        help="Target control frequency (Hz); 0 disables rate limiting.",
    )
    p.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="How many full passes through all demos to run. "
             "Demos cycle repeatedly until num_rounds * n total episodes complete. "
             "Default=1 (each demo played once).",
    )
    return p


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------

def main():
    args = _build_parser().parse_args()

    # ---- 1. Load demos ------------------------------------------------------ #
    demos = _load_demos(args.demo_dir)
    n = len(demos)
    m = args.num_envs

    # ---- 2. Load config ----------------------------------------------------- #
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(_RLINF_ROOT, cfg_path)
    if not os.path.exists(cfg_path):
        sys.exit(f"[replay_sim_demos] Config not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    with open_dict(cfg):
        cfg.init_params.num_envs = m

    action_dim: int = cfg.init_params.get("action_dim", 14)
    model_action_dim: int = cfg.init_params.get("model_action_dim", action_dim)

    # ---- 3. Instantiate env ------------------------------------------------- #
    from rlinf.envs.geniesim import REGISTER_GENIESIM_ENVS
    from rlinf.envs.geniesim.tasks import JunpuPlaceWorkpieceEnv  # noqa: F401

    task_id = cfg.init_params.id
    if task_id not in REGISTER_GENIESIM_ENVS:
        sys.exit(
            f"[replay_sim_demos] Unknown task_id={task_id!r}. "
            f"Registered: {list(REGISTER_GENIESIM_ENVS.keys())}"
        )

    EnvCls = REGISTER_GENIESIM_ENVS[task_id]
    print(f"[replay_sim_demos] Creating {EnvCls.__name__} "
          f"(num_envs={m}, task={task_id})")

    env = EnvCls(
        cfg,
        num_envs=m,
        seed_offset=0,
        total_num_processes=1,
        worker_info={},
    )
    atexit.register(env.close)

    # ---- 4. Initialise replay state ----------------------------------------- #
    num_rounds: int = args.num_rounds
    total_target: int = n * num_rounds   # total episodes to complete before exit

    # queue_pos: monotonically increasing counter of assigned episodes;
    #            demos[queue_pos % n] gives the actual demo (cycles forever).
    queue_pos = min(m, n)
    # current_demo[i]: the demo dict assigned to env i, or None when idle
    current_demo: List[Optional[dict]] = [None] * m
    # current_step[i]: the next action index to send for env i's active demo
    current_step: List[int] = [0] * m
    # last_action[i]: last sent action for env i (used as hold when idle)
    last_action: List[Optional[torch.Tensor]] = [None] * m

    for i in range(min(m, n)):
        current_demo[i] = demos[i]
        current_step[i] = 0

    completed_demos = 0

    # ---- 5. Initial full reset ---------------------------------------------- #
    env.reset()
    print(f"[replay_sim_demos] Starting replay: {n} demo(s) × {num_rounds} round(s) "
          f"= {total_target} total episodes, {m} parallel env(s).")

    step_dt = 1.0 / args.step_hz if args.step_hz > 0 else 0.0
    t_last = time.monotonic()
    total_steps = 0

    # ---- 6. Main loop ------------------------------------------------------- #
    while any(d is not None for d in current_demo):
        actions = torch.zeros(m, model_action_dim, dtype=torch.float32)

        for i in range(m):
            if current_demo[i] is None:
                # Idle env: hold last known position
                if last_action[i] is not None:
                    actions[i] = last_action[i]
                continue

            act = _get_action_for_step(current_demo[i], current_step[i])
            if act is not None:
                actions[i] = act
                last_action[i] = act.clone()
            # If act is None, the trajectory ended — will be caught below

        env.step(actions, auto_reset=False)
        total_steps += 1

        # Rate limiting
        if step_dt > 0:
            elapsed = time.monotonic() - t_last
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)
        t_last = time.monotonic()

        # Advance step counters and detect env completions
        for i in range(m):
            if current_demo[i] is None:
                continue

            current_step[i] += 1
            demo_len = len(current_demo[i]["actions"])

            if current_step[i] >= demo_len:
                # This env finished its demo
                completed_demos += 1
                demo_id = current_demo[i].get("episode_id", "?")
                round_num = (completed_demos - 1) // n + 1
                print(f"[replay_sim_demos] env_{i} finished demo "
                      f"episode_id={demo_id} "
                      f"({completed_demos}/{total_target}, round {round_num}/{num_rounds})")

                if completed_demos < total_target:
                    # Cycle demos: wrap queue_pos around using modulo
                    current_demo[i] = demos[queue_pos % n]
                    current_step[i] = 0
                    queue_pos += 1
                    env.reset(env_ids=np.array([i]))
                    print(f"[replay_sim_demos] env_{i} reset → starting demo "
                          f"episode_id={current_demo[i].get('episode_id', '?')}")
                else:
                    # Reached total_target; park this env
                    current_demo[i] = None

    print(
        f"\n[replay_sim_demos] Done. "
        f"Replayed {completed_demos}/{total_target} episode(s) "
        f"({num_rounds} round(s) × {n} demo(s)) in {total_steps} total steps."
    )
    env.close()


if __name__ == "__main__":
    main()
