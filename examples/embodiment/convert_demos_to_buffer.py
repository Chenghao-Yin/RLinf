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
# Convert recorded GeneSim demo .pkl files to TrajectoryReplayBuffer checkpoint
# format so they can be loaded as a demo_buffer for SAC training.
#
# Demo pkl format (per file):
#   observations : list[dict]  — length T+1 (includes final obs)
#     states        : Tensor[40]
#     main_images   : Tensor[H, W, C]  (optional)
#     task_descriptions : list[str]
#   actions      : list[Tensor[14]]   — length T
#   rewards      : list[Tensor[()]]   — length T
#   terminated   : list[Tensor[bool]] — length T
#   truncated    : list[Tensor[bool]] — length T
#   infos        : list[dict]         — length T
#
# Output checkpoint layout (one dir):
#   metadata.json
#   trajectory_index.json
#   trajectory_<id>_<uuid>.pt   (one per demo)
#
# Usage:
#   cd RLinf
#   python examples/embodiment/convert_demos_to_buffer.py \
#       --demo-dir my_demos \
#       --output-dir /tmp/junpu_demo_buffer \
#       [--include-images]        # include main_images (large!)
#       [--placeholder-reward 0]  # override reward value (default: use recorded)

import argparse
import glob
import json
import os
import pickle
import sys
import uuid

import numpy as np
import torch

# ── Make rlinf importable when run from the repo root ──────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RLINF_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _RLINF_ROOT not in sys.path:
    sys.path.insert(0, _RLINF_ROOT)

from rlinf.data.embodied_io_struct import Trajectory  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _to_float_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float().cpu()
    return torch.tensor(np.asarray(x, dtype=np.float32))


def _to_bool_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.bool().cpu()
    return torch.tensor(bool(x))


def _demo_to_trajectory(ep: dict, include_images: bool,
                         placeholder_reward: float | None) -> Trajectory:
    """Convert a single demo episode dict to a Trajectory object.

    Shape convention used by TrajectoryReplayBuffer: [T, B, ...] where B=1.
    curr_obs[t] is the obs before step t; next_obs[t] is the obs after step t.
    """
    obs_list = ep["observations"]   # length T+1 if final obs present, else T
    act_list = ep["actions"]        # length T
    rew_list = ep["rewards"]
    term_list = ep["terminated"]
    trunc_list = ep["truncated"]

    T = len(act_list)

    # ---- actions ----
    actions = torch.stack([_to_float_tensor(a) for a in act_list], dim=0)  # [T, 14]
    actions = actions.unsqueeze(1)  # [T, 1, 14]

    # ---- rewards ----
    if placeholder_reward is not None:
        rewards = torch.full((T, 1, 1), float(placeholder_reward))
    else:
        rewards = torch.stack(
            [_to_float_tensor(r).reshape(1) for r in rew_list], dim=0
        )  # [T, 1]
        rewards = rewards.unsqueeze(1)  # [T, 1, 1]

    # ---- terminations / truncations / dones ----
    terminations = torch.stack(
        [_to_bool_tensor(t).reshape(1) for t in term_list], dim=0
    ).unsqueeze(1)  # [T, 1, 1]

    truncations = torch.stack(
        [_to_bool_tensor(t).reshape(1) for t in trunc_list], dim=0
    ).unsqueeze(1)  # [T, 1, 1]

    dones = (terminations | truncations)  # [T, 1, 1]

    # ---- observations ----
    # curr_obs[t] = obs_list[t], next_obs[t] = obs_list[t+1]
    # If obs_list has exactly T entries (no final obs), duplicate the last one.
    if len(obs_list) >= T + 1:
        curr_obs_raw = obs_list[:T]
        next_obs_raw = obs_list[1:T + 1]
    else:
        curr_obs_raw = obs_list[:T]
        next_obs_raw = obs_list[:T]   # fallback: same as curr
        if T > 1:
            next_obs_raw = obs_list[1:T] + [obs_list[-1]]

    def _build_obs_dict(obs_seq):
        states = torch.stack(
            [_to_float_tensor(o["states"]) for o in obs_seq], dim=0
        ).unsqueeze(1)  # [T, 1, 40]
        d = {"states": states}
        if include_images:
            imgs = torch.stack(
                [o["main_images"].float().cpu() / 255.0 if o["main_images"].dtype == torch.uint8
                 else o["main_images"].float().cpu()
                 for o in obs_seq], dim=0
            ).unsqueeze(1)  # [T, 1, H, W, C]
            d["main_images"] = imgs
        return d

    curr_obs = _build_obs_dict(curr_obs_raw)
    next_obs = _build_obs_dict(next_obs_raw)

    # ---- model_weights_id — demos have no policy; use a fixed human-demo UUID ----
    _HUMAN_DEMO_UUID = "human-demonstration-fixed-seed-00000000"

    traj = Trajectory(
        max_episode_length=T,
        model_weights_id=_HUMAN_DEMO_UUID,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        truncations=truncations,
        dones=dones,
        curr_obs=curr_obs,
        next_obs=next_obs,
    )
    return traj


def _save_trajectory(traj: Trajectory, traj_id: int, save_dir: str) -> dict:
    """Save one Trajectory to disk as a .pt file; return its index entry."""
    filename = f"trajectory_{traj_id}_{traj.model_weights_id}.pt"
    path = os.path.join(save_dir, filename)

    traj_dict = {}
    for field_name in traj.__dataclass_fields__.keys():
        value = getattr(traj, field_name, None)
        if value is not None:
            traj_dict[field_name] = value

    torch.save(traj_dict, path)

    T, B = traj.rewards.shape[:2]
    num_samples = T * B

    return {
        "num_samples": num_samples,
        "trajectory_id": traj_id,
        "max_episode_length": traj.max_episode_length,
        "shape": list(traj.rewards.shape),
        "model_weights_id": traj.model_weights_id,
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert GeneSim demo .pkl files to TrajectoryReplayBuffer checkpoint."
    )
    parser.add_argument(
        "--demo-dir", default="my_demos",
        help="Directory containing recorded .pkl demo files.",
    )
    parser.add_argument(
        "--output-dir", default="/tmp/junpu_demo_buffer",
        help="Output directory for the buffer checkpoint.",
    )
    parser.add_argument(
        "--include-images", action="store_true",
        help="Include main_images in the buffer (warning: large files).",
    )
    parser.add_argument(
        "--placeholder-reward", type=float, default=None,
        help="Override all rewards with this constant value. "
             "If omitted, recorded rewards are used.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed recorded in metadata.json.",
    )
    args = parser.parse_args()

    # ── discover demo files ────────────────────────────────────────────────
    pattern = os.path.join(args.demo_dir, "*.pkl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        sys.exit(f"[convert] No .pkl files found in {args.demo_dir!r}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[convert] Found {len(paths)} demo(s) in {args.demo_dir!r}")
    print(f"[convert] Output → {args.output_dir!r}")
    print(f"[convert] include_images={args.include_images}, "
          f"placeholder_reward={args.placeholder_reward}")

    trajectory_index = {}
    trajectory_id_list = []
    total_samples = 0

    for traj_id, pkl_path in enumerate(paths):
        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)

        T = len(ep["actions"])
        ep_id = ep.get("episode_id", traj_id)
        success = ep.get("success", "?")

        traj = _demo_to_trajectory(ep, args.include_images, args.placeholder_reward)
        info = _save_trajectory(traj, traj_id, args.output_dir)

        trajectory_index[traj_id] = info
        trajectory_id_list.append(traj_id)
        total_samples += info["num_samples"]

        print(f"[convert]   [{traj_id:3d}] {os.path.basename(pkl_path)}: "
              f"T={T} steps, success={success}")

    # ── write metadata.json ────────────────────────────────────────────────
    metadata = {
        "trajectory_format": "pt",
        "size": len(paths),
        "total_samples": total_samples,
        "trajectory_counter": len(paths),
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ── write trajectory_index.json ────────────────────────────────────────
    index_data = {
        "trajectory_index": trajectory_index,
        "trajectory_id_list": trajectory_id_list,
    }
    with open(os.path.join(args.output_dir, "trajectory_index.json"), "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"\n[convert] Done. {len(paths)} trajectories, "
          f"{total_samples} total samples → {args.output_dir!r}")
    print("[convert] Verify with:")
    print(f"  python rlinf/data/replay_buffer.py "
          f"--load-path {args.output_dir} "
          f"--num-chunks 32 --cache-size 20 --enable-cache")


if __name__ == "__main__":
    main()
