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
#       --output-dir /tmp/geniesim_demo_buffer \
#       [--include-images]        # include main_images (large!)
#       [--placeholder-reward 0]  # override reward value (default: use recorded)

import argparse
import glob
import json
import os
import pickle
import sys

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

_STATE_INDICES = np.arange(40, 52).astype(np.intp)


def _to_float_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float().cpu()
    return torch.tensor(np.asarray(x, dtype=np.float32))


def _quat_angle_diff_scalar(q1, q2):
    dot = abs(float(np.sum(q1 * q2)))
    return 2.0 * np.arccos(min(dot, 1.0))


# Reward constants — must stay in sync with place_workpiece.py.
# (Source of truth: rlinf/envs/geniesim/tasks/place_workpiece.py)
_REWARD_TARGET_REL_POS = np.array([-0.073, 0.007, 1.185], dtype=np.float32)
_REWARD_TARGET_WP_QUAT = np.array([0.1807, 0.6802, 0.6847, 0.1896], dtype=np.float32)
_REWARD_TARGET_WP_QUAT /= np.linalg.norm(_REWARD_TARGET_WP_QUAT)
_REWARD_XY_TOL = 0.02
_REWARD_Z_TOL = 0.01
_REWARD_ORIENT_TOL = 0.15
_REWARD_STILL_SPEED = 0.02
_REWARD_STILL_STEPS = 5
_REWARD_R_BELOW_COEF = -20.0
_REWARD_BELOW_THRESH = 0.01
_REWARD_R_PROGRESS_SCALE = 10.0
_REWARD_R_Z_INSERT_COEF = -2.0
_REWARD_TERMINAL_SUCCESS = 50.0
_REWARD_TERMINAL_FAIL = -10.0

_EE_R_RESET_POS = np.array([0.4833, 0.0051, 1.2548], dtype=np.float32)
_EE_R_RESET_RPY = np.array([2.5633, 0.0261, 1.5791], dtype=np.float32)
_POS_SCALE = np.float32(0.015)
_RPY_SCALE = np.float32(0.05)


def _recompute_rewards(ep: dict) -> torch.Tensor:
    """Replay place_workpiece.py:_compute_reward offline.

    Mirrors the runtime reward in place_workpiece.py:
      r_progress  = (prev_dist_3d - cur_dist_3d) * 10  (PBRS, 0 on first step)
      r_z_insert  = -2 * |diff_z|  iff xy_aligned, else 0
      r_below     = -20 * max(-diff_z - 0.01, 0)
    Plus terminal overrides applied at the success / fail step:
      success step -> +50  (replaces dense reward at that index)
      fail    step -> -10  (replaces dense reward at that index)

    Note on info indexing: infos[0] is the post-reset state (empty dict, no
    body_poses), and infos[1..T] hold the post-step poses for actions[0..T-1].
    So transition t uses infos[t+1].
    """
    T = len(ep["actions"])
    infos = ep.get("infos", [])
    tgt = _REWARD_TARGET_REL_POS
    tgt_q = _REWARD_TARGET_WP_QUAT
    prev_wp = None
    prev_dist_3d = None
    still_cnt = 0
    success_step = -1
    rewards = []
    for t in range(T):
        # transition t corresponds to action[t] which produced infos[t+1].
        info_idx = t + 1
        bp = infos[info_idx].get("body_poses", {}) if info_idx < len(infos) else {}
        wp = bp.get("workpiece_r")
        ws = bp.get("/World/workspace01")
        if wp is None or ws is None:
            rewards.append(0.0)
            prev_dist_3d = None
            continue
        wp_pos, wp_q = wp[:3].copy(), wp[3:7].copy()
        ws_pos = ws[:3].copy()
        rel = wp_pos - ws_pos
        dxy = float(np.linalg.norm(rel[:2] - tgt[:2]))
        dz = float(rel[2] - tgt[2])
        d3d = float(np.sqrt(dxy * dxy + dz * dz))
        odiff = _quat_angle_diff_scalar(wp_q, tgt_q)

        if prev_wp is not None:
            wp_spd = float(np.linalg.norm(wp_pos - prev_wp) * 30.0)
        else:
            wp_spd = 0.0
        prev_wp = wp_pos.copy()

        if prev_dist_3d is None:
            r_progress = 0.0
        else:
            r_progress = (prev_dist_3d - d3d) * _REWARD_R_PROGRESS_SCALE
        prev_dist_3d = d3d

        r_z_insert = (_REWARD_R_Z_INSERT_COEF * abs(dz)) if dxy < _REWARD_XY_TOL else 0.0

        overshoot = max(-dz - _REWARD_BELOW_THRESH, 0.0)
        r_below = _REWARD_R_BELOW_COEF * overshoot

        near = (
            dxy < _REWARD_XY_TOL
            and abs(dz) < _REWARD_Z_TOL
            and odiff < _REWARD_ORIENT_TOL
        )
        still = wp_spd < _REWARD_STILL_SPEED
        if near and still:
            still_cnt += 1
        else:
            still_cnt = 0
        if still_cnt == _REWARD_STILL_STEPS and success_step < 0:
            success_step = t

        rewards.append(r_progress + r_z_insert + r_below)

    # Terminal overrides at the last step. demos here are all *success* episodes
    # (filename suffix _success.pkl), so the final transition is the success
    # step; non-success demos would receive the fail bonus.
    if T > 0:
        is_success = bool(ep.get("success", success_step >= 0))
        rewards[-1] = (
            _REWARD_TERMINAL_SUCCESS if is_success else _REWARD_TERMINAL_FAIL
        )
    return torch.tensor(rewards, dtype=torch.float32)


def _to_bool_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.bool().cpu()
    return torch.tensor(bool(x))


def _demo_to_trajectory(
    ep: dict,
    include_images: bool,
    placeholder_reward: float | None,
    recompute_reward: bool = False,
) -> Trajectory:
    """Convert a single demo episode dict to a Trajectory object.

    Shape convention used by TrajectoryReplayBuffer: [T, B, ...] where B=1.
    curr_obs[t] is the obs before step t; next_obs[t] is the obs after step t.
    """
    obs_list = ep["observations"]  # length T+1 if final obs present, else T
    act_list = ep["actions"]  # length T
    rew_list = ep["rewards"]
    term_list = ep["terminated"]
    trunc_list = ep["truncated"]
    infos_list = ep.get(
        "infos", []
    )  # per-step info dicts (may contain intervene_action)

    T = len(act_list)

    # ---- actions ----
    # When demos are collected via SpacemouseSimIntervention the caller passes
    # all-zero policy actions, so the recorded act_list may be all zeros.
    # The actual EEF targets are stored in infos_list[t]["intervene_action"].
    # Use intervene_action when present (same fallback logic as replay_sim_demos.py).
    def _effective_action(t: int) -> torch.Tensor:
        info_t = infos_list[t] if t < len(infos_list) else {}
        if isinstance(info_t, dict) and "intervene_action" in info_t:
            ia = info_t["intervene_action"]
            return (
                ia.float().cpu()
                if isinstance(ia, torch.Tensor)
                else torch.tensor(np.asarray(ia, dtype=np.float32))
            )
        return _to_float_tensor(act_list[t])

    actions = torch.stack([_effective_action(t) for t in range(T)], dim=0)  # [T, 7]
    if actions.shape[-1] == 14:
        right_arm = torch.cat(
            [
                actions[:, 6:9],
                actions[:, 9:12],
                actions[:, 13:14],
            ],
            dim=-1,
        )
        actions = right_arm

    actions = actions.clamp(-1.0, 1.0)

    actions = actions.unsqueeze(1)  # [T, 1, 7]

    # ---- rewards ----
    if placeholder_reward is not None:
        rewards = torch.full((T, 1, 1), float(placeholder_reward))
    elif recompute_reward:
        r_vec = _recompute_rewards(ep)
        rewards = r_vec.reshape(T, 1, 1)
    else:
        rewards = torch.stack(
            [_to_float_tensor(r).reshape(1) for r in rew_list[:T]], dim=0
        )  # [T, 1]
        # If the source recorded an extra terminal reward (rew_list has T+1
        # entries, e.g. the r_success bonus fired on the final terminal frame),
        # fold it into the last transition so the bonus is preserved.
        if len(rew_list) > T:
            extra = sum(
                _to_float_tensor(r).reshape(1) for r in rew_list[T:]
            )
            rewards[-1] = rewards[-1] + extra
        rewards = rewards.unsqueeze(1)  # [T, 1, 1]

    # ---- terminations / truncations / dones ----
    # Cap to T to match actions/observations length. Raw lists may contain T+1
    # entries (one per state including the terminal); the buffer expects all
    # transition-aligned fields to have length T.
    terminations = torch.stack(
        [_to_bool_tensor(t).reshape(1) for t in term_list[:T]], dim=0
    ).unsqueeze(1)  # [T, 1, 1]

    truncations = torch.stack(
        [_to_bool_tensor(t).reshape(1) for t in trunc_list[:T]], dim=0
    ).unsqueeze(1)  # [T, 1, 1]

    dones = terminations | truncations  # [T, 1, 1]

    # ---- observations ----
    # curr_obs[t] = obs_list[t], next_obs[t] = obs_list[t+1]
    # If obs_list has exactly T entries (no final obs), duplicate the last one.
    if len(obs_list) >= T + 1:
        curr_obs_raw = obs_list[:T]
        next_obs_raw = obs_list[1 : T + 1]
    else:
        curr_obs_raw = obs_list[:T]
        next_obs_raw = obs_list[:T]  # fallback: same as curr
        if T > 1:
            next_obs_raw = obs_list[1:T] + [obs_list[-1]]

    def _build_obs_dict(obs_seq):
        states = torch.stack([_to_float_tensor(o["states"]) for o in obs_seq], dim=0)
        raw_dim = states.shape[-1]
        if raw_dim > len(_STATE_INDICES):
            valid_idx = _STATE_INDICES[_STATE_INDICES < raw_dim]
            states = states[..., valid_idx]
        states = states.unsqueeze(1)
        d = {"states": states}
        if include_images:
            img_keys = [
                k
                for k in obs_seq[0].keys()
                if k.endswith("_images") and obs_seq[0][k] is not None
            ]
            if not img_keys:
                img_keys = ["main_images"]
            first_key = img_keys[0]
            img_list = []
            for o in obs_seq:
                img = o.get(first_key)
                if img is None:
                    continue
                if img.dtype != torch.uint8:
                    img = (img.float().cpu().clamp(0, 1) * 255).to(torch.uint8)
                else:
                    img = img.cpu()
                img_list.append(img)
            if img_list:
                imgs = torch.stack(img_list, dim=0).unsqueeze(1)
                d["main_images"] = imgs
            if len(img_keys) > 1:
                extras = []
                for extra_key in img_keys[1:]:
                    extra_list = []
                    for o in obs_seq:
                        img = o.get(extra_key)
                        if img is None:
                            continue
                        if img.dtype != torch.uint8:
                            img = (img.float().cpu().clamp(0, 1) * 255).to(torch.uint8)
                        else:
                            img = img.cpu()
                        extra_list.append(img)
                    if extra_list:
                        extras.append(torch.stack(extra_list, dim=0))
                if extras:
                    d["extra_view_images"] = torch.stack(extras, dim=2).unsqueeze(1)
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
        "--demo-dir",
        default="my_demos",
        help="Directory containing recorded .pkl demo files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/geniesim_demo_buffer",
        help="Output directory for the buffer checkpoint.",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Include main_images in the buffer (warning: large files).",
    )
    parser.add_argument(
        "--placeholder-reward",
        type=float,
        default=None,
        help="Override all rewards with this constant value. "
        "If omitted, recorded rewards are used.",
    )
    parser.add_argument(
        "--recompute-reward",
        action="store_true",
        help="Recompute rewards using the current reward function "
        "(requires body_poses in infos).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
    print(
        f"[convert] include_images={args.include_images}, "
        f"placeholder_reward={args.placeholder_reward}, "
        f"recompute_reward={args.recompute_reward}"
    )

    trajectory_index = {}
    trajectory_id_list = []
    total_samples = 0

    for traj_id, pkl_path in enumerate(paths):
        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)

        T = len(ep["actions"])
        ep_id = ep.get("episode_id", traj_id)  # noqa: F841
        success = ep.get("success", "?")

        traj = _demo_to_trajectory(
            ep,
            args.include_images,
            args.placeholder_reward,
            recompute_reward=args.recompute_reward,
        )
        info = _save_trajectory(traj, traj_id, args.output_dir)

        trajectory_index[traj_id] = info
        trajectory_id_list.append(traj_id)
        total_samples += info["num_samples"]

        print(
            f"[convert]   [{traj_id:3d}] {os.path.basename(pkl_path)}: "
            f"T={T} steps, success={success}"
        )

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

    print(
        f"\n[convert] Done. {len(paths)} trajectories, "
        f"{total_samples} total samples → {args.output_dir!r}"
    )
    print("[convert] Verify with:")
    print(
        f"  python rlinf/data/replay_buffer.py "
        f"--load-path {args.output_dir} "
        f"--num-chunks 32 --cache-size 20 --enable-cache"
    )


if __name__ == "__main__":
    main()
