#!/usr/bin/env python3
"""Replay a dumped trajectory in a real sim env and print per-step reward detail.

Usage (inside the docker container):
    python examples/embodiment/replay_traj_in_sim.py \
        --config examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml \
        --traj-file /path/to/step_000010_traj_0.pt \
        --env-id 3 \
        --step-hz 10

The script creates a 1-env sim, finds the first episode boundary (done=True)
in the trajectory for the chosen env-id, then replays the actions from that
fresh episode start so that _ee_target is consistent.  Each step prints the
real reward_detail from the task env.
"""

import argparse
import atexit
import os
import sys
import time

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RLINF_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _RLINF_ROOT not in sys.path:
    sys.path.insert(0, _RLINF_ROOT)

if "GENIESIM_ROOT" not in os.environ:
    _default_gs_root = os.path.abspath(os.path.join(_RLINF_ROOT, ".."))
    os.environ["GENIESIM_ROOT"] = _default_gs_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml")
    parser.add_argument("--traj-file", required=True)
    parser.add_argument("--env-id", type=int, default=0,
                        help="Which env in the trajectory batch to replay")
    parser.add_argument("--step-hz", type=float, default=10.0)
    parser.add_argument("--episode", type=int, default=0,
                        help="Which episode within the trajectory to replay (0=first)")
    args = parser.parse_args()

    d = torch.load(args.traj_file, map_location="cpu")
    T = d["actions"].shape[0]
    B = d["actions"].shape[1]
    eid = args.env_id
    assert 0 <= eid < B, f"env_id {eid} out of range [0, {B})"

    actions_all = d["actions"][:, eid].numpy()
    rewards_all = d["rewards"][:, eid, 0].numpy()
    dones_all = d["dones"][:, eid, 0].numpy()
    intervene_all = d["intervene_flags"][:, eid].numpy()

    episodes = []
    ep_start = 0
    for t in range(T):
        if dones_all[t]:
            episodes.append((ep_start, t + 1))
            ep_start = t + 1
    if ep_start < T:
        episodes.append((ep_start, T))

    print(f"Trajectory: {args.traj_file}")
    print(f"Env: {eid}/{B},  T={T},  Episodes found: {len(episodes)}")
    for i, (s, e) in enumerate(episodes):
        print(f"  ep {i}: steps [{s}, {e})  len={e-s}  "
              f"return={rewards_all[s:e].sum():.3f}  "
              f"done={bool(dones_all[e-1]) if e-1 < T else '?'}")

    ep_idx = args.episode
    if ep_idx >= len(episodes):
        print(f"Episode {ep_idx} not found (only {len(episodes)} episodes). Using last.")
        ep_idx = len(episodes) - 1

    ep_start, ep_end = episodes[ep_idx]
    ep_actions = actions_all[ep_start:ep_end]
    ep_rewards = rewards_all[ep_start:ep_end]
    ep_intervene = intervene_all[ep_start:ep_end]
    ep_len = ep_end - ep_start
    print(f"\nReplaying episode {ep_idx}: steps [{ep_start}, {ep_end}), len={ep_len}")

    from omegaconf import OmegaConf, open_dict
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(_RLINF_ROOT, cfg_path)
    cfg = OmegaConf.load(cfg_path)
    with open_dict(cfg):
        cfg.init_params.num_envs = 1

    from rlinf.envs.geniesim import REGISTER_GENIESIM_ENVS
    from rlinf.envs.geniesim.tasks import junpu_place_workpiece  # noqa: F401

    task_id = cfg.init_params.id
    EnvCls = REGISTER_GENIESIM_ENVS[task_id]
    print(f"Creating {EnvCls.__name__} (num_envs=1, task={task_id})")

    env = EnvCls(cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info={})
    atexit.register(env.close)

    env.reset()
    print(f"\nEnv reset complete. Starting replay...\n")

    action_dim = ep_actions.shape[-1]
    step_dt = 1.0 / args.step_hz if args.step_hz > 0 else 0.0

    header = (f"{'step':>4s} {'orig_r':>8s} {'sim_r':>8s} {'r_alive':>8s} {'r_appr':>8s} "
              f"{'r_oappr':>8s} {'r_spd':>8s} {'r_below':>8s} {'r_succ':>8s} "
              f"{'d3d':>8s} {'dxy':>8s} {'dz':>8s} {'odiff':>8s} "
              f"{'intv':>4s} {'done':>4s}")
    print(header)
    print("-" * len(header))

    ep_return_sim = 0.0

    for t in range(ep_len):
        act = torch.from_numpy(ep_actions[t:t+1].copy()).float()
        t_start = time.monotonic()

        obs, rewards, terminated, truncated, infos = env.step(act, auto_reset=False)

        sim_r = rewards[0].item() if isinstance(rewards, torch.Tensor) else float(rewards[0])
        ep_return_sim += sim_r
        orig_r = ep_rewards[t]
        is_intv = bool(ep_intervene[t].any())
        done = bool((terminated | truncated)[0]) if isinstance(terminated, torch.Tensor) else False

        rd = getattr(env, '_last_reward_detail', None)
        if rd is not None:
            r_alive = rd['r_alive'][0].item() if isinstance(rd['r_alive'], torch.Tensor) else float(rd['r_alive'][0])
            r_appr = rd['r_approach'][0].item() if isinstance(rd['r_approach'], torch.Tensor) else float(rd['r_approach'][0])
            r_oappr = rd['r_orient_approach'][0].item() if isinstance(rd['r_orient_approach'], torch.Tensor) else float(rd['r_orient_approach'][0])
            r_spd = rd['r_speed'][0].item() if isinstance(rd['r_speed'], torch.Tensor) else float(rd['r_speed'][0])
            r_below = rd['r_below'][0].item() if isinstance(rd['r_below'], torch.Tensor) else float(rd['r_below'][0])
            r_succ = rd['r_success'][0].item() if isinstance(rd['r_success'], torch.Tensor) else float(rd['r_success'][0])
            d3d = rd['dist_3d'][0].item() if isinstance(rd['dist_3d'], torch.Tensor) else float(rd['dist_3d'][0])
            dxy = rd['dist_xy'][0].item() if isinstance(rd['dist_xy'], torch.Tensor) else float(rd['dist_xy'][0])
            dz = rd['diff_z'][0].item() if isinstance(rd['diff_z'], torch.Tensor) else float(rd['diff_z'][0])
            odiff = rd['orient_diff'][0].item() if isinstance(rd['orient_diff'], torch.Tensor) else float(rd['orient_diff'][0])
        else:
            r_alive = r_appr = r_oappr = r_spd = r_below = r_succ = 0.0
            d3d = dxy = dz = odiff = 0.0

        print(f"{t:4d} {orig_r:8.3f} {sim_r:8.3f} {r_alive:8.4f} {r_appr:8.4f} "
              f"{r_oappr:8.4f} {r_spd:8.4f} {r_below:8.4f} {r_succ:8.4f} "
              f"{d3d:8.4f} {dxy:8.4f} {dz:8.4f} {odiff:8.4f} "
              f"{'Y' if is_intv else '.':>4s} {'D' if done else '.':>4s}")

        if done:
            print(f"  >>> Sim terminated at step {t}, sim_return={ep_return_sim:.3f}")
            break

        if step_dt > 0:
            elapsed = time.monotonic() - t_start
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

    print(f"\nReplay complete. "
          f"Original return={ep_rewards[:ep_len].sum():.3f}, "
          f"Sim return={ep_return_sim:.3f}, "
          f"Steps played={min(t+1, ep_len)}/{ep_len}")

    env.close()


if __name__ == "__main__":
    main()
