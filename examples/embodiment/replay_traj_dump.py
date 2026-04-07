#!/usr/bin/env python3
"""Replay dumped trajectories and print per-step reward decomposition.

Usage:
    python replay_traj_dump.py --traj-file <path_to_step_NNNNNN_traj_N.pt> [--env-id 0]

Each trajectory file has shape [T=100, B=num_envs, ...].  This script picks
one env (--env-id) and walks through all T steps, printing the reward
breakdown that matches the task code in junpu_place_workpiece.py.

The script also detects episode boundaries (done=True) and restarts the
per-episode accumulator.
"""

import argparse
import math

import torch
import numpy as np


_TARGET_REL_POS = np.array([-0.073, 0.007, 1.185], dtype=np.float32)
_TARGET_WP_QUAT = np.array([-0.00587199, 0.00482861, -0.7230842, 0.69078833], dtype=np.float32)

_XY_TOLERANCE = 0.01
_Z_TOLERANCE = 0.01
_ORIENT_TOLERANCE = 0.15
_EE_SPEED_THRESH = 0.10
_STILL_SPEED_THRESH = 0.002
_STILL_STEPS_REQUIRED = 30

_TERM_XY_DIST = 0.12
_TERM_Z_DROP = -0.04


def quat_angle_diff(q1, q2):
    q1_n = q1 / (np.linalg.norm(q1) + 1e-8)
    q2_n = q2 / (np.linalg.norm(q2) + 1e-8)
    dot = min(abs(np.dot(q1_n, q2_n)), 1.0)
    return 2.0 * math.acos(dot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-file", required=True)
    parser.add_argument("--env-id", type=int, default=0)
    args = parser.parse_args()

    d = torch.load(args.traj_file, map_location="cpu")
    T = d["actions"].shape[0]
    B = d["actions"].shape[1]
    eid = args.env_id
    assert 0 <= eid < B, f"env_id {eid} out of range [0, {B})"

    actions = d["actions"][:, eid].numpy()
    rewards = d["rewards"][:, eid, 0].numpy()
    dones = d["dones"][:, eid, 0].numpy() if d["dones"].shape[0] > T else d["dones"][:T, eid, 0].numpy()
    terms = d["terminations"][:, eid, 0].numpy() if d["terminations"].shape[0] > T else d["terminations"][:T, eid, 0].numpy()
    intervene = d["intervene_flags"][:, eid].numpy()

    states = d["curr_obs"]["states"][:, eid].numpy()
    next_states = d["next_obs"]["states"][:, eid].numpy()

    target_rel = _TARGET_REL_POS
    target_quat = _TARGET_WP_QUAT

    prev_dist_3d = None
    prev_orient_diff = None
    ep_return = 0.0
    ep_step = 0
    ep_id = 0

    print(f"Trajectory: {args.traj_file}")
    print(f"Env: {eid}/{B},  T={T}")
    print(f"{'step':>4s} {'reward':>8s} {'r_alive':>8s} {'r_appr':>8s} {'r_oapp':>8s} "
          f"{'r_speed':>8s} {'r_below':>8s} {'dist3d':>8s} {'dxy':>8s} {'dz':>8s} "
          f"{'odiff':>8s} {'eespd':>8s} {'act_norm':>8s} {'intv':>4s} {'done':>4s} {'ep_ret':>8s}")
    print("-" * 140)

    for t in range(T):
        s = states[t]
        ns = next_states[t]

        wp_pos = s[7:10]
        wp_quat_s = s[10:14]
        ws_x, ws_y, ws_z = 0.0, 0.0, 0.0

        rel_pos = wp_pos - np.array([ws_x, ws_y, ws_z])
        dist_xy = np.linalg.norm(rel_pos[:2] - target_rel[:2])
        diff_z = rel_pos[2] - target_rel[2]
        dist_3d = math.sqrt(dist_xy**2 + diff_z**2)

        orient_diff = quat_angle_diff(wp_quat_s, target_quat)

        r_alive = 1.0 * math.exp(-5.0 * dist_3d) * math.exp(-2.0 * orient_diff)

        if prev_dist_3d is not None:
            r_approach = 5.0 * (prev_dist_3d - dist_3d)
            r_orient_approach = 1.0 * (prev_orient_diff - orient_diff)
        else:
            r_approach = 0.0
            r_orient_approach = 0.0

        ee_lin_vel = s[6:9]
        ee_speed = np.linalg.norm(ee_lin_vel)
        excess_speed = max(ee_speed - _EE_SPEED_THRESH, 0.0)
        r_speed = -2.0 * excess_speed

        overshoot = max(-diff_z - 0.01, 0.0)
        r_below = -20.0 * overshoot

        act_norm = np.linalg.norm(actions[t])
        is_intervene = intervene[t].any()

        done_t = bool(dones[t]) if t < len(dones) else False
        term_t = bool(terms[t]) if t < len(terms) else False

        computed_r = r_alive + r_approach + r_orient_approach + r_speed + r_below
        actual_r = rewards[t]

        ep_return += actual_r
        ep_step += 1

        flags = ""
        if dist_xy > _TERM_XY_DIST:
            flags += "XY! "
        if diff_z < _TERM_Z_DROP:
            flags += "DRP! "

        print(f"{t:4d} {actual_r:8.3f} {r_alive:8.4f} {r_approach:8.4f} {r_orient_approach:8.4f} "
              f"{r_speed:8.4f} {r_below:8.4f} {dist_3d:8.4f} {dist_xy:8.4f} {diff_z:8.4f} "
              f"{orient_diff:8.4f} {ee_speed:8.4f} {act_norm:8.4f} {'Y' if is_intervene else '.':>4s} "
              f"{'D' if done_t else '.':>4s} {ep_return:8.2f}  {flags}")

        prev_dist_3d = dist_3d
        prev_orient_diff = orient_diff

        if done_t:
            print(f"  >>> Episode {ep_id} ended at step {t}, return={ep_return:.3f}, len={ep_step}")
            ep_return = 0.0
            ep_step = 0
            ep_id += 1
            prev_dist_3d = None
            prev_orient_diff = None


if __name__ == "__main__":
    main()
