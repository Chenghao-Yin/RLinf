#!/usr/bin/env python3
"""Replay demo episodes through new reward function to verify behaviour."""
import pickle, glob, os, sys, numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rlinf.envs.geniesim.tasks.junpu_place_workpiece import (
    _TARGET_REL_POS, _TARGET_WP_QUAT, _RIVET_HEIGHT, _PRE_PLACE_DZ,
    _STILL_SPEED_THRESH, _XY_TOLERANCE, _Z_TOLERANCE, _ORIENT_TOLERANCE,
    _EE_SPEED_THRESH, _quat_angle_diff,
)

data_dir = '/home/zy/code/rlinf_open_source/sim_demos'
files = sorted(glob.glob(os.path.join(data_dir, '*.pkl')))

def to_np(x):
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except:
        pass
    return np.asarray(x)

target_rel = torch.from_numpy(_TARGET_REL_POS).unsqueeze(0)
target_quat = torch.from_numpy(_TARGET_WP_QUAT).unsqueeze(0)

for f in files:
    with open(f, 'rb') as fh:
        ep = pickle.load(fh)
    fname = os.path.basename(f)
    ep_num = fname.split('episode_')[1].split('_')[0]
    n_steps = len(ep.get('actions', []))
    if n_steps < 10:
        continue

    infos = ep.get('infos', [])
    observations = ep.get('observations', [])
    prev_wp_pos = None
    rewards_list = []
    components = {k: [] for k in ['r_xy', 'r_z_app', 'r_z_ins', 'r_below', 'r_lat', 'r_ori', 'r_ee', 'r_still', 'r_suc']}

    for t in range(n_steps):
        info = infos[t] if t < len(infos) else {}
        bp = info.get('body_poses', {})
        wp = to_np(bp.get('workpiece_r', np.full((1, 7), np.nan)))
        ws = to_np(bp.get('/World/workspace01', np.full((1, 7), np.nan)))
        if wp.ndim == 1:
            wp = wp[None]
        if ws.ndim == 1:
            ws = ws[None]

        wp_pos = torch.from_numpy(wp[:, :3].copy().astype(np.float32))
        wp_quat = torch.from_numpy(wp[:, 3:7].copy().astype(np.float32))
        ws_pos = torch.from_numpy(ws[:, :3].copy().astype(np.float32))

        rel_pos = wp_pos - ws_pos
        dist_xy = torch.linalg.norm(rel_pos[:, :2] - target_rel[:, :2], dim=-1)
        diff_z = rel_pos[:, 2] - target_rel[:, 2]

        r_xy = -5.0 * dist_xy
        r_z_approach = torch.where(diff_z > _RIVET_HEIGHT, -3.0 * diff_z, torch.zeros_like(diff_z))

        xy_aligned = dist_xy < _XY_TOLERANCE
        in_insert = (diff_z > -_Z_TOLERANCE) & (diff_z <= _RIVET_HEIGHT)
        r_z_insert = torch.where(xy_aligned & in_insert, torch.ones_like(diff_z), torch.zeros_like(diff_z))

        r_below = torch.where(diff_z < -_Z_TOLERANCE, torch.full_like(diff_z, -10.0), torch.zeros_like(diff_z))

        below_pp = diff_z < _RIVET_HEIGHT
        r_lateral = torch.where(below_pp & (~xy_aligned), -8.0 * dist_xy, torch.zeros_like(dist_xy))

        orient_diff = _quat_angle_diff(wp_quat, target_quat)
        r_orient = -2.0 * orient_diff

        obs = observations[t] if t < len(observations) else {}
        st = to_np(obs.get('states', np.zeros(26))).flatten().astype(np.float32)
        ee_speed = float(np.linalg.norm(st[20:23]))
        r_ee = -2.0 * max(ee_speed - _EE_SPEED_THRESH, 0.0)

        if prev_wp_pos is not None:
            wp_speed = float(torch.linalg.norm(wp_pos - prev_wp_pos, dim=-1).item()) * 30.0
        else:
            wp_speed = 0.0
        prev_wp_pos = wp_pos.clone()

        orient_ok = orient_diff.item() < _ORIENT_TOLERANCE
        near = (dist_xy.item() < _XY_TOLERANCE) and (abs(diff_z.item()) < _Z_TOLERANCE) and orient_ok
        still = wp_speed < _STILL_SPEED_THRESH
        r_still = 1.0 if (near and still) else 0.0

        total = r_xy.item() + r_z_approach.item() + r_z_insert.item() + r_below.item() + r_lateral.item() + r_orient.item() + r_ee + r_still
        rewards_list.append(total)
        components['r_xy'].append(r_xy.item())
        components['r_z_app'].append(r_z_approach.item())
        components['r_z_ins'].append(r_z_insert.item())
        components['r_below'].append(r_below.item())
        components['r_lat'].append(r_lateral.item())
        components['r_ori'].append(r_orient.item())
        components['r_ee'].append(r_ee)
        components['r_still'].append(r_still)
        components['r_suc'].append(0.0)

    rewards_arr = np.array(rewards_list)
    print(f"\n=== Episode {ep_num} ({n_steps} steps) ===")
    print(f"  Total reward: {rewards_arr.sum():.2f}  Mean/step: {rewards_arr.mean():.4f}")
    print(f"  First 5 steps: {rewards_arr[:5]}")
    print(f"  Last  5 steps: {rewards_arr[-5:]}")
    for k, v in components.items():
        arr = np.array(v)
        nonzero = (arr != 0).sum()
        print(f"  {k:>8s}: sum={arr.sum():>8.2f}  mean={arr.mean():>8.4f}  nonzero_steps={nonzero}")
