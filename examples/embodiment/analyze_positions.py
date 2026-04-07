#!/usr/bin/env python3
import pickle, glob, os, numpy as np

data_dir = '/home/zy/code/rlinf_open_source/sim_demos'
files = sorted(glob.glob(os.path.join(data_dir, '*.pkl')))

TARGET_DX = -0.073
TARGET_DY = 0.007
TARGET_DZ = 1.185
PRE_PLACE_DZ = TARGET_DZ + 0.01

def to_np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except:
        pass
    return np.asarray(x)

def extract_wp_ws_traj(ep):
    infos = ep.get('infos', [])
    wp_list, ws_list = [], []
    for info in infos:
        if not isinstance(info, dict):
            wp_list.append(np.full(7, np.nan))
            ws_list.append(np.full(7, np.nan))
            continue
        bp = info.get('body_poses', {})
        wp_list.append(to_np(bp.get('workpiece_r', np.full(7, np.nan))).flatten()[:7])
        ws_list.append(to_np(bp.get('/World/workspace01', np.full(7, np.nan))).flatten()[:7])
    return np.array(wp_list), np.array(ws_list)

print("=" * 100)
print("PART 1: 末端执行器(EE)和工件(WP)速度分析")
print("=" * 100)

for f in files:
    with open(f, 'rb') as fh:
        ep = pickle.load(fh)
    fname = os.path.basename(f)
    ep_num = fname.split('episode_')[1].split('_')[0]
    n_steps = len(ep.get('actions', []))
    if n_steps < 10:
        continue

    wp_traj, ws_traj = extract_wp_ws_traj(ep)
    rel_traj = wp_traj[:, :3] - ws_traj[:, :3]

    wp_vel = np.diff(wp_traj[:, :3], axis=0) * 30.0
    wp_speed = np.linalg.norm(wp_vel, axis=-1)
    wp_speed_xy = np.linalg.norm(wp_vel[:, :2], axis=-1)
    wp_speed_z = np.abs(wp_vel[:, 2])

    obs_list = ep.get('observations', [])
    states_list = []
    for obs in obs_list:
        if isinstance(obs, dict):
            s = obs.get('states')
        else:
            s = obs
        if s is not None:
            states_list.append(to_np(s).flatten())
    states = np.array(states_list) if states_list else None

    ee_speed_arr = None
    if states is not None and states.shape[1] >= 26:
        ee_linvel = states[:, 20:23]
        ee_speed_arr = np.linalg.norm(ee_linvel, axis=-1)

    dist_to_target_xy = np.sqrt((rel_traj[:, 0] - TARGET_DX)**2 + (rel_traj[:, 1] - TARGET_DY)**2)
    dist_to_target_z = rel_traj[:, 2] - TARGET_DZ

    print(f"\n--- Episode {ep_num} ({n_steps} steps) ---")
    print(f"  WP speed (m/s):   mean={wp_speed.mean():.4f}  max={wp_speed.max():.4f}  p95={np.percentile(wp_speed, 95):.4f}")
    print(f"  WP speed_xy:      mean={wp_speed_xy.mean():.4f}  max={wp_speed_xy.max():.4f}")
    print(f"  WP speed_z:       mean={wp_speed_z.mean():.4f}  max={wp_speed_z.max():.4f}")
    if ee_speed_arr is not None:
        print(f"  EE speed (m/s):   mean={ee_speed_arr.mean():.4f}  max={ee_speed_arr.max():.4f}  p95={np.percentile(ee_speed_arr, 95):.4f}")

    below_target = dist_to_target_z < -0.005
    n_below = below_target.sum()
    print(f"  Steps below target_z-0.5cm: {n_below}/{len(dist_to_target_z)}")

    approaching_from_above = (dist_to_target_xy < 0.02) & (dist_to_target_z > 0.01)
    n_above = approaching_from_above.sum()
    print(f"  Steps at target_xy & above target_z+1cm (pre-place): {n_above}")

    near_target = (dist_to_target_xy < 0.02) & (np.abs(dist_to_target_z) < 0.01)
    n_near = near_target.sum()
    print(f"  Steps near target (xy<2cm & |dz|<1cm): {n_near}")

    print(f"  Final rel pos: dx={rel_traj[-1,0]:.4f} dy={rel_traj[-1,1]:.4f} dz={rel_traj[-1,2]:.4f}")
    print(f"  Final dist_to_target: xy={dist_to_target_xy[-1]:.4f}  z={dist_to_target_z[-1]:.4f}")

print("\n")
print("=" * 100)
print("PART 2: 轨迹阶段分析 (每个episode的dz随时间变化)")
print("=" * 100)

for f in files:
    with open(f, 'rb') as fh:
        ep = pickle.load(fh)
    fname = os.path.basename(f)
    ep_num = fname.split('episode_')[1].split('_')[0]
    n_steps = len(ep.get('actions', []))
    if n_steps < 10:
        continue

    wp_traj, ws_traj = extract_wp_ws_traj(ep)
    rel_traj = wp_traj[:, :3] - ws_traj[:, :3]
    dz_traj = rel_traj[:, 2]
    dx_traj = rel_traj[:, 0]
    dy_traj = rel_traj[:, 1]
    dist_xy = np.sqrt((dx_traj - TARGET_DX)**2 + (dy_traj - TARGET_DY)**2)

    print(f"\n--- Episode {ep_num} ---")
    checkpoints = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]
    print(f"  {'step':>5s}  {'dx':>7s} {'dy':>7s} {'dz':>7s}  {'dxy_tgt':>7s} {'dz_tgt':>7s}")
    for s in checkpoints:
        if s < len(dz_traj):
            dz_t = dz_traj[s] - TARGET_DZ
            print(f"  {s:>5d}  {dx_traj[s]:>7.4f} {dy_traj[s]:>7.4f} {dz_traj[s]:>7.4f}  {dist_xy[s]:>7.4f} {dz_t:>7.4f}")

print("\n")
print("=" * 100)
print("PART 3: 速度分位数统计（所有episode合并）")
print("=" * 100)

all_wp_speeds = []
all_ee_speeds = []
for f in files:
    with open(f, 'rb') as fh:
        ep = pickle.load(fh)
    n_steps = len(ep.get('actions', []))
    if n_steps < 10:
        continue
    wp_traj, ws_traj = extract_wp_ws_traj(ep)
    wp_vel = np.diff(wp_traj[:, :3], axis=0) * 30.0
    all_wp_speeds.extend(np.linalg.norm(wp_vel, axis=-1).tolist())

    obs_list = ep.get('observations', [])
    for obs in obs_list:
        if isinstance(obs, dict):
            s = obs.get('states')
        else:
            s = obs
        if s is not None:
            s_np = to_np(s).flatten()
            if len(s_np) >= 26:
                all_ee_speeds.append(np.linalg.norm(s_np[20:23]))

all_wp_speeds = np.array(all_wp_speeds)
all_ee_speeds = np.array(all_ee_speeds)

print(f"WP speed (m/s) across all demos:")
for p in [50, 75, 90, 95, 99, 100]:
    v = np.percentile(all_wp_speeds, p) if p < 100 else all_wp_speeds.max()
    print(f"  p{p:>3d}: {v:.4f}")

print(f"\nEE speed (m/s) across all demos:")
for p in [50, 75, 90, 95, 99, 100]:
    v = np.percentile(all_ee_speeds, p) if p < 100 else all_ee_speeds.max()
    print(f"  p{p:>3d}: {v:.4f}")

print(f"\nSuggested speed penalty threshold (p95 of demos): WP={np.percentile(all_wp_speeds, 95):.4f}  EE={np.percentile(all_ee_speeds, 95):.4f}")

print("\n")
print("=" * 100)
print("PART 4: 朝向分析 (工件 vs 工作台 四元数)")
print("=" * 100)

def quat_angle(q1, q2):
    q1 = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2, axis=-1, keepdims=True) + 1e-8)
    if q1.ndim == 1:
        dot = abs(np.dot(q1, q2))
    else:
        dot = np.abs(np.sum(q1 * q2, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0, 1.0))

def quat_to_euler(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.arcsin(sinp))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=-1)

print("\n--- 最终时刻: 工件四元数(wxyz) & 工作台四元数(wxyz) & 相对角度 ---")
print(f"{'Ep':>4s}  {'WP quat (wxyz)':>40s}  {'WS quat (wxyz)':>40s}  {'angle':>7s}")
print('-' * 105)

all_final_wp_quat = []
all_final_ws_quat = []
all_final_rel_angle = []

for f in files:
    with open(f, 'rb') as fh:
        ep = pickle.load(fh)
    fname = os.path.basename(f)
    ep_num = fname.split('episode_')[1].split('_')[0]
    n_steps = len(ep.get('actions', []))
    if n_steps < 10:
        continue
    wp_traj, ws_traj = extract_wp_ws_traj(ep)
    wp_q = wp_traj[-1, 3:7]
    ws_q = ws_traj[-1, 3:7]
    angle = quat_angle(wp_q, ws_q)
    all_final_wp_quat.append(wp_q)
    all_final_ws_quat.append(ws_q)
    all_final_rel_angle.append(angle)
    print(f"{ep_num:>4s}  [{wp_q[0]:>7.4f} {wp_q[1]:>7.4f} {wp_q[2]:>7.4f} {wp_q[3]:>7.4f}]  [{ws_q[0]:>7.4f} {ws_q[1]:>7.4f} {ws_q[2]:>7.4f} {ws_q[3]:>7.4f}]  {angle:>7.4f}")

all_final_wp_quat = np.array(all_final_wp_quat)
all_final_ws_quat = np.array(all_final_ws_quat)
all_final_rel_angle = np.array(all_final_rel_angle)

print('-' * 105)
mean_wp_q = all_final_wp_quat.mean(axis=0)
mean_wp_q /= np.linalg.norm(mean_wp_q)
mean_ws_q = all_final_ws_quat.mean(axis=0)
mean_ws_q /= np.linalg.norm(mean_ws_q)
print(f"Mean WP quat: [{mean_wp_q[0]:.4f} {mean_wp_q[1]:.4f} {mean_wp_q[2]:.4f} {mean_wp_q[3]:.4f}]")
print(f"Mean WS quat: [{mean_ws_q[0]:.4f} {mean_ws_q[1]:.4f} {mean_ws_q[2]:.4f} {mean_ws_q[3]:.4f}]")
print(f"Rel angle: mean={all_final_rel_angle.mean():.4f} std={all_final_rel_angle.std():.4f} min={all_final_rel_angle.min():.4f} max={all_final_rel_angle.max():.4f}")

print(f"\nMean WP euler (rad): {quat_to_euler(mean_wp_q)}")
print(f"Mean WS euler (rad): {quat_to_euler(mean_ws_q)}")

print("\n--- 全程朝向变化: 工件相对目标朝向(mean final)的偏差轨迹 ---")
target_wp_quat = mean_wp_q.copy()
print(f"Target WP quat (目标朝向): [{target_wp_quat[0]:.4f} {target_wp_quat[1]:.4f} {target_wp_quat[2]:.4f} {target_wp_quat[3]:.4f}]")

for f in files:
    with open(f, 'rb') as fh:
        ep = pickle.load(fh)
    fname = os.path.basename(f)
    ep_num = fname.split('episode_')[1].split('_')[0]
    n_steps = len(ep.get('actions', []))
    if n_steps < 10:
        continue
    wp_traj, _ = extract_wp_ws_traj(ep)
    wp_quats = wp_traj[:, 3:7]
    orient_diffs = quat_angle(wp_quats, target_wp_quat[None])

    print(f"\n  Episode {ep_num}: orient_diff_to_target_quat (rad)")
    print(f"    init={orient_diffs[0]:.4f}  mid={orient_diffs[len(orient_diffs)//2]:.4f}  final={orient_diffs[-1]:.4f}  max={orient_diffs.max():.4f}  mean={orient_diffs.mean():.4f}")

print("\n--- 全程朝向变化: 工件相对初始朝向的偏差轨迹 ---")
for f in files:
    with open(f, 'rb') as fh:
        ep = pickle.load(fh)
    fname = os.path.basename(f)
    ep_num = fname.split('episode_')[1].split('_')[0]
    n_steps = len(ep.get('actions', []))
    if n_steps < 10:
        continue
    wp_traj, _ = extract_wp_ws_traj(ep)
    wp_quats = wp_traj[:, 3:7]
    init_quat = wp_quats[0]
    orient_diffs_from_init = quat_angle(wp_quats, init_quat[None])
    print(f"  Episode {ep_num}: orient_drift_from_init: max={orient_diffs_from_init.max():.4f}  final={orient_diffs_from_init[-1]:.4f}  mean={orient_diffs_from_init.mean():.4f}")
