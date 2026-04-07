#!/usr/bin/env python3
"""
Analyze collected demonstration episodes for reward function optimization.

Reads pickle files saved by CollectEpisode and produces:
  1. Per-step trajectories of workpiece pose, EE pose, actions, rewards
  2. Summary statistics (min/max/mean) for key quantities
  3. Reward component breakdown over time
  4. Matplotlib plots saved to --output-dir

Usage:
  cd RLinf
  python examples/embodiment/analyze_demo_data.py \
      --data-dir /tmp/sim_demos \
      --output-dir /tmp/demo_analysis
"""

import argparse
import glob
import os
import pickle
import sys
from typing import Any, Dict, List, Optional

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RLINF_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _RLINF_ROOT not in sys.path:
    sys.path.insert(0, _RLINF_ROOT)


def _to_numpy(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _load_episodes(data_dir: str) -> List[Dict[str, Any]]:
    pattern = os.path.join(data_dir, "*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No .pkl files found in {data_dir}")
    episodes = []
    for f in files:
        with open(f, "rb") as fh:
            ep = pickle.load(fh)
        ep["_filename"] = os.path.basename(f)
        episodes.append(ep)
    print(f"Loaded {len(episodes)} episodes from {data_dir}")
    return episodes


def _extract_body_pose_trajectory(episode: Dict, body_name: str) -> Optional[np.ndarray]:
    infos = episode.get("infos", [])
    poses = []
    for info in infos:
        if not isinstance(info, dict):
            poses.append(np.full(7, np.nan))
            continue
        bp = info.get("body_poses")
        if bp is None:
            poses.append(np.full(7, np.nan))
            continue
        p = bp.get(body_name)
        if p is None:
            poses.append(np.full(7, np.nan))
            continue
        arr = _to_numpy(p).flatten()[:7]
        poses.append(arr)
    if not poses:
        return None
    return np.array(poses)


def _extract_reward_detail_trajectory(episode: Dict) -> Optional[Dict[str, np.ndarray]]:
    infos = episode.get("infos", [])
    keys = None
    all_data: Dict[str, list] = {}
    for info in infos:
        if not isinstance(info, dict):
            if keys:
                for k in keys:
                    all_data[k].append(np.nan)
            continue
        rd = info.get("reward_detail")
        if rd is None:
            if keys:
                for k in keys:
                    all_data[k].append(np.nan)
            continue
        if keys is None:
            keys = list(rd.keys())
            all_data = {k: [] for k in keys}
        for k in keys:
            v = rd.get(k)
            if v is not None:
                v = float(_to_numpy(v).flatten()[0])
            else:
                v = np.nan
            all_data[k].append(v)
    if not all_data:
        return None
    return {k: np.array(v) for k, v in all_data.items()}


def _extract_state_trajectory(episode: Dict) -> Optional[np.ndarray]:
    observations = episode.get("observations", [])
    states_list = []
    for obs in observations:
        if isinstance(obs, dict):
            s = obs.get("states")
        else:
            s = obs
        if s is not None:
            states_list.append(_to_numpy(s).flatten())
    if not states_list:
        return None
    return np.array(states_list)


def _extract_action_trajectory(episode: Dict) -> Optional[np.ndarray]:
    actions = episode.get("actions", [])
    if not actions:
        return None
    act_list = [_to_numpy(a).flatten() for a in actions]
    return np.array(act_list)


def _extract_reward_trajectory(episode: Dict) -> Optional[np.ndarray]:
    rewards = episode.get("rewards", [])
    if not rewards:
        return None
    return np.array([float(_to_numpy(r).flatten()[0]) for r in rewards])


def _quat_angle_diff_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1_n = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + 1e-8)
    q2_n = q2 / (np.linalg.norm(q2, axis=-1, keepdims=True) + 1e-8)
    dot = np.abs(np.sum(q1_n * q2_n, axis=-1)).clip(max=1.0)
    return 2.0 * np.arccos(dot)


def analyze_episode(episode: Dict, ep_idx: int) -> Dict[str, Any]:
    fname = episode.get("_filename", f"episode_{ep_idx}")
    success = episode.get("success", False)

    wp_traj = _extract_body_pose_trajectory(episode, "workpiece_r")
    ws_traj = _extract_body_pose_trajectory(episode, "/World/workspace01")
    rd_traj = _extract_reward_detail_trajectory(episode)
    state_traj = _extract_state_trajectory(episode)
    action_traj = _extract_action_trajectory(episode)
    reward_traj = _extract_reward_trajectory(episode)

    result: Dict[str, Any] = {
        "filename": fname,
        "success": success,
        "num_steps": len(episode.get("actions", [])),
    }

    if wp_traj is not None:
        result["wp_pos"] = wp_traj[:, :3]
        result["wp_quat"] = wp_traj[:, 3:7]

        wp_init_pos = wp_traj[0, :3]
        target_pos = wp_init_pos.copy()
        target_pos[2] -= 0.05

        dist_xy = np.linalg.norm(wp_traj[:, :2] - target_pos[:2], axis=-1)
        dist_z = wp_traj[:, 2] - target_pos[2]

        result["target_pos"] = target_pos
        result["dist_xy"] = dist_xy
        result["dist_z"] = dist_z
        result["wp_init_pos"] = wp_init_pos

        if wp_traj.shape[0] > 1:
            wp_velocity = np.diff(wp_traj[:, :3], axis=0) * 30.0
            wp_speed = np.linalg.norm(wp_velocity, axis=-1)
            result["wp_velocity"] = wp_velocity
            result["wp_speed"] = wp_speed

        wp_init_quat = wp_traj[0, 3:7]
        orient_diff = _quat_angle_diff_np(wp_traj[:, 3:7], wp_init_quat[None])
        result["orient_diff"] = orient_diff

    if ws_traj is not None:
        result["ws_pos"] = ws_traj[:, :3]

    if rd_traj is not None:
        result["reward_detail"] = rd_traj

    if state_traj is not None:
        result["states"] = state_traj

    if action_traj is not None:
        result["actions"] = action_traj
        if action_traj.shape[0] > 1:
            action_delta = np.diff(action_traj, axis=0)
            result["action_smoothness"] = np.linalg.norm(action_delta, axis=-1)

    if reward_traj is not None:
        result["rewards"] = reward_traj
        result["total_reward"] = float(np.nansum(reward_traj))
        result["mean_reward"] = float(np.nanmean(reward_traj))

    return result


def print_summary(analyses: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("DEMONSTRATION DATA ANALYSIS SUMMARY")
    print("=" * 80)

    for i, a in enumerate(analyses):
        print(f"\n--- Episode {i}: {a['filename']} ---")
        print(f"  Success: {a['success']}  |  Steps: {a['num_steps']}")
        if "total_reward" in a:
            print(f"  Total reward: {a['total_reward']:.2f}  |  Mean reward/step: {a['mean_reward']:.4f}")

        if "dist_xy" in a:
            print(f"  Target pos: {a['target_pos']}")
            print(f"  WP init pos: {a['wp_init_pos']}")
            dxy = a["dist_xy"]
            dz = a["dist_z"]
            print(f"  dist_xy: min={dxy.min():.4f}  max={dxy.max():.4f}  final={dxy[-1]:.4f}")
            print(f"  dist_z:  min={dz.min():.4f}  max={dz.max():.4f}  final={dz[-1]:.4f}")

        if "orient_diff" in a:
            od = a["orient_diff"]
            print(f"  orient_diff (rad): min={od.min():.4f}  max={od.max():.4f}  final={od[-1]:.4f}")

        if "wp_speed" in a:
            sp = a["wp_speed"]
            print(f"  wp_speed (m/s): min={sp.min():.4f}  max={sp.max():.4f}  final={sp[-1]:.4f}")

        if "action_smoothness" in a:
            sm = a["action_smoothness"]
            print(f"  action_smoothness: mean={sm.mean():.4f}  max={sm.max():.4f}")

        if "actions" in a:
            acts = a["actions"]
            print(f"  action range: pos_x=[{acts[:,0].min():.3f}, {acts[:,0].max():.3f}]  "
                  f"pos_y=[{acts[:,1].min():.3f}, {acts[:,1].max():.3f}]  "
                  f"pos_z=[{acts[:,2].min():.3f}, {acts[:,2].max():.3f}]")
            print(f"                rpy_r=[{acts[:,3].min():.3f}, {acts[:,3].max():.3f}]  "
                  f"rpy_p=[{acts[:,4].min():.3f}, {acts[:,4].max():.3f}]  "
                  f"rpy_y=[{acts[:,5].min():.3f}, {acts[:,5].max():.3f}]")
            if acts.shape[1] > 6:
                print(f"                grip=[{acts[:,6].min():.3f}, {acts[:,6].max():.3f}]")

        if "states" in a:
            st = a["states"]
            ee_r_pos = st[:, 14:17] if st.shape[1] >= 17 else None
            ee_r_rpy = st[:, 17:20] if st.shape[1] >= 20 else None
            if ee_r_pos is not None:
                print(f"  EE_R pos range: x=[{ee_r_pos[:,0].min():.3f}, {ee_r_pos[:,0].max():.3f}]  "
                      f"y=[{ee_r_pos[:,1].min():.3f}, {ee_r_pos[:,1].max():.3f}]  "
                      f"z=[{ee_r_pos[:,2].min():.3f}, {ee_r_pos[:,2].max():.3f}]")
            if ee_r_rpy is not None:
                print(f"  EE_R rpy range: r=[{ee_r_rpy[:,0].min():.3f}, {ee_r_rpy[:,0].max():.3f}]  "
                      f"p=[{ee_r_rpy[:,1].min():.3f}, {ee_r_rpy[:,1].max():.3f}]  "
                      f"y=[{ee_r_rpy[:,2].min():.3f}, {ee_r_rpy[:,2].max():.3f}]")

    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)
    success_eps = [a for a in analyses if a.get("success")]
    print(f"  Total episodes: {len(analyses)}  |  Successful: {len(success_eps)}")
    if success_eps:
        total_rewards = [a["total_reward"] for a in success_eps if "total_reward" in a]
        if total_rewards:
            print(f"  Success total rewards: mean={np.mean(total_rewards):.2f}  "
                  f"std={np.std(total_rewards):.2f}  "
                  f"min={np.min(total_rewards):.2f}  max={np.max(total_rewards):.2f}")
        steps = [a["num_steps"] for a in success_eps]
        print(f"  Success episode lengths: mean={np.mean(steps):.1f}  "
              f"min={np.min(steps)}  max={np.max(steps)}")


def save_plots(analyses: List[Dict[str, Any]], output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not available, skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for i, a in enumerate(analyses):
        label = "success" if a.get("success") else "fail"
        prefix = f"ep{i}_{label}"
        n_steps = a["num_steps"]
        t = np.arange(n_steps)

        if "dist_xy" in a and "dist_z" in a:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

            axes[0].plot(t[:len(a["dist_xy"])], a["dist_xy"], label="dist_xy")
            axes[0].axhline(y=0.02, color="r", linestyle="--", alpha=0.5, label="threshold=0.02")
            axes[0].set_ylabel("dist_xy (m)")
            axes[0].legend()
            axes[0].set_title(f"Episode {i} ({label}) - Distance to Target")

            dz = a["dist_z"]
            axes[1].plot(t[:len(dz)], dz, label="dist_z (wp_z - target_z)")
            axes[1].axhline(y=0.01, color="r", linestyle="--", alpha=0.5, label="+threshold")
            axes[1].axhline(y=-0.01, color="r", linestyle="--", alpha=0.5, label="-threshold")
            axes[1].axhline(y=0.0, color="g", linestyle="-", alpha=0.3)
            axes[1].set_ylabel("dist_z (m)")
            axes[1].legend()

            if "orient_diff" in a:
                axes[2].plot(t[:len(a["orient_diff"])], a["orient_diff"], label="orient_diff")
                axes[2].axhline(y=0.15, color="r", linestyle="--", alpha=0.5, label="threshold=0.15")
                axes[2].set_ylabel("orient_diff (rad)")
                axes[2].legend()

            axes[-1].set_xlabel("Step")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_distances.png"), dpi=100)
            plt.close()

        if "wp_speed" in a:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            sp = a["wp_speed"]
            ax.plot(t[:len(sp)], sp, label="wp_speed")
            ax.axhline(y=0.02, color="r", linestyle="--", alpha=0.5, label="still_thresh=0.02")
            ax.set_xlabel("Step")
            ax.set_ylabel("Workpiece speed (m/s)")
            ax.set_title(f"Episode {i} ({label}) - Workpiece Speed")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_speed.png"), dpi=100)
            plt.close()

        if "rewards" in a:
            fig, ax = plt.subplots(1, 1, figsize=(12, 4))
            ax.plot(t[:len(a["rewards"])], a["rewards"], label="reward")
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.set_title(f"Episode {i} ({label}) - Per-step Reward (total={a.get('total_reward', 0):.2f})")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_rewards.png"), dpi=100)
            plt.close()

        if "wp_pos" in a:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            wp = a["wp_pos"]
            for dim, name in enumerate(["x", "y", "z"]):
                axes[dim].plot(t[:len(wp)], wp[:, dim], label=f"wp_{name}")
                if dim < 2:
                    axes[dim].axhline(y=a["target_pos"][dim], color="r", linestyle="--",
                                      alpha=0.5, label=f"target_{name}")
                else:
                    axes[dim].axhline(y=a["target_pos"][dim], color="r", linestyle="--",
                                      alpha=0.5, label=f"target_z (init-0.05)")
                axes[dim].set_ylabel(f"{name} (m)")
                axes[dim].legend()
            axes[0].set_title(f"Episode {i} ({label}) - Workpiece Position")
            axes[-1].set_xlabel("Step")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_wp_pos.png"), dpi=100)
            plt.close()

        if "actions" in a:
            acts = a["actions"]
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

            for dim, name in enumerate(["x", "y", "z"]):
                axes[0].plot(t[:len(acts)], acts[:, dim], label=f"pos_{name}")
            axes[0].set_ylabel("Position (m)")
            axes[0].legend()
            axes[0].set_title(f"Episode {i} ({label}) - Actions")

            for dim, name in enumerate(["roll", "pitch", "yaw"]):
                axes[1].plot(t[:len(acts)], acts[:, 3 + dim], label=name)
            axes[1].set_ylabel("Rotation (rad)")
            axes[1].legend()

            if acts.shape[1] > 6:
                axes[2].plot(t[:len(acts)], acts[:, 6], label="gripper")
                axes[2].set_ylabel("Gripper cmd")
                axes[2].legend()

            axes[-1].set_xlabel("Step")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_actions.png"), dpi=100)
            plt.close()

        if "states" in a and a["states"].shape[1] >= 26:
            st = a["states"]
            fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

            ee_pos = st[:, 14:17]
            for dim, name in enumerate(["x", "y", "z"]):
                axes[0].plot(range(len(ee_pos)), ee_pos[:, dim], label=f"ee_{name}")
            axes[0].set_ylabel("EE position (m)")
            axes[0].legend()
            axes[0].set_title(f"Episode {i} ({label}) - Right-arm State")

            ee_rpy = st[:, 17:20]
            for dim, name in enumerate(["roll", "pitch", "yaw"]):
                axes[1].plot(range(len(ee_rpy)), ee_rpy[:, dim], label=f"ee_{name}")
            axes[1].set_ylabel("EE rotation (rad)")
            axes[1].legend()

            ee_linvel = st[:, 20:23]
            ee_speed = np.linalg.norm(ee_linvel, axis=-1)
            axes[2].plot(range(len(ee_speed)), ee_speed, label="EE linear speed")
            axes[2].set_ylabel("EE speed (m/s)")
            axes[2].legend()

            ee_angvel = st[:, 23:26]
            ee_angspeed = np.linalg.norm(ee_angvel, axis=-1)
            axes[3].plot(range(len(ee_angspeed)), ee_angspeed, label="EE angular speed")
            axes[3].set_ylabel("EE ang speed (rad/s)")
            axes[3].legend()

            axes[-1].set_xlabel("Step (observation index)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_ee_state.png"), dpi=100)
            plt.close()

    print(f"\nPlots saved to {output_dir}")


def save_csv_summary(analyses: List[Dict[str, Any]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "episode_summary.csv")
    with open(csv_path, "w") as f:
        header = [
            "episode", "filename", "success", "num_steps", "total_reward", "mean_reward",
            "final_dist_xy", "final_dist_z", "min_dist_xy", "min_abs_dist_z",
            "final_orient_diff", "max_wp_speed", "mean_action_smoothness",
        ]
        f.write(",".join(header) + "\n")
        for i, a in enumerate(analyses):
            row = [
                str(i),
                a["filename"],
                str(a.get("success", "")),
                str(a["num_steps"]),
                f"{a.get('total_reward', ''):.4f}" if "total_reward" in a else "",
                f"{a.get('mean_reward', ''):.4f}" if "mean_reward" in a else "",
                f"{a['dist_xy'][-1]:.4f}" if "dist_xy" in a else "",
                f"{a['dist_z'][-1]:.4f}" if "dist_z" in a else "",
                f"{a['dist_xy'].min():.4f}" if "dist_xy" in a else "",
                f"{np.abs(a['dist_z']).min():.4f}" if "dist_z" in a else "",
                f"{a['orient_diff'][-1]:.4f}" if "orient_diff" in a else "",
                f"{a['wp_speed'].max():.4f}" if "wp_speed" in a else "",
                f"{a['action_smoothness'].mean():.4f}" if "action_smoothness" in a else "",
            ]
            f.write(",".join(row) + "\n")
    print(f"CSV summary saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze collected demo episodes.")
    parser.add_argument("--data-dir", required=True, help="Directory with .pkl episode files.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for analysis outputs (plots, CSV). Default: <data-dir>/analysis")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "analysis")

    episodes = _load_episodes(args.data_dir)
    analyses = [analyze_episode(ep, i) for i, ep in enumerate(episodes)]

    print_summary(analyses)
    save_csv_summary(analyses, args.output_dir)

    if not args.no_plots:
        save_plots(analyses, args.output_dir)


if __name__ == "__main__":
    main()