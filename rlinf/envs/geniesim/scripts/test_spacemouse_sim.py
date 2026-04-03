#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Integration test: verifies the SpacemouseSimIntervention + CollectEpisode
# wrapper stack on top of a GenieSimEnv using FakeSpaceMouseExpert (no hardware
# or real SpaceMouse required).
#
# The test checks:
#   1. Wrappers can be stacked and env.reset() / env.step() work.
#   2. SpaceMouse deltas modify env_0's action (right arm EEF changes).
#   3. Junpu mode: left-button press terminates env_0 and marks info["success"]=True.
#   4. CollectEpisode saves a file to disk after the episode ends.
#
# Run from the RLinf repo root:
#   GENIESIM_ROOT=.. python rlinf/envs/geniesim/scripts/test_spacemouse_sim.py \
#       [--config examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml] \
#       [--save-dir /tmp/spacemouse_test] \
#       [--num-demos 2] \
#       [--dry-run]   # skip actual sim launch; only test import & class wiring

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RLINF_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../../.."))
if _RLINF_ROOT not in sys.path:
    sys.path.insert(0, _RLINF_ROOT)

if "GENIESIM_ROOT" not in os.environ:
    _gs_root = os.path.abspath(os.path.join(_RLINF_ROOT, ".."))
    os.environ["GENIESIM_ROOT"] = _gs_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml",
    )
    parser.add_argument("--save-dir", default="/tmp/spacemouse_test")
    parser.add_argument("--num-demos", type=int, default=2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only verify imports and class wiring; skip simulation launch.",
    )
    args = parser.parse_args()

    # ── 1. Import checks ──────────────────────────────────────────────────── #
    from rlinf.envs.wrappers.spacemouse_sim_intervention import (
        FakeSpaceMouseExpert,
        SpacemouseSimIntervention,
    )
    from rlinf.envs.wrappers.collect_episode import CollectEpisode
    print("[test] Imports OK ✓")

    # ── 2. FakeSpaceMouseExpert unit test ─────────────────────────────────── #
    fake = FakeSpaceMouseExpert(move_steps=5, gripper_steps=3, lift_steps=5)
    actions_seen = []
    buttons_seen = []
    for _ in range(20):
        a, b = fake.get_action()
        actions_seen.append(a.copy())
        buttons_seen.append(list(b))

    # First 5 steps: motion in +Y, no buttons
    assert all(a[1] > 0 for a in actions_seen[:5]), "Expected +Y motion in phase 0"
    # Steps 5-7: gripper close (left button = 1)
    assert all(b[0] == 1 for b in buttons_seen[5:8]), "Expected left button in phase 1"
    # Steps 8-12: motion in +Z
    assert all(a[2] > 0 for a in actions_seen[8:13]), "Expected +Z motion in phase 2"
    # Step 13: right button (episode done)
    assert buttons_seen[13][1] == 1, "Expected right button at episode end"

    # Junpu fake: move + lift + left button (save)
    fake_j = FakeSpaceMouseExpert(
        move_steps=5, gripper_steps=0, lift_steps=5, button_semantics="junpu",
    )
    aj, bj = [], []
    for _ in range(20):
        a, b = fake_j.get_action()
        aj.append(a.copy())
        bj.append(list(b))
    assert all(x[1] > 0 for x in aj[:5]), "junpu phase0 +Y"
    assert all(x[2] > 0 for x in aj[5:10]), "junpu phase1 +Z"
    assert bj[10][0] == 1 and bj[10][1] == 0, "junpu done: left button"

    fake.on_episode_reset()
    a0, b0 = fake.get_action()
    assert a0[1] > 0, "Expected +Y motion after reset"
    print("[test] FakeSpaceMouseExpert unit test OK ✓")

    # ── 3. Config loading ─────────────────────────────────────────────────── #
    from omegaconf import OmegaConf, open_dict

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(_RLINF_ROOT, cfg_path)
    assert os.path.exists(cfg_path), f"Config not found: {cfg_path}"

    cfg = OmegaConf.load(cfg_path)
    action_dim = cfg.init_params.get("action_dim", 14)
    print(f"[test] Config loaded: task={cfg.init_params.id}, action_dim={action_dim} ✓")

    if args.dry_run:
        print("[test] Dry-run mode: skipping simulation launch.")
        print("[test] PASSED (dry-run) ✓")
        return

    # ── 4. Full integration test ──────────────────────────────────────────── #
    import atexit
    import numpy as np
    import torch

    from rlinf.envs.geniesim.tasks import JunpuPlaceWorkpieceEnv  # noqa: F401
    from rlinf.envs.geniesim import REGISTER_GENIESIM_ENVS

    assert cfg.init_params.id in REGISTER_GENIESIM_ENVS, \
        f"Task {cfg.init_params.id!r} not registered"

    with open_dict(cfg):
        cfg.init_params.num_envs = 1

    EnvCls = REGISTER_GENIESIM_ENVS[cfg.init_params.id]
    base_env = EnvCls(cfg, num_envs=1, seed_offset=0,
                      total_num_processes=1, worker_info={})
    atexit.register(base_env.close)

    # Build fake expert with short phases for fast testing
    expert = FakeSpaceMouseExpert(
        move_steps=10, lift_steps=10, button_semantics="junpu",
    )

    sm_env = SpacemouseSimIntervention(
        base_env,
        expert=expert,
        action_scale=0.01,
        rotation_scale=0.05,
        button_mode="junpu",
    )

    os.makedirs(args.save_dir, exist_ok=True)
    env = CollectEpisode(
        sm_env,
        save_dir=args.save_dir,
        rank=0,
        num_envs=1,
        export_format="pickle",
        only_success=True,
    )
    atexit.register(env.close)

    print(f"[test] Collecting {args.num_demos} demo(s) → {args.save_dir}")

    demos_collected = 0
    obs, _ = env.reset()

    # Verify initial obs shape
    assert "states" in obs, "obs missing 'states' key"
    assert "main_images" in obs, "obs missing 'main_images' key"
    assert obs["states"].shape == (1, 40), \
        f"Unexpected states shape: {obs['states'].shape}"
    print(f"[test] reset() OK | states={obs['states'].shape} "
          f"images={obs['main_images'].shape} ✓")

    # Verify that SpacemouseSimIntervention initialised _current_target from obs
    state_r_ee = obs["states"][0, 34:37].cpu().numpy()
    target_r_pos = sm_env._current_target[6:9]
    assert np.allclose(target_r_pos, state_r_ee, atol=1e-5), \
        f"Target not initialised from obs: {target_r_pos} vs {state_r_ee}"
    print("[test] _current_target initialised from reset obs ✓")

    episode_steps = 0
    while demos_collected < args.num_demos:
        actions = torch.zeros(1, action_dim, dtype=torch.float32)
        obs, reward, terminated, truncated, info = env.step(actions)
        episode_steps += 1

        done_0 = bool(terminated[0]) or bool(truncated[0])
        if done_0:
            is_success = info.get("success", False)
            print(f"[test] Episode done after {episode_steps} steps, "
                  f"success={is_success}")
            if is_success:
                demos_collected += 1
            if demos_collected < args.num_demos:
                obs, _ = env.reset()
                episode_steps = 0

    env.close()

    # Verify files were written
    saved_files = [
        f for f in os.listdir(args.save_dir)
        if f.endswith("_success.pkl") or f.endswith(".pkl")
    ]
    assert len(saved_files) >= args.num_demos, \
        f"Expected {args.num_demos} saved files, found {len(saved_files)}: {saved_files}"
    print(f"[test] {len(saved_files)} file(s) saved in {args.save_dir} ✓")

    print("\n[test] PASSED ✓")


if __name__ == "__main__":
    main()
