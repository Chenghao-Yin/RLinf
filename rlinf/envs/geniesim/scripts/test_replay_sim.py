#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Integration test: verifies the replay_sim_demos script logic using demo data
# collected by test_spacemouse_sim.py (or collect_sim_data.py --fake-spacemouse).
#
# The test checks:
#   1. Demo loading (_load_demos) finds .pkl files and parses them correctly.
#   2. _get_action_for_step returns intervene_action when available, falls back
#      to actions[step], and returns None when step is out of range.
#   3. Full replay: all n demos are replayed across m envs, individual-env
#      reset is used when an env finishes early.
#   4. The replay loop terminates (no hang) and reports correct counts.
#
# Run from the RLinf repo root:
#   GENIESIM_ROOT=.. python rlinf/envs/geniesim/scripts/test_replay_sim.py \
#       [--config examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml] \
#       [--demo-dir /tmp/spacemouse_test] \
#       [--num-envs 1] \
#       [--dry-run]   # skip actual sim launch; only test logic

import argparse
import os
import pickle
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RLINF_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../../.."))
if _RLINF_ROOT not in sys.path:
    sys.path.insert(0, _RLINF_ROOT)

if "GENIESIM_ROOT" not in os.environ:
    _gs_root = os.path.abspath(os.path.join(_RLINF_ROOT, ".."))
    os.environ["GENIESIM_ROOT"] = _gs_root


def _make_fake_demo(episode_id: int, num_steps: int = 10) -> dict:
    """Create a minimal fake episode dict that mimics CollectEpisode pickle format."""
    import numpy as np
    import torch

    actions = [torch.zeros(14, dtype=torch.float32) for _ in range(num_steps)]
    # Simulate SpaceMouse: put intervene_action in every info
    base_pos = np.array([0.1, -0.93, 1.34, 1.57, -1.57, 0.0,
                          0.1, -0.93, 1.34, 1.57, -1.57, 0.0,
                          1.0, 1.0], dtype=np.float32)
    infos = []
    for step in range(num_steps):
        target = base_pos.copy()
        target[7] += step * 0.01  # simulate +Y motion on right arm
        infos.append({
            "intervene_action": torch.tensor(target),
            "intervene_flag": torch.ones(1, dtype=torch.bool),
            "episode": {"success_once": torch.tensor([False])},
        })

    return {
        "rank": 0,
        "env_idx": 0,
        "episode_id": episode_id,
        "success": True,
        "observations": [],
        "actions": actions,
        "rewards": [torch.zeros(1) for _ in range(num_steps)],
        "terminated": [torch.zeros(1, dtype=torch.bool) for _ in range(num_steps)],
        "truncated": [torch.zeros(1, dtype=torch.bool) for _ in range(num_steps)],
        "infos": infos,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml",
    )
    parser.add_argument(
        "--demo-dir",
        default="/tmp/spacemouse_test",
        help="Directory with recorded .pkl demo files (used in full-sim mode).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel envs for full-sim replay test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only verify imports, demo loading, and action extraction logic; "
             "skip simulation launch.",
    )
    args = parser.parse_args()

    # ── 1. Import checks ──────────────────────────────────────────────────── #
    _examples_dir = os.path.join(_RLINF_ROOT, "examples", "embodiment")
    if _examples_dir not in sys.path:
        sys.path.insert(0, _examples_dir)

    from replay_sim_demos import _load_demos, _get_action_for_step
    print("[test] Imports OK ✓")

    # ── 2. Demo loading unit test ─────────────────────────────────────────── #
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write 3 fake demos with different lengths
        lengths = [5, 8, 12]
        for i, L in enumerate(lengths):
            ep = _make_fake_demo(episode_id=i, num_steps=L)
            path = os.path.join(tmpdir, f"rank_0_env_0_episode_{i}_success.pkl")
            with open(path, "wb") as f:
                pickle.dump(ep, f)

        loaded = _load_demos(tmpdir)
        assert len(loaded) == 3, f"Expected 3 demos, got {len(loaded)}"
        assert [len(d["actions"]) for d in loaded] == lengths, \
            f"Action lengths mismatch: {[len(d['actions']) for d in loaded]}"

    print("[test] _load_demos unit test OK ✓")

    # ── 3. _get_action_for_step unit test ─────────────────────────────────── #
    fake_demo = _make_fake_demo(episode_id=99, num_steps=5)

    # Step within range: should return intervene_action (not zeros)
    act0 = _get_action_for_step(fake_demo, 0)
    assert act0 is not None, "Expected action at step 0"
    assert isinstance(act0, torch.Tensor), f"Expected Tensor, got {type(act0)}"
    assert act0.shape == (14,), f"Unexpected shape: {act0.shape}"
    # intervene_action has non-zero values; policy action is all zeros
    assert act0.abs().sum() > 0.1, \
        f"Expected intervene_action (non-zero) at step 0, got {act0}"

    # Step out of range: should return None
    act_oor = _get_action_for_step(fake_demo, 100)
    assert act_oor is None, f"Expected None for out-of-range step, got {act_oor}"

    # Demo with no intervene_action: should fall back to actions[step]
    plain_demo = {
        "actions": [torch.zeros(14), torch.ones(14)],
        "infos": [{}, {"other_key": 42}],
    }
    act_plain_0 = _get_action_for_step(plain_demo, 0)
    assert act_plain_0 is not None
    assert act_plain_0.abs().sum() == 0.0, "Expected zeros (policy action)"

    act_plain_1 = _get_action_for_step(plain_demo, 1)
    assert act_plain_1 is not None
    assert act_plain_1.abs().sum() == 14.0, "Expected ones (policy action)"

    print("[test] _get_action_for_step unit test OK ✓")

    # ── 4. Config loading ─────────────────────────────────────────────────── #
    from omegaconf import OmegaConf

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

    # ── 5. Full integration test ──────────────────────────────────────────── #
    import atexit
    import numpy as np
    from omegaconf import open_dict

    # Verify demo_dir has files; if not, fail clearly
    import glob
    demo_files = sorted(glob.glob(os.path.join(args.demo_dir, "*.pkl")))
    if not demo_files:
        sys.exit(
            f"[test] No .pkl files found in {args.demo_dir}. "
            f"Run test_spacemouse_sim.py first to collect demos."
        )
    print(f"[test] Found {len(demo_files)} demo file(s) in {args.demo_dir}")

    from rlinf.envs.geniesim.tasks import JunpuPlaceWorkpieceEnv  # noqa: F401
    from rlinf.envs.geniesim import REGISTER_GENIESIM_ENVS

    assert cfg.init_params.id in REGISTER_GENIESIM_ENVS, \
        f"Task {cfg.init_params.id!r} not registered"

    with open_dict(cfg):
        cfg.init_params.num_envs = args.num_envs

    EnvCls = REGISTER_GENIESIM_ENVS[cfg.init_params.id]
    env = EnvCls(cfg, num_envs=args.num_envs, seed_offset=0,
                 total_num_processes=1, worker_info={})
    atexit.register(env.close)

    n = len(demo_files)
    m = args.num_envs
    demos = _load_demos(args.demo_dir)

    # ── Replay logic (mirrors replay_sim_demos.main but inlined for assertions) ──
    from replay_sim_demos import _get_action_for_step as gaf

    queue_pos = min(m, n)
    current_demo = [None] * m
    current_step = [0] * m
    last_action = [None] * m

    for i in range(min(m, n)):
        current_demo[i] = demos[i]
        current_step[i] = 0

    completed_demos = 0

    env.reset()
    print(f"[test] Replaying {n} demo(s) with {m} env(s)...")

    steps_taken = 0
    max_steps = sum(len(d["actions"]) for d in demos) * 3 + 100  # safety cap

    while any(d is not None for d in current_demo):
        import torch as _torch
        actions = _torch.zeros(m, action_dim, dtype=_torch.float32)

        for i in range(m):
            if current_demo[i] is None:
                if last_action[i] is not None:
                    actions[i] = last_action[i]
                continue
            act = gaf(current_demo[i], current_step[i])
            if act is not None:
                actions[i] = act
                last_action[i] = act.clone()

        env.step(actions, auto_reset=False)
        steps_taken += 1

        if steps_taken > max_steps:
            raise RuntimeError(
                f"[test] Replay did not terminate after {max_steps} steps — "
                "possible infinite loop!"
            )

        for i in range(m):
            if current_demo[i] is None:
                continue
            current_step[i] += 1
            if current_step[i] >= len(current_demo[i]["actions"]):
                completed_demos += 1
                ep_id = current_demo[i].get("episode_id", "?")
                print(f"[test] env_{i} finished demo episode_id={ep_id} "
                      f"({completed_demos}/{n})")

                if queue_pos < n:
                    current_demo[i] = demos[queue_pos]
                    current_step[i] = 0
                    queue_pos += 1
                    env.reset(env_ids=np.array([i]))
                    print(f"[test] env_{i} individual reset OK ✓")
                else:
                    current_demo[i] = None

    assert completed_demos == n, \
        f"Expected {n} demos completed, got {completed_demos}"
    print(f"[test] All {n} demo(s) replayed in {steps_taken} steps ✓")

    env.close()
    print("\n[test] PASSED ✓")


if __name__ == "__main__":
    main()
