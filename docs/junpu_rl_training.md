# Junpu RL Training Pipeline

Complete guide for the `junpu_place_workpiece` task: data collection, replay, and RL training with SAC + demo buffer.

---

## Overview

| Phase | Description | Key files |
|-------|-------------|-----------|
| 0 | Data collection (SpaceMouse) | `collect_sim_demos.py` |
| 1 | Demo replay & validation | `replay_sim_demos.py` |
| 2 | Convert demos → buffer | `convert_demos_to_buffer.py` |
| 3 | SAC RL training | `geniesim_junpu_sac.yaml` |
| 4 | PPO baseline | `geniesim_junpu_ppo_mlp.yaml` |
| 5 | Eval & video | `--only_eval True` |

---

## Prerequisites

```bash
# Working directory
cd /home/zy/code/rlinf_open_source/RLinf

# Environment variables
export GENIESIM_ROOT=/home/zy/code/rlinf_open_source
export EMBODIED_PATH=examples/embodiment

# Activate RLinf venv
source .venv/bin/activate
```

---

## Phase 0: Data Collection

See existing SpaceMouse collection script (already working).  
Demos land in `my_demos/` as `.pkl` files.

---

## Phase 1: Replay Demos

Verify that recorded demos replay correctly in simulation:

```bash
GENIESIM_ROOT=.. python examples/embodiment/replay_sim_demos.py \
    --config examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml \
    --demo-dir my_demos \
    --num-envs 1 \
    --num-rounds 1 \
    --step-hz 10
```

**Options:**
| Flag | Default | Meaning |
|------|---------|---------|
| `--num-envs` | 1 | Parallel simulation instances |
| `--num-rounds` | 1 | How many full passes through all demos |
| `--step-hz` | 10 | Control frequency; 0 = unlimited |

---

## Phase 2: Convert Demos to Replay Buffer

Convert demo `.pkl` files to the `TrajectoryReplayBuffer` checkpoint format
required by the SAC demo buffer:

```bash
python examples/embodiment/convert_demos_to_buffer.py \
    --demo-dir my_demos \
    --output-dir /tmp/junpu_demo_buffer
```

**Options:**
| Flag | Default | Meaning |
|------|---------|---------|
| `--demo-dir` | `my_demos` | Input directory with `.pkl` demos |
| `--output-dir` | `/tmp/junpu_demo_buffer` | Output checkpoint directory |
| `--include-images` | off | Also store `main_images` (large files) |
| `--placeholder-reward VALUE` | use recorded | Override all rewards with a constant |
| `--seed` | 42 | Seed written to metadata |

**Verify the buffer:**
```bash
python rlinf/data/replay_buffer.py \
    --load-path /tmp/junpu_demo_buffer \
    --num-chunks 32 \
    --cache-size 30 \
    --enable-cache
# Expected output: [sample] keys: ['actions', 'rewards', 'terminations', ...]
```

---

## Phase 3: SAC RL Training

### 3a. SAC with demo buffer (recommended)

```bash
GENIESIM_ROOT=.. \
python examples/embodiment/train_embodied_agent.py \
    --config-path examples/embodiment/config \
    --config-name geniesim_junpu_sac \
    ++algorithm.demo_buffer.load_path=/tmp/junpu_demo_buffer
```

This will:
- Collect online transitions from the simulation (1 env).
- Mix 50% online transitions + 50% demo transitions in each training batch.
- Run 8 SAC critic+actor update steps per rollout epoch.
- Log to TensorBoard at `../results/`.

### 3b. Pure online SAC (no demos)

```bash
GENIESIM_ROOT=.. \
python examples/embodiment/train_embodied_agent.py \
    --config-path examples/embodiment/config \
    --config-name geniesim_junpu_sac
```

### 3c. Resume from checkpoint

```bash
GENIESIM_ROOT=.. \
python examples/embodiment/train_embodied_agent.py \
    --config-path examples/embodiment/config \
    --config-name geniesim_junpu_sac \
    ++runner.resume_dir=../results/junpu_sac_mlp/checkpoints/global_step_500
```

### Key SAC hyperparameters (in `geniesim_junpu_sac.yaml`)

```yaml
algorithm:
  gamma: 0.99          # discount factor
  tau: 0.005           # target network soft update rate
  target_entropy: -7   # ≈ -action_dim; controls exploration/exploitation
  update_epoch: 8      # update steps per rollout epoch
  critic_actor_ratio: 2

  replay_buffer:
    cache_size: 200     # reduced for image observations
    min_buffer_size: 2  # start training after 2 rollout epochs

actor:
  model:
    model_type: cnn_policy   # ResNet10 encoder + MLP head
    image_size: [3, 128, 128]
    state_dim: 26            # right arm only
    action_dim: 7
    num_q_heads: 10
  enable_drq: true           # DRQ image augmentation

env:
  train:
    total_num_envs: 4
    max_steps_per_rollout_epoch: 100
```

### Monitoring training

```bash
tensorboard --logdir ../results/junpu_sac_mlp
```

Key metrics to watch:
- `q_data` — should increase over time
- `q_pi` — should track `q_data`
- `actor_loss` / `critic_loss` — should decrease and stabilize
- `entropy_temp/alpha` — auto-tunes exploration temperature

---

## Phase 4: PPO Baseline

```bash
GENIESIM_ROOT=.. \
python examples/embodiment/train_embodied_agent.py \
    --config-path examples/embodiment/config \
    --config-name geniesim_junpu_ppo_mlp
```

---

## Phase 5: Evaluation Only

```bash
GENIESIM_ROOT=.. \
python examples/embodiment/train_embodied_agent.py \
    --config-path examples/embodiment/config \
    --config-name geniesim_junpu_sac \
    ++runner.only_eval=True \
    ++runner.resume_dir=../results/junpu_sac_mlp/checkpoints/global_step_1000 \
    ++env.eval.video_cfg.save_video=True
```

Videos are saved to `../results/junpu_sac_mlp/video/eval/`.

---

## Reward Function

The reward in `JunpuPlaceWorkpieceEnv._compute_reward()` is a **dense multi-component reward**:

| Component | Weight | Description |
|-----------|--------|-------------|
| xy distance | −5.0 × ‖wp_xy − target_xy‖ | Penalize horizontal deviation |
| z distance (asymmetric) | −10.0 below / −5.0 above target | Penalize dropping harder |
| orientation | −2.0 × angle_diff | Keep workpiece upright |
| stillness bonus | +0.5 | When velocity < 0.02 m/s |
| success bonus | +5.0 | All criteria met for 15 consecutive steps (0.5s) |

Target position: 5cm below initial workpiece position. Success requires xy < 2cm, z < 1cm, angle < 0.15 rad, velocity < 0.02 m/s for 15 steps.

Workpiece position is obtained from `infos["body_poses"]["workpiece_r"]` (ground truth from sim via SHM).

---

## Environment Config Reference

**`examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml`**

```yaml
init_params:
  id: junpu_place_workpiece
  state_dim: 52         # 26 right-arm (joint pos+vel + EE) + 26 left-arm (unused, reserved)
  action_dim: 14        # [pos_l(3), rpy_l(3), pos_r(3), rpy_r(3), grip_l, grip_r]
  control_mode: ee      # EE-space; IK solved in MuJoCo
  cam_width: 480
  cam_height: 480
  wrist_cam_prim: /robot/Right_Camera   # right wrist camera for CNN policy
  total_num_envs: 4     # parallel simulation instances
```

**`examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml`**  
This is the simulation-side config (container/standalone use), with `init_qpos` for arm reset positions.

---

## File Reference

| File | Purpose |
|------|---------|
| `examples/embodiment/replay_sim_demos.py` | Replay recorded demos in simulation |
| `examples/embodiment/convert_demos_to_buffer.py` | Convert demos → SAC buffer format |
| `examples/embodiment/config/geniesim_junpu_sac.yaml` | SAC training config |
| `examples/embodiment/config/geniesim_junpu_ppo_mlp.yaml` | PPO baseline config |
| `examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml` | Unified env/sim config (`init_qpos`, host + `container_cfg`) |
| `rlinf/envs/geniesim/tasks/junpu_place_workpiece.py` | Task env with reward override |

---

## Troubleshooting

**`state_dim mismatch`** — the demo states have shape `(40,)`. Both env configs now have `state_dim: 40`. If you see this error, check that neither config file still says `28`.

**Demo buffer not ready** — ensure `min_demo_buffer_size: 1` and `load_path` is set correctly. The buffer needs at least 1 trajectory before mixing begins.

**Container startup timeout** — `startup_timeout_sec: 300` (5 min) in the env config. On first run, Isaac Sim needs ~3 min cold start. Subsequent runs are faster (`reuse_running: true`).

**`FileExistsError` on SHM** — fixed in `mujoco_ros_node.py`. If you see it, manually clean up: `python3 -c "from multiprocessing import shared_memory; shared_memory.SharedMemory(name='geniesim_ctrl_0', create=False).unlink()"`.
