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
  target_entropy: -14  # ≈ -action_dim; controls exploration/exploitation
  update_epoch: 8      # update steps per rollout epoch
  critic_actor_ratio: 2

  replay_buffer:
    min_buffer_size: 2  # start training after 2 episodes collected

  demo_buffer:
    load_path: null     # set via CLI: ++algorithm.demo_buffer.load_path=...
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

The current reward in `JunpuPlaceWorkpieceEnv.step()` is a **placeholder**:

```python
# rlinf/envs/geniesim/tasks/junpu_place_workpiece.py
def _placeholder_reward(self, obs, terminated):
    # -0.01 per timestep (time pressure)
    return torch.full((n,), -0.01, dtype=torch.float32)
```

**To replace it**, edit `rlinf/envs/geniesim/tasks/junpu_place_workpiece.py`:

```python
def _placeholder_reward(self, obs, terminated):
    states = obs["states"]   # [B, 40]
    # State layout:
    #   [0:7]   arm_l joint positions
    #   [7:14]  arm_r joint positions
    #   [14:21] arm_l joint velocities
    #   [21:28] arm_r joint velocities
    #   [28:31] left  EE position  (x, y, z)  in base_link frame
    #   [31:34] left  EE euler XYZ
    #   [34:37] right EE position  (x, y, z)  in base_link frame
    #   [37:40] right EE euler XYZ
    ee_r_pos = states[:, 34:37]  # right end-effector position

    # Example: distance to a target position (meters)
    target = torch.tensor([0.55, -0.12, 1.05], device=states.device)
    dist = torch.norm(ee_r_pos - target, dim=-1)
    reward = -dist                    # dense distance reward
    reward += 1.0 * terminated.float().squeeze(-1)  # +1 on success
    return reward
```

---

## Environment Config Reference

**`examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml`**

```yaml
init_params:
  id: junpu_place_workpiece
  state_dim: 40         # 28 joint (pos+vel) + 12 EE (pos+rpy × 2)
  action_dim: 14        # [pos_l(3), rpy_l(3), pos_r(3), rpy_r(3), grip_l, grip_r]
  control_mode: ee      # EE-space; IK solved in MuJoCo
  ee_body_l: gripper_l_base_link
  ee_body_r: gripper_r_base_link
  gripper_ctrl_l: 52
  gripper_ctrl_r: 53
  physics_hz: 1000
  render_hz: 30.0
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
