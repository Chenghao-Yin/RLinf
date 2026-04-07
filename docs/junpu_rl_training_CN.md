# Junpu 机器人 RL 训练用户手册

> 面向不熟悉强化学习的用户。本手册介绍两种训练模式（SAC 和 PPO）的原理差异、适用场景和完整操作步骤。

---

## 目录

1. [强化学习基础概念](#1-强化学习基础概念)
2. [两种训练算法：SAC 与 PPO](#2-两种训练算法sac-与-ppo)
3. [如何选择算法](#3-如何选择算法)
4. [环境准备](#4-环境准备)
5. [SAC 训练完整流程](#5-sac-训练完整流程)
6. [PPO 训练完整流程](#6-ppo-训练完整流程)
7. [监控训练进度](#7-监控训练进度)
8. [自定义 Reward 函数](#8-自定义-reward-函数)
9. [恢复训练与评测](#9-恢复训练与评测)
10. [常见问题与排错](#10-常见问题与排错)
11. [参数速查表](#11-参数速查表)

---

## 1. 强化学习基础概念

### 1.1 RL 在做什么？

强化学习（Reinforcement Learning，RL）的目标是让机器人**通过不断尝试来学习完成任务**。整个过程类比于训练小狗：

```
机器人（策略）
    │
    ├── 观察当前状态（关节角度、末端位置）
    │
    ├── 执行动作（移动手臂、开合夹爪）
    │
    ├── 收到奖励信号（任务做得好 → 正奖励，做得差 → 负奖励）
    │
    └── 根据奖励更新策略（下次遇到相同情况，做更好的选择）
```

反复循环几千次后，机器人逐渐学会如何最大化累计奖励——即完成任务。

### 1.2 Junpu 任务的关键元素

| 概念 | 对应内容 |
|------|---------|
| **状态（obs）** | 26 维向量（右臂）：7 关节角度 + 7 关节速度 + 6 EE 位置/姿态 + 6 EE 速度；外加 128×128×3 右手腕部相机 RGB 图像 |
| **动作（action）** | 7 维向量：右手末端目标位置(3) + 姿态(3) + 夹爪(1) |
| **奖励（reward）** | 密集奖励：xy 距离惩罚 + 非对称 z 惩罚 + 姿态偏差惩罚 + 静止奖励 + 成功奖励 |
| **回合（episode）** | 一次完整尝试，最多 300 步（约 10 秒仿真时间，30Hz 控制频率） |
| **策略（policy）** | CNN（ResNet10 图像编码器）+ MLP 头，输入图像+状态，输出动作 |

### 1.3 什么是"示范数据"？

示范数据是人类用 SpaceMouse 遥操作机器人完成任务时录下的轨迹（存储在 `my_demos/` 中的 `.pkl` 文件）。每条轨迹包含：
- 每步的状态观测
- 每步实际执行的动作
- 每步的奖励和是否结束

SAC 算法可以利用这些示范数据加速学习，PPO 则不使用。

---

## 2. 两种训练算法：SAC 与 PPO

### 2.1 PPO（近端策略优化）

**直觉**：PPO 就像一个纯粹靠自己摸索的学生——完全从零开始，只依靠自己的亲身尝试。

**工作方式**：
```
① 让机器人跑完整个回合（300 步）
② 计算这个回合里每一步"做得有多好"（GAE 优势估计）
③ 用这些数据更新策略，但更新幅度不能太大（"近端"约束）
④ 丢掉这批数据，重新收集新的经验
⑤ 循环
```

**核心特点**：
- ✅ 简单稳定，参数少，容易调
- ✅ 不需要任何先验数据
- ✅ 对噪声奖励比较鲁棒
- ❌ 样本效率低：每条轨迹只用一次就丢掉
- ❌ 稀疏奖励下探索困难（机器人很难随机碰对动作）
- ❌ 需要较长训练时间（通常需要数百到数千个回合）

**PPO 配置**（`geniesim_junpu_ppo_mlp.yaml`）关键参数：
```yaml
algorithm:
  gamma: 0.99          # 未来奖励折扣：0.99 表示重视长期回报
  gae_lambda: 0.95     # 优势估计的平滑系数
  clip_ratio_high: 0.2 # 策略更新幅度上限（±20%）
  update_epoch: 8      # 每批数据重复利用 8 次

env:
  train:
    max_steps_per_rollout_epoch: 300  # 每轮收集 300 步（一整个回合）
```

---

### 2.2 SAC（软演员-评论家）

**直觉**：SAC 就像一个有历史案例库的学生——除了自己的亲身经历，还能反复从记忆库（包括人类示范）中提取经验学习。

**工作方式**：
```
① 让机器人走几步（不需要跑完整回合，默认 4 步）
② 把这些经验存入"回放缓冲区"（记忆库）
③ 从记忆库中随机采样（50% 自己的经验 + 50% 人类示范）
④ 同时训练两个网络：
   - Critic（评论家）：评估"某个状态下做某个动作有多好"→ Q 值
   - Actor（演员）：根据 Q 值改善动作选择
⑤ 循环
```

**SAC 的"软"意味着什么**：SAC 在最大化累计奖励的同时，还最大化策略的**熵**（随机性）。这确保机器人持续探索，不会过早收敛到一个局部最优解。

**核心特点**：
- ✅ 样本效率高：每条经验可以反复使用
- ✅ 人类示范数据大幅加速早期探索
- ✅ 持续探索，不容易陷入局部最优
- ✅ 适合连续动作空间（机械臂任务非常合适）
- ❌ 超参数稍多，调参复杂度略高
- ❌ 需要提前准备并转换示范数据（一次性工作）

**SAC 配置**（`geniesim_junpu_sac.yaml`）关键参数：
```yaml
algorithm:
  gamma: 0.99          # 未来奖励折扣
  tau: 0.005           # 目标网络软更新速率（每步小幅同步）
  target_entropy: -7   # 目标熵值 = -action_dim，控制探索程度
  update_epoch: 8      # 每轮更新 8 次 Critic+Actor
  critic_actor_ratio: 2 # 每更新 1 次 Actor，先更新 2 次 Critic

  replay_buffer:
    cache_size: 200     # 记忆库最多存 200 条轨迹（图像数据较大）

env:
  train:
    total_num_envs: 4                    # 4 个并行仿真环境
    max_steps_per_rollout_epoch: 100     # 每轮收集 100 步/环境

actor:
  model:
    model_type: cnn_policy               # CNN 策略（ResNet10 + MLP）
    image_size: [3, 128, 128]            # 右手腕部相机 128×128 RGB
    state_dim: 26                        # 右臂状态
    action_dim: 7                        # 右臂动作
  enable_drq: true                       # DRQ 图像数据增强
```

---

### 2.3 核心差异对比

| 维度 | PPO | SAC |
|------|-----|-----|
| **类型** | On-policy（只用当前策略的新数据） | Off-policy（可以复用历史数据） |
| **需要示范数据** | 否 | 可选（有数据时强烈推荐） |
| **每轮收集步数** | 300 步（整回合） | 100 步 × 4 环境 = 400 转移 |
| **数据复用** | 不复用（每批数据用完即丢） | 大量复用（回放缓冲区） |
| **探索机制** | ε-greedy 式随机动作 + 温度采样 | 熵正则化（自动维持探索） |
| **适合阶段** | 任何阶段（尤其是没有示范数据时） | 有示范数据时效率更高 |
| **调参难度** | ⭐⭐（简单） | ⭐⭐⭐（适中） |
| **样本效率** | 低（需要更多仿真交互） | 高（利用历史数据） |
| **配置文件** | `geniesim_junpu_ppo_mlp.yaml` | `geniesim_junpu_sac.yaml` |

---

## 3. 如何选择算法

```
你有人类示范数据（my_demos/*.pkl）吗？
│
├── 有（≥5 条成功演示）
│     └── 推荐 SAC + demo buffer
│           原因：示范数据帮助机器人快速了解"成功是什么样的"
│
└── 没有
      ├── 任务奖励比较密集（每步都有反馈）
      │     └── SAC 或 PPO 均可
      │
      └── 任务奖励稀疏（只有成功才有奖励）
            └── 推荐先采集一些示范数据再用 SAC
                 （PPO 在稀疏奖励下探索非常困难）
```

**快速建议**：

- **第一次上手，想快速跑通**：用 PPO，配置少，命令简单
- **认真训练，有示范数据**：用 SAC + demo buffer，效果更好
- **没有示范数据但想用 SAC**：也可以，只是早期探索会慢

---

## 4. 环境准备

### 4.1 前置要求

- Docker 已安装且可访问 GPU（`nvidia-docker2`）
- 镜像 `geniesim-rlinf:latest` 已构建
- RLinf `.venv` 已创建（`RLinf/.venv/`）

### 4.2 环境变量

每次打开新终端，执行：

```bash
cd /home/zy/code/rlinf_open_source/RLinf

export GENIESIM_ROOT=/home/zy/code/rlinf_open_source
export EMBODIED_PATH=$(pwd)/examples/embodiment
```

或者直接在命令前加前缀（本手册均采用此方式）：

```bash
GENIESIM_ROOT=.. EMBODIED_PATH=$(pwd)/examples/embodiment .venv/bin/python ...
```

### 4.3 验证环境（不需要 Docker）

```bash
.venv/bin/python -c "
from rlinf.envs.geniesim.tasks import JunpuPlaceWorkpieceEnv
print('环境模块加载成功')
"
```

期望输出：`环境模块加载成功`

---

## 5. SAC 训练完整流程

SAC 有两种子模式：
- **SAC + demo buffer**（推荐）：利用人类示范数据加速训练
- **纯 SAC**：无示范数据，完全从零开始

### 步骤 1：准备示范数据（可选但推荐）

如果已有 `my_demos/` 目录下的演示文件，执行以下命令将其转换为 SAC 可用的格式：

```bash
cd /home/zy/code/rlinf_open_source/RLinf

EMBODIED_PATH=$(pwd)/examples/embodiment \
.venv/bin/python examples/embodiment/convert_demos_to_buffer.py \
    --demo-dir my_demos \
    --output-dir /tmp/junpu_demo_buffer
```

**期望输出**：
```
转换完成：30 条轨迹，共 1338 步
输出目录：/tmp/junpu_demo_buffer
```

> **转换只需做一次**。只要 `my_demos/` 不变，缓冲区可以反复使用。

### 步骤 2：启动仿真容器

SAC 训练启动时会**自动管理容器**。首次运行时，Isaac Sim 冷启动需要约 3 分钟，请耐心等待。

如果容器已在运行（`reuse_running: true`），则会直接复用，启动约 10 秒。

### 步骤 3：启动 SAC 训练

**有示范数据**（推荐）：

```bash
cd /home/zy/code/rlinf_open_source/RLinf

GENIESIM_ROOT=.. \
EMBODIED_PATH=$(pwd)/examples/embodiment \
.venv/bin/python examples/embodiment/train_embodied_agent.py \
    --config-path config \
    --config-name geniesim_junpu_sac \
    ++algorithm.demo_buffer.load_path=/tmp/junpu_demo_buffer
```

**无示范数据**：

```bash
cd /home/zy/code/rlinf_open_source/RLinf

GENIESIM_ROOT=.. \
EMBODIED_PATH=$(pwd)/examples/embodiment \
.venv/bin/python examples/embodiment/train_embodied_agent.py \
    --config-path config \
    --config-name geniesim_junpu_sac
```

### 步骤 4：确认训练正常启动

训练开始后，每隔几秒会打印一次指标面板。看到以下输出说明一切正常：

```
[GenieSimShmClient] Initialised | num_envs=4 state_dim=52 action_dim=14
sac/critic_loss=x.xxx             ← Critic 正在训练
sac/actor_loss=x.xxx              ← Actor 正在训练
sac/alpha=0.01                    ← 熵系数（自动调节）
actor/q_value_0=-x.xxx            ← Q 值估计（10 个 Q-head）
```

### SAC 训练内部流程

每个训练 Epoch 包含以下步骤：

```
[Epoch N]
  ① 环境交互（100 步 × 4 并行环境 = 400 转移）
       机器人执行动作 → 收集 (s, a, r, s') 转移
       存入回放缓冲区（在线数据）
  ② 采样训练批次（从回放缓冲区随机采样 128 个样本）
  ③ Critic 更新（8 次中每次都更新）
       用贝尔曼方程计算目标 Q 值（10 个 Q-head，取最小值）
       最小化 Q 值预测误差（均方误差）
  ④ Actor 更新（每 2 次 Critic 更新后更新 1 次）
       最大化期望 Q 值 + 熵
  ⑤ Alpha（温度）更新
       自动调节，使策略熵接近目标值（-7）
  ⑥ 目标网络软更新
       τ=0.005，缓慢向主网络靠拢（稳定训练）
  ⑦ DRQ 图像增强
       训练时对图像应用随机裁剪（pad=4），提高视觉泛化
  [每 100 Epoch] Eval：完整跑 300 步，记录成功率
```

---

## 6. PPO 训练完整流程

PPO 不需要示范数据，一条命令搞定：

```bash
cd /home/zy/code/rlinf_open_source/RLinf

GENIESIM_ROOT=.. \
EMBODIED_PATH=$(pwd)/examples/embodiment \
.venv/bin/python examples/embodiment/train_embodied_agent.py \
    --config-path config \
    --config-name geniesim_junpu_ppo_mlp
```

### PPO 训练内部流程

每个训练 Epoch 包含以下步骤：

```
[Epoch N]
  ① 环境交互（300 步 = 1 个完整回合）
       机器人从头到尾走完一个完整 episode
       记录每步的 (状态, 动作, 奖励, 是否结束)
  ② 计算优势（GAE）
       从后往前推算每一步"比平均水平好多少"
       γ=0.99 折扣未来奖励，λ=0.95 平滑估计
  ③ 策略更新（8 次）
       同一批数据重复使用 8 次（提高利用率）
       clip_ratio=0.2：每次更新幅度不超过 20%（稳定性保证）
  ④ 值函数更新
       Actor 和 Critic 一起训练
       Critic 预测"这个状态能获得多少总奖励"
  [丢弃本批数据，重新收集]
  [每 50 Epoch] Eval：完整跑 300 步，记录成功率
```

---

## 7. 监控训练进度

### 7.1 启动 TensorBoard

```bash
# 新开一个终端
tensorboard --logdir /home/zy/code/rlinf_open_source/results
```

然后在浏览器中访问 `http://localhost:6006`。

### 7.2 关键指标解读

#### SAC 指标

| 指标 | 含义 | 健康趋势 |
|------|------|---------|
| `sac/critic_loss` | Critic 预测误差 | 快速下降后稳定在低值 |
| `sac/actor_loss` | Actor 优化目标（负 Q 值） | 通常为负数，逐渐变小（更负） |
| `sac/alpha` | 熵温度系数 | 自动调整，通常在 0.01~0.1 稳定 |
| `actor/q_value_0..9` | Q 值估计（10 个 Q-head） | 随训练应逐渐反映真实回报 |
| `actor/q_pi` | 策略期望 Q 值 | 应接近 `q_value` |
| `actor/entropy` | 策略熵 | 维持在合理范围，过低说明过度收敛 |
| `replay_buffer/total_samples` | 回放缓冲区样本数 | 持续增加 |

#### PPO 指标

| 指标 | 含义 | 健康趋势 |
|------|------|---------|
| `actor/grad_norm` | 梯度范数 | 在 1~10 范围内稳定 |
| `actor/entropy` | 策略熵 | 初期高，随训练缓慢降低 |
| `actor/lr` | 学习率 | 固定值（3e-4） |
| `env/episode_len` | 回合平均长度 | 任务成功后应减少 |

#### 通用指标（Eval 时）

| 指标 | 含义 | 目标 |
|------|------|------|
| `eval/success_once` | 评测回合成功率 | 越高越好（目标 >0.8） |
| `eval/return` | 评测累计奖励 | 越高越好 |
| `eval/episode_len` | 评测回合步数 | 成功任务应 <300 |

### 7.3 如何判断训练是否正常？

**正常**：
- 前几个 Epoch：指标波动较大，属正常现象
- `critic_loss` 持续下降
- `q_value` 在训练数百 Epoch 后开始上升
- Eval 中偶尔出现成功（`success_once > 0`）

**异常**：
- `critic_loss` 一直不降甚至上升 → 检查 reward 函数是否返回 `nan`
- `alpha` 迅速变得很大（>1.0）→ 奖励尺度可能过大，考虑缩小
- `grad_norm` 持续很大（>100）→ 降低学习率或检查 reward 是否有异常值

---

## 8. Reward 函数设计

当前 reward 已实现为密集奖励函数，位于 `JunpuPlaceWorkpieceEnv._compute_reward()`。

### 8.1 当前 Reward 设计

任务目标：将工件放置在初始位置下方 5cm 处，保持竖直姿态，静止 0.5 秒。

| 奖励分量 | 公式 | 权重 |
|---------|------|------|
| **xy 距离惩罚** | −‖wp_xy − target_xy‖ | ×5.0 |
| **z 距离惩罚（非对称）** | 低于目标 ×10.0，高于目标 ×5.0 | — |
| **姿态偏差惩罚** | −angle_diff（与竖直方向夹角） | ×2.0 |
| **静止奖励** | 工件速度 < 0.02 m/s 时 | +0.5 |
| **成功奖励** | 所有条件满足并持续 15 步（0.5 秒） | +5.0 |

成功判定条件（同时满足 15 个连续步）：
- xy 偏差 < 2cm
- z 偏差 < 1cm
- 姿态偏差 < 0.15 rad
- 速度 < 0.02 m/s

### 8.2 修改位置

编辑文件：`RLinf/rlinf/envs/geniesim/tasks/junpu_place_workpiece.py`

工件位置通过 `infos["body_poses"]` 中的 `workpiece_r` 获取世界坐标（由 sim 侧通过 SHM 传递）。

---

## 9. 恢复训练与评测

### 9.1 查看已有检查点

训练会定期保存检查点（SAC 每 500 Epoch，PPO 需手动配置）：

```bash
ls /home/zy/code/rlinf_open_source/results/junpu_sac_mlp/checkpoints/
# global_step_500/
# global_step_1000/
# ...
```

### 9.2 从检查点恢复训练

```bash
GENIESIM_ROOT=.. \
EMBODIED_PATH=$(pwd)/examples/embodiment \
.venv/bin/python examples/embodiment/train_embodied_agent.py \
    --config-path config \
    --config-name geniesim_junpu_sac \
    ++algorithm.demo_buffer.load_path=/tmp/junpu_demo_buffer \
    ++runner.resume_dir=../results/junpu_sac_mlp/checkpoints/global_step_500
```

### 9.3 仅评测（不训练）

```bash
GENIESIM_ROOT=.. \
EMBODIED_PATH=$(pwd)/examples/embodiment \
.venv/bin/python examples/embodiment/train_embodied_agent.py \
    --config-path config \
    --config-name geniesim_junpu_sac \
    ++runner.only_eval=true \
    ++runner.ckpt_path=../results/junpu_sac_mlp/checkpoints/global_step_1000 \
    ++env.eval.video_cfg.save_video=true
```

评测视频保存到：`../results/junpu_sac_mlp/video/eval/`

---

## 10. 常见问题与排错

### Q1：训练启动后等了很久没有开始？

**原因**：Isaac Sim 首次冷启动需要 3-5 分钟（加载 USD 场景、初始化 GPU）。

**解决**：耐心等待，看到 `[GenieSimShmClient] Initialised` 才算启动成功。后续重启训练时，若容器仍在运行（`reuse_running: true`），启动只需约 10 秒。

---

### Q2：出现 `Container startup timeout` 错误？

**原因**：仿真容器在规定时间内（300 秒）未就绪。

**解决**：
```bash
# 查看容器日志
docker logs geniesim_rlinf_sim --tail 50

# 手动重启容器
docker restart geniesim_rlinf_sim

# 再重试训练
```

---

### Q3：出现 `FileExistsError` 或 SHM 相关错误？

**原因**：上一次训练崩溃留下了残留的共享内存段。

**解决**：
```bash
# 清理共享内存（根据实际 SHM 名称调整）
python3 -c "
from multiprocessing import shared_memory
for name in ['geniesim_frames', 'geniesim_frames_ctrl_0']:
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
        shm.unlink()
        print(f'已清理: {name}')
    except FileNotFoundError:
        print(f'不存在: {name}')
"
```

---

### Q4：训练一直没有 Eval 成功（`success_once=0.0`）？

**可能原因与解决**：

1. **Reward 函数问题**：当前 reward 为 `-0.01/步`，机器人无法判断成功，需要实现真实 reward（见 §8）
2. **训练轮数不足**：机械臂任务通常需要 500~2000 Epoch，继续训练
3. **探索不足（SAC）**：尝试增大 `target_entropy`（如从 -14 改为 -10），增加探索
4. **探索不足（PPO）**：尝试增大 `temperature_train`（如从 1.0 改为 1.5）

---

### Q5：`sac/critic_loss` 或 `sac/actor_loss` 出现 `nan`？

**原因**：奖励值中有 `nan` 或 `inf`，或者梯度爆炸。

**解决**：
1. 检查 reward 函数中是否有除零或 `log(0)` 操作
2. 降低学习率（从 3e-4 改为 1e-4）
3. 降低 `clip_grad`（从 10.0 改为 1.0）

---

### Q6：SAC 训练时提示 `demo_buffer not ready`？

**原因**：示范数据路径不正确，或格式不对。

**解决**：
```bash
# 检查路径
ls /tmp/junpu_demo_buffer/
# 应看到: metadata.json, trajectory_index.json, trajectory_0_*.pt, ...

# 重新转换
.venv/bin/python examples/embodiment/convert_demos_to_buffer.py \
    --demo-dir my_demos --output-dir /tmp/junpu_demo_buffer
```

---

### Q7：如何调整训练速度（Epoch 数）？

通过 CLI 覆盖配置：

```bash
# 快速测试 10 个 Epoch
++runner.max_epochs=10

# 正式训练 5000 个 Epoch
++runner.max_epochs=5000

# 调整 Eval 频率（每 50 Epoch 评测一次）
++runner.val_check_interval=50

# 调整保存频率（每 200 Epoch 保存一次）
++runner.save_interval=200
```

---

## 11. 参数速查表

### SAC 核心超参数

| 参数 | 默认值 | 含义 | 调整建议 |
|------|--------|------|---------|
| `algorithm.gamma` | 0.99 | 未来奖励折扣 | 通常不需要改 |
| `algorithm.tau` | 0.005 | 目标网络更新速率 | 更大 → 学习更快但不稳定 |
| `algorithm.target_entropy` | -7 | 目标熵（=-action_dim） | 更大（如 -5）→ 更多探索 |
| `algorithm.update_epoch` | 8 | 每轮更新次数 | 更多 → 更新更充分，但更慢 |
| `algorithm.critic_actor_ratio` | 2 | Critic/Actor 更新比 | 通常不需要改 |
| `actor.optim.lr` | 3e-4 | Actor 学习率 | 不稳定时减小到 1e-4 |
| `actor.critic_optim.lr` | 3e-4 | Critic 学习率 | 同上 |
| `env.train.max_steps_per_rollout_epoch` | 100 | 每轮收集步数/环境 | 与 total_num_envs 一起决定总数据量 |
| `env.train.total_num_envs` | 4 | 并行仿真环境数 | 更多 → 数据收集更快，但内存增加 |
| `algorithm.replay_buffer.cache_size` | 200 | 在线缓冲区大小 | 图像策略时不宜太大（内存限制） |

### PPO 核心超参数

| 参数 | 默认值 | 含义 | 调整建议 |
|------|--------|------|---------|
| `algorithm.gamma` | 0.99 | 未来奖励折扣 | 通常不需要改 |
| `algorithm.gae_lambda` | 0.95 | GAE 平滑系数 | 通常不需要改 |
| `algorithm.clip_ratio_high` | 0.2 | 策略更新上限 | 更小 → 更保守稳定 |
| `algorithm.update_epoch` | 8 | 数据复用次数 | 过多会过拟合 |
| `actor.optim.lr` | 3e-4 | 学习率 | 不稳定时减小 |
| `env.train.max_steps_per_rollout_epoch` | 300 | 每轮收集步数（=1 回合） | 通常不需要改 |

### 训练流程控制

| 参数 | 作用 | 示例 |
|------|------|------|
| `++runner.max_epochs=N` | 训练总轮数 | `++runner.max_epochs=2000` |
| `++runner.val_check_interval=N` | 每 N 轮评测一次 | `++runner.val_check_interval=50` |
| `++runner.save_interval=N` | 每 N 轮保存一次 | `++runner.save_interval=200` |
| `++runner.resume_dir=PATH` | 从检查点恢复 | `++runner.resume_dir=../results/.../checkpoints/global_step_500` |
| `++runner.only_eval=true` | 仅评测，不训练 | — |
| `++env.eval.video_cfg.save_video=true` | 评测时录制视频 | — |

---

*如有问题，请参考 `DEVELOPER_NOTES_CN.md`（技术细节）或提交 Issue。*
