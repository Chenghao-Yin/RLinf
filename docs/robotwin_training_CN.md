# RoboTwin 任务训练说明手册

> 面向初次使用 RLinf 进行 RoboTwin 机械臂任务强化学习训练的用户

---

## 目录

1. [RoboTwin 任务简介](#1-robotwin-任务简介)
2. [支持的模型](#2-支持的模型)
3. [训练流程概览：两阶段训练](#3-训练流程概览两阶段训练)
4. [是否需要演示数据？](#4-是否需要演示数据)
5. [核心算法：GRPO vs PPO](#5-核心算法grpo-vs-ppo)
6. [环境准备](#6-环境准备)
7. [第一阶段：SFT 预训练（加载现成模型）](#7-第一阶段sft-预训练加载现成模型)
8. [第二阶段：强化学习训练（GRPO）](#8-第二阶段强化学习训练grpo)
9. [使用 PPO 替代 GRPO](#9-使用-ppo-替代-grpo)
10. [使用 LingbotVLA 模型](#10-使用-lingbotvla-模型)
11. [训练监控与 TensorBoard](#11-训练监控与-tensorboard)
12. [性能基准](#12-性能基准)
13. [常见问题 Q&A](#13-常见问题-qa)
14. [配置参数速查表](#14-配置参数速查表)

---

## 1. RoboTwin 任务简介

RoboTwin 是一套基于 MuJoCo 物理仿真的双臂机器人操作任务集，涵盖多种日常物品操作场景。RLinf 为这些任务提供了完整的强化学习训练支持。

### 已支持任务列表

| 任务名称 | 配置文件前缀 | 任务描述 |
|---------|------------|---------|
| `place_empty_cup` | `robotwin_place_empty_cup_*` | 将空杯子放置到指定位置 |
| `place_shoe` | `robotwin_place_shoe_*` | 将鞋子放置到鞋架 |
| `lift_pot` | `robotwin_lift_pot_*` | 双手抬起锅具 |
| `pick_dual_bottles` | `robotwin_pick_dual_bottles_*` | 双手各抓取一个瓶子 |
| `place_container_plate` | `robotwin_place_container_plate_*` | 将容器放入托盘 |
| `move_can_pot` | `robotwin_move_can_pot_*` | 移动罐子到锅中 |
| `beat_block_hammer` | `robotwin_beat_block_hammer_*` | 用锤子敲击木块 |
| `handover_block` | `robotwin_handover_block_*` | 左右手传递积木 |
| `click_bell` | `robotwin_click_bell_*` | 按压铃铛 |

### 机器人配置

- 机器人：双臂 Piper 机械臂
- 摄像头：双 Intel D435 深度相机（主相机 + 腕部相机）
- 动作空间维度：14（左臂7维 + 右臂7维）
- 本体感知维度：14（关节位置）

---

## 2. 支持的模型

RLinf 为 RoboTwin 任务支持三种视觉-语言-动作（VLA）模型：

### 2.1 OpenVLA-OFT（主要推荐）

**OpenVLA-OFT（Open Vision-Language-Action with Open-loop Fine-Tuning）** 是 RLinf 在 RoboTwin 任务上的主要模型。

| 属性 | 参数 |
|------|------|
| 基础架构 | LLaMA (7B) + 视觉编码器 |
| 版本 | `official`（推荐用于 RoboTwin） |
| 精度 | BF16 |
| 动作块长度 | 25 步（`num_action_chunks: 25`） |
| LoRA 微调 | 是（`is_lora: True`） |
| 本体感知输入 | 是（`use_proprio: True`，维度 14） |
| FiLM 条件 | 否（RoboTwin 不使用）|
| 动作量化 | 256 个 bin 的 token 化动作 |

OpenVLA-OFT 接收图像 + 语言指令 + 本体关节状态作为输入，输出 25 步预测动作序列（chunk）。

### 2.2 LingbotVLA

**LingbotVLA** 是基于 Qwen2.5-VL-3B 的流式扩散策略模型。

| 属性 | 参数 |
|------|------|
| 基础架构 | Qwen2.5-VL-3B |
| 动作生成方式 | Flow-based SDE（随机微分方程流模型） |
| 动作块长度 | 50 步（`num_action_chunks: 50`） |
| 去噪步数 | 10 步（`num_steps: 10`） |
| 噪声强度 | 0.5–0.7（`noise_level`） |
| 支持任务 | `place_shoe`、`click_bell` |

LingbotVLA 的动作生成过程类似扩散模型：从随机噪声出发，经过 10 步去噪，得到平滑的动作序列。

### 2.3 π₀ / OpenPI（评估用）

**π₀（pi-zero）** 是基于 Flow Matching 的扩散策略，RLinf 目前以评估模式支持：

| 属性 | 参数 |
|------|------|
| 基础架构 | OpenPI（流式扩散） |
| 动作块长度 | 10 步（`num_action_chunks: 10`） |
| 用途 | 仅评估（`only_eval: True`） |
| 配置文件 | `robotwin_place_empty_cup_openpi_eval.yaml` |

> π₀ 当前仅支持 eval 模式，不支持 GRPO/PPO 训练。

---

## 3. 训练流程概览：两阶段训练

与 Junpu 任务（MLP 策略 + SAC）不同，RoboTwin 任务使用的 VLA 模型参数量巨大（3B–7B），**不能从零开始强化学习**。必须先进行**监督微调（SFT）预训练**，再进行**强化学习（RL）精炼**。

```
第一阶段：SFT 预训练
  ┌─────────────────────────────────────────┐
  │  演示数据（约 1000 条人工示范轨迹）       │
  │        ↓                                │
  │  监督学习（行为克隆）                    │
  │        ↓                                │
  │  SFT 模型（能完成约 28% 的任务）         │
  └─────────────────────────────────────────┘
             ↓（加载为 RL 的起点）
第二阶段：RL 精炼（GRPO / PPO）
  ┌─────────────────────────────────────────┐
  │  仿真环境奖励（任务完成信号）             │
  │        ↓                                │
  │  强化学习（探索 + 改进）                 │
  │        ↓                                │
  │  RL 模型（能完成约 86% 的任务）          │
  └─────────────────────────────────────────┘
```

**好消息**：RLinf 团队已在 HuggingFace 上发布了所有任务的 SFT 预训练模型，用户**无需自己进行 SFT 训练**，直接下载即可。

---

## 4. 是否需要演示数据？

**RoboTwin RL 训练阶段不需要演示数据。**

| 阶段 | 是否需要演示数据 |
|------|----------------|
| SFT 预训练 | 需要（但 RLinf 已提供训练好的 SFT 模型） |
| GRPO 强化学习 | **不需要** |
| PPO 强化学习 | **不需要** |

RL 训练阶段的奖励信号完全来自仿真环境——机器人完成任务则获得正奖励，失败则获得零奖励。这与 Junpu 任务的 SAC（需要演示 buffer）有本质区别。

---

## 5. 核心算法：GRPO vs PPO

### 5.1 什么是 GRPO？

**GRPO（Group Relative Policy Optimization）** 是 RLinf 为 RoboTwin 推荐的主要算法，专为大型语言/动作模型设计。

#### 核心思想：组内相对排名

对于同一初始状态，GRPO 同时运行 **8 条轨迹**（`group_size: 8`）。这 8 条轨迹得到的奖励相互比较：奖励高于平均的轨迹被"鼓励"，低于平均的被"抑制"。

```
同一起点，并行执行 8 条轨迹：
  轨迹1: 奖励 = 1.0  ← 高于平均，被鼓励 ↑
  轨迹2: 奖励 = 0.0  ← 低于平均，被抑制 ↓
  轨迹3: 奖励 = 1.0  ← 高于平均，被鼓励 ↑
  轨迹4: 奖励 = 0.0  ← 低于平均，被抑制 ↓
  轨迹5: 奖励 = 1.0  ← 高于平均，被鼓励 ↑
  轨迹6: 奖励 = 1.0  ← 高于平均，被鼓励 ↑
  轨迹7: 奖励 = 0.0  ← 低于平均，被抑制 ↓
  轨迹8: 奖励 = 1.0  ← 高于平均，被鼓励 ↑
  
  平均奖励 = 0.625，相对优势 ∝ (奖励 - 平均)
```

#### GRPO 的优点

- **不需要 Critic 网络**：节省一半的 GPU 显存（相比 PPO）
- **优势估计更稳定**：用组内相对排名，不受绝对奖励尺度影响
- **适合稀疏奖励**：只要 8 条中有几条成功，就能获得有效的梯度信号
- **gamma=1.0**：片段奖励不打折扣，整条成功轨迹等权重对待

#### GRPO 关键参数

```yaml
algorithm:
  adv_type: grpo          # 使用 GRPO 优势估计
  group_size: 8           # 每组并行轨迹数
  gamma: 1.0              # 奖励不折扣（片段任务推荐）
  loss_type: actor        # 只训练 Actor，无 Critic
  kl_beta: 0.0            # 不限制与参考模型的 KL 散度
  clip_ratio_high: 0.28   # 策略更新幅度上限
  temperature_train: 1.6  # 采样温度（增加探索）
  reward_coef: 5.0        # 奖励缩放系数
```

### 5.2 什么是 PPO？

**PPO（Proximal Policy Optimization）** 是 RL 经典算法，也可用于 RoboTwin。

#### PPO 与 GRPO 的区别

PPO 为每条轨迹训练一个独立的 **Critic 网络**来估计状态价值（Value Function），然后用 GAE 方法计算优势。

```yaml
algorithm:
  adv_type: gae           # GAE 优势估计
  group_size: 1           # 每次只跑 1 条轨迹（不需要组）
  gamma: 0.99             # 奖励折扣因子
  loss_type: actor_critic # 同时训练 Actor 和 Critic

actor:
  model:
    add_value_head: True  # 模型头部增加 Value Head
```

### 5.3 GRPO vs PPO 对比

| 对比维度 | GRPO（推荐） | PPO |
|---------|------------|-----|
| 算法类型 | 组内相对优势 | 广义优势估计 (GAE) |
| 是否需要 Critic | 否 | 是（`add_value_head: True`） |
| 并行轨迹数 | 8条（`group_size=8`） | 1条（`group_size=1`） |
| 奖励折扣 | 1.0（不折扣） | 0.99 |
| GPU 显存占用 | 较低（无 Critic） | 较高（需要 Value Head）|
| 批量大小 | 1024（并行高效） | 32（受显存限制）|
| 训练稳定性 | 高（组内归一化） | 中等 |
| 适合场景 | 片段稀疏奖励 | 密集奖励连续任务 |
| RoboTwin 表现 | **推荐** | 可用，但性能通常低于 GRPO |
| 配置文件后缀 | `_grpo_openvlaoft` | `_ppo_openvlaoft` |

---

## 6. 环境准备

### 6.1 硬件要求

| 配置 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU 数量 | 4 × A100 40G | 8 × A100 80G |
| 显存 | 160 GB 总显存 | 640 GB 总显存 |
| 内存 | 128 GB | 256 GB |
| 训练时间（GRPO 1000 epoch） | 约 48 小时 | 约 24 小时 |

> RoboTwin VLA 模型（7B 参数）需要使用 FSDP 进行多 GPU 分片训练。

### 6.2 软件依赖

确保已安装 RLinf venv（`.venv/`）：

```bash
cd /path/to/RLinf
python -m venv .venv
source .venv/bin/activate
pip install -e ".[robotwin]"
```

### 6.3 下载 RoboTwin 资产

RoboTwin 任务需要仿真场景资产（3D 模型、纹理等）：

```bash
# 克隆 RoboTwin 资产仓库（约 10 GB）
git clone https://github.com/TianxingChen/RoboTwin.git /path/to/robotwin_assets
```

资产路径将在训练配置中通过 `assets_path` 指定。

---

## 7. 第一阶段：SFT 预训练（加载现成模型）

### 7.1 下载预训练 SFT 模型

RLinf 团队已在 HuggingFace 上发布了所有任务的 SFT 模型，**直接下载即可，无需自行训练**：

```bash
# 示例：下载 place_empty_cup 任务的 SFT 模型
# HuggingFace 地址：https://huggingface.co/RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup

huggingface-cli download \
  RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup \
  --local-dir /path/to/models/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup
```

### 7.2 各任务 SFT 模型

| 任务 | HuggingFace 模型 ID |
|------|-------------------|
| place_empty_cup | `RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup` |
| place_shoe | `RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_shoe` |
| lift_pot | `RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot` |
| pick_dual_bottles | `RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-pick_dual_bottles` |
| place_container_plate | `RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_container_plate` |
| move_can_pot | `RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot` |
| beat_block_hammer | `RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-beat_block_hammer` |
| handover_block | `RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-handover_block` |

> LingbotVLA 的 SFT 模型需自行准备，基础权重从 `Qwen2.5-VL-3B-Instruct` 微调得到。

---

## 8. 第二阶段：强化学习训练（GRPO）

### 8.1 完整训练步骤

以 `place_empty_cup` 任务为例：

**步骤 1：准备目录结构**

```
/path/to/
├── RLinf/                    # RLinf 代码仓库
├── robotwin_assets/          # RoboTwin 仿真资产
└── models/
    └── RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup/  # SFT 模型
```

**步骤 2：确认种子文件存在**

```bash
ls $REPO_PATH/rlinf/envs/robotwin/seeds/
# 应看到：train_seeds.json  eval_seeds.json
```

**步骤 3：启动 GRPO 训练**

```bash
cd /path/to/RLinf

REPO_PATH=$(pwd) \
EMBODIED_PATH=$(pwd)/examples/embodiment \
  .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name robotwin_place_empty_cup_grpo_openvlaoft \
  ++env.train.assets_path=/path/to/robotwin_assets \
  ++env.eval.assets_path=/path/to/robotwin_assets \
  ++actor.model.model_path=/path/to/models/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup
```

### 8.2 多卡训练（推荐 8 卡）

RoboTwin GRPO 训练默认使用全部可用 GPU（通过 FSDP 分片）：

```bash
# 配置文件中默认设置：
# cluster.component_placement: actor, env, rollout: 0-7  (使用 GPU 0-7)

# 如果 GPU 数量不足 8 张，修改配置：
REPO_PATH=$(pwd) \
EMBODIED_PATH=$(pwd)/examples/embodiment \
  .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name robotwin_place_empty_cup_grpo_openvlaoft \
  ++cluster.component_placement."actor, env, rollout"="0-3" \  # 使用 GPU 0-3
  ++env.train.assets_path=/path/to/robotwin_assets \
  ++env.eval.assets_path=/path/to/robotwin_assets \
  ++actor.model.model_path=/path/to/models/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup
```

### 8.3 其他任务

```bash
# place_shoe（OpenVLA-OFT）
--config-name robotwin_place_shoe_grpo_openvlaoft   # （需自行创建，参考 place_empty_cup 模板）

# lift_pot
--config-name robotwin_lift_pot_grpo_openvlaoft

# pick_dual_bottles
--config-name robotwin_pick_dual_bottles_grpo_openvlaoft

# place_container_plate
--config-name robotwin_place_container_plate_grpo_openvlaoft

# move_can_pot
--config-name robotwin_move_can_pot_grpo_openvlaoft

# beat_block_hammer
--config-name robotwin_beat_block_hammer_grpo_openvlaoft

# handover_block
--config-name robotwin_handover_block_grpo_openvlaoft
```

### 8.4 从 checkpoint 恢复训练

```bash
REPO_PATH=$(pwd) \
EMBODIED_PATH=$(pwd)/examples/embodiment \
  .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name robotwin_place_empty_cup_grpo_openvlaoft \
  ++runner.resume_dir=../results/checkpoints/global_step_200 \
  ++env.train.assets_path=/path/to/robotwin_assets \
  ++env.eval.assets_path=/path/to/robotwin_assets \
  ++actor.model.model_path=/path/to/models/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup
```

### 8.5 仅评估（不训练）

```bash
REPO_PATH=$(pwd) \
EMBODIED_PATH=$(pwd)/examples/embodiment \
  .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name robotwin_place_empty_cup_grpo_openvlaoft \
  ++runner.only_eval=True \
  ++runner.ckpt_path=../results/checkpoints/global_step_500/actor.pt \
  ++env.eval.assets_path=/path/to/robotwin_assets \
  ++actor.model.model_path=/path/to/models/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup
```

---

## 9. 使用 PPO 替代 GRPO

目前 RLinf 提供了 `place_empty_cup` 任务的 PPO 配置：

```bash
cd /path/to/RLinf

REPO_PATH=$(pwd) \
EMBODIED_PATH=$(pwd)/examples/embodiment \
  .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name robotwin_place_empty_cup_ppo_openvlaoft \
  ++env.train.assets_path=/path/to/robotwin_assets \
  ++env.eval.assets_path=/path/to/robotwin_assets \
  ++actor.model.model_path=/path/to/models/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup
```

> PPO 配置中 `group_size: 1`，`global_batch_size: 32`，显存占用与 GRPO 相当，但训练效率低于 GRPO。

---

## 10. 使用 LingbotVLA 模型

LingbotVLA 目前支持 `place_shoe` 和 `click_bell` 任务：

```bash
# place_shoe（LingbotVLA）
cd /path/to/RLinf

REPO_PATH=$(pwd) \
EMBODIED_PATH=$(pwd)/examples/embodiment \
LINGBOT_VLA_PATH=/path/to/lingbot-vla-4b \
  .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name robotwin_place_shoe_grpo_lingbotvla \
  ++env.train.assets_path=/path/to/robotwin_assets \
  ++env.eval.assets_path=/path/to/robotwin_assets \
  ++actor.model.model_path=/path/to/lingbot_sft_model \
  ++actor.model.tokenizer_path=/path/to/Qwen2.5-VL-3B-Instruct \
  ++actor.model.lingbotvla.config_path=/path/to/lingbot-vla-4b
```

**LingbotVLA 与 OpenVLA-OFT 的主要区别：**

| 维度 | OpenVLA-OFT | LingbotVLA |
|------|------------|------------|
| 动作生成 | 自回归 token 化 | Flow-based SDE 扩散 |
| 动作块长度 | 25 步 | 50 步 |
| 推理速度 | 较快 | 较慢（10 步去噪） |
| 动作平滑度 | 中等 | 更平滑 |
| 需要配置 | `LINGBOT_VLA_PATH` 环境变量 | 否 |

---

## 11. 训练监控与 TensorBoard

### 11.1 启动 TensorBoard

```bash
tensorboard --logdir /path/to/RLinf/../results --port 6006
# 访问 http://localhost:6006
```

### 11.2 关键指标说明

| 指标名称 | 含义 | 健康范围 |
|---------|------|---------|
| `eval/success_rate` | 评估集任务成功率 | 逐步上升至 >0.5 |
| `train/reward` | 训练集平均奖励 | 与成功率正相关 |
| `actor/loss` | Actor 策略损失 | 应稳定下降，避免发散 |
| `actor/kl` | 与参考策略的 KL 散度 | 保持较小（<0.1） |
| `actor/clip_ratio` | 策略更新被 clip 的比例 | 正常 <30% |
| `actor/lr` | 当前学习率 | 按调度曲线变化 |
| `rollout/group_rewards_mean` | 每组轨迹平均奖励 | 逐步上升 |
| `rollout/group_rewards_std` | 组内奖励标准差 | 越大代表探索越充分 |

### 11.3 如何判断训练健康

**正常训练迹象：**
- `eval/success_rate` 在 100-300 epoch 内从约 28% 开始上升
- `actor/loss` 缓慢下降，无突变
- `actor/kl` 保持在 0.0–0.1 之间

**需要干预的异常：**

| 异常现象 | 可能原因 | 建议操作 |
|---------|---------|---------|
| `eval/success_rate` 持续不上升 | 奖励太稀疏 | 增大 `reward_coef`（如改为 10.0）|
| `actor/loss` 突然暴增 | 学习率过高 | 降低 `optim.lr`（如 5e-5）|
| `actor/kl` 持续增大 | 策略偏移过大 | 增大 `kl_beta`（如 0.01）|
| CUDA OOM | 显存不足 | 减小 `micro_batch_size`（如改为 16）|
| `clip_ratio` 持续为 1.0 | clip 过于保守 | 适当增大 `clip_ratio_high`|

---

## 12. 性能基准

基于 RLinf 官方在 RoboTwin 任务上的实验结果：

| 任务 | SFT 成功率 | GRPO 成功率 | 提升幅度 |
|------|-----------|------------|---------|
| place_empty_cup | ~28% | ~86% | +58% |
| place_shoe | ~25% | ~82% | +57% |
| lift_pot | ~30% | ~88% | +58% |
| pick_dual_bottles | ~22% | ~80% | +58% |
| 平均 | ~28.79% | ~86.16% | **+57.37%** |

> 以上数据基于 8×A100 80G，GRPO 训练约 500 epoch。实际结果因硬件和随机种子有所差异。

---

## 13. 常见问题 Q&A

**Q1: 为什么不能从零开始 RL 训练，必须先 SFT？**

A: VLA 模型（7B 参数）的动作空间极其复杂。如果从随机初始化开始 RL，模型会随机生成动作，几乎永远无法完成任务，也就无法获得任何正奖励，训练无法收敛。SFT 预训练给模型注入了任务知识，使其有约 28% 的成功率。在这个基础上，RL 才能有效地探索和改进。

**Q2: GRPO 的 group_size=8 意味着什么？实际需要多少并发环境？**

A: `group_size=8` 意味着每个初始状态并行执行 8 条轨迹。配置中 `total_num_envs: 128` 表示总共有 128 个并发环境，即同时处理 128/8=16 个不同初始状态，每个状态各跑 8 条轨迹。

**Q3: GRPO gamma=1.0 是否合理？**

A: 对于 RoboTwin 这种片段式任务（任务结束时才给奖励），gamma=1.0 是合理的——它意味着每一步的动作对最终结果的贡献权重相等，不会因为"步骤离成功时刻较远"而打折扣。这有助于 VLA 模型学习整条轨迹的策略。

**Q4: 显存不够怎么办？**

A: 可以尝试以下方案（按效果排序）：
1. 减小 `actor.micro_batch_size`（如 32→16→8）
2. 减少 GPU 并行数（`component_placement: "0-3"` 使用 4 卡）
3. 减少 `total_num_envs`（如 128→64）
4. 开启梯度检查点：`fsdp_config.gradient_checkpointing: True`

**Q5: 训练结果保存在哪里？**

A: 默认保存在 `../results/`（相对于 RLinf 目录的父目录）：
```
../results/
├── checkpoints/
│   ├── global_step_20/
│   │   └── actor.pt      # 模型权重
│   ├── global_step_40/
│   └── ...
├── video/eval/           # 评估时的轨迹视频
└── events.out.tfevents.* # TensorBoard 日志
```

**Q6: 如何只跑评估，不训练？**

A: 使用 `only_eval: True` 并指定 checkpoint 路径：
```bash
++runner.only_eval=True \
++runner.ckpt_path=../results/checkpoints/global_step_500/actor.pt
```

**Q7: LingbotVLA 和 OpenVLA-OFT 哪个效果更好？**

A: 在 RoboTwin 任务上，两者性能相当，但各有适用场景：
- **OpenVLA-OFT**：推理速度更快，现有任务支持更全面（9个任务），推荐首选
- **LingbotVLA**：动作轨迹更平滑（flow-based 生成），适合对运动平滑性要求高的任务（目前支持 `place_shoe` 和 `click_bell`）

---

## 14. 配置参数速查表

### GRPO 关键参数

| 参数路径 | 默认值 | 说明 |
|---------|-------|------|
| `algorithm.group_size` | `8` | 每组并行轨迹数（GRPO 核心） |
| `algorithm.rollout_epoch` | `8` | 每次 rollout 迭代数 |
| `algorithm.gamma` | `1.0` | 奖励折扣因子 |
| `algorithm.clip_ratio_high` | `0.28` | PPO clip 上限 |
| `algorithm.kl_beta` | `0.0` | KL 惩罚系数（0=不限制）|
| `algorithm.reward_coef` | `5.0` | 奖励缩放倍数 |
| `algorithm.temperature_train` | `1.6` | 采样温度（越高越随机）|
| `algorithm.filter_rewards` | `True` | 过滤极端奖励（稳定训练）|

### OpenVLA-OFT 模型参数

| 参数路径 | 默认值 | 说明 |
|---------|-------|------|
| `actor.model.model_path` | （需指定）| SFT 预训练模型路径 |
| `actor.model.implement_version` | `"official"` | 使用官方版本（RoboTwin）|
| `actor.model.num_action_chunks` | `25` | 每次预测的动作步数 |
| `actor.model.use_proprio` | `True` | 使用本体感知（关节状态）|
| `actor.model.is_lora` | `True` | 使用 LoRA 微调 |
| `actor.model.unnorm_key` | （任务相关）| 动作反归一化统计键 |

### 环境参数

| 参数路径 | 默认值 | 说明 |
|---------|-------|------|
| `env.train.total_num_envs` | `128` | 训练并发环境总数 |
| `env.train.max_episode_steps` | `200` | 最大片段步数 |
| `env.train.assets_path` | （需指定）| RoboTwin 资产目录 |
| `env.eval.total_num_envs` | `128` | 评估并发环境总数 |
| `runner.val_check_interval` | `20` | 每 20 epoch 评估一次 |
| `runner.save_interval` | `20` | 每 20 epoch 保存一次 |
| `actor.micro_batch_size` | `32` | 单卡 batch 大小（调显存用）|
| `actor.global_batch_size` | `1024` | 全局 batch 大小 |

---

## 附录：配置文件一览

| 配置文件名 | 算法 | 模型 | 任务 | 用途 |
|-----------|------|------|------|------|
| `robotwin_place_empty_cup_grpo_openvlaoft.yaml` | GRPO | OpenVLA-OFT | place_empty_cup | **主要训练** |
| `robotwin_place_empty_cup_ppo_openvlaoft.yaml` | PPO | OpenVLA-OFT | place_empty_cup | PPO 对比 |
| `robotwin_lift_pot_grpo_openvlaoft.yaml` | GRPO | OpenVLA-OFT | lift_pot | 主要训练 |
| `robotwin_pick_dual_bottles_grpo_openvlaoft.yaml` | GRPO | OpenVLA-OFT | pick_dual_bottles | 主要训练 |
| `robotwin_place_container_plate_grpo_openvlaoft.yaml` | GRPO | OpenVLA-OFT | place_container_plate | 主要训练 |
| `robotwin_move_can_pot_grpo_openvlaoft.yaml` | GRPO | OpenVLA-OFT | move_can_pot | 主要训练 |
| `robotwin_beat_block_hammer_grpo_openvlaoft.yaml` | GRPO | OpenVLA-OFT | beat_block_hammer | 主要训练 |
| `robotwin_handover_block_grpo_openvlaoft.yaml` | GRPO | OpenVLA-OFT | handover_block | 主要训练 |
| `robotwin_place_shoe_grpo_lingbotvla.yaml` | GRPO | LingbotVLA | place_shoe | LingbotVLA 训练 |
| `robotwin_click_bell_grpo_lingbotvla.yaml` | GRPO | LingbotVLA | click_bell | LingbotVLA 训练 |
| `robotwin_place_empty_cup_openpi_eval.yaml` | — | π₀/OpenPI | place_empty_cup | 仅评估 |
| `robotwin_place_shoe_grpo_lingbotvla_eval.yaml` | — | LingbotVLA | place_shoe | 仅评估 |
| `robotwin_click_bell_grpo_lingbotvla_eval.yaml` | — | LingbotVLA | click_bell | 仅评估 |
| `robotwin_place_empty_cup_ppo_openvlaoft_eval.yaml` | — | OpenVLA-OFT | place_empty_cup | 仅评估 |
