# GeneSim/Junpu 任务 SFT+RL 训练可行性分析

> 分析 GeneSim Junpu 任务能否像 RoboTwin 一样采用 VLA 模型（OpenVLA-OFT）进行 SFT 预训练 + GRPO 强化学习训练

---

## 一句话结论

**框架层面：完全支持，无需改任何代码。**
**落地层面：主要障碍是 SFT 预训练模型——RoboTwin 有官方 HuggingFace 模型可以直接下载，GeneSim/Junpu 没有，需要自行训练。**

---

## 框架兼容性验证

| 检查项 | 状态 | 说明 |
|--------|------|------|
| obs 格式兼容 | ✅ | `GenieSimBaseEnv._wrap_obs()` 已返回 `main_images` / `wrist_images` / `states` / `task_descriptions`，与 VLA 模型需求完全一致 |
| `chunk_step` 接口 | ✅ | `GenieSimBaseEnv` 已有 `chunk_step()`，返回格式与 `env_worker` 期望一致 |
| 图像 SHM 传输 | ✅ | `GenieSimShmClient` 已支持 `main_images` / `wrist_images` 从容器传到 host |
| 动作 pass-through | ✅ | `action_utils.py:257-259` 对 `GENIESIM` env_type 直接透传，与模型类型无关 |
| Worker 路由 | ✅ | `rollout_worker` / `env_worker` 完全由 config 的 `model_type` 决定，无 env-model 硬编码约束 |
| 配置验证函数 | ✅ | `validate_embodied_cfg()` 只检查 `"embodied"` 类别，不限制 env-model 组合 |

**关键代码证据：**
- `rlinf/envs/geniesim/geniesim_env.py:191-212` — `_wrap_obs()` 已返回完整 VLA 所需 obs
- `rlinf/envs/action_utils.py:257-259` — GENIESIM 动作直接透传
- `rlinf/config.py:746-883` — `validate_embodied_cfg` 无 env-model 约束

---

## RoboTwin vs GeneSim/Junpu 落地对比

| 维度 | RoboTwin + VLA（当前） | GeneSim/Junpu + VLA（待实现） |
|------|----------------------|------------------------------|
| 模型架构 | OpenVLA-OFT (7B) / LingbotVLA (3B) | 同样可用 OpenVLA-OFT (7B) |
| SFT 模型来源 | HuggingFace 官方下载，开箱即用 | **需要自行训练**（无公开模型）|
| SFT 演示数据 | ~1000 条（RLinf 团队已收集）| `my_demos/` 已有 30 条，数据量偏少 |
| RL 算法 | GRPO（group_size=8） | 同样可用 GRPO |
| 奖励信号 | 任务完成奖励（env 内实现）| 当前仅有 -0.01/step 占位符，**需实现真实奖励**|
| 图像来源 | Isaac Sim 渲染 → SHM → host | 同样通过 SHM 传输（已支持）|
| 配置文件 | 14 个现成配置 | **需新建配置** |
| 硬件需求 | 4-8× A100（与 RoboTwin 相同）| 相同 |

---

## 实施路线

### 选项 A：快速实验（跳过 SFT）

**目的**：验证 GeneSim + VLA 框架是否打通，效果会较差。

**需要做的事**：
1. 新建配置文件 `examples/embodiment/config/geniesim_junpu_grpo_openvlaoft.yaml`
   - 参考 `robotwin_place_empty_cup_grpo_openvlaoft.yaml`
   - `defaults` 改为 `env/geniesim_junpu_place_workpiece`
   - `actor.model.model_path` 指向 OpenVLA-OFT 基础权重（未经 SFT 微调）
   - 算法改为 GRPO
2. 实现真实奖励函数（否则 GRPO 无法学习）

**无需改任何现有代码。**

---

### 选项 B：标准路线（SFT + GRPO）

#### Phase A：SFT 预训练

**A1. 将 my_demos 转为 SFT 格式**

现有 `convert_demos_to_buffer.py` 输出的是 SAC replay buffer 格式。SFT 需要不同格式：每条轨迹的 (image, state, action) 时间步序列，输出为 HuggingFace dataset。

需新建 `examples/embodiment/convert_demos_to_sft.py`：
- 输入：`my_demos/*.pkl`（每个文件包含 observations, actions 等字段）
- 输出：HuggingFace `datasets` 格式（包含图像、状态、动作、任务描述）
- 参考：`convert_demos_to_buffer.py` 中的 pkl 解析逻辑

**A2. SFT 微调**

使用 RLinf 已有的 `fsdp_vla_sft_worker.py` 对 OpenVLA-OFT 基础权重进行微调：
```bash
# 示意（具体配置需新建）
REPO_PATH=$(pwd) EMBODIED_PATH=$(pwd)/examples/embodiment \
  .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name geniesim_junpu_sft_openvlaoft
```

**注意**：30 条演示数据量偏少（RoboTwin 用了 ~1000 条），过拟合风险较高。可考虑：
- 数据增强（图像色彩抖动、随机裁剪）
- 更小的 LoRA rank
- 更短的训练 epoch

#### Phase B：GRPO 强化学习

**B1. 实现真实奖励函数**（GRPO 必要条件）

当前 `rlinf/envs/geniesim/tasks/junpu_place_workpiece.py` 只有占位符奖励 `-0.01/step`。GRPO 依赖稀疏成功奖励，需要实现：
- 工件是否到达目标位置的检测（通过 GeneSim 的物体位姿查询接口）
- 成功时返回正奖励（如 `+1.0`），失败返回 `0`

**B2. 新建 GRPO 配置文件**

新建 `examples/embodiment/config/geniesim_junpu_grpo_openvlaoft.yaml`：

```yaml
defaults:
  - env/geniesim_junpu_place_workpiece@env.train
  - env/geniesim_junpu_place_workpiece@env.eval
  - model/openvla_oft@actor.model
  - training_backend/fsdp@actor.fsdp_config

algorithm:
  adv_type: grpo
  loss_type: actor        # 无 Critic，节省显存
  group_size: 8           # 每组 8 条并行轨迹
  gamma: 1.0              # 片段任务不折扣
  kl_beta: 0.0
  reward_coef: 5.0
  temperature_train: 1.6

env:
  train:
    group_size: 8         # 与 algorithm.group_size 一致
    # container_cfg 等保持不变

actor:
  model:
    model_path: "/path/to/junpu_sft_model"
    implement_version: "official"
    action_dim: 14
    num_action_chunks: 25
    use_proprio: True
    proprio_dim: 14       # state_dim = 40，但 proprio 取前 14（关节位置）
    is_lora: True
    unnorm_key: junpu_place_workpiece  # 需与 SFT 时一致
```

---

## 需要解决的问题

### 问题 1：演示数据量不足

30 条演示用于 SFT 7B 模型，数量远低于 RoboTwin 的 1000 条。选项：
- **收集更多演示**：通过 GeneSim 系统录制更多成功轨迹
- **降低模型规模**：考虑用 LingbotVLA (3B) 代替 OpenVLA-OFT (7B)
- **接受较低 SFT 质量**：SFT 后成功率可能只有 10-20%，再用 GRPO 提升

### 问题 2：真实奖励函数

当前占位符奖励无法用于 GRPO。需要通过 GeneSim 的状态查询接口实现：
- 工件（workpiece）的当前位姿
- 目标位置的定义
- 成功判定阈值（如位置误差 < 2cm，姿态误差 < 10°）

### 问题 3：unnorm_key（动作归一化）

OpenVLA-OFT 需要一个 `unnorm_key` 来反归一化预测动作。SFT 时需要统计 `my_demos` 中的动作均值/方差并注册这个 key。

---

## 决策参考

| 目标 | 推荐方案 |
|------|---------|
| 快速验证框架打通 | 选项 A（跳过 SFT，直接 GRPO） |
| 完整流程验证 | 选项 B（先收集更多演示 → SFT → GRPO） |
| 暂不实施，留待后续 | 不需要任何代码改动，框架已就绪 |

---

## 关键文件索引

| 文件 | 说明 |
|------|------|
| `rlinf/envs/geniesim/geniesim_env.py:191-212` | `_wrap_obs()`，obs 格式已兼容 VLA |
| `rlinf/envs/geniesim/tasks/junpu_place_workpiece.py` | 需实现真实奖励/终止条件 |
| `rlinf/envs/action_utils.py:257-259` | GENIESIM 动作透传，无需改动 |
| `rlinf/config.py:746-883` | `validate_embodied_cfg`，无 env-model 约束 |
| `examples/embodiment/config/robotwin_place_empty_cup_grpo_openvlaoft.yaml` | GRPO 配置参考模板 |
| `examples/embodiment/convert_demos_to_buffer.py` | demo pkl 解析参考 |
| `rlinf/workers/vla/fsdp_vla_sft_worker.py` | SFT 训练 worker（已有，无需改）|
