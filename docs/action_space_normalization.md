# Action Space 归一化方案记录

## 背景

系统中存在两个 action 空间：

| 空间 | 范围 | 使用者 |
|------|------|--------|
| 绝对坐标 | `[_ACTION_LOW, _ACTION_HIGH]`（如 pos ~0.33-0.63, rpy ~1.38-1.78） | env.step()、SpaceMouse、仿真环境 |
| 归一化坐标 | `[-1, 1]` | policy 输入/输出、replay buffer、demo buffer、critic、BC loss |

## 数据流

```
Policy.predict_action_batch()
  └── action = tanh(raw) * action_scale + action_bias   → [-1, 1]
  └── forward_inputs["action"] = action                  → [-1, 1] 存入 trajectory

env_worker.env_interact_step()
  └── prepare_actions_for_geniesim(raw_chunk_actions)     → [-1,1] → 绝对坐标
  └── env.chunk_step(chunk_actions)                       → 绝对坐标传入 env

SpacemouseSimIntervention.step()
  └── actions 已是绝对坐标（prepare_actions 转换后）
  └── SpaceMouse active 时覆盖 actions[eid] 为绝对坐标
  └── info["intervene_action"] = actions.clone()          → 绝对坐标

env_worker._run_interact_once()
  └── env_output.intervene_actions                        → 绝对坐标
  └── update_last_actions(intervene_actions, flags)       → 用绝对坐标替换 buffer 中的 [-1,1] action
```

## 当前问题

`update_last_actions` (embodied_io_struct.py L573-613) 的行为：
1. 将 `intervene_actions`（绝对坐标）按 flag 替换 `self.actions[-1]`（原本 [-1,1]）
2. 同时替换 `self.forward_inputs[-1]["action"]`

结果：replay buffer 和提取的 intervene trajectory (demo buffer) 中，被人类接管的 step 的 action 是绝对坐标，其他 step 是 [-1,1]。

训练时 `forward_critic()` 和 `forward_actor()` 的 BC loss 直接使用 `batch["actions"]`，空间不一致会导致 Q 值和 BC loss 计算错误。

## 当前处理方式

**在 `env_worker.py` 调用 `update_last_actions` 之前，对 `intervene_actions` 做归一化**，将绝对坐标转为 [-1,1]。

### 修改的文件

1. **`rlinf/envs/action_utils.py`**
   - 新增 `normalize_actions_for_geniesim(absolute_actions)`: 绝对坐标 → [-1,1]
     ```python
     mid = torch.as_tensor(_ACTION_MID, ...)
     half_range = torch.as_tensor(_ACTION_RANGE / 2.0, ...)
     return ((absolute_actions - mid) / half_range).clamp(-1.0, 1.0)
     ```
   - 新增 `normalize_intervene_actions(intervene_actions, env_type)`: 按 env_type 分发的通用入口

2. **`rlinf/workers/env/env_worker.py`**
   - 两处 `update_last_actions` 调用（L734、L796）前加 `normalize_intervene_actions()`
   - 保证 intervene_action 写入 trajectory 之前已经是 [-1,1]

### 架构原则

- `env.step()` 始终接收绝对坐标
- SpaceMouse 输出绝对坐标，不做转换
- `prepare_actions_for_geniesim()`: [-1,1] → 绝对坐标（policy → env 方向）
- `normalize_intervene_actions()`: 绝对坐标 → [-1,1]（env → buffer 方向）
- replay buffer / demo buffer 中所有 action 始终为 [-1,1]

## 待讨论的替代方案

### 方案 A: 全部使用绝对坐标（buffer 也存绝对坐标）

让 buffer 统一存绝对坐标，只在 policy forward 的输入/输出边界做转换。

**需要改动**：
- `append_step_result()` 中存入的 `forward_inputs["action"]` 需要反归一化为绝对坐标
- `forward_critic()` 中 `actions = batch["actions"]` 传入 Q 网络前需要归一化
- `forward_actor()` 中 BC loss 的 `demo_actions` 需要归一化
- demo buffer 的离线数据也需要全部用绝对坐标

**优点**：
- 概念上更统一——"buffer 反映真实 env 交互"
- SpaceMouse intervene_action 无需任何转换

**缺点**：
- 改动面更大，涉及 policy worker 的训练循环
- critic 需要知道 action 空间范围做归一化，增加耦合
- 与 SAC 的 tanh squashing 假设不一致（policy 的 log_prob 计算基于 [-1,1]）

### 方案 B: 当前方案（buffer 存 [-1,1]，intervene_action 写入前归一化）

**优点**：
- 改动最小（2 个文件，单点修改）
- 与 SAC 的 tanh policy 天然一致
- critic/actor 训练代码零改动
- 离线 demo buffer 已经是 [-1,1]，无需再改

**缺点**：
- 在 env_worker 层面多了一个归一化调用
- 如果未来有其他 env_type 也需要类似处理，需要在 `normalize_intervene_actions` 中添加分支

### 方案 C: 在 SpaceMouse wrapper 中直接输出 [-1,1] 的 intervene_action

在 `spacemouse_sim_intervention.py` 的 `info["intervene_action"]` 赋值时做归一化。

**优点**：
- 归一化在数据源头完成

**缺点**：
- SpaceMouse wrapper 需要知道 action 归一化参数
- wrapper 的 `intervene_action` 语义变了（不再是 "env 实际执行的 action"）
- 与 CollectEpisode 数据采集可能冲突（采集时可能希望记录绝对坐标）
