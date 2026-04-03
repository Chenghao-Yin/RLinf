# LIBERO Spatial + OpenPI（Pi0 / Pi0.5）PPO 训练链路说明

本文档将 `examples/embodiment/config/libero_spatial_ppo_openpi_quickstart.yaml` 所代表的 **LIBERO 仿真 + OpenPI（Pi0 系列）+ PPO** 训练链路整理为可落地的说明，便于对照代码与排障。

---

## 1. 配置如何拼装（Hydra）

训练入口为 `examples/embodiment/train_embodied_agent.py`，通过 Hydra 读取配置；**必须**设置环境变量 **`EMBODIED_PATH`** 指向 `examples/embodiment/config`（与 YAML 中 `searchpath` 一致），否则子配置（`env/`、`model/` 等）无法解析。

示例：

```bash
export EMBODIED_PATH=/path/to/RLinf/examples/embodiment/config
cd /path/to/RLinf/examples/embodiment
python train_embodied_agent.py --config-path config --config-name libero_spatial_ppo_openpi_quickstart
```

### 1.1 `libero_spatial_ppo_openpi_quickstart.yaml` 引用的 defaults

| 引用 | 文件位置（相对 `EMBODIED_PATH`） | 作用 |
|------|----------------------------------|------|
| `env/libero_spatial@env.train` / `@env.eval` | `env/libero_spatial.yaml` | LIBERO spatial 任务、并行环境数、步长、视频路径等 |
| `model/pi0@actor.model` | `model/pi0.yaml` | **OpenPI / Pi0**：`model_type: openpi`、`openpi.config_name: pi0_libero` 等 |
| `training_backend/fsdp@actor.fsdp_config` | `training_backend/fsdp.yaml` | Actor 使用 FSDP；OpenPI 侧 **不要开启 gradient checkpointing**（与 `pi0` 训练约束一致） |

### 1.2 Pi0 与 Pi0.5（Pi05）的差异（换链路时改这里）

| 项目 | Pi0（当前 quickstart 默认） | Pi0.5 |
|------|-----------------------------|--------|
| `defaults` 中 model | `model/pi0@actor.model` | 改为 `model/pi0_5@actor.model` |
| `openpi.config_name` | `pi0_libero`（见 `model/pi0.yaml`） | `pi05_libero`（见 `model/pi0_5.yaml`） |
| 其它 | `num_steps` 等以各自 `model/pi0*.yaml` 为准 | `precision` 等对 Pi0.5 的约束见 `pi0_5.yaml` 注释 |

业务链路（Env → Rollout → Actor → PPO 更新 → 权重同步）**相同**，区别主要在 **OpenPI 配置名、默认超参与模型权重路径**。

### 1.3 本 quickstart 中需要重点关注的覆盖项

- **`actor.model.model_path` / `rollout.model.model_path`**：预训练 Pi0（或 Pi0.5）权重；**建议二者指向同一目录**（或你有意分离时再区分）。
- **`actor.model.openpi`**：例如 `detach_critic_input: True`，会合并进 `OpenPi0Config`。
- **`rollout.unnorm_key: libero_10`**：动作反归一化使用的统计 key，需与 OpenPI 数据/检查点一致。
- **`actor.model.num_action_chunks: 5`**：一次策略输出多步 action chunk；Rollout 用 `max_steps_per_rollout_epoch // num_action_chunks` 计算 chunk 步数。

---

## 2. OpenPI 模型在哪里加载、加载步骤

统一入口：**`rlinf.models.get_model`** → 当 `model_type` 为 `openpi` 时，调用 **`rlinf.models.embodiment.openpi.get_model`**。

### 2.1 加载顺序（逻辑）

1. **`get_openpi_config(config_name, model_path=..., data_kwargs=...)`**  
   - `config_name` 来自 `actor.model.openpi.config_name`（Pi0 常为 `pi0_libero`，Pi0.5 为 `pi05_libero`）。  
   - 与 YAML 中 `actor.model.openpi` 的其它字段合并为 `OpenPi0Config`。

2. **`openpi.shared.download.maybe_download(model_path)`**  
   - 本地目录或从 Hugging Face 拉取到可访问路径。

3. **构建 `OpenPi0ForRLActionPrediction`**（定义于 `openpi_action_model.py`，底层为 `openpi.models_pytorch.pi0_pytorch.PI0Pytorch`）。

4. **权重**  
   - 若存在 `model_state_dict/full_weights.pt` 或 `actor/model_state_dict/full_weights.pt`：用 `torch.load` + `load_state_dict`。  
   - 否则：按目录下 **`*.safetensors`**（或 `model.safetensors`）加载。

5. **`openpi.training.checkpoints.load_norm_stats`**  
   - 按 `asset_id` 从 checkpoint 读归一化统计，保证与预训练一致。

6. **`model.setup_wrappers`**  
   - 组装 `Normalize` / `Unnormalize` 与 **`libero_dataconfig`** 中的 **`LiberoInputs` / `LiberoOutputs`**。

### 2.2 核心代码文件

| 文件 | 作用 |
|------|------|
| `rlinf/models/__init__.py` | `get_model` 按 `model_type` 分发到 OpenPI |
| `rlinf/models/embodiment/openpi/__init__.py` | OpenPI：`get_openpi_config`、下载、加载权重、norm、wrappers |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | `OpenPi0ForRLActionPrediction`、PPO 价值头、前向等 |
| `rlinf/models/embodiment/openpi/dataconfig/__init__.py` | 注册 `pi0_libero` / `pi05_libero` 等训练配置 |
| `rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py` | LIBERO 数据变换与 `DataConfig` |
| `rlinf/models/embodiment/openpi/policies/libero_policy.py` | LIBERO 观测 → 模型输入、输出反变换 |

运行时依赖 Python 包 **`openpi`**（如 `openpi.models_pytorch.pi0_pytorch`、`openpi.training.checkpoints`）。

---

## 3. 训练进程架构（三 Worker + Runner）

| 组件 | 类 / 文件 | 职责 |
|------|-----------|------|
| 入口 | `examples/embodiment/train_embodied_agent.py` | Hydra、`Cluster`、`EmbodiedRunner` |
| Actor | `EmbodiedFSDPActor`（`rlinf/workers/actor/fsdp_actor_worker.py`） | `get_model(self.cfg.actor.model)`，FSDP 包装，优化器，PPO 更新 |
| Rollout | `MultiStepRolloutWorker`（`rlinf/workers/rollout/hf/huggingface_worker.py`） | 使用 **`rollout.model`**（`model_path` / `precision`）再次 `get_model`，**eval**，在环境里前向采样 |
| Env | `EnvWorker`（`rlinf/workers/env/env_worker.py`） | LIBERO 向量环境交互 |
| 编排 | `EmbodiedRunner`（`rlinf/runners/embodied_runner.py`） | 同步权重、驱动 interact / generate / 训练一步 |

**权重同步**：Actor 将完整 `state_dict` 发给 Rollout（`sync_model_to_rollout` / `sync_model_from_actor`），保证采样与训练使用同一套参数（在同步周期内）。

---

## 4. 单步训练循环（与 OpenPI 的关系）

`EmbodiedRunner.run()` 主循环大致为：

1. **同步权重**（按 `weight_sync_interval`）：Rollout 上 OpenPI 与 Actor 对齐。  
2. **`env.interact` + `rollout.generate`**：环境给观测 → Rollout 用 OpenPI 生成 chunk 动作与 logprob 等 → 轨迹送入 Actor。  
3. **`actor.recv_rollout_trajectories`**：接收并整理 batch。  
4. **`actor.compute_advantages_and_returns`**：GAE 等（与 YAML 中 `adv_type: gae` 等一致）。  
5. **`actor.run_training()`**：在 FSDP 包裹的 OpenPI 上做 PPO（`loss_type: actor_critic`）；价值头、`detach_critic_input` 等由 `OpenPi0Config` + `openpi_action_model.py` 实现。

详细时序见 `rlinf/runners/embodied_runner.py` 中 `run()`。

---

## 5. 环境与数据侧（LIBERO）

- 环境类型与任务套件在 `env/libero_spatial.yaml` 中定义（如 `env_type: libero`、`task_suite_name: libero_spatial`）。  
- 具体仿真与向量化封装在 **`rlinf/envs/libero/`**（如 `venv.py` 等），根据是否使用 Libero Pro/Plus 可能有分支导入。

---

## 6. 使用前检查清单

1. **`EMBODIED_PATH`** 指向 `examples/embodiment/config`。  
2. **`actor.model.model_path` 与 `rollout.model.model_path`** 已改为本机或 HF 上的真实 Pi0/Pi0.5 目录。  
3. **OpenPI / CUDA / FSDP** 与官方 Dockerfile或 `requirements` 一致；Pi0 系列注意 **勿随意打开 gradient checkpointing**（与 `openpi` 限制一致）。  
4. 若改用 **Pi0.5**：将 defaults 改为 `model/pi0_5@actor.model`，并准备对应 **`pi05_libero`** 权重与配置。

---

## 7. 相关文件索引（便于检索）

```
examples/embodiment/config/libero_spatial_ppo_openpi_quickstart.yaml  # 主配置
examples/embodiment/config/env/libero_spatial.yaml
examples/embodiment/config/model/pi0.yaml
examples/embodiment/config/model/pi0_5.yaml
examples/embodiment/train_embodied_agent.py

rlinf/runners/embodied_runner.py
rlinf/workers/actor/fsdp_actor_worker.py
rlinf/workers/rollout/hf/huggingface_worker.py

rlinf/models/__init__.py
rlinf/models/embodiment/openpi/__init__.py
rlinf/models/embodiment/openpi/openpi_action_model.py
rlinf/models/embodiment/openpi/dataconfig/__init__.py
rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py
rlinf/models/embodiment/openpi/policies/libero_policy.py

rlinf/envs/libero/
```

---

## 8. 可选：从 JAX 检查点转换

若手边是 OpenPI 的 JAX 格式，可参考仓库内转换脚本（路径以仓库为准）：

- `rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py`

转换后再将输出目录填到 `model_path` 即可接入上述链路。
