# GenieSim × RLinf 用户手册

> 面向希望在 GenieSim 仿真环境中用 RLinf 训练机器人策略的用户。
> **推荐运行方式**：Mode 2（全容器内运行），一条 `docker run` 即可完成采集、回放、训练。

---

## 目录

1. [架构概览](#1-架构概览)
2. [运行模式](#2-运行模式)
3. [环境准备](#3-环境准备)
4. [快速开始（容器内）](#4-快速开始容器内)
5. [采集示范数据](#5-采集示范数据)
6. [转换示范数据为回放缓冲区（SAC 专用）](#6-转换示范数据为回放缓冲区sac-专用)
7. [SAC 训练](#7-sac-训练)
8. [PPO 训练](#8-ppo-训练)
9. [监控训练](#9-监控训练)
10. [常见问题](#10-常见问题)

---

## 1. 架构概览

```
geniesim-rlinf-train:latest 容器（推荐）
┌─────────────────────────────────────────────────────┐
│  RLinf 训练/采集（Python 3.11 venv）                 │
│  └── SimContainerManager（mode: auto）               │
│             │ 自动检测 GENIESIM_CONTAINER 环境变量    │
│             │ → 容器内自动使用 local 模式             │
│             │ subprocess.Popen                       │
│             ▼                                        │
│  sim_server.py（系统 Python 3.12 + ROS Jazzy）       │
│  ├── MuJoCo 物理（N 个进程，1000 Hz）                │
│  └── IsaacSim 渲染（GridCloner，30 Hz）              │
│                      ↕  POSIX SHM（同机器）          │
└─────────────────────────────────────────────────────┘
```

**关键设计原则**：
- 容器内自给自足，**不需要 Docker daemon 权限**
- RLinf 训练侧**无 geniesim Python 依赖**，只用 stdlib 的 POSIX SHM
- sim-server 通过**哨兵文件协议**通知就绪/空闲/错误状态
- `keep_alive: true` 让 sim-server 在回合间保持运行，重启只需秒级（vs 首次 ~90 秒冷启动）
- **自动模式检测**：配置文件统一一份，代码自动判断容器内/外运行

---

## 2. 运行模式

系统支持两种运行模式，**推荐使用 Mode 2**：

| | Mode 2（推荐，容器内） | Mode 1（经典，宿主机） |
|--|--|--|
| **RLinf 运行位置** | 容器内 `/opt/rlinf_venv/rlinf` | 宿主机 `.venv` |
| **GenieSim 运行位置** | 同一容器内子进程 | 单独 Docker 容器 |
| **适用场景** | 日常开发、CI、部署 | 需要宿主机 IDE 调试仿真代码时 |
| **Docker 权限需求** | 不需要（容器内无 Docker-in-Docker） | 需要（`docker run` 启动 sim） |
| **启动脚本** | 容器内用 `run_local.sh` | 宿主机直接运行 |

**自动检测机制**：配置文件只有一份（无需区分 `_local` 后缀），代码通过 `GENIESIM_CONTAINER` 环境变量自动选择：
- 容器内 → `local` 模式（subprocess 启动 sim_server）
- 宿主机 → `docker` 模式（Docker API 启动容器）

---

## 3. 环境准备

### 3.1 构建 Docker 镜像

```bash
# Step 1：构建基础 GenieSim 镜像
cd /path/to/rlinf_open_source
bash main/scripts/build_geniesim_rlinf_image.sh
# 得到 geniesim-rlinf:latest（~38.5 GB）

# Step 2：构建合并镜像（包含 RLinf 训练栈）
cd /path/to/rlinf_open_source/RLinf
docker build -f docker/Dockerfile.geniesim -t geniesim-rlinf-train:latest .
# 得到 geniesim-rlinf-train:latest（~57.6 GB）
# 构建时间约 20-40 分钟（编译 flash-attn 等）
```

验证：
```bash
docker images | grep geniesim-rlinf
# geniesim-rlinf-train  latest  ...  57.6GB
# geniesim-rlinf        latest  ...  38.5GB
```

### 3.2 启动容器

**Headless 模式（默认，训练用）**：
```bash
cd /path/to/rlinf_open_source

docker run --rm -it --gpus all --ipc=host \
    -v $(pwd):/geniesim/main:rw \
    -v $(pwd)/main:/geniesim/main/main:rw \
    geniesim-rlinf-train:latest bash
```

**可视化模式（调试/演示用）**：
```bash
cd /path/to/rlinf_open_source

docker run --rm -it --gpus all --ipc=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd):/geniesim/main:rw \
    -v $(pwd)/main:/geniesim/main/main:rw \
    geniesim-rlinf-train:latest bash
```

> **可视化说明**：挂载 X11 后，在 YAML 中设 `headless: false`（或通过命令行 `init_params.headless=false`），MuJoCo（仅 env_0）和 Isaac Sim 都会打开可视化窗口。

**卷挂载说明**：

| 宿主机路径 | 容器内路径 | 说明 |
|--|--|--|
| `rlinf_open_source/` | `/geniesim/main` | geniesim_root，含 RLinf、哨兵文件、MuJoCo 资产 |
| `rlinf_open_source/main/` | `/geniesim/main/main` | GenieSim 主仓库（sim_server.py） |

---

## 4. 快速开始（容器内）

以下操作均在容器内执行，使用 `run_local.sh` 自动设置所有环境变量。

### 4.1 仅测试 Python 接口（dry-run，无需 GPU 渲染）

```bash
run_local.sh python rlinf/envs/geniesim/scripts/test_spacemouse_sim.py --dry-run
```

预期输出：
```
[test] Imports OK ✓
[test] FakeSpaceMouseExpert unit test OK ✓
[test] Config loaded: task=junpu_place_workpiece, action_dim=14 ✓
[test] Dry-run mode: skipping simulation launch.
[test] PASSED (dry-run) ✓
```

### 4.2 完整集成测试（需要 GPU，首次约 2-3 分钟）

```bash
run_local.sh python rlinf/envs/geniesim/scripts/test_spacemouse_sim.py \
    --save-dir /tmp/spacemouse_test \
    --num-demos 2
```

> **提示**：第二次运行时（sim-server 仍保持空闲），启动时间约 10-17 秒。

### 4.3 Mode 2 的工作原理

```
run_local.sh python train_embodied_agent.py
     │
     └── SimContainerManager（mode: auto → local）
              │ 检测到 GENIESIM_CONTAINER=1
              │
              ├── 检查 .geniesim_idle 或 .geniesim_ready 哨兵
              │     └── 若已有进程：复用（秒级重启）
              │     └── 若无：subprocess.Popen("bash -c 'source /opt/ros/jazzy/setup.bash; python3 sim_server.py'")
              │
              └── 注意：VIRTUAL_ENV 已从子进程 env 中剥离
                        （sim_server.py 需要系统 Python 3.12 + ROS rclpy）
```

---

## 5. 采集示范数据

### 5.1 脚本专家（--fake-spacemouse，流程验证）

```bash
# 容器内
run_local.sh python examples/embodiment/collect_sim_data.py \
    --config examples/embodiment/config/env/geniesim_place_block_into_box.yaml \
    --save-dir /tmp/demos \
    --num-demos 5 \
    --fake-spacemouse
```

脚本专家会自动执行固定轨迹（用于测试流程是否通畅）。

### 5.2 真实 SpaceMouse

```bash
# 容器内（需要将 SpaceMouse 设备透传到容器）
run_local.sh python examples/embodiment/collect_sim_data.py \
    --config examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml \
    --save-dir /tmp/demos \
    --num-demos 20
```

操作说明：
- 推/拉 SpaceMouse 控制末端执行器
- **左键**：标记当前轨迹为**成功**并保存
- **右键**：放弃当前轨迹，重新开始

### 5.3 Mode 1（宿主机运行，可选）

如需在宿主机运行采集（例如调试 IDE），确保已配置宿主机 `.venv`：

```bash
cd /path/to/rlinf_open_source/RLinf
export GENIESIM_ROOT=/path/to/rlinf_open_source

.venv/bin/python examples/embodiment/collect_sim_data.py \
    --config examples/embodiment/config/env/geniesim_place_block_into_box.yaml \
    --save-dir my_demos --num-demos 5 --fake-spacemouse
```

代码会自动检测到宿主机环境（无 `GENIESIM_CONTAINER`），使用 docker 模式。

---

## 6. 转换示范数据为回放缓冲区（SAC 专用）

SAC 算法需要将 `.pkl` 示范文件转换为统一格式的回放缓冲区：

```bash
# 容器内
run_local.sh python examples/embodiment/convert_demos_to_buffer.py \
    --config examples/embodiment/config/env/geniesim_place_block_into_box.yaml \
    --demo-dir /tmp/demos \
    --output-path /tmp/demo_buffer.pkl
```

---

## 7. SAC 训练

```bash
# 容器内
run_local.sh python examples/embodiment/train_embodied_agent.py \
    --config-path config \
    --config-name geniesim_junpu_sac

# 带 demo buffer
run_local.sh python examples/embodiment/train_embodied_agent.py \
    --config-path config \
    --config-name geniesim_junpu_sac \
    ++algorithm.demo_buffer.load_path=/tmp/demo_buffer.pkl
```

> **注意**：`--config-path config` 是相对于 `train_embodied_agent.py` 所在目录（`examples/embodiment/`）的路径。

**SAC 关键参数（`geniesim_junpu_sac.yaml`）**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algorithm.gamma` | 0.99 | 未来奖励折扣率 |
| `algorithm.tau` | 0.005 | 目标网络软更新速率 |
| `algorithm.target_entropy` | -14 | 目标熵（约等于 -action_dim） |
| `algorithm.update_epoch` | 8 | 每轮梯度更新次数 |

---

## 8. PPO 训练

```bash
# 容器内，place_block_into_box 任务
run_local.sh python examples/embodiment/train_embodied_agent.py \
    --config-path config \
    --config-name geniesim_ppo_mlp

# 容器内，junpu_place_workpiece 任务
run_local.sh python examples/embodiment/train_embodied_agent.py \
    --config-path config \
    --config-name geniesim_junpu_ppo_mlp
```

**PPO 关键参数（`geniesim_ppo_mlp.yaml`）**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algorithm.gamma` | 0.99 | 未来奖励折扣率 |
| `algorithm.gae_lambda` | 0.95 | GAE 优势估计平滑系数 |
| `algorithm.clip_ratio_high/low` | 0.2 | 策略更新幅度限制 |
| `algorithm.update_epoch` | 8 | 每批数据重复利用次数 |
| `env.train.max_steps_per_rollout_epoch` | 4 | 每训练轮采集步数 |
| `env.eval.max_steps_per_rollout_epoch` | 300 | 每评估轮最大步数（完整回合） |

---

## 9. 监控训练

### TensorBoard

```bash
# 训练日志默认保存在 ../results/（相对于 RLinf 根目录）
tensorboard --logdir ../results

# 或指定具体实验
tensorboard --logdir ../results/geniesim_ppo_mlp
```

### 关键指标

| 指标 | 说明 |
|------|------|
| `train/reward_mean` | 平均回报（越高越好） |
| `train/episode_success_rate` | 任务成功率 |
| `train/value_loss` | Critic 损失（应逐渐下降） |
| `train/policy_loss` | Actor 损失 |
| `train/kl_divergence` | 策略更新幅度（PPO 应 <0.05） |

### sim-server 日志

```bash
# 容器内查看 sim-server 日志
tail -f /geniesim/main/sim_server_local.log

# Mode 1（宿主机运行时）查看容器日志
docker logs -f geniesim_rlinf_sim
```

---

## 10. 常见问题

### sim-server 启动超时（300 秒）

**原因**：Isaac Sim 首次冷启动需要加载 USD 场景和 GPU 初始化，约 90 秒。
**解决**：
- 确认 GPU 可用：`nvidia-smi`
- 检查 sim-server 日志：`tail -100 /geniesim/main/sim_server_local.log`
- 延长超时：在 YAML 中设置 `startup_timeout_sec: 600`

### `rclpy` 导入失败

**原因**：RLinf venv（Python 3.11）与 ROS Jazzy（需要 Python 3.12 系统 rclpy）版本冲突。
**解决**：已在代码中自动处理——子进程启动前会剥离 `VIRTUAL_ENV` 环境变量。如果手动启动 sim_server.py，需先 `deactivate` RLinf venv。

### `4 is not divisible by <batch_size>`

**原因**：训练配置中的 `micro_batch_size` 或 `global_batch_size` 大于单轮实际采集的步数。
**解决**：在命令行覆盖批大小，例如：
```bash
run_local.sh python examples/embodiment/train_embodied_agent.py \
    --config-path config \
    --config-name geniesim_ppo_mlp \
    actor.micro_batch_size=4 actor.global_batch_size=4
```

### 容器内找不到 `run_local.sh`

`run_local.sh` 已在 Dockerfile 构建时复制到 `/usr/local/bin/run_local.sh`（需要镜像重新构建）。
临时替代方案：
```bash
source /opt/rlinf_venv/rlinf/bin/activate
export GENIESIM_ROOT=/geniesim/main
export GENIESIM_CONTAINER=1
export PYTHONPATH=/geniesim/main/RLinf:${PYTHONPATH:-}
source /opt/ros/jazzy/setup.bash
source /geniesim/ros_ws_build/install/setup.bash
cd /geniesim/main/RLinf
```

### `AMENT_TRACE_SETUP_FILES: unbound variable`

**原因**：`run_local.sh` 之前使用 `set -euo pipefail`，`-u` 导致 ROS setup 脚本中的未绑定变量报错。
**修复**：已改为 `set -eo pipefail`（去掉 `-u`）。

### 可视化窗口无法显示

**现象**：YAML 中 `headless: false` 但没有弹出 MuJoCo viewer 窗口。
**检查清单**：
1. 确认容器启动时挂载了 X11：`-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw`
2. 确认宿主机允许 X11 转发：`xhost +local:`
3. 检查容器内 `echo $DISPLAY` 不为空
4. 查看 MuJoCo 日志：`cat /tmp/geniesim_logs/mujoco_env_0.log`，如果有 `GLFW` 错误说明 X11 连接失败

> **提示**：多 env 并行时只有 env_0 的 MuJoCo 打开 viewer 窗口，其余 env 始终 headless。Isaac Sim renderer 也会根据 `headless` 设置决定是否显示 GUI。

### MuJoCo 窗口/进程在运行结束后没有关闭

**原因**（历史 bug，已修复）：`_shutdown_local()` 仅检查 `self._local_proc` 是否为 None，但在哨兵-idle 复用路径下 `_local_proc` 不会被设置，导致 shutdown 提前返回。
**修复**：现已改为检查 `_ready_file.exists()`，与 `_local_proc` 状态无关。
