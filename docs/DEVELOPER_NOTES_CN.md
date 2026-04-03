# GenieSim × RLinf 开发者笔记

> 面向需要理解或扩展 GenieSim 集成的开发者。  
> 涵盖架构设计、核心代码路径、已知 bug 与修复、新任务接入方法。

---

## 目录

1. [整体架构](#1-整体架构)
2. [哨兵文件协议](#2-哨兵文件协议)
3. [SHM 通信布局](#3-shm-通信布局)
4. [SimContainerManager 详解](#4-simcontainermanager-详解)
5. [Mode 2（本地模式）实现细节](#5-mode-2本地模式实现细节)
6. [已修复的关键 Bug](#6-已修复的关键-bug)
7. [如何接入新任务](#7-如何接入新任务)
8. [镜像构建](#8-镜像构建)
9. [关键文件速查](#9-关键文件速查)

---

## 1. 整体架构

### 进程拓扑

```
RLinf 训练进程（Python 3.11）
    │
    ├── GenieSimVectorEnv.step()
    │       └── GenieSimShmClient  ← 写 Ctrl SHM，读 Frame SHM（stdlib 纯 SHM，无 ROS）
    │
    └── SimContainerManager        ← 管理 sim-server 生命周期
            │
            ├── Mode 1（docker）: docker run → 独立容器
            │       Container（Python 3.12）：
            │           sim_server.py
            │               ├── mujoco_ros_node.py×N  ← MuJoCo 物理（1000 Hz）
            │               └── Isaac Sim renderer    ← 渲染（30 Hz）→ 写 Frame SHM
            │
            └── Mode 2（local）: subprocess.Popen → 同容器内子进程
                    VIRTUAL_ENV 已剥离，子进程用系统 Python 3.12 + ROS rclpy
```

### 通信层

| 通信类型 | 实现 | 说明 |
|----------|------|------|
| 状态/动作 | POSIX SHM (`geniesim_frames_ctrl_N`) | 每个 env 实例一个 SHM，原子读写 |
| 图像帧 | POSIX SHM (`geniesim_frames`) | Isaac Sim → 训练侧，30 Hz |
| 生命周期 | 哨兵文件（见第 2 节） | 异步事件通知，不需要轮询网络 |
| Mode 2 子进程 | `subprocess.Popen` | sim_server.py 以 bash -c 启动，stdout → log 文件 |

---

## 2. 哨兵文件协议

所有哨兵文件位于 `geniesim_root/`（Mode 1: 容器内的 `/geniesim/main`；Mode 2: `/geniesim/main`）：

| 文件名 | 写入方 | 含义 |
|--------|--------|------|
| `.geniesim_ready` | sim_server.py | 仿真已就绪，可以开始 step |
| `.geniesim_idle` | sim_server.py | 仿真已停止（MuJoCo/Isaac Sim 进程已退出），等待下次启动 |
| `.geniesim_start` | SimContainerManager | 通知 sim_server 启动仿真进程 |
| `.geniesim_stop` | SimContainerManager | 通知 sim_server 停止仿真进程（但 sim_server.py 本身继续运行） |
| `.geniesim_error` | sim_server.py | 启动过程中发生错误 |
| `.geniesim_progress` | sim_server.py | 启动进度信息（JSON，用于进度提示） |
| `sim_config.json` | SimContainerManager | 传递 init_params（mjcf_path、physics_hz 等）给 sim_server |

### 协议状态机

```
           写 sim_config.json + .geniesim_start
                    ↓
[等待]  ──────────────────────────────────────→  [启动中]
                                                      │
                            sim_server 写 .geniesim_ready
                                                      ↓
                                                   [就绪]  ← step() 在这里运行
                                                      │
                          SimContainerManager 写 .geniesim_stop
                                                      ↓
                                                  [停止中]
                                                      │
                            sim_server 写 .geniesim_idle
                                                      ↓
                                                   [空闲]  ← keep_alive=true 时在这里等待
```

---

## 3. SHM 通信布局

相关代码：`rlinf/envs/geniesim/shm_layout.py`、`rlinf/envs/geniesim/shm_client.py`

### Frame SHM（图像）

名称：`geniesim_frames`（可通过 `shm_name` 配置）

```
[帧头 header_size bytes] [图像数据 N×H×W×C bytes]
```

### Ctrl SHM（状态/动作）

名称：`geniesim_frames_ctrl_<env_idx>`

每个 env 实例有独立的 Ctrl SHM，包含：
- 状态观测（`state_dim` 维浮点）
- 动作指令（`action_dim` 维浮点）
- 重置标志、终止标志、奖励

**注意**：两侧（RLinf 训练 + sim_server）必须使用相同的 `state_dim` / `action_dim` / `shm_name`。G2 机器人：`state_dim=40, action_dim=14`。

---

## 4. SimContainerManager 详解

**文件**：`rlinf/envs/geniesim/container_manager.py`

### 初始化参数

来自 `container_cfg`（YAML 中的 `container_cfg` 节点）：

| 参数 | 类型 | 说明 |
|------|------|------|
| `mode` | str | `"auto"`（默认）、`"docker"` 或 `"local"`；auto 自动检测 `GENIESIM_CONTAINER` 环境变量 |
| `image` | str | Docker 镜像名（docker mode） |
| `name` | str | 容器名称（docker mode） |
| `geniesim_root` | str | rlinf_open_source/ 的路径（`${oc.env:GENIESIM_ROOT}`） |
| `container_geniesim_root` | str | 容器内 GENIESIM_ROOT（默认 `/geniesim/main`，docker mode 自动路径替换） |
| `ros_ws_install` | str | ROS 2 workspace install 路径（local mode） |
| `keep_alive` | bool | true=训练结束后只停止仿真进程，sim_server 保持运行 |
| `reuse_running` | bool | true=如已有运行中的 sim 则复用 |
| `startup_timeout_sec` | int | 等待 .geniesim_ready 的超时（秒） |
| `ros_domain_id` | int | ROS 2 域 ID（需与 init_params 一致） |

### 核心方法调用链

```
GenieSimVectorEnv.__init__()
    └── SimContainerManager.ensure_running(pm_kwargs)
            ├── Mode 1: _check_docker_available() → docker run/reuse
            └── Mode 2: _ensure_running_local(pm_kwargs)

GenieSimVectorEnv.close()
    └── SimContainerManager.shutdown()
            ├── Mode 1: docker stop/rm
            └── Mode 2: _shutdown_local()
```

### `_wait_ready()` 逻辑

轮询 `.geniesim_ready` 文件，每 5 秒读取 `.geniesim_progress` 打印进度，直到超时：

```python
while not ready_file.exists():
    if error_file.exists():
        raise RuntimeError(...)
    if mode == "local" and self._local_proc.poll() is not None:
        raise RuntimeError("sim_server.py exited prematurely")
    _print_progress()
    time.sleep(1)
```

---

## 5. Mode 2（本地模式）实现细节

### `_ensure_running_local()` 执行路径

```
_ensure_running_local(pm_kwargs)
    │
    ├── [reuse_running=true] 检查哨兵文件（sentinel-first，不依赖 _local_proc）
    │       ├── .geniesim_ready 存在 + SHM 可访问 → return（直接复用）
    │       └── .geniesim_idle 存在 → 删除哨兵，写 .geniesim_start → _wait_ready() → return
    │                                   （注意：此路径不设置 _local_proc！）
    │
    ├── 若 _local_proc 非 None 且仍在运行 → terminate
    │
    ├── _kill_orphan_sim_servers()
    │       └── 扫描 /proc，杀掉所有 sim_server.py 进程
    │           （仅在无哨兵文件时执行，避免误杀受管理的进程）
    │
    ├── 清理旧哨兵文件和旧 SHM
    │
    ├── 写 sim_config.json（从 pm_kwargs 构建）
    │
    ├── 构建 bash_cmd：
    │       source /opt/ros/jazzy/setup.bash
    │       source <ros_ws_install>/setup.bash
    │       exec python3 sim_server.py --config-json ... --ready-file ...
    │
    ├── 剥离 VIRTUAL_ENV（关键！避免 Python 3.11 venv 遮蔽系统 Python 3.12）
    │       env.pop("VIRTUAL_ENV")
    │       env["PATH"] = <PATH 中去掉 venv bin 目录>
    │
    ├── subprocess.Popen(["bash", "-c", bash_cmd], env=env, stdout=log_file)
    │       → self._local_proc = <Popen 对象>
    │
    └── _wait_ready()
```

### `_shutdown_local()` 执行路径

```
_shutdown_local()
    │
    ├── [keep_alive=true]
    │       proc_alive = _local_proc is not None and _local_proc.poll() is None
    │       sim_active = _ready_file.exists() or proc_alive      ← 关键修复点
    │       if not sim_active: return
    │       if _idle_file.exists(): return
    │       写 .geniesim_stop → 等待 .geniesim_idle（最多 60 秒）
    │
    └── [keep_alive=false]
            _ready_file.unlink()
            if _local_proc is not None: terminate/kill _local_proc
            _kill_orphan_sim_servers()
```

### Python 版本冲突（为什么要剥离 VIRTUAL_ENV）

| 组件 | Python 版本 | 原因 |
|------|-------------|------|
| RLinf 训练代码 | 3.11（venv） | torch 2.6+, flash-attn, ray 在 3.11 下编译 |
| sim_server.py | 3.12（系统） | ROS Jazzy rclpy 只有 Python 3.12 版本 |
| mujoco_ros_node.py | 3.12（系统） | 同上 |
| Isaac Sim Python | 3.11.13（独立） | NVIDIA Isaac Sim 内置，独立路径 `/isaac-sim/python.sh` |

如果 RLinf venv 激活状态下直接 `subprocess.Popen(["python3", "sim_server.py"])`，`PATH` 中 venv 的 `bin/python3`（3.11）会排在系统 Python 3.12 前面，导致 `import rclpy` 失败。

**修复**：`_ensure_running_local()` 在 `subprocess.Popen` 前执行：
```python
_venv = env.pop("VIRTUAL_ENV", "")
env.pop("VIRTUAL_ENV_PROMPT", None)
if _venv:
    env["PATH"] = ":".join(
        p for p in env.get("PATH", "").split(":")
        if not p.startswith(_venv)
    )
```

---

## 5.1 可视化架构（headless 控制链路）

`headless` 参数完全由 YAML 配置文件决定，控制链路如下：

```
env YAML: headless: false
    │
    └── geniesim_env.py → pm_kwargs["headless"] = False
            │
            └── sim_manager_base._write_sim_config() → .sim_server_config.json
                    │
                    └── sim_server.py → ProcessManager(headless=False)
                            │
                            ├── _start_renderer(): IsaacSim 不传 --headless → GUI 窗口
                            │
                            └── _launch_mujoco(env_id):
                                    ├── env_id == 0 且 headless == False → --viewer → MuJoCo GUI
                                    └── env_id > 0 → 不传 --viewer → headless（无 GUI）
```

**多 env 并行时只有 env_0 可视化**：这是因为多个 MuJoCo 窗口同时运行会争抢 GPU 资源，且对调试无额外价值。

**容器可视化前提**：
```bash
docker run ... -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw ...
```

**MuJoCo viewer 参数传递**（`process_manager.py`）：
```python
if not self.headless and env_id == 0:
    cmd.append("--viewer")
```

**MuJoCo 节点处理**（`mujoco_ros_node.py`）：
```python
# --viewer 默认 False（store_true），纯粹由 CLI 参数决定
if args.viewer:
    handle = mujoco.viewer.launch_passive(...)   # GUI 窗口
else:
    _run_headless(...)                            # 无窗口
```

---

## 6. 已修复的关键 Bug

### Bug 1：孤儿进程积累（哨兵检查被 `_local_proc` guard 封锁）

**症状**：每次运行都启动新的 sim_server.py，旧进程变孤儿，多个 MuJoCo 窗口同时运行。

**根因**：`_ensure_running_local()` 将哨兵文件检查放在 `if self._local_proc is not None` 块内。由于 `_local_proc` 是实例变量，每个新 `SimContainerManager` 实例创建时都为 None，哨兵检查永远被跳过。

**修复**（`_ensure_running_local()` 顶部）：
```python
# sentinel-first 复用检查，不依赖 _local_proc（mirrors docker mode）
if self.reuse_running:
    if self._ready_file.exists():
        if self._check_shm_accessible(shm_name):
            return  # 直接复用
    elif self._idle_file.exists():
        # 唤醒空闲 sim_server
        self._idle_file.unlink(missing_ok=True)
        self._write_sim_config(pm_kwargs)
        self._start_file.write_text("start")
        self._wait_ready()
        return
```

### Bug 2：`_shutdown_local()` 在哨兵-idle 复用路径下提前返回

**症状**：第二次运行后 MuJoCo 进程没有关闭（`.geniesim_ready` 仍然存在，MuJoCo 进程仍在运行）。

**根因**：哨兵-idle 复用路径（`_ensure_running_local()` 的 `elif idle_file` 分支）不设置 `self._local_proc`。`_shutdown_local()` 检查：
```python
if self._local_proc is None or self._local_proc.poll() is not None:
    return  # ← 提前返回，没有写 .geniesim_stop！
```

**修复**（`_shutdown_local()` 中）：
```python
proc_alive = self._local_proc is not None and self._local_proc.poll() is None
sim_active = self._ready_file.exists() or proc_alive
if not sim_active:
    return
```

### Bug 3：`_kill_orphan_sim_servers()` 误杀 PID 1（容器 init）

**症状**：容器启动时，`_kill_orphan_sim_servers()` 尝试 `os.kill(1, 9)`，导致权限错误（PID 1 是容器 entrypoint，不可杀）。

**修复**：
```python
if pid <= 1 or pid == os.getpid():
    continue
```

### Bug 4：`set -u` 导致 ROS setup 失败

**症状**：`run_local.sh` 执行 `source /opt/ros/jazzy/setup.bash` 时报错 `AMENT_TRACE_SETUP_FILES: unbound variable`。

**修复**：`run_local.sh` 中 `set -euo pipefail` 改为 `set -eo pipefail`（去掉 `-u`，ROS setup 脚本使用了大量未绑定变量）。

### Bug 5：MuJoCo viewer 在容器内强制启用导致 GLFW 崩溃

**症状**：容器内运行时 `mujoco_ros_node.py` 持续崩溃重启（进程 crash → ProcessManager 重启 → 再次 crash），日志中出现 `ERROR: could not initialize GLFW`。ctrl SHM 在 resource_tracker 清理后丢失，shm_client 等待 180 秒超时。

**根因**：`mujoco_ros_node.py` 的 `main()` 中硬编码 `args.viewer = True`，无视命令行参数和容器环境。

**修复**：
1. `mujoco_ros_node.py`：移除 `args.viewer = True` 硬编码，完全依赖 `--viewer` CLI 参数
2. `process_manager.py`：`_launch_mujoco()` 中仅当 `headless=False` 且 `env_id==0` 时传 `--viewer`
3. YAML 默认 `headless: true`（安全默认值），需要可视化时用户显式设为 `false`

---

## 7. 如何接入新任务

### 7.1 创建环境配置（`examples/embodiment/config/env/`）

配置文件只需一份，路径使用 `${oc.env:GENIESIM_ROOT}` 自动适配容器内/外：

```yaml
init_params:
  id: my_new_task
  task_description: "描述任务目标"
  mjcf_path: ${oc.env:GENIESIM_ROOT}/main/source/geniesim/assets/mujoco_scenes/my_task.xml
  robot_type: G2
  robot_cfg: G2_omnipicker
  state_dim: 40        # 必须与 SHM layout 一致
  action_dim: 14       # 必须与 SHM layout 一致
  state_joint_offset: 16
  ctrl_offset: 24
  ctrl_offset_r: 38
  control_mode: joint
  ...

container_cfg:
  geniesim_root: ${oc.env:GENIESIM_ROOT}
  image: geniesim-rlinf:latest
  name: geniesim_rlinf_sim
  ros_ws_install: /geniesim/ros_ws_build/install
  keep_alive: true
  reuse_running: true
  startup_timeout_sec: 300
```

mode 默认为 `auto`，在容器内自动使用 local 模式，在宿主机自动使用 docker 模式。

### 7.2 创建训练配置（`examples/embodiment/config/`）

参考 `geniesim_ppo_mlp.yaml`，修改：
- `defaults` 中的 env 配置名
- `actor.model.obs_dim`、`actor.model.action_dim`
- `runner.logger.experiment_name`

### 7.3 实现 Reward 函数

在 `rlinf/envs/geniesim/` 中添加 reward 计算逻辑，或在任务的 Python 类中重写 `compute_reward()`。

参考 `rlinf/envs/geniesim/place_block_env.py`（place_block_into_box 的 reward 实现）。

### 7.4 注册任务

如果任务需要特殊的 env 类（非 `GenieSimVectorEnv` 默认行为），在 `rlinf/envs/geniesim/__init__.py` 中注册：

```python
TASK_ENV_MAP = {
    "place_block_into_box": PlaceBlockIntoBoxEnv,
    "my_new_task": MyNewTaskEnv,  # ← 新增
}
```

---

## 8. 镜像构建

### geniesim-rlinf-train:latest

```bash
cd /home/zy/code/rlinf_open_source/RLinf
docker build -f docker/Dockerfile.geniesim -t geniesim-rlinf-train:latest .
```

**构建层说明**：

```
geniesim-rlinf:latest（base，约 38.5 GB）
    ├── Ubuntu 24.04
    ├── Isaac Sim 5.1（/isaac-sim，Python 3.11.13，torch 2.7+cu128）
    ├── ROS 2 Jazzy（/opt/ros/jazzy）
    └── MuJoCo runtime

Dockerfile.geniesim 新增层（约 19 GB）
    ├── Python 3.11 venv（/opt/rlinf_venv/rlinf）
    │       torch 2.6.0+cu124, flash-attn 2.7.4, ray 2.54.1
    │       docker SDK 7.1.0, openvla-oft, transformers, prismatic
    ├── /usr/local/bin/run_local.sh（COPY from RLinf/docker/）
    └── /usr/local/bin/switch_env
```

### 版本关系

| 组件 | 版本 | 路径 |
|------|------|------|
| Isaac Sim Python | 3.11.13 + torch 2.7.0+cu128 | `/isaac-sim/python.sh` |
| RLinf venv | 3.11.14 + torch 2.6.0+cu124 | `/opt/rlinf_venv/rlinf` |
| 系统 Python（ROS） | 3.12 | `/usr/bin/python3` |
| ROS Jazzy rclpy | 系统包 | `/opt/ros/jazzy` |

---

## 9. 关键文件速查

| 文件 | 说明 |
|------|------|
| `rlinf/envs/geniesim/container_manager.py` | SimContainerManager：生命周期管理、自动模式检测、哨兵协议 |
| `rlinf/envs/geniesim/shm_client.py` | GenieSimShmClient：SHM 读写（纯 stdlib） |
| `rlinf/envs/geniesim/shm_layout.py` | SHM 内存布局常量 |
| `rlinf/envs/geniesim/env.py` | GenieSimVectorEnv：gym 接口包装器 |
| `examples/embodiment/config/env/geniesim_place_block_into_box.yaml` | place_block 环境配置（统一，自动适配两种模式） |
| `examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml` | junpu 环境配置（统一，自动适配两种模式） |
| `examples/embodiment/config/geniesim_ppo_mlp.yaml` | PPO 训练配置（place_block） |
| `examples/embodiment/config/geniesim_junpu_ppo_mlp.yaml` | PPO 训练配置（junpu） |
| `examples/embodiment/config/geniesim_junpu_sac.yaml` | SAC 训练配置（junpu） |
| `docker/Dockerfile.geniesim` | 合并镜像构建文件 |
| `docker/run_local.sh` | Mode 2 便捷启动脚本 |
