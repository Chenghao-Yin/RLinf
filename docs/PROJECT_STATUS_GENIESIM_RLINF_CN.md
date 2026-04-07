# GenieSim × RLinf 开源轻量化联合仿真 — 项目状态文档

> **目的**：全面记录本仓库"MuJoCo 物理 + IsaacSim 渲染 + 容器化 sim-server + RLinf 训练"方案的实现状态，方便新开发者接手和继续迭代。
>
> 本文档聚焦 **RL 训练用的轻量化开源方案**，不涉及 GenieSim benchmark/teleop 全量系统。
>
> **最后更新**：2026-04-04

---

## 1. 项目目标

把 GenieSim 的任务资产/配置能力复用到可并发的 VectorEnv 中，对接 RLinf embodied 训练框架，使其能像 robotwin/libero/isaaclab 一样被 `train_embodied_agent.py` 驱动，支持 RL 训练、评测、以及人类示范数据采集。

核心设计目标：
- **N 并发环境**：MuJoCo 多进程物理（每 env 独立进程），IsaacSim 单进程渲染（GridCloner）
- **高带宽图像不走 ROS**：相机帧通过 POSIX 共享内存（SHM）零拷贝传递
- **容器化 sim-server**：仿真进程全部运行在 Docker 容器内，宿主机无需安装 ROS/IsaacSim
- **Gym VectorEnv 兼容**：下游拿到标准 gymnasium 5-tuple（obs, reward, terminated, truncated, info），支持 auto-reset
- **多种控制模式**：关节空间（joint）和末端执行器空间（ee + IK）
- **人类示范采集**：SpaceMouse 遥操作 + CollectEpisode 录制，可直接生成训练数据

---

## 2. 当前架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│  宿主机（Host）                                                   │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  RLinf 训练 / 数据采集                                    │   │
│  │  train_embodied_agent.py / collect_sim_data.py           │   │
│  │                                                           │   │
│  │  GenieSimBaseEnv (RLinf/rlinf/envs/geniesim/)            │   │
│  │  ├── GenieSimShmClient — 薄客户端，读写 Step SHM         │   │
│  │  ├── JunpuPlaceWorkpieceEnv — 覆盖稠密 reward 计算      │   │
│  │  ├── SpacemouseSimIntervention (可选, 人类控制)            │   │
│  │  ├── CollectEpisode (可选, 示范录制)                      │   │
│  │  └── RecordVideo (可选, 视频录制)                         │   │
│  │                                                           │   │
│  │  SimContainerManager ──── Docker API ────────────────►   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                    │                    │                         │
│            POSIX SHM (3种)         sentinel files                │
│         /dev/shm/geniesim_*       /geniesim/main/                │
└───────────────────────────────────────────────────────────────── ┘
                    │                    │
┌─────────────────────────────────────────────────────────────────┐
│  Docker 容器（geniesim-rlinf:latest）                             │
│                                                                   │
│  sim_server.py (进程监督器 + step loop)                           │
│  ├── GenieSimVectorEnv（组织 obs/reward/term/trunc/info）         │
│  │   ├── ProcessManager                                          │
│  │   │   ├── mujoco_ros_node.py × N  （信号量同步模式）           │
│  │   │   └── rl_renderer.py × 1     （IsaacSim，GridCloner）     │
│  │   │                                                            │
│  │   │   进程间通信：                                              │
│  │   │   ├── ROS2 Jazzy: joint_command / joint_states / tf_render│
│  │   │   └── POSIX SHM: 帧 + 控制 + 步进 (3种)                  │
│  │   │                                                            │
│  │   ├── run_step_loop(): 读取 Step SHM 请求 → 步进 → 写回结果   │
│  │   └── 监督线程: 检测崩溃/停止信号                               │
│  └─────────────────────────────────────────────────────────────  │
└─────────────────────────────────────────────────────────────────┘
```

**通信方式说明（三种 SHM）：**
- **Frame SHM**（`geniesim_frames`，全局共享）：IsaacSim 渲染的相机帧，RLinf 只读
- **Ctrl SHM**（`geniesim_frames_ctrl_{i}`，每 env 独立）：MuJoCo 写 joint states + body poses，GenieSimVectorEnv 写 actions + 同步信号
- **Step SHM**（`geniesim_frames_step`，全局共享，**新增**）：请求-应答通道，RLinf 写 actions + reset_mask，sim 侧写 rewards/terminated/truncated/info
- **容器内**：MuJoCo ↔ IsaacSim 通过 ROS2 Jazzy（`/env_i/tf_render`、`/env_i/joint_states` 等）
- **宿主机不需要 ROS**：所有必要数据均通过共享内存交换

**同步步进协议（新增）：**
1. RLinf 侧写 actions 到 Step SHM，设 `STEP_PHASE_STEP_REQUEST`
2. GenieSimVectorEnv 读 actions → 写入 Ctrl SHM → 设 `MUJOCO_PHASE_GO`
3. MuJoCo 执行 N 步物理 → 写 states/body_poses → 设 `MUJOCO_PHASE_DONE`
4. GenieSimVectorEnv 读结果 → 计算 reward/terminated/truncated → 写回 Step SHM → 设 `STEP_PHASE_STEP_DONE`
5. RLinf 侧读取结果（GenieSimShmClient 只做读取，不二次加工）

---

## 3. 代码仓库结构

### 3.1 `main/`（仿真侧）

| 路径 | 作用 |
|------|------|
| `main/source/geniesim/rl/mujoco/mujoco_ros_node.py` | MuJoCo 物理节点：信号量同步模式（等待 GO → 执行 N 步 → 设 DONE）+ ROS2 接口 |
| `main/source/geniesim/rl/renderer/rl_renderer.py` | IsaacSim 渲染器：GridCloner 多环境克隆 + TF 驱动 + SHM 相机帧输出（自由运行） |
| `main/source/geniesim/rl/envs/geniesim_vec_env.py` | `GenieSimVectorEnv`：sim 侧核心，组织 obs/reward/term/trunc/info，同步 MuJoCo 步进，管理 Step SHM |
| `main/source/geniesim/rl/envs/process_manager.py` | `ProcessManager`：子进程启停、日志重定向、就绪检测，传递 sync_mode/steps_per_step |
| `main/source/geniesim/rl/scripts/sim_server.py` | 容器内入口：创建 GenieSimVectorEnv + run_step_loop()，监督线程检测崩溃 |
| `main/source/geniesim/rl/renderer/shm_layout.py` | SHM 布局常量：Frame/Ctrl/Step 三种 SHM 的偏移量和大小计算 |
| `main/rl_ros_ws/` | ROS2 接口包工作区（`geniesim_rl_interfaces`） |
| `main/scripts/dockerfile_geniesim_rlinf` | Docker 镜像定义（IsaacSim + ROS Jazzy + MuJoCo） |
| `main/scripts/entrypoint_geniesim_rlinf.sh` | 容器 entrypoint（colcon build + 环境初始化） |
| `main/scripts/build_geniesim_rlinf_image.sh` | 镜像构建脚本 |

### 3.2 `RLinf/`（训练侧）

| 路径 | 作用 |
|------|------|
| `RLinf/rlinf/envs/geniesim/geniesim_env.py` | `GenieSimBaseEnv`：RLinf 侧 gym.Env 包装器，转发 Step SHM 结果，子类可覆盖 reward |
| `RLinf/rlinf/envs/geniesim/shm_client.py` | `GenieSimShmClient`：薄客户端，通过 Step SHM 发送 actions/reset，读取预组织好的结果 |
| `RLinf/rlinf/envs/geniesim/shm_layout.py` | SHM 布局常量本地副本（与容器侧 `shm_layout.py` 保持一致，含 Step SHM 定义） |
| `RLinf/rlinf/envs/geniesim/container_manager.py` | `SimContainerManager`：Docker 生命周期管理 + 就绪握手 |
| `RLinf/rlinf/envs/geniesim/tasks/` | 任务子类注册（`place_block_into_box`、`junpu_place_workpiece`） |
| `RLinf/examples/embodiment/config/env/` | GenieSim 任务级 YAML（含 EE/joint 控制参数、相机、容器路径） |
| `RLinf/rlinf/envs/wrappers/collect_episode.py` | `CollectEpisode`：示范数据录制（pickle / LeRobot 格式） |
| `RLinf/rlinf/envs/wrappers/record_video.py` | `RecordVideo`：MP4 视频录制 |
| `RLinf/rlinf/envs/wrappers/spacemouse_sim_intervention.py` | `SpacemouseSimIntervention` + `FakeSpaceMouseExpert` |
| `RLinf/examples/embodiment/collect_sim_data.py` | 人类示范采集主脚本 |
| `RLinf/examples/embodiment/convert_demos_to_buffer.py` | 将 pickle demo 转换为 `TrajectoryReplayBuffer` checkpoint（demo buffer 训练用） |
| `RLinf/examples/embodiment/config/geniesim_junpu_sac.yaml` | Junpu SAC + demo buffer 训练配置 |
| `RLinf/examples/embodiment/config/geniesim_junpu_ppo_mlp.yaml` | Junpu PPO + MLP 训练配置 |
| `RLinf/examples/embodiment/config/geniesim_ppo_mlp.yaml` | 通用 PPO + MLP 训练配置示例 |
| `RLinf/examples/embodiment/config/env/geniesim_*.yaml` | Hydra env 片段（供训练配置 defaults 引用） |
| `RLinf/docs/junpu_rl_training.md` | Junpu RL 训练完整使用指南 |
| `RLinf/rlinf/envs/geniesim/scripts/test_rlinf_integration.py` | RLinf 集成测试（reset + step 接口验证） |
| `RLinf/rlinf/envs/geniesim/scripts/test_spacemouse_sim.py` | SpaceMouse 采集集成测试（fake 模式 + 真实 sim） |

### 3.3 `main/source/geniesim/assets/`（MuJoCo 资产）

| 路径 | 作用 |
|------|------|
| `main/source/geniesim/assets/mujoco_scenes/` | MJCF 场景文件（junpu_place_workpiece.xml 等） + 机器人/环境 mesh |

---

## 4. 已实现功能（当前状态）

### 4.1 ✅ 容器化 sim-server（完整可用）

**这是与旧文档最大的变化**：仿真不再依赖宿主机的 ROS/IsaacSim 环境，全部在 Docker 容器内运行。

- `SimContainerManager`：
  - Docker 容器完整生命周期管理（创建、启动、等待就绪、停止、清理）
  - 文件哨兵握手协议（`.geniesim_ready`/`.geniesim_start`/`.geniesim_idle`）
  - `keep_alive` 模式：训练重启时不需要 Isaac Sim 重新冷启动（节省 1-2 分钟）
  - `reuse_running` 模式：复用已就绪的容器
  - 故障诊断：自动识别 GPU/CUDA、SHM 冲突、MJCF 路径错误等启动失败原因
  - 陈旧 SHM 清理（自动检测并通过临时容器删除遗留 SHM 段，含 step SHM）

- `sim_server.py`（容器内）：
  - 创建 `GenieSimVectorEnv` 实例，调用 `run_step_loop()` 处理来自 RLinf 的步进请求
  - 监督线程检测 IsaacSim 崩溃和停止信号，通过 `STEP_PHASE_CLOSE` 退出步进循环
  - SHM 权限管理（Isaac Sim 创建后 chmod 0o666 供宿主机访问）
  - 启动时自动清理残留的 Ctrl SHM 和 Step SHM

### 4.2 ✅ 双控制模式（joint + ee）

**关节模式（joint）**：
- 动作直接映射到 MuJoCo ctrl 数组（arm position actuators）
- 用于 `place_block_into_box` 任务

**末端执行器模式（ee）**：
- 动作格式：`[pos_l(3), euler_l(3), pos_r(3), euler_r(3), gripper_l, gripper_r]`，共 14 维
- 仿真侧（`mujoco_ros_node.py`）运行阻尼最小二乘雅可比 IK（`ik_max_iter=10`，`ik_damp=0.05`）
- 用于 `junpu_place_workpiece` 任务
- 状态向量（state_dim=40）额外包含 EE 位姿：末尾 12 维为 `[ee_l_pos(3), ee_l_rpy(3), ee_r_pos(3), ee_r_rpy(3)]`

### 4.3 ✅ 两个已注册任务

| 任务 ID | 类名 | 控制模式 | state_dim | Reward | 策略类型 |
|---------|------|----------|-----------|--------|---------|
| `place_block_into_box` | `PlaceBlockIntoBoxEnv` | joint（14D） | 28 | ADER（内置，可启用） | MLP |
| `junpu_place_workpiece` | `JunpuPlaceWorkpieceEnv` | ee（14D，IK） | 52（使用右臂 26 维） | 密集 reward：xy/z 距离 + 姿态 + 静止 + 成功奖励 | CNN（ResNet10）+ MLP |

`JunpuPlaceWorkpieceEnv` 特性：
- 通过 `info["body_poses"]` 中 `workpiece_r` 获取工件世界坐标，计算密集 reward
- 右手腕部相机（480×480 → 128×128）输入 CNN 策略
- 4 并行 MuJoCo 环境，DRQ 图像增强
- 10 个 Q-head 的 SAC 训练已验证通过

### 4.4 ✅ SpaceMouse 示范数据采集

新增完整的仿真内人类遥操作数据采集方案：
- `SpacemouseSimIntervention`：封装向量化 sim env，将 SpaceMouse 6D delta 转换为右臂绝对 EEF 目标，仅控制 env_0
- `FakeSpaceMouseExpert`：无需硬件的确定性测试模拟器，用于 CI/集成测试
- `collect_sim_data.py`：主采集脚本，支持 `--fake-spacemouse` 模式
- 与 `CollectEpisode` 深度集成：保存每步 `intervene_action`，兼容 LeRobot 格式

### 4.5 ✅ RLinf 训练框架对接

- `GenieSimBaseEnv` 完整实现 RLinf 要求的接口（reset/step/chunk_step/close/elapsed_steps）
- sim 侧 `GenieSimVectorEnv` 负责组织 obs/reward/terminated/truncated/info
- RLinf 侧 `GenieSimShmClient` 通过 Step SHM 读取预组织好的结果，无需二次加工
- 子类（如 `JunpuPlaceWorkpieceEnv`）可覆盖 reward 计算（通过 `info["body_poses"]` 获取 ground truth）
- `infos["episode"]` 结构与 IsaaclabBaseEnv 完全一致，RLinf logger 可直接复用
- 提供 PPO + MLP 训练配置示例（`geniesim_ppo_mlp.yaml`）
- 提供两个 Hydra env 片段（`geniesim_place_block_into_box.yaml`、`geniesim_junpu_place_workpiece.yaml`）

### 4.6 ✅ Junpu SAC RL 训练（完整端到端验证）

- **SAC + CNN 策略**：`geniesim_junpu_sac.yaml`，使用 ResNet10 图像编码器 + MLP 策略头
- **密集 Reward**：xy/z 距离 + 姿态偏差 + 静止奖励 + 成功奖励（通过 body_poses 计算）
- **右手腕部相机**：480×480 → 128×128 缩放，DRQ 图像增强
- **4 并行环境**：每 rollout epoch 收集 400 转移（100 步 × 4 环境）
- **10 个 Q-head**：更稳定的 Q 值估计
- **已修复的训练 bug**：
  - demo_buffer 死循环（`load_path: null` 时 DataLoader 无限等待 → 注释掉未使用的 demo_buffer 配置段）
  - replay buffer OOM（图像数据导致 1.38TB 预分配 → `_extract_images` 过滤 + 缩小 cache_size）
  - `final_info` KeyError（自定义 termination 未被 base env auto_reset 处理 → 移除 early termination）
  - ROS 接口未编译（Docker entrypoint 路径不匹配 → 手动 colcon build）
  - Stale sentinel files（`.geniesim_ready` 残留导致复用已死 sim_server → 启动前清理）
- 参考训练文档：`RLinf/docs/junpu_rl_training.md`

### 4.7 ✅ 宿主机侧 geniesim 依赖完全消除

- 新增 `shm_client.py`：纯 stdlib SHM 客户端，`GenieSimShmClient` 替代宿主机侧 `GenieSimVectorEnv`
- 新增 `shm_layout.py`：SHM 布局常量本地副本，无外部依赖
- `geniesim_env.py` 无任何 `from geniesim.*` import（含延迟 import 和 sys.path 注入均已移除）
- 用户运行 RLinf 训练**只需要** Docker + `.venv`，无需安装 ROS/geniesim/rclpy

### 4.8 ✅ Mode 2（全容器内运行）+ 合并镜像 + 统一配置

新增**合并镜像** `geniesim-rlinf-train:latest`（57.6 GB），将 GenieSim sim-server 与 RLinf 完整训练栈打包为单一容器，实现两种运行模式：

| 模式 | 说明 |
|------|------|
| **Mode 1（原方案）** | RLinf 在宿主机 `.venv` 运行，GenieSim 在独立 Docker 容器运行，通过 POSIX SHM + 哨兵文件通信 |
| **Mode 2（新方案）** | RLinf 和 GenieSim sim-server 均在同一容器内运行，sim-server 以 subprocess 方式启动 |

**Mode 2 核心实现**：
- `SimContainerManager` 新增 `mode: auto` 自动检测（通过 `GENIESIM_CONTAINER` 环境变量自动选择 local/docker）
- 子进程启动前剥离 `VIRTUAL_ENV`（避免 Python 3.11 venv 遮蔽系统 Python 3.12/rclpy）
- 哨兵文件协议与 Mode 1 完全一致，复用逻辑相同
- 新增 `run_local.sh` 便捷脚本（已 COPY 入镜像 `/usr/local/bin/`）
- 统一 env YAML（消除 `_local` 后缀），路径使用 `${oc.env:GENIESIM_ROOT}`，自动适配两种模式
- Docker mode 自动路径替换：`container_geniesim_root` 自动将宿主机路径转为容器内路径

**已修复的 Mode 2 bug**：
- 哨兵检查被 `_local_proc` guard 封锁 → 孤儿进程积累（每次启动新 sim）
- `_shutdown_local()` 在哨兵-idle 复用路径下提前返回 → MuJoCo 进程不关闭
- `_kill_orphan_sim_servers()` 误杀 PID 1（容器 init）
- `run_local.sh` 的 `set -u` 导致 ROS setup 失败
- MuJoCo viewer 在容器内强制启用导致 GLFW 崩溃 → `--viewer` 仅在 `headless=false` 且 `env_id==0` 时传递

### 4.9 ✅ 已通过的测试

| 测试 | 脚本/方式 | 状态 |
|------|-----------|------|
| 容器启动 + 基础 step | `test_rlinf_integration.py` | ✅ PASSED |
| SpaceMouse 采集（fake 模式，2 episodes） | `test_spacemouse_sim.py` | ✅ PASSED |
| pickle 文件写入验证 | 集成测试内 | ✅ PASSED（2×24MB） |
| EEF 位置变化验证 | 手动检查 | ✅ Y/Z 轴移动正确 |
| Demo 转换（30 demos → 1338 samples） | `convert_demos_to_buffer.py` | ✅ PASSED |
| SAC 训练 12 epoch（含 eval） | `train_embodied_agent.py` | ✅ PASSED |
| 无 geniesim 依赖下模块 import | 手动验证 | ✅ PASSED |
| **Mode 2** collect（2 demos） | 容器内手动验证 | ✅ PASSED（2026-04-02） |
| **Mode 2** convert_demos_to_buffer | 容器内手动验证 | ✅ PASSED |
| **Mode 2** PPO 训练（place_block，4 epoch，25→100%） | 容器内手动验证 | ✅ PASSED |
| **Mode 2** idle 复用（17s vs 87s 冷启动） | 容器内手动验证 | ✅ PASSED |
| **Mode 2** shutdown 后 sim 进程干净退出 | 容器内手动验证（bug 修复后） | ✅ PASSED |
| **Mode 2** 统一 YAML 自动检测模式（GENIESIM_CONTAINER） | 容器内手动验证 | ✅ PASSED |
| **Mode 2** 可视化（headless=false + X11 挂载）MuJoCo viewer | 容器内手动验证 | ✅ PASSED（2026-04-03） |
| **Mode 2** 可视化 Isaac Sim GUI | 容器内手动验证 | ✅ PASSED |
| **Mode 1** collect + SAC 训练（keep_alive 复用） | 宿主机手动验证 | ✅ PASSED |
| **Mode 2** SAC MLP 训练（junpu，4 env，20 epoch） | 容器内手动验证 | ✅ PASSED（2026-04-04） |
| **Mode 2** SAC CNN 训练（junpu，ResNet10，4 env，DRQ） | 容器内手动验证 | ✅ PASSED（2026-04-04） |
| **Mode 2** multi-env（4 并行 MuJoCo 进程） | 容器内手动验证 | ✅ PASSED（2026-04-04） |
| **Mode 2** 腕部相机（Right_Camera，480×480→128×128） | 容器内手动验证 | ✅ PASSED（2026-04-04） |

---

## 5. 待完成（TODO）

### 5.1 🔴 高优先级

| 项目 | 说明 |
|------|------|
| `_build_infos` 重复计算 bug | `geniesim_vec_env.py` 中 `_build_infos()` 二次调用 `compute_reward()` 推进 ADER 状态机，`enable_reward=true` 时 reward 会算错，需缓存上次结果 |
| IsaacSim early return 风险 | 当 RL step 频率 < 30Hz 时（如同步步进导致 step 耗时增加），IsaacSim 的 `_render_callback` 会更频繁地 early return（因为 TF dirty flag 未更新），导致渲染帧不及时。需监控实际 step 频率与渲染帧率，必要时调整 `render_hz` 或 `steps_per_step` |

### 5.2 🟡 中优先级

| 项目 | 说明 |
|------|------|
| 新任务支持 | 每加一个任务需：MJCF + USD + body_map.json + 任务子类 + Hydra env 片段 |
| SHM 帧同步优化 | 当前忙等（self-spin），渲染器卡顿时会堵塞 2 秒，可改为 Unix socket 通知 |
| LeRobot 格式端到端测试 | `export_format=lerobot` 路径尚未完整测试 |
| shm_layout.py 同步机制 | 容器侧布局变更时宿主机需手动同步，建议 CI 加检查 |

### 5.3 🟢 低优先级

| 项目 | 说明 |
|------|------|
| `body_name_map` 自动生成工具 | 目前需手动配置 MuJoCo body → USD prim 映射 |
| `init_qpos` 按 body name 指定 | 当前只支持数组形式，复杂场景不方便 |
| Demo buffer 支持修复 | `embodied_buffer_dataset.py` 中 `demo_buffer is not None` 但 buffer 为空时会死循环，需增加 `load_path` 有效性检查 |

---

## 6. 快速验证指南

### 6.1 镜像构建（首次）

```bash
cd /home/zy/code/rlinf_open_source
bash main/scripts/build_geniesim_rlinf_image.sh
```

### 6.2 RLinf 集成测试（dry-run，无需 Docker）

```bash
cd /home/zy/code/rlinf_open_source/RLinf
GENIESIM_ROOT=.. .venv/bin/python3 \
  rlinf/envs/geniesim/scripts/test_rlinf_integration.py \
  --config examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml \
  --dry-run
```

### 6.3 RLinf 集成测试（完整，需要 Docker + GPU）

```bash
cd /home/zy/code/rlinf_open_source/RLinf
GENIESIM_ROOT=.. .venv/bin/python3 \
  rlinf/envs/geniesim/scripts/test_rlinf_integration.py \
  --config examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml \
  --num-envs 1 --steps 10
```

### 6.4 SpaceMouse 采集测试（fake 模式）

```bash
cd /home/zy/code/rlinf_open_source/RLinf
GENIESIM_ROOT=.. .venv/bin/python3 \
  rlinf/envs/geniesim/scripts/test_spacemouse_sim.py \
  --save-dir /tmp/spacemouse_test \
  --num-demos 2
```

### 6.5 Mode 2：全部在容器内运行

```bash
cd /home/zy/code/rlinf_open_source

docker run --rm -it --gpus all --ipc=host \
    -v $(pwd):/geniesim/main:rw \
    -v $(pwd)/main:/geniesim/main/main:rw \
    geniesim-rlinf-train:latest bash

# 容器内
run_local.sh python examples/embodiment/collect_sim_data.py \
    --config examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml \
    --save-dir /tmp/demos --num-demos 5 --fake-spacemouse

run_local.sh python examples/embodiment/train_embodied_agent.py \
    --config-path config \
    --config-name geniesim_ppo_mlp
```

### 6.6 Junpu SAC RL 训练（推荐）

```bash
cd /home/zy/code/rlinf_open_source/RLinf

# 1. 转换示范数据（一次性，需要 my_demos/*.pkl）
EMBODIED_PATH=$(pwd)/examples/embodiment \
  .venv/bin/python examples/embodiment/convert_demos_to_buffer.py \
  --demo-dir my_demos --output-dir /tmp/junpu_demo_buffer

# 2. SAC 训练（含 demo buffer）
GENIESIM_ROOT=/home/zy/code/rlinf_open_source \
EMBODIED_PATH=$(pwd)/examples/embodiment \
  .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name geniesim_junpu_sac \
  ++algorithm.demo_buffer.load_path=/tmp/junpu_demo_buffer

# 3. PPO 基准（不需要 demo）
GENIESIM_ROOT=/home/zy/code/rlinf_open_source \
EMBODIED_PATH=$(pwd)/examples/embodiment \
  .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path config \
  --config-name geniesim_junpu_ppo_mlp
```

**注意**：`--config-path` 必须是相对于脚本的路径（`config`，不是 `examples/embodiment/config`）；`EMBODIED_PATH` 必须指向 `examples/embodiment` 目录（Hydra 用于解析 `${oc.env:EMBODIED_PATH}`）。

---

## 7. 关键技术约束（给接手者的提示）

- **两种运行模式**：Mode 1（宿主机 `.venv` + Docker 容器 sim）和 Mode 2（`geniesim-rlinf-train:latest` 容器内全运行）；两种模式共用哨兵文件协议和 SHM 通信
- **Python 版本冲突（Mode 2 关键）**：RLinf venv 用 Python 3.11，sim_server.py 需要系统 Python 3.12（ROS Jazzy rclpy）；`_ensure_running_local()` 启动子进程前会自动剥离 `VIRTUAL_ENV`，手动启动 sim_server 时需先 `deactivate`
- **宿主机无 geniesim/ROS 依赖（Mode 1）**：宿主机只需 Docker + `.venv`；所有 geniesim/ROS 通信在容器内进行，宿主机通过 `GenieSimShmClient`（纯 stdlib）接入
- **shm_layout.py 需保持同步**：`RLinf/rlinf/envs/geniesim/shm_layout.py` 是容器侧 `shm_layout.py` 的副本，包含 Frame/Ctrl/Step 三种 SHM 布局，若容器侧常量变更，必须同步更新
- **三种 SHM 通信**：Frame SHM（相机帧）、Ctrl SHM（states + actions + body_poses + 同步信号）、Step SHM（请求-应答：actions/reset → rewards/terminated/truncated/info）
- **同步步进协议**：MuJoCo 进程现在等待 `MUJOCO_PHASE_GO` 信号，执行 `steps_per_step` 步物理后设 `MUJOCO_PHASE_DONE`，由 `GenieSimVectorEnv` 控制步进节奏
- **IsaacSim 自由运行**：IsaacSim 渲染器不受同步步进影响，仍然自由运行并通过 TF dirty flag 触发渲染
- **sim 侧 vs RLinf 侧职责分离**：sim 侧 `GenieSimVectorEnv` 负责组织 obs/reward/terminated/truncated/info 并写入 Step SHM；RLinf 侧 `GenieSimShmClient` 只负责读取，不做二次加工；但 RLinf 侧任务子类（如 `JunpuPlaceWorkpieceEnv`）可覆盖 reward 计算
- **SHM 跨容器**：容器启动时需 `--ipc=host`（共享 `/dev/shm`），SimContainerManager 已自动处理
- **Isaac Sim 冷启动**：首次/重启容器后 Isaac Sim 需要 1-3 分钟加载 USD 场景；`keep_alive=true` 可跳过这个等待
- **GIL 与 1000Hz**：Python MuJoCo + ROS spin 因 GIL 实际约 800-900Hz，对 RL 训练够用
- **EE 模式精度**：阻尼最小二乘 IK（10 次迭代）对普通操作精度足够，但大奇异位形附近可能不收敛
- **state_dim 统一为 40**：两个 junpu 配置文件（`configs/` 和 `examples/embodiment/config/env/`）现在均为 `state_dim=40`，MLP 策略 `obs_dim` 必须对应设为 40
- **chunk_step 不能委托给 self.env**：`GenieSimBaseEnv.chunk_step()` 必须循环调用 `self.step()`，不能直接调 `self.env.chunk_step()`，原因见 DEVELOPER_NOTES §5.4
- **SAC FSDP 参数**：`use_orig_params` 必须为 `false`，否则 Q-head 与 actor 参数形状冲突
- **Ray worker 任务注册**：任务子类必须在 `rlinf/envs/geniesim/__init__.py` 的 `_import_all_tasks()` 中 import，否则 Ray subprocess 中 `@register_geniesim_env` 不生效

---

## 8. 文件索引（最常改）

| 功能 | 文件 |
|------|------|
| 添加新任务 | `RLinf/rlinf/envs/geniesim/tasks/<task>.py` + `tasks/__init__.py`（`_import_all_tasks()`） |
| 任务配置（统一） | `RLinf/examples/embodiment/config/env/geniesim_<task>.yaml`（`mode: auto`，自动适配 Mode 1/2） |
| 训练 env 片段（统一） | `RLinf/examples/embodiment/config/env/geniesim_<task>.yaml` |
| 容器配置 | `container_cfg` 节在任务 YAML 内（`mode: auto`，自动检测 `GENIESIM_CONTAINER` 环境变量） |
| 训练配置 | `RLinf/examples/embodiment/config/geniesim_*.yaml` |
| Junpu reward 实现 | `RLinf/rlinf/envs/geniesim/tasks/junpu_place_workpiece.py`（通过 info["body_poses"] 计算 −L2 距离） |
| SHM 客户端（薄客户端） | `RLinf/rlinf/envs/geniesim/shm_client.py`（通过 Step SHM 请求-应答） |
| SHM 布局（双侧同步） | `RLinf/rlinf/envs/geniesim/shm_layout.py` ↔ `main/source/geniesim/rl/renderer/shm_layout.py`（含 Frame/Ctrl/Step 三种） |
| 容器管理（Mode 1 + Mode 2） | `RLinf/rlinf/envs/geniesim/container_manager.py` |
| Mode 2 便捷脚本 | `RLinf/docker/run_local.sh`（容器内路径 `/usr/local/bin/run_local.sh`） |
| 合并镜像 Dockerfile | `RLinf/docker/Dockerfile.geniesim` |
| 物理仿真 | `main/source/geniesim/rl/mujoco/mujoco_ros_node.py` |
| 渲染器 | `main/source/geniesim/rl/renderer/rl_renderer.py` |
| VectorEnv（容器侧） | `main/source/geniesim/rl/envs/geniesim_vec_env.py`（同步步进 + Step SHM 请求-应答循环） |
| RLinf 包装 | `RLinf/rlinf/envs/geniesim/geniesim_env.py`（转发 Step SHM 结果，子类可覆盖 reward） |
| SpaceMouse wrapper | `RLinf/rlinf/envs/wrappers/spacemouse_sim_intervention.py` |
| 数据采集脚本 | `RLinf/examples/embodiment/collect_sim_data.py` |
| Demo 转换工具 | `RLinf/examples/embodiment/convert_demos_to_buffer.py` |
| 训练使用指南（Junpu） | `RLinf/docs/junpu_rl_training.md` |
| 用户手册 | `USER_GUIDE_CN.md`（本仓库根目录） |
| 开发者文档 | `DEVELOPER_NOTES_CN.md`（本仓库根目录） |
| 基础镜像 Dockerfile | `main/scripts/dockerfile_geniesim_rlinf` |
