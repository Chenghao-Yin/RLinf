# GenieSim × RLinf 项目状态

> 最后更新：2026-04-03

---

## 当前状态：已完成并验证

GenieSim 仿真环境与 RLinf 训练框架的集成**全部完成**，两种运行模式均经过端到端验证。

---

## 已完成功能

### 阶段 1：基础集成（Mode 1）

| 功能 | 状态 | 说明 |
|------|------|------|
| GenieSimShmClient（SHM 通信） | ✅ 完成 | 纯 stdlib，无 geniesim 依赖 |
| SHM 布局定义 | ✅ 完成 | state_dim=40, action_dim=14 |
| SimContainerManager（docker mode） | ✅ 完成 | 自动启动/复用/关闭容器 |
| 哨兵文件协议 | ✅ 完成 | ready/idle/start/stop/error/progress |
| place_block_into_box 环境配置 | ✅ 完成 | `examples/embodiment/config/env/` |
| collect_sim_data 脚本 | ✅ 完成 | 支持真实 SpaceMouse 和脚本专家 |
| convert_demos_to_buffer 脚本 | ✅ 完成 | 示范数据 → SAC 回放缓冲区 |

### 阶段 2：RL 训练配置（junpu 任务）

| 功能 | 状态 | 说明 |
|------|------|------|
| state_dim 修复（40 维） | ✅ 完成 | 验证 arm 关节 pos+vel 布局 |
| SAC 训练配置 | ✅ 完成 | `geniesim_junpu_sac.yaml` |
| PPO 训练配置 | ✅ 完成 | `geniesim_junpu_ppo_mlp.yaml` |
| Reward 占位实现 | ✅ 完成 | 时间惩罚 -0.01/步，成功占位 |
| MLP 策略（G2-qpos） | ✅ 完成 | obs=40, action=14, hidden=256 |

### 阶段 3：合并镜像（geniesim-rlinf-train:latest）

| 功能 | 状态 | 说明 |
|------|------|------|
| Dockerfile.geniesim | ✅ 完成 | Base geniesim-rlinf:latest + RLinf venv |
| Python 3.11 venv | ✅ 完成 | torch 2.6+cu124, flash-attn, ray 2.54 |
| docker SDK 集成 | ✅ 完成 | SimContainerManager 在容器内可调用 docker |
| 镜像大小 | ✅ 57.6 GB | base 38.5 GB + RLinf 层 ~19 GB |
| 构建验证 | ✅ 完成 | 2026-04-02 |

### 阶段 4：Mode 2（全部在容器内运行）+ 统一配置

| 功能 | 状态 | 说明 |
|------|------|------|
| `mode: auto` 自动检测 | ✅ 完成 | 通过 `GENIESIM_CONTAINER` 环境变量自动选择 local/docker |
| Python 版本冲突修复 | ✅ 完成 | 剥离 VIRTUAL_ENV，子进程用系统 Python 3.12 |
| `run_local.sh` 便捷脚本 | ✅ 完成 | 设置全部 env var，`exec "$@"` |
| Dockerfile 集成 run_local.sh | ✅ 完成 | COPY 到 `/usr/local/bin/run_local.sh` |
| 统一 env YAML（消除 `_local` 后缀） | ✅ 完成 | 路径使用 `${oc.env:GENIESIM_ROOT}`，自动适配两种模式 |
| Docker mode 自动路径替换 | ✅ 完成 | `container_geniesim_root` 自动将宿主机路径转为容器内路径 |

### 阶段 5：孤儿进程 bug 修复

| Bug | 状态 | 说明 |
|-----|------|------|
| 哨兵检查被 `_local_proc` guard 封锁 | ✅ 修复 | sentinel-first 检查，不依赖 _local_proc |
| `_shutdown_local()` 在 idle 复用路径提前返回 | ✅ 修复 | 改用 `_ready_file.exists()` 判断 sim 状态 |
| `_kill_orphan_sim_servers()` 误杀 PID 1 | ✅ 修复 | 添加 `if pid <= 1: continue` guard |
| `set -u` 导致 ROS setup 失败 | ✅ 修复 | `run_local.sh` 改为 `set -eo pipefail` |
| MuJoCo viewer 容器内强制启用致 GLFW 崩溃 | ✅ 修复 | `--viewer` 仅在 `headless=false` 且 `env_id==0` 时传递 |

### 阶段 6：可视化支持

| 功能 | 状态 | 说明 |
|------|------|------|
| `headless` 完全由 YAML 配置决定 | ✅ 完成 | 不再自动检测 DISPLAY，配置文件即最终设定 |
| MuJoCo viewer 只开 env_0 | ✅ 完成 | `process_manager.py` 仅 env_id==0 传 `--viewer` |
| Isaac Sim GUI 联动 | ✅ 完成 | `headless=false` 时 renderer 也非 headless |
| 容器 X11 挂载验证 | ✅ 通过 | `-e DISPLAY -v /tmp/.X11-unix` |

---

## 端到端验证结果（2026-04-02 / 2026-04-03）

### Mode 1（宿主机 RLinf + 容器 GenieSim）

| 测试项 | 结果 |
|--------|------|
| collect_sim_data（place_block，2 demos） | ✅ 通过 |
| SAC 训练（launch + rollout epochs） | ✅ 通过 |
| 容器从 idle 状态重启 sim 进程 | ✅ 通过 |

### Mode 2（全部在容器内）

| 测试项 | 结果 |
|--------|------|
| collect_sim_data（place_block，2 demos） | ✅ 通过 |
| convert_demos_to_buffer（2 demos → buffer） | ✅ 通过 |
| PPO 训练（4 epochs，25→100%，metrics 正常） | ✅ 通过 |
| sim_server.py 子进程干净启动和停止 | ✅ 通过 |
| VIRTUAL_ENV 剥离（Python 3.12 正确用于 rclpy） | ✅ 通过 |
| 第二次运行复用空闲 sim_server（17s vs 87s 冷启动） | ✅ 通过 |
| 运行结束后 sim 进程正确停止（不留孤儿） | ✅ 通过（修复后） |
| 统一 YAML 自动检测模式（GENIESIM_CONTAINER） | ✅ 通过 |
| 可视化模式（headless=false + X11 挂载）MuJoCo viewer 正常 | ✅ 通过 |
| 可视化模式 Isaac Sim GUI 正常 | ✅ 通过 |

---

## 架构设计决策

### 为什么不直接合并镜像（单一 Dockerfile）

见 `docs/geniesim_docker_integration_feasibility_CN.md`。简要：
- Isaac Sim 需要 Ubuntu 24.04 + NVIDIA Driver 550+，历史 RLinf 镜像基于 Ubuntu 22.04（已升级）
- Isaac Sim 镜像 38.5 GB，加上 RLinf 依赖总计 57.6 GB，已属合理范围
- **GPU 资源竞争是主要顾虑**：Isaac Sim 渲染与 VLA 训练共享 GPU，实际测试中 MLP 策略（非 VLA）不存在显存冲突

### 为什么选择哨兵文件而非 socket/RPC

- 哨兵文件对网络配置零依赖
- sim_server.py 和训练代码在不同 Python 环境（3.12 vs 3.11），共享文件系统是最简单的 IPC
- 原子性：文件存在/不存在比 socket 连接更易于调试和状态恢复

### 为什么 `keep_alive: true` 是默认值

Isaac Sim 冷启动约 90 秒。训练过程中每个 epoch 的 sim 关闭/重启代价极高。`keep_alive: true` 让 sim 进程在 epoch 间保持运行（sim_server.py 写 `.geniesim_idle`），重启只需约 17 秒（重新加载 MuJoCo 和 USD 场景，Isaac Sim 本身不重启）。

---

## 已知限制

| 限制 | 说明 | 是否计划修复 |
|------|------|-------------|
| Reward 为占位实现 | place_block 和 junpu 任务的 reward 目前只有时间惩罚 | 需按任务具体实现 |
| 单节点训练 | 当前配置 `num_nodes: 1`，未测试多节点 | 按需 |
| 无 Docker Compose 配置 | 多容器编排需手动管理 | 低优先级 |
| Isaac Sim 日志量大 | 冷启动时 Isaac Sim 输出大量 GPU 初始化日志 | 不影响功能 |

---

## 文件变更汇总

### 新增文件

| 文件 | 说明 |
|------|------|
| `docker/run_local.sh` | Mode 2 便捷启动脚本 |
| `docker/Dockerfile.geniesim` | 合并镜像 Dockerfile |
| `docs/USER_GUIDE_CN.md` | 用户操作手册 |
| `docs/DEVELOPER_NOTES_CN.md` | 开发者参考文档 |
| `docs/PROJECT_STATUS_GENIESIM_RLINF_CN.md` | 本文档 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `rlinf/envs/geniesim/geniesim_env.py` | mode 默认改为 `auto`，通过 `GENIESIM_CONTAINER` 环境变量自动检测 |
| `rlinf/envs/geniesim/container_manager.py` | 新增 `container_geniesim_root` 配置、自动路径前缀替换；含 `mode: local` 支持、哨兵协议 bug 修复 |
| `examples/embodiment/config/env/geniesim_*.yaml` | 统一为一份配置（消除 `_local` 后缀），路径使用 `${oc.env:GENIESIM_ROOT}`，默认 `headless: true` |
| `examples/embodiment/config/geniesim_*.yaml` | 训练配置统一引用非 `_local` 的 env YAML |
| `main/source/geniesim/rl/envs/process_manager.py` | `_launch_mujoco()` 只在 `headless=false` 且 `env_id==0` 时传 `--viewer` |
| `main/source/geniesim/rl/mujoco/mujoco_ros_node.py` | 移除 `args.viewer = True` 硬编码，viewer 完全由 `--viewer` CLI 参数决定 |

### 删除文件（配置合并）

| 文件 | 说明 |
|------|------|
| `examples/embodiment/config/env/geniesim_place_block_into_box_local.yaml` | 合并入 `geniesim_place_block_into_box.yaml` |
| `examples/embodiment/config/env/geniesim_junpu_place_workpiece_local.yaml` | 合并入 `geniesim_junpu_place_workpiece.yaml` |
| `examples/embodiment/config/geniesim_ppo_mlp_local.yaml` | 合并入 `geniesim_ppo_mlp.yaml` |

---

## 下一步计划

| 优先级 | 任务 | 说明 |
|--------|------|------|
| 高 | 实现真实 Reward 函数 | place_block 任务：block 在 box 内 → +1；junpu 任务：放置成功 → +1 |
| 中 | 域随机化（randomization） | 物体位置/机器人初始位置随机化，提高策略泛化性 |
| 中 | Docker Compose 配置 | 一条命令拉起 Mode 1 架构（geniesim 容器 + rlinf 容器） |
| 低 | 多节点训练验证 | 测试 `num_nodes > 1` 的分布式 PPO/SAC |
| 低 | VLA 策略（OpenVLA-OFT）接入 | 当前 MLP 策略验证完毕，可接入 VLA 进行微调 |
