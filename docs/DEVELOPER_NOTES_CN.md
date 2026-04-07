# GenieSim × RLinf — 开发者技术文档

> 记录核心模块的设计决策、接口细节、已知 bug 及后续开发指引。
>
> **最后更新**：2026-04-04

---

## 目录

1. [整体架构与设计选择](#1-整体架构与设计选择)
2. [容器化 sim-server](#2-容器化-sim-server)
3. [MuJoCo 物理节点](#3-mujoco-物理节点-mujoco_ros_nodepy)
4. [IsaacSim 渲染器](#4-isaacsim-渲染器-rl_rendererpy)
5. [GenieSimVectorEnv](#5-geniesimvectorenv-geniesim_vec_envpy)
6. [RLinf 侧 GenieSimEnv](#6-rlinf-侧-geniesimenv)
7. [控制模式：joint 与 ee](#7-控制模式joint-与-ee)
8. [SpaceMouse 示范采集](#8-spacemouse-示范采集)
9. [ADER Reward 计算器](#9-ader-reward-计算器)
10. [数据格式](#10-数据格式)
11. [已知 Bug 与待修复项](#11-已知-bug-与待修复项)
12. [添加新任务完整清单](#12-添加新任务完整清单)
13. [数据流时序图](#13-数据流时序图)
14. [Mode 2：本地模式实现细节](#14-mode-2本地模式实现细节)

---

## 1. 整体架构与设计选择

### 1.1 为什么用 Python MuJoCo

旧版 `mujoco_geniesim` 是 C++ 实现，需要 colcon 编译，耦合 CosineOS 和底盘控制，开源成本高。本方案改用 `mujoco` Python 包（`pip install mujoco`）：
- 零编译，直接 `python mujoco_ros_node.py`
- 物理精度与 C++ 完全一致（同一套 mjModel/mjData）
- 代价：Python GIL 导致 1000Hz 物理线程与 ROS spin 线程有轻微争用，实测约 800-900Hz

### 1.2 为什么相机帧不走 ROS

N=8 环境、640×480 RGB、30Hz 时 ROS topic 带宽约 1.4 GB/s，会打满本地 loopback。改为 POSIX 共享内存（`/dev/shm`）：
- IsaacSim 渲染器写 `[N, 2, H, W, 3]` uint8 数组
- 宿主机 VectorEnv 直接 `numpy view`，零拷贝

### 1.3 为什么做容器化

- 宿主机无需安装 ROS/IsaacSim，降低部署门槛
- Isaac Sim `keep_alive` 模式：训练重启时跳过 1-3 分钟冷启动
- 隔离性：不同项目可以用不同 ROS_DOMAIN_ID 同时运行
- 主要代价：SHM 跨容器需要 `--ipc=host`；容器 restart 后要重建 SHM

### 1.4 宿主机不需要 ROS

容器内 MuJoCo 进程 ↔ IsaacSim 通过 ROS2 Jazzy 通信（`/env_i/tf_render`、`/env_i/joint_states` 等），但宿主机与容器之间完全通过 POSIX SHM 交换数据：
- **ctrl_shm**（每 env 独立）：宿主机下发 action、触发 reset；容器侧写回 joint_states
- **frames_shm**（全局共享）：IsaacSim 将相机帧写入，宿主机直接读取

---

## 2. 容器化 sim-server

### 2.1 文件握手协议

`SimContainerManager`（RLinf 侧）和 `sim_server.py`（仿真侧）通过 `geniesim_root/` 目录下的哨兵文件通信（Mode 1 是 bind-mount 目录，Mode 2 是直接共享文件系统）：

```
<geniesim_root>/
├── .geniesim_ready    sim 写：仿真已就绪，训练侧可连接 SHM
├── .geniesim_start    训练侧写：请求 sim 启动/重启仿真进程
├── .geniesim_stop     训练侧写：请求 sim 停止仿真进程（sim_server 本身继续运行）
├── .geniesim_idle     sim 写：仿真进程已停止，等待下次启动
├── .geniesim_error    sim 写：启动失败
└── .geniesim_progress sim 写：启动阶段进度（JSON），供训练侧显示进度条
```

`ensure_running()` 流程（**Mode 1** docker 模式）：
1. 检查容器是否运行（`docker inspect`）
2. 若未运行：`docker run`（完整冷启动）
3. 若运行但无 `.geniesim_ready`：发 `.geniesim_start` 信号，等待 `.geniesim_ready`
4. 若运行且有 `.geniesim_ready`：直接 attach（`reuse_running=True` 时）

`ensure_running()` 流程（**Mode 2** local 模式，见 §14）：
1. sentinel-first 检查（不依赖 `_local_proc` Python 对象状态）
2. `.geniesim_ready` 存在且 SHM 可访问 → 直接复用
3. `.geniesim_idle` 存在 → 写 `.geniesim_start`，等待就绪（~17 秒）
4. 否则：`subprocess.Popen("bash -c '...'")` 冷启动（~90 秒）

`shutdown()` 流程（两种模式相同语义）：
- `keep_alive=True`：写 `.geniesim_stop`，等 `.geniesim_idle`，主进程/容器继续存活
- `keep_alive=False`：Mode 1 `docker stop`；Mode 2 terminate subprocess + kill orphans

### 2.2 SimContainerManager 配置参数

```yaml
container_cfg:
  geniesim_root: ${oc.env:GENIESIM_ROOT}   # 自动适配容器内/外

  # Docker mode 设置（local mode 下忽略）
  image:               geniesim-rlinf:latest  # Docker 镜像名
  name:                geniesim_rlinf_sim     # 容器名（同一台机器唯一）
  isaac_cache_root:    ~/docker/isaac-sim     # Isaac Sim 缓存目录（bind-mount）
  extra_docker_args:   []                     # 额外的 docker run 参数

  # Local mode 设置（docker mode 下忽略）
  ros_ws_install: /geniesim/ros_ws_build/install

  # 共享设置
  keep_alive:          true                   # 关闭时保留 sim 进程（跳过 Isaac Sim 重启）
  reuse_running:       true                   # 复用已就绪的 sim 进程
  startup_timeout_sec: 300                    # 等待就绪的超时（秒）
  ros_domain_id:       0                      # 容器内 ROS_DOMAIN_ID
```

> **自动模式检测**：`mode` 默认为 `auto`，通过 `GENIESIM_CONTAINER` 环境变量自动选择：容器内 → local，宿主机 → docker。可显式设置 `mode: docker` 或 `mode: local` 覆盖。
>
> **Docker mode 路径自动替换**：`_write_sim_config()` 会自动将 `init_params` 中的宿主机 `GENIESIM_ROOT` 前缀替换为 `container_geniesim_root`（默认 `/geniesim/main`），无需手动配置 `container_paths`。

### 2.3 启动阶段进度报告

容器写入 `.geniesim_progress` 的内容（JSON），宿主机解析并展示：

| 阶段 | 含义 |
|------|------|
| `mujoco_launching` | MuJoCo 进程正在启动 |
| `mujoco_ready` | 所有 MuJoCo 节点就绪 |
| `isaac_launching` | IsaacSim 进程已 fork |
| `isaac_loading` | IsaacSim GPU 初始化 + USD 场景加载（最耗时） |
| `isaac_ready` | IsaacSim SHM 就绪，整体 READY |

### 2.4 已知问题：多次崩溃后容器状态异常

若连续多次崩溃（每次崩溃后 atexit 发 stop 信号），容器可能陷入 start→stop→start 循环。
**解决方法**：`docker restart geniesim_rlinf_sim`，然后重新运行测试。

---

## 3. MuJoCo 物理节点 (`mujoco_ros_node.py`)

**路径**：`main/source/geniesim/rl/mujoco/mujoco_ros_node.py`

### 3.1 线程架构

```
main()
├── physics_thread → run_physics_loop()
│     ├── sync_mode=false: _run_physics_free() @ 1000Hz（旧行为）
│     └── sync_mode=true:  _run_physics_sync()（新行为，默认）
│           等待 ctrl_shm.mujoco_phase == MUJOCO_PHASE_GO
│           执行 steps_per_step 步物理（默认 33 步 ≈ 30Hz）
│           每步：apply ctrl → mj_step → write states
│           最后一步：publish tf_render + write body_poses to info_buf
│           设 ctrl_shm.mujoco_phase = MUJOCO_PHASE_DONE
└── ROS executor (SingleThreadedExecutor)
      处理 /joint_command 订阅（legacy ROS 路径）
      处理 /get_object_pose、/get_object_aabb、/reset_scene 服务
```

**注意**：`_ctrl` 数组受 `threading.Lock` 保护，避免物理线程与 ROS 回调竞态。

### 3.2 ctrl_shm 布局

每个 env 有独立的 ctrl_shm（`{shm_name}_ctrl_{env_id}`）：

```
字节偏移            内容
0                   state_counter (uint32)    每次 joint_state 更新后自增
4                   reset_flag (uint32)       宿主机写 RESET_REQUESTED 触发 reset
8                   state data (float32×S)    MuJoCo 写入 joint_states（+ EE states）
8+4S                action data (float32×A)   写入 action（ctrl 数组）
8+4S+4A             info data (float32×I)     body_poses（info_body_names 对应的 7D poses）
8+4S+4A+4I          mujoco_phase (uint32)     同步信号：WAIT/GO/DONE
8+4S+4A+4I+4        steps_per_step (uint32)   每次 GO 执行的物理步数
```

- `CTRL_HEADER_BYTES = 8`（state_counter + reset_flag）
- `CTRL_SYNC_BYTES = 8`（mujoco_phase + steps_per_step）
- `EE_STATE_DIM = 12`（ee 模式下额外追加到 state：`[ee_l_pos(3), ee_l_rpy(3), ee_r_pos(3), ee_r_rpy(3)]`）
- `BODY_POSE_DIM = 7`（x, y, z, qw, qx, qy, qz）

### 3.3 同步步进模式（sync_mode，新增）

默认启用（`--sync-mode` CLI 参数）。MuJoCo 不再自由运行 1000Hz，而是：

1. 等待 `mujoco_phase == MUJOCO_PHASE_GO`（busy-poll，0.1ms 间隔）
2. 读 `steps_per_step` 值（默认 33，对应 1000Hz/30Hz）
3. 执行该步数的物理循环（`mj_step`），最后一步发布 `tf_render`
4. 将 body_poses 写入 ctrl_shm 的 info_buf（`_write_ctrl_info()`）
5. 设 `mujoco_phase = MUJOCO_PHASE_DONE`

检测到 `reset_flag == RESET_REQUESTED` 时：
- 调用 `mj_resetData()` + `mj_forward()`
- 写入 reset 后的 states/info
- 设 `reset_flag = RESET_DONE`

### 3.4 body_poses 写入（info_buf）

`_write_ctrl_info()` 方法将 `--info-body-names` 指定的 MuJoCo body 世界坐标写入 ctrl_shm info 区：
- 位置：`data.xpos[body_id]`（3 floats）
- 姿态：通过 `mju_mat2Quat(data.xmat[body_id])` 转为四元数（qw, qx, qy, qz，4 floats）
- 每个 body 占 7 个 float32

### 3.5 ee 控制模式实现

`_apply_ee_actions()` 方法：
1. 从 action 读取右臂目标 `[pos_r(3), euler_r(3)]`（base_link 系，米/弧度）
2. 用阻尼最小二乘（DLS）IK 迭代求解关节角：
   ```
   Δθ = Jᵀ (JJᵀ + λ²I)⁻¹ Δx
   ```
   其中 λ = `ik_damp`，最多迭代 `ik_max_iter` 次
3. 类似地处理左臂（若 action[:6] 非零）
4. 将解出的关节角写入 `data.ctrl[ctrl_offset : ctrl_offset+7]`
5. gripper 直接写入 `data.ctrl[gripper_ctrl_r]`

**IK 注意事项**：
- 默认只求解右臂，左臂保持当前位置
- 大奇异位形附近（关节接近极限或臂完全伸直）IK 可能不收敛，机器人会停在最近一次成功位置
- 增加 `ik_max_iter` 可提升精度，但会增加每步计算时间

### 3.6 AABB 计算（Reward 用）

`compute_aabb_world()` 遍历所有 geom，取 union。近似值：box/sphere/cylinder/capsule 精确，mesh geom 取 `geom_size` 近似。

### 3.7 可视化（viewer）

`headless` 参数完全由 YAML 配置文件决定，控制链路：

1. `geniesim_env.py` 读取 `init_params.headless` → 传入 `pm_kwargs`
2. `process_manager.py` 的 `_launch_mujoco(env_id)` 中：
   - `headless=False` 且 `env_id==0` → 传 `--viewer` CLI 参数
   - 其余情况 → 不传（headless）
3. `mujoco_ros_node.py` 的 `main()` 纯粹依赖 `--viewer` 参数：
   - 有 `--viewer` → `mujoco.viewer.launch_passive()` 打开 GUI
   - 无 `--viewer` → `_run_headless()` 无窗口运行

**多 env 并行时只有 env_0 可视化**，避免多个 MuJoCo 窗口争抢 GPU 资源。

**容器可视化**：启动容器时需挂载 X11：
```bash
docker run ... -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw ...
```

---

## 4. IsaacSim 渲染器 (`rl_renderer.py`)

**路径**：`main/source/geniesim/rl/renderer/rl_renderer.py`

### 4.1 物理引擎禁用

`physics_dt=0.0`：IsaacSim 物理完全禁用，USD prims 仅作为渲染骨架，被 MuJoCo 的 tf_render 消息驱动。

### 4.2 GridCloner 多环境

```python
cloner = GridCloner(spacing=3.0)          # 每 clone 间隔 3 米（防相机视野重叠）
env_paths = cloner.generate_paths("/World/envs/env", N)
add_reference_to_stage(scene_usd, env_paths[0])  # 只加载 env_0 作为原型
cloner.clone(source=env_paths[0], prim_paths=env_paths,
             copy_from_source=False, replicate_physics=False)
```

### 4.3 body_name_map — 最易踩坑处

MuJoCo body name（如 `box_body`）必须正确映射到 USD prim 路径后缀（如 `Objects/box`），否则 tf_render 消息被忽略，物体不动。

优先级（从高到低）：
1. 读取 `<scene_usd>_body_map.json`（推荐）
2. 自动扫描 USD stage 构建 `leaf_name → rel_path` 映射
3. identity 映射（body name = prim name，几乎肯定不对）

`_body_map.json` 格式：
```json
{
  "G2_base_link": "robot/base_link",
  "G2_arm_l_link1": "robot/arm_l_link1",
  "box_body": "Objects/box"
}
```

**推荐做法**：写一个工具脚本，对比 MJCF 中的 body name 列表和 USD stage 中的 prim 路径，半自动生成这个文件。

### 4.4 帧同步

IsaacSim 每渲染一帧后将 `frame_counter`（uint32）自增，并将图像写入 SHM。`GenieSimVectorEnv._wait_new_frame()` 自旋等待计数器变化，超时 2 秒。

**已知问题**：忙等消耗 CPU；IsaacSim 卡顿时会堵塞整个 step()。后续可改为 Unix socket 通知。

---

## 5. GenieSimShmClient / GenieSimVectorEnv

### 5.1 架构概览（重构后）

```
RLinf 侧（宿主机 / Mode 2 venv）           Sim 侧（容器内 / Mode 2 系统 Python）
────────────────────────────             ─────────────────────────────────
GenieSimBaseEnv                          sim_server.py
  └─ GenieSimShmClient                     └─ GenieSimVectorEnv
       │   ┌─── Step SHM ───────┐              ├─ ProcessManager
       ├──>│ actions + reset_mask│──>           │    ├─ mujoco_ros_node × N (sync)
       │<──│ rewards/term/trunc  │<──           │    └─ rl_renderer × 1
       │   │ episode stats       │              ├─ 信号量同步 MuJoCo (GO/DONE)
       │   │ body_poses (info)   │              ├─ 组织 obs/reward/term/trunc/info
       │   └────────────────────┘              └─ 写入 Step SHM
       ├─── Ctrl SHM ──── states (只读)
       └─── Frame SHM ─── images (只读)
```

**职责分离原则**：
- **sim 侧 GenieSimVectorEnv**：管理 MuJoCo + IsaacSim 生命周期，同步 MuJoCo 步进，组织完整的 gymnasium 5-tuple 输出，写入 Step SHM
- **RLinf 侧 GenieSimShmClient**：薄客户端，只负责写 actions/reset_mask 和读取预组织好的结果，**不做二次加工**
- **RLinf 侧 GenieSimBaseEnv**：转发结果为 torch tensor，追踪 RLinf 指标，允许子类覆盖 reward
- **任务子类**（如 JunpuPlaceWorkpieceEnv）：通过 `info["body_poses"]` 计算稠密 reward，覆盖 sim 侧的稀疏 reward

### 5.2 Step SHM 通信协议

Step SHM（`{shm_name}_step`）是一个全局共享的请求-应答通道：

**布局**：
```
[0:4]              uint32  step_phase     (STEP_PHASE_*)
[4:4+N*4]          float32 reset_mask     (num_envs,) 1.0=reset, 0.0=keep
[+:+N*A*4]         float32 actions        (num_envs * action_dim)
--- 以下为 sim 侧写入的输出 ---
[+:+N*4]           float32 rewards        (num_envs,)
[+:+N*4]           float32 terminated     (num_envs,) 0.0/1.0
[+:+N*4]           float32 truncated      (num_envs,) 0.0/1.0
[+:+N*4]           float32 elapsed_steps  (num_envs,)
[+:+N*4]           float32 episode_returns(num_envs,)
[+:+N*4]           float32 success_once   (num_envs,) 0.0/1.0
[+:+N*I*4]         float32 body_poses     (num_envs * info_dim)
```

**协议**：
| 阶段 | step_phase 值 | 发起方 | 描述 |
|------|---------------|--------|------|
| 空闲 | `STEP_PHASE_IDLE` (0) | — | 无请求 |
| 步进请求 | `STEP_PHASE_STEP_REQUEST` (1) | RLinf | 已写入 actions |
| 步进完成 | `STEP_PHASE_STEP_DONE` (2) | Sim | 已写入 rewards/term/trunc/info |
| 重置请求 | `STEP_PHASE_RESET_REQUEST` (3) | RLinf | 已写入 reset_mask |
| 重置完成 | `STEP_PHASE_RESET_DONE` (4) | Sim | 已写入重置后的 states/info |
| 关闭 | `STEP_PHASE_CLOSE` (5) | 任意 | 步进循环退出 |

### 5.3 宿主机侧：`GenieSimShmClient`

**路径**：`RLinf/rlinf/envs/geniesim/shm_client.py`

宿主机只使用 `GenieSimShmClient`，**不再依赖 geniesim Python 包**。功能：
- attach 三种 SHM：Frame（相机帧）、Ctrl（states 只读）、Step（请求-应答）
- `step(actions)` → 写 actions 到 Step SHM → 设 STEP_REQUEST → 等 STEP_DONE → 读结果
- `reset(env_idx)` → 写 reset_mask → 设 RESET_REQUEST → 等 RESET_DONE → 读结果
- `close()` → 设 STEP_PHASE_CLOSE

**SHM 布局常量**：`RLinf/rlinf/envs/geniesim/shm_layout.py`（从容器侧复制，零外部依赖）。

### 5.4 容器侧：`GenieSimVectorEnv`

**路径**：`main/source/geniesim/rl/envs/geniesim_vec_env.py`

容器侧核心类，负责：
- 创建 ProcessManager 启动 MuJoCo + IsaacSim 子进程
- 创建并管理 Step SHM
- attach Frame SHM 和 Ctrl SHM
- 同步 MuJoCo 步进（`_trigger_mujoco_step()` / `_wait_mujoco_done()`）
- 组织 obs/reward/terminated/truncated/info 并写入 Step SHM
- `run_step_loop()`：主循环，处理 RLinf 的步进和重置请求

**关键方法**：
- `step(actions, auto_reset=False)`：写 actions → 触发 MuJoCo GO → 等 DONE → 读 states/body_poses → 计算 reward/truncated → 写 Step SHM
- `reset(env_idx)`：对指定 env 写 RESET_REQUESTED → 触发 MuJoCo GO → 等 DONE → 写 Step SHM
- `run_step_loop()`：无限循环 poll step_phase，处理 STEP_REQUEST 和 RESET_REQUEST

### 5.5 auto_reset 语义

auto_reset 完全由 **RLinf 侧** 控制：
- sim 侧 `run_step_loop()` 调用 `step(auto_reset=False)`，只报告 dones
- RLinf 侧 `GenieSimBaseEnv.step()` 检测 dones → 保存 `final_observation` → 发送 RESET_REQUEST → 读取重置后的 obs

**不要让 sim 侧做 auto_reset**，否则会与 RLinf 侧双重 reset。

### 5.6 `chunk_step` 实现（重要）

`GenieSimBaseEnv.chunk_step()` **不再委托给** `self.env.chunk_step()`，而是循环调用 `self.step()`：

```python
for i in range(num_action_chunks):
    obs, rewards, terminated, truncated, infos = self.step(chunk_actions[:, i, :])
    ...
# 返回 [num_envs, num_action_chunks] 形状的 terminations/rewards
```

原因：
1. 确保子类的 `step()` reward override（如 `JunpuPlaceWorkpieceEnv._compute_reward()`）被正确调用
2. 确保 terminations/truncations 返回 `[num_envs, num_action_chunks]` 形状，符合 `env_worker.py` 对 `chunk_dones[:, -1]` 的切片预期
3. 确保 episode 指标和 auto-reset 正确执行

**不要**恢复为委托 `self.env.chunk_step()`，否则会出现 shape mismatch 和 reward bypass 两个 bug。

### 5.7 观测格式（numpy，from VectorEnv）

```python
{
    "main_images":       np.ndarray  [N, H, W, 3] uint8
    "wrist_images":      np.ndarray  [N, H, W, 3] uint8  # 或 None（wrist_cam_prim=""）
    "states":            np.ndarray  [N, state_dim] float32
    "task_descriptions": list[str]   length N
}
```

`GenieSimBaseEnv._wrap_obs()` 将 numpy 转为 CPU torch tensor，上游 RLinf 直接使用。

### 5.8 joint mode 的 state layout

```
states[i][0:7]   左臂关节位置
states[i][7:14]  右臂关节位置
states[i][14:21] 左臂关节速度
states[i][21:28] 右臂关节速度
```
总 state_dim = 28

### 5.9 ee mode 的 state layout（额外 12 维 EE poses）

```
states[i][0:7]   左臂关节位置
states[i][7:14]  右臂关节位置
states[i][14:21] 左臂关节速度
states[i][21:28] 右臂关节速度
states[i][28:31] 左末端 EE 位置 (base_link 系, 米)
states[i][31:34] 左末端 EE 姿态 (RPY, 弧度)
states[i][34:37] 右末端 EE 位置 (base_link 系, 米)
states[i][37:40] 右末端 EE 姿态 (RPY, 弧度)
```
总 state_dim = 40

---

## 6. RLinf 侧 GenieSimEnv

**路径**：`RLinf/rlinf/envs/geniesim/geniesim_env.py`

### 6.0 依赖隔离（重要）

`geniesim_env.py` **不再有任何 `from geniesim.*` import**，包括运行时延迟 import。宿主机运行 RLinf 只需要：
- `rlinf` 包本身（`.venv` 已包含）
- Docker（用于容器管理）
- 无需 geniesim Python 包、无需 ROS、无需 ROS 接口包

如需修改此文件，确保不引入新的 geniesim 依赖。

### 6.1 `GenieSimBaseEnv` 与 `GenieSimShmClient` 的职责边界

- `GenieSimBaseEnv.step()` 调用 `self.env.step(actions_np)` 获取 numpy 结果 → 转 torch → 追踪本地指标 → 允许子类覆盖 reward → 处理 auto_reset
- `GenieSimShmClient.step()` 只做：写 actions → 设 STEP_REQUEST → 等 STEP_DONE → 读取预组织好的 rewards/terminated/truncated/body_poses → 返回
- 子类（如 `JunpuPlaceWorkpieceEnv`）可在 `step()` 中覆盖 `rewards_t`，通过 `infos["body_poses"]` 计算自定义稠密 reward

### 6.2 与 IsaaclabBaseEnv 的对比

| | IsaaclabBaseEnv | GenieSimBaseEnv |
|---|---|---|
| obs 数组类型 | CUDA tensor | CPU numpy → torch tensor |
| VecEnv 层 | `SubProcIsaacLabEnv` | `GenieSimVectorEnv` |
| 并行机制 | IsaacLab 内置 | ProcessManager（容器内） |
| reset 参数 | `env_ids: torch.Tensor` | `env_idx: list[int]` |
| device | CUDA | CPU |
| 容器管理 | 无 | `SimContainerManager` |

### 6.3 metrics 结构（与 IsaaclabBaseEnv 一致）

```python
infos["episode"] = {
    "success_once": torch.Tensor [N] bool,   # 此 episode 是否曾获得 reward>0
    "return":       torch.Tensor [N] float,  # 累计 reward
    "episode_len":  torch.Tensor [N] int,    # 步数
    "reward":       torch.Tensor [N] float,  # 平均 reward（return/steps）
}
```

### 6.4 添加新任务子类

```python
from rlinf.envs.geniesim import register_geniesim_env
from rlinf.envs.geniesim.geniesim_env import GenieSimBaseEnv

@register_geniesim_env("my_task")
class MyTaskEnv(GenieSimBaseEnv):
    # 若需要特殊相机/state 配置，override _make_vec_env_config()
    pass
```

然后在 `tasks/__init__.py` 加一行 import。

### 6.5 任务配置文件

`junpu_place_workpiece` 使用单一 YAML：`RLinf/examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml`（`state_dim: 40`，含 EE states）。采集/回放/测试与 Hydra 训练（`defaults` 引用同一路径）共用该文件；MLP 策略的 `obs_dim: 40` 须与之一致。

### 6.6 Junpu SAC 训练要点

SAC + CNN 策略训练（`geniesim_junpu_sac.yaml`）的关键配置：

```yaml
actor:
  model:
    model_type: cnn_policy    # ResNet10 编码器 + MLP 头
    image_size: [3, 128, 128] # 右手腕部相机，480×480 渲染后缩放
    state_dim: 26             # 右臂 only（从 52 维 state 中取右半）
    action_dim: 7             # [pos_r(3), rpy_r(3), grip_r(1)]
    add_q_head: true          # SAC 用 Q-head，不用 value head
    add_value_head: false
    num_q_heads: 10           # 10 个 Q-head（取 min 更稳定）
  enable_drq: true            # DRQ 图像增强（随机裁剪 pad=4）
  fsdp_config:
    use_orig_params: true     # CNN 策略需要 true
algorithm:
  replay_buffer:
    cache_size: 200           # 图像数据较大，减小缓存
    min_buffer_size: 2        # 2 个 rollout epoch 后开始训练
  # demo_buffer: 注释掉（若未设置 load_path，DataLoader 会死循环）
env:
  train:
    total_num_envs: 4         # 4 并行 MuJoCo 环境
    max_steps_per_rollout_epoch: 100  # 每 rollout epoch 收集 400 转移
```

**已知 bug 与注意事项**：
- `demo_buffer` 配置存在但 `load_path: null` 时，`ReplayBufferDataset.__iter__` 会死循环（demo buffer 永远不 ready）。解决方法：注释掉整个 `demo_buffer` 段。
- `use_orig_params: true` 用于 CNN 策略（与 MLP 策略的 `false` 相反）。
- 启动容器后需手动 `colcon build` ROS 接口（`geniesim_rl_interfaces`），Docker 镜像的 entrypoint 路径不匹配。
- 上次训练崩溃后可能残留 `.geniesim_ready` 哨兵文件，需清理后重启。

---

## 7. 控制模式：joint 与 ee

### 7.1 动作格式对比

**joint 模式**（`place_block_into_box`）：
```
action[0:7]   左臂 7 关节目标位置（弧度）
action[7:14]  右臂 7 关节目标位置（弧度）
```

**ee 模式**（`junpu_place_workpiece`）：
```
action[0:3]   左末端 EE 目标位置（base_link 系，米）
action[3:6]   左末端 EE 目标姿态（intrinsic XYZ Euler，弧度）
action[6:9]   右末端 EE 目标位置（base_link 系，米）
action[9:12]  右末端 EE 目标姿态（intrinsic XYZ Euler，弧度）
action[12]    左夹爪指令（-1 关闭 ~ +1 打开）
action[13]    右夹爪指令（-1 关闭 ~ +1 打开）
```

### 7.2 SpacemouseSimIntervention 的 action 注入

SpaceMouse 输出 6D delta（`[-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]`），wrapper 将其累积为绝对目标：

```python
# 初始化：从 reset obs 中读当前 EE 位姿（states[env_0, 34:40]）
self._current_target[6:9]  = obs["states"][0, 34:37]   # 右臂 EE 位置
self._current_target[9:12] = obs["states"][0, 37:40]   # 右臂 EE 姿态

# 每步：
self._current_target[6:9]  += sm_delta[:3] * action_scale      # 默认 0.01 m/unit
self._current_target[9:12] += sm_delta[3:] * rotation_scale    # 默认 0.05 rad/unit

# 左臂保持初始位置不动（从 obs 初始化后不再更新）
# 右夹爪：左键=关闭（-1），默认保持
```

---

## 8. SpaceMouse 示范采集

### 8.1 Wrapper 层次结构

```
collect_sim_data.py
  └── CollectEpisode                    # 录制 + 保存
        └── SpacemouseSimIntervention   # SpaceMouse 注入（env_0）
              └── GenieSimBaseEnv       # 向量化仿真环境
```

### 8.2 FakeSpaceMouseExpert 行为

每个 episode 固定序列：

| 阶段 | 步数 | 行为 |
|------|------|------|
| Phase 0 | `move_steps`（默认 40） | 右臂沿 +Y 移动（`[0, 1, 0, 0, 0, 0]`） |
| Phase 1 | `gripper_steps`（默认 15） | 保持位置，左键按下（关夹爪） |
| Phase 2 | `lift_steps`（默认 40） | 右臂沿 +Z 上升（`[0, 0, 1, 0, 0, 0]`） |
| Done | 1 步 | 右键按下（episode 结束信号） |

`on_episode_reset()` 后序列重置，可用于多 episode 采集。

### 8.3 CollectEpisode 的 success 识别

`SpacemouseSimIntervention.step()` 在右键按下时：
1. 克隆 `terminated` 张量并设 `terminated[env_0] = True`
2. 设 `info["success"] = True`
3. 将 `info["episode"]["success_once"][env_0]` 设为 True（**必须**，因为 `_get_episode_success` 优先扫描 `info["episode"]["success_once"]`，若不设置会短路返回 False 而跳过 `info["success"]`）

### 8.4 保存的数据格式（pickle）

```python
{
    "rank":         int,
    "env_idx":      int,
    "episode_id":   int,
    "success":      bool,
    "observations": list[dict],  # 每步 {main_images, states, task_descriptions, wrist_images}
    "actions":      list[Tensor],  # 每步 shape (14,)——注意这是原始 policy action
    "rewards":      list[Tensor],
    "terminated":   list[Tensor],
    "truncated":    list[Tensor],
    "infos":        list[dict],   # 含 intervene_action（实际执行动作）、intervene_flag
}
```

实际执行的动作从 `infos[i]["intervene_action"]` 取，shape `(14,)`，是经过 SpaceMouse 修改后发给仿真的绝对 EEF 目标。

---

## 9. ADER Reward 计算器

**路径**：`main/source/geniesim/rl/reward/reward_computer.py`

**状态**：框架已就位，但 `junpu_place_workpiece` 尚无对应任务 JSON（`enable_reward=false`）。

### 9.1 _build_infos 重复计算 bug（已知，待修复）

**问题**：`geniesim_vec_env.py` 的 `_build_infos()` 调用了 `rc.compute_reward()`，导致 ADER 状态机 update 两次。

**修复方法**：
```python
# step() 里缓存
self._last_reward_infos = [rc.compute_reward() for rc in self._reward_computers]

# _build_infos() 里读缓存
task_progress = [info.task_progress for info in self._last_reward_infos]
```

### 9.2 prim_path → body_name 映射假设

`MuJoCoAPICore._prim_path_to_body_name()` 假设 ADER 动作配置里的 `obj_name` 与 MuJoCo MJCF 中的 body name 相同。若不同，需在该方法中加 alias 字典。

---

## 10. 数据格式

### 10.1 观测（Observation）

**来自 VectorEnv**（numpy）→ **被 GenieSimBaseEnv 转换**（torch tensor）：

```python
obs = {
    "main_images":       torch.Tensor  [N, H, W, 3]  uint8（注意：未归一化）
    "wrist_images":      torch.Tensor  [N, H, W, 3]  uint8  或 None
    "states":            torch.Tensor  [N, state_dim] float32
    "task_descriptions": list[str]     length N
}
```

归一化（`/ 255.0` 或其他变换）由策略模型侧处理，env 侧不做。

### 10.2 Reward

- `junpu_place_workpiece`：密集 reward，多分量（xy 距离 + 非对称 z 惩罚 + 姿态偏差 + 静止奖励 + 成功奖励），通过 `info["body_poses"]["workpiece_r"]` 计算工件世界坐标（在 RLinf 侧 `JunpuPlaceWorkpieceEnv._compute_reward()` 中）
- `place_block_into_box`：稀疏，+1.0 当 Inside ADER action 成功
- 类型：`torch.Tensor [N] float32`

### 10.3 infos 结构

```python
infos = {
    "episode": {
        "success_once": torch.Tensor [N] bool,
        "return":       torch.Tensor [N] float32,
        "episode_len":  torch.Tensor [N] int32,
        "reward":       torch.Tensor [N] float32,
    },
    "body_poses": {                              # 来自 Step SHM（sim 侧写入）
        "workpiece_r":        np.ndarray [N, 7],  # [x, y, z, qw, qx, qy, qz]
        "/World/workspace01": np.ndarray [N, 7],
    },
    # （SpaceMouse 采集时）
    "intervene_action": torch.Tensor [14] float32,
    "intervene_flag":   torch.Tensor [1]  bool,
    "success":          bool,
}
```

---

## 11. 已知 Bug 与待修复项

#### [F12] demo_buffer 死循环导致训练挂起（2026-04-04）
**原问题**：`geniesim_junpu_sac.yaml` 中 `demo_buffer` 段存在但 `load_path: null`。`fsdp_sac_policy_worker.py` 的 `setup_sac_components()` 检测到 `demo_buffer` 配置后创建了空的 `TrajectoryReplayBuffer`。`ReplayBufferDataset.__iter__()` 的 `while True` 循环同时检查 `replay_buffer.is_ready()` 和 `demo_buffer.is_ready(min_demo_buffer_size=1)`。replay buffer 在收集数据后 ready，但 demo buffer 永远为空（size=0 < 1），导致循环永不 yield batch。同时没有 `time.sleep()`，CPU 100% 空转。
**修复**：注释掉未使用的 `demo_buffer` 配置段。根本修复应在 `setup_sac_components()` 中检查 `load_path` 有效性，若 null 则不创建 demo buffer。

#### [F13] replay buffer OOM（图像数据 1.38TB 预分配）（2026-04-04）
**原问题**：`_FlatTrajectoryCache._ensure_capacity()` 为 `cache_size * traj_length` 个 step 预分配张量。当 obs 中包含 640×480×3 图像、`cache_size=5000`、`traj_length=300` 时，预分配需 5000×300×640×480×3×4=1.38TB 内存。
**修复**：在 `JunpuPlaceWorkpieceEnv._extract_images()` 中将 480×480 腕部相机图像缩放为 128×128 后存入 `main_images`，移除未使用的 `wrist_images` 和原始 `main_images`。同时将 `cache_size` 减小到 200。

#### [F14] `final_info` KeyError（自定义 termination）（2026-04-04）
**原问题**：`JunpuPlaceWorkpieceEnv._compute_reward()` 最初额外设置 `terminated` 标志（early termination），但 `GenieSimBaseEnv.step()` 的 auto_reset 逻辑只为 sim 侧返回的 `terminated` 设置 `final_info`。env_worker.py 在 `chunk_dones[:, -1]` 检测到 done 后无条件访问 `infos["final_info"]`，KeyError。
**修复**：移除 early termination，episode 只通过 `max_episode_steps` truncation 结束。

### ✅ 已修复

#### [F7] Mode 2：哨兵检查被 `_local_proc` guard 封锁 → 孤儿进程积累（2026-04-02/03）
**原问题**：`_ensure_running_local()` 将哨兵文件复用检查放在 `if self._local_proc is not None` 块内。由于 `_local_proc` 是实例变量，每次新建 `SimContainerManager` 对象时都为 None，哨兵检查被跳过，每次都启动新的 sim_server.py 子进程，旧进程变孤儿。
**修复**：将哨兵文件检查提到函数顶部（sentinel-first），完全不依赖 `_local_proc` 状态，与 docker 模式的 `_get_status()` 检查对称。

#### [F8] Mode 2：`_shutdown_local()` 在哨兵-idle 复用路径下提前返回 → MuJoCo 进程不关闭（2026-04-03）
**原问题**：哨兵-idle 复用路径（`_ensure_running_local()` 的 `elif idle_file` 分支）不设置 `self._local_proc`。`_shutdown_local()` 检查 `if self._local_proc is None: return`，导致提前返回，MuJoCo/IsaacSim 进程保持运行。
**修复**：`_shutdown_local()` 改为：
```python
proc_alive = self._local_proc is not None and self._local_proc.poll() is None
sim_active = self._ready_file.exists() or proc_alive
if not sim_active:
    return
```

#### [F9] Mode 2：`_kill_orphan_sim_servers()` 误杀 PID 1（2026-04-02）
**原问题**：容器内 PID 1 是 entrypoint（实际是 docker init），其 cmdline 中含 `sim_server.py` 字样，导致 `os.kill(1, 9)` 被调用。
**修复**：添加 `if pid <= 1: continue` guard。

#### [F10] Mode 2：`run_local.sh` 的 `set -u` 导致 ROS setup 脚本失败（2026-04-02）
**原问题**：`set -euo pipefail` 中的 `-u` 导致 `source /opt/ros/jazzy/setup.bash` 报错 `AMENT_TRACE_SETUP_FILES: unbound variable`。
**修复**：改为 `set -eo pipefail`（去掉 `-u`，ROS setup 脚本使用了大量未绑定变量）。

#### [F11] MuJoCo viewer 容器内强制启用导致 GLFW 崩溃（2026-04-03）
**原问题**：`mujoco_ros_node.py` 的 `main()` 中硬编码 `args.viewer = True`，导致在无 X11 DISPLAY 的容器环境下 MuJoCo 进程初始化 GLFW 窗口失败 → crash → ProcessManager 无限重启 → ctrl SHM 被 resource_tracker 清理后丢失 → shm_client 超时。
**修复**：
1. `mujoco_ros_node.py`：移除 `args.viewer = True` 硬编码，viewer 完全依赖 `--viewer` CLI 参数（默认 `store_true`=False）
2. `process_manager.py`：`_launch_mujoco(env_id)` 仅在 `headless=False` 且 `env_id==0` 时传 `--viewer`，多 env 时只可视化第一个
3. YAML 默认 `headless: true`（安全默认值）

#### [F1] `GenieSimBaseEnv.chunk_step` 形状 + reward bypass（2026-04-02）
**原问题**：委托 `self.env.chunk_step()` 导致 terminations 形状 `[num_steps, num_envs]` 与 `env_worker.py` 期望的 `[num_envs, num_steps]` 不符；子类 reward override 被跳过。
**修复**：`chunk_step` 改为循环调用 `self.step()`，见 §5.4。

#### [F2] `prepare_actions` 缺少 GENIESIM case（2026-04-02）
**原问题**：`action_utils.py` 无 `SupportedEnvType.GENIESIM` 分支，抛 `NotImplementedError`。
**修复**：添加 pass-through 分支。

#### [F3] `concat_batch` 跳过 dict key（2026-04-02）
**原问题**：online buffer 有 `forward_inputs`（dict），demo buffer 没有，`concat_batch` 只跳过缺失的 tensor key，dict key 会 KeyError。
**修复**：`nested_dict_process.py` 中 dict 分支也加 `if key not in data2: continue`。

#### [F4] FSDP `use_orig_params: true` 导致 shape writeback error（2026-04-02）
**修复**：SAC MLP 配置改为 `use_orig_params: false`。

#### [F5] Ray worker 任务注册失败（2026-04-02）
**原问题**：Ray subprocess 重新导入模块，`@register_geniesim_env` 装饰器不触发。
**修复**：`rlinf/envs/geniesim/__init__.py` 加 `_import_all_tasks()` 在模块加载时自动 import 所有任务类。

#### [F6] 宿主机依赖 geniesim Python 包（2026-04-02）
**原问题**：`geniesim_env.py` 通过 `sys.path` 注入 + 延迟 import 依赖 `geniesim`、`rclpy`、`geniesim_rl_interfaces`。
**修复**：创建 `shm_layout.py`（本地常量副本）和 `shm_client.py`（`GenieSimShmClient` + `GenieSimVectorEnvConfig`），完全消除宿主机侧 geniesim 依赖。

### 🔴 高优先级（影响功能正确性）

#### [B1] `_build_infos` 重复推进 ADER 状态机
**位置**：`main/source/geniesim/rl/envs/geniesim_vec_env.py`
**影响**：`enable_reward=true` 时 reward 计算错误（目前 junpu 任务 `enable_reward=false`，暂无影响）
**修复**：如 §9.1 所述

#### [B2] `junpu_place_workpiece` 缺少 ADER 任务 JSON
**位置**：无
**影响**：无法通过 ADER 系统获取 reward 和成功终止信号。当前使用自定义密集 reward（`_compute_reward`）替代。
**修复**：编写对应任务 JSON，参考 `place_block_into_box.json`；启用后设 `enable_reward: true`

#### [B7] demo_buffer 配置存在但 load_path 为 null 时的死循环
**位置**：`RLinf/rlinf/data/embodied_buffer_dataset.py` 第 90 行
**影响**：`ReplayBufferDataset.__iter__` 的 `while True` 循环永不 yield batch，训练挂起，CPU 100%
**修复建议**：在 `fsdp_sac_policy_worker.py` 的 `setup_sac_components()` 中检查 demo_buffer 的 `load_path`，若为 null 则跳过 demo buffer 创建

### 🟡 中优先级（影响性能/稳定性）

#### [B3] SHM 帧同步忙等
**位置**：`GenieSimShmClient._wait_new_frame()`
**影响**：消耗一个 CPU 核；IsaacSim 卡顿时 step() 堵塞 2 秒
**改进**：改为 Unix socket 或 `threading.Event` 通知

#### [B4] `body_name_map` 手动维护成本高
**位置**：`rl_renderer.py`
**影响**：添加新任务时容易漏配或配错
**改进**：开发自动对比工具，或在首次启动时自动扫描并生成

### 🟢 低优先级（质量/易用性）

#### [B5] `init_qpos` 仅支持数组格式
**改进**：支持 `{"body_name": {"pos": ..., "quat": ...}}` 字典格式

#### [B6] shm_layout.py 需手动同步
**位置**：`RLinf/rlinf/envs/geniesim/shm_layout.py`
**影响**：若容器侧 SHM 布局变更而宿主机未同步，会导致数据错位
**改进**：CI 中加检查步骤，或改为从共享路径自动 import

---

## 12. 添加新任务完整清单

1. **MJCF 场景文件**（`main/source/geniesim/assets/mujoco_scenes/<task>.xml`）
   - 包含机器人、操作物体、桌面的完整场景
   - 为评测物体添加合理的 geom size（用于 AABB 计算）

2. **USD 场景文件**（IsaacSim 渲染用，如有需要）

3. **body_name_map JSON**（`<scene>_body_map.json`）
   - 格式见 §4.3

4. **任务评测 JSON**（ADER 用，若需要 reward）
   - 参考 `place_block_into_box.json` 格式

5. **任务 YAML 配置**
   - 只需一份：`RLinf/examples/embodiment/config/env/geniesim_<task>.yaml`
   - 路径使用 `${oc.env:GENIESIM_ROOT}` 自动适配容器内/外，无需分 Mode 1/Mode 2 两份
   - 参考 `place_block_into_box.yaml`
   - 注意：ee 模式需要配置 `ee_body_l/r`、`gripper_ctrl_l/r`、`ik_max_iter`、`ik_damp`
   - 若需要 body_poses（用于稠密 reward），配置 `info_body_names`
   - 关键字段示例：
     ```yaml
     init_params:
       id: my_task
       mjcf_path: ${oc.env:GENIESIM_ROOT}/main/source/geniesim/assets/mujoco_scenes/my_task.xml
       state_dim: 40              # 需与 SHM layout 一致
       action_dim: 14
       state_joint_offset: 16     # 根据机器人 XML 中的关节排列确定
       ctrl_offset: 24
       info_body_names:           # 需要通过 info 传递的 MuJoCo body 名
         - workpiece_body
         - target_body
     container_cfg:
       geniesim_root: ${oc.env:GENIESIM_ROOT}
       image: geniesim-rlinf:latest
       name: geniesim_rlinf_sim
       ros_ws_install: /geniesim/ros_ws_build/install
       keep_alive: true
       reuse_running: true
       startup_timeout_sec: 300
     ```

6. **任务子类**（`RLinf/rlinf/envs/geniesim/tasks/<task>.py`）
   - 继承 `GenieSimBaseEnv`，用 `@register_geniesim_env("<task_id>")` 装饰
   - 在 `tasks/__init__.py` 加 import

7. **Hydra env 片段**
   - 统一一份：`RLinf/examples/embodiment/config/env/geniesim_<task>.yaml`
   - 路径使用 `${oc.env:GENIESIM_ROOT}`，自动适配 Mode 1/Mode 2
   - 供训练配置 defaults 列表引用

8. **（可选）训练配置**（`RLinf/examples/embodiment/config/geniesim_<algo>_<task>.yaml`）

---

## 13. 数据流时序图

### 13.1 单步 `step()` 时序（容器模式，同步步进）

```
宿主机 GenieSimShmClient          容器 GenieSimVectorEnv       容器 MuJoCo         容器 IsaacSim
         │                                │                        │                     │
         ├── 写 step_shm.actions ─────────►│                        │                     │
         │   设 STEP_PHASE_STEP_REQUEST    │                        │                     │
         │                                 ├── 写 ctrl_shm.action ─►│                     │
         │                                 │   设 MUJOCO_PHASE_GO   │ apply data.ctrl     │
         │                                 │                        │ mj_step × N         │
         │                                 │                        │ 最后一步:            │
         │                                 │                        │ publish tf_render ──►│ set_pose
         │                                 │                        │ write body_poses     │ render
         │                                 │                        │ 写 states            │ write frames
         │                                 │                        │ 设 MUJOCO_PHASE_DONE │
         │                                 ├── 读 states/info ◄─────┤                     │
         │                                 ├── 读 frames ◄──────────┼─────────────────────┤
         │                                 ├── 计算 reward/term/trunc                     │
         │                                 ├── 写 step_shm output   │                     │
         │                                 │   设 STEP_PHASE_STEP_DONE                    │
         │◄──────── 读 step_shm output ────┤                        │                     │
         │   rewards/terminated/truncated/body_poses                 │                     │
```

**关键变化**：
- MuJoCo 不再自由运行 1000Hz，而是等待 GO 信号，执行 N 步后设 DONE
- GenieSimVectorEnv 负责组织所有输出，ShmClient 只做读取
- IsaacSim 仍自由运行（TF dirty flag 触发渲染，不受同步步进影响）

### 13.2 episode reset 时序（同步步进）

```
宿主机 GenieSimShmClient          容器 GenieSimVectorEnv       容器 MuJoCo
         │                                │                        │
         ├── 写 step_shm.reset_mask ──────►│                        │
         │   设 STEP_PHASE_RESET_REQUEST   │                        │
         │                                 ├── 写 ctrl_shm.reset=REQ►│ mj_resetData()
         │                                 │   设 MUJOCO_PHASE_GO    │ mj_forward()
         │                                 │                         │ 写 states/info
         │                                 │                         │ 设 MUJOCO_PHASE_DONE
         │                                 ├── 读 reset 后 states ◄──┤
         │                                 ├── 写 step_shm output    │
         │                                 │   设 STEP_PHASE_RESET_DONE
         │◄──────── 读 step_shm output ────┤                        │
         │   重置后的 states + body_poses   │                        │
```

### 13.3 SpaceMouse 采集调用链

```
collect_sim_data.py
  env.step(zero_actions)
    └─ CollectEpisode.step(zero_actions)
          │ 记录 action=zero, obs, info 到 buffer
          └─ SpacemouseSimIntervention.step(zero_actions)
                │ expert.get_action() → sm_delta, buttons
                │ _current_target[6:12] += sm_delta * scale
                │ actions[0, :] = _current_target  (替换 env_0)
                └─ GenieSimBaseEnv.step(modified_actions, auto_reset=False)
                      └─ GenieSimShmClient.step()
                            └─ Step SHM: STEP_REQUEST → 等待 STEP_DONE → 读取结果
          │ 右键按下？terminated[0]=True, info["success"]=True
          │ _maybe_flush() → 若 success → 异步写 pickle
```

---

## 14. Mode 2：本地模式实现细节

**文件**：`RLinf/rlinf/envs/geniesim/container_manager.py`

### 14.1 Python 版本关系

| 组件 | Python 版本 | 路径 |
|------|-------------|------|
| RLinf 训练代码 | 3.11（venv） | `/opt/rlinf_venv/rlinf/bin/python3` |
| sim_server.py | 3.12（系统） | `/usr/bin/python3` |
| mujoco_ros_node.py | 3.12（系统） | 同上，ROS rclpy 需要 |
| Isaac Sim Python | 3.11.13（独立） | `/isaac-sim/python.sh` |

如果 RLinf venv 激活状态下 `subprocess.Popen(["python3", ...])` 启动 sim_server.py，`PATH` 中 `/opt/rlinf_venv/rlinf/bin/python3`（3.11）排在系统 Python 3.12 前面，导致 `import rclpy` 失败。

**解决**：`_ensure_running_local()` 在 Popen 前剥离 venv：
```python
_venv = env.pop("VIRTUAL_ENV", "")
env.pop("VIRTUAL_ENV_PROMPT", None)
if _venv:
    env["PATH"] = ":".join(
        p for p in env.get("PATH", "").split(":")
        if not p.startswith(_venv)
    )
```

### 14.2 `_ensure_running_local()` 执行路径

```
_ensure_running_local(pm_kwargs)
    │
    ├── [reuse_running=true] sentinel-first 检查（不依赖 _local_proc）
    │       ├── .geniesim_ready 存在 + SHM 可访问
    │       │       → return（直接复用，~0 秒）
    │       └── .geniesim_idle 存在
    │               → 清理旧 SHM，写 sim_config.json，写 .geniesim_start
    │               → _wait_ready()（~17 秒，Isaac Sim 已启动无需重启）
    │               → return  ← 注意：此路径不设置 self._local_proc！
    │
    ├── [_local_proc 非 None 且仍在运行] → terminate/wait/kill
    │
    ├── _kill_orphan_sim_servers()
    │       扫描 /proc（跳过 PID≤1 和自身），SIGKILL 所有 sim_server.py 进程
    │       仅在无哨兵文件时执行（避免误杀受管理进程）
    │
    ├── 清理旧哨兵文件 + 旧 SHM
    ├── 写 sim_config.json（从 pm_kwargs 构建）
    ├── 构建 bash_cmd（source ROS setup → exec python3 sim_server.py）
    ├── 剥离 VIRTUAL_ENV（见 §14.1）
    ├── subprocess.Popen(["bash", "-c", bash_cmd], ...) → self._local_proc
    └── _wait_ready()（~90 秒，Isaac Sim 冷启动）
```

### 14.3 `_shutdown_local()` 执行路径

```
_shutdown_local()
    │
    ├── [keep_alive=true]
    │       proc_alive = _local_proc is not None and _local_proc.poll() is None
    │       sim_active = _ready_file.exists() or proc_alive   ← 关键：不依赖 _local_proc
    │       if not sim_active: return
    │       if _idle_file.exists(): return
    │       写 .geniesim_stop
    │       等待 .geniesim_idle（最多 60 秒）
    │       return  ← sim_server.py 主进程继续运行，等待下次 _start
    │
    └── [keep_alive=false]
            _ready_file.unlink()
            if _local_proc is not None: terminate → wait(15s) → kill
            _kill_orphan_sim_servers()
```

### 14.4 合并镜像（geniesim-rlinf-train:latest）层次

```
geniesim-rlinf:latest（base，~38.5 GB）
    Ubuntu 24.04 + Isaac Sim 5.1 + ROS 2 Jazzy + MuJoCo

Dockerfile.geniesim 新增（~19 GB）
    /opt/rlinf_venv/rlinf/        Python 3.11 venv
        torch 2.6.0+cu124
        flash-attn 2.7.4.post1（预编译 wheel）
        ray 2.54.1
        docker SDK 7.1.0
        openvla-oft, transformers, prismatic...
    /usr/local/bin/run_local.sh   Mode 2 便捷脚本
    /usr/local/bin/switch_env     venv 切换工具
```

构建命令：
```bash
cd /home/zy/code/rlinf_open_source/RLinf
docker build -f docker/Dockerfile.geniesim -t geniesim-rlinf-train:latest .
```

### 14.5 run_local.sh 做了什么

```bash
source /opt/rlinf_venv/rlinf/bin/activate   # 激活 RLinf Python 3.11 venv
export GENIESIM_ROOT=/geniesim/main          # rlinf_open_source bind-mount
export PYTHONPATH=/geniesim/main/RLinf:...  # 加入 RLinf 源码路径
source /opt/ros/jazzy/setup.bash            # ROS 2 Jazzy（训练侧通常不需要，但无害）
source /geniesim/ros_ws_build/install/setup.bash
cd /geniesim/main/RLinf
exec "$@"                                    # 运行传入的命令
```

注意：`run_local.sh` 激活了 RLinf venv，但 `_ensure_running_local()` 会在启动 sim_server.py 子进程时自动剥离该 venv，确保 sim_server.py 用系统 Python 3.12。

---

## 15. Mode 2 完整测试流程

Mode 2 指 RLinf 和 GenieSim 都运行在同一个 Docker 容器内（`GENIESIM_CONTAINER=1`）。

### 15.1 前置条件

1. 构建合并镜像：
```bash
cd /home/zy/code/rlinf_open_source/RLinf
docker build -f docker/Dockerfile.geniesim -t geniesim-rlinf-train:latest .
```

2. 目录结构要求：
   - `/home/zy/code/rlinf_open_source` — RLinf 代码仓库
   - `/home/zy/code/main` — GenieSim 代码仓库（符号链接目标）

### 15.2 Docker 运行模板

```bash
docker run --rm --ipc=host --user root \
  -v /home/zy/code/rlinf_open_source:/geniesim/main:rw \
  -v /home/zy/code/main:/geniesim/main/main:rw \
  -v /home/zy/.cache/isaacsim:/root/.cache/isaacsim:rw \
  -v /home/zy/.cache/nvidia:/root/.cache/nvidia:rw \
  -v /home/zy/.cache/ov:/root/.cache/ov:rw \
  -e GENIESIM_CONTAINER=1 \
  --network host \
  --gpus all \
  geniesim-rlinf-train:latest \
  bash /geniesim/main/RLinf/docker/run_local.sh <command>
```

**重要**：`-v /home/zy/code/main:/geniesim/main/main:rw` 是必需的，因为 `rlinf_open_source/main` 是符号链接。

### 15.3 Test 1：数据采集

```bash
docker run --rm --ipc=host --user root \
  -v /home/zy/code/rlinf_open_source:/geniesim/main:rw \
  -v /home/zy/code/main:/geniesim/main/main:rw \
  -v /home/zy/.cache/isaacsim:/root/.cache/isaacsim:rw \
  -v /home/zy/.cache/nvidia:/root/.cache/nvidia:rw \
  -v /home/zy/.cache/ov:/root/.cache/ov:rw \
  -e GENIESIM_CONTAINER=1 \
  --network host \
  --gpus all \
  geniesim-rlinf-train:latest \
  bash /geniesim/main/RLinf/docker/run_local.sh \
    python examples/embodiment/collect_sim_data.py \
      --config-path config \
      --config-name geniesim_junpu_sac \
      ++output_dir=/geniesim/main/test_output/sim_demos \
      ++num_demos=2 \
      ++max_steps_per_demo=30
```

预期输出：
- 2 个 pkl 文件保存到 `/geniesim/main/test_output/sim_demos/`
- 每个文件约 20MB

### 15.4 Test 2A：数据转换

```bash
docker run --rm --ipc=host --user root \
  -v /home/zy/code/rlinf_open_source:/geniesim/main:rw \
  -v /home/zy/code/main:/geniesim/main/main:rw \
  -v /home/zy/.cache/isaacsim:/root/.cache/isaacsim:rw \
  -v /home/zy/.cache/nvidia:/root/.cache/nvidia:rw \
  -v /home/zy/.cache/ov:/root/.cache/ov:rw \
  -e GENIESIM_CONTAINER=1 \
  --network host \
  --gpus all \
  geniesim-rlinf-train:latest \
  bash /geniesim/main/RLinf/docker/run_local.sh \
    python examples/embodiment/convert_demos_to_buffer.py \
      --config-path config \
      --config-name geniesim_junpu_sac \
      ++demo_dir=/geniesim/main/test_output/sim_demos \
      ++output_dir=/geniesim/main/test_output/demo_buffer
```

预期输出：
- `trajectory_*.pt` 文件保存到 `/geniesim/main/test_output/demo_buffer/`

### 15.5 Test 2B：Demo 回放

```bash
docker run --rm --ipc=host --user root \
  -v /home/zy/code/rlinf_open_source:/geniesim/main:rw \
  -v /home/zy/code/main:/geniesim/main/main:rw \
  -v /home/zy/.cache/isaacsim:/root/.cache/isaacsim:rw \
  -v /home/zy/.cache/nvidia:/root/.cache/nvidia:rw \
  -v /home/zy/.cache/ov:/root/.cache/ov:rw \
  -e GENIESIM_CONTAINER=1 \
  --network host \
  --gpus all \
  geniesim-rlinf-train:latest \
  bash /geniesim/main/RLinf/docker/run_local.sh \
    python examples/embodiment/replay_sim_demos.py \
      --config-path config \
      --config-name geniesim_junpu_sac \
      ++demo_dir=/geniesim/main/test_output/sim_demos
```

预期输出：
- 所有 episode 回放成功

### 15.6 Test 3：SAC 训练

注意：启动容器后需先 `colcon build` ROS 接口。推荐使用 detached 容器 + `docker exec`：

```bash
# 启动容器（detached）
docker run --rm -itd --name geniesim_sac_train \
  --gpus all --ipc=host --network=host \
  -e TORCHDYNAMO_DISABLE=1 -e PYTHONUNBUFFERED=1 \
  -v /home/zy/code/rlinf_open_source:/geniesim/main:rw \
  -v "$(realpath /home/zy/code/rlinf_open_source/main)":/geniesim/main/main:rw \
  -v ~/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw \
  -v ~/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw \
  geniesim-rlinf-train:latest

# 构建 ROS 接口
docker exec geniesim_sac_train bash -c '
source /opt/ros/jazzy/setup.bash
WS_BUILD="/geniesim/ros_ws_build"
mkdir -p ${WS_BUILD}/src
ln -sfn /geniesim/main/main/source/geniesim/rl/ros_interfaces ${WS_BUILD}/src/geniesim_rl_interfaces
cd ${WS_BUILD}
colcon build --packages-select geniesim_rl_interfaces --symlink-install
'

# 启动训练
docker exec -e TORCHDYNAMO_DISABLE=1 -e PYTHONUNBUFFERED=1 geniesim_sac_train \
  run_local.sh python examples/embodiment/train_embodied_agent.py \
    --config-name geniesim_junpu_sac
```

预期输出：
- 4 个 MuJoCo 环境启动（~75s）
- 每 rollout epoch ~6.2s（400 转移/epoch）
- Training step ~0.2s（8 update epochs × ~26ms）
- Critic loss 持续下降
- 10 个 Q-head 值正常输出

### 15.7 常见问题

1. **Stale SHM 文件**：如果 `/dev/shm/geniesim_*` 文件是 root 创建的旧文件，需要先清理：
   ```bash
   sudo rm -f /dev/shm/geniesim*
   ```

2. **Stale sentinel 文件**：如果 `.geniesim_idle` 存在但没有 sim_server 进程：
   ```bash
   rm -f /home/zy/code/rlinf_open_source/.geniesim_*
   ```

3. **符号链接问题**：确保 `/home/zy/code/main` 存在且包含 GenieSim 代码。

