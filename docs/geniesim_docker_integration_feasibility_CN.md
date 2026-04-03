# GeneSim 镜像整合到 RLinf Dockerfile 可行性评估

> 评估能否将 GeneSim 容器（Isaac Sim + MuJoCo + ROS2）整合进 RLinf 的 Dockerfile，以及替代方案

---

## 一句话结论

**技术上可行，但代价极大，不推荐。**  
现有双容器架构（RLinf 训练 + GeneSim 仿真）是正确的设计，应当保留。  
若想改善用户体验（减少手动操作），推荐用 **Docker Compose** 方案代替镜像合并。

---

## 当前架构说明

```
Host（宿主机）                     Docker 容器（GeneSim）
────────────────────               ──────────────────────────────────
RLinf 训练进程                      sim_server.py（进程管理）
  │                                  ├── MuJoCo 物理节点 ×N（1000 Hz）
  │   ① 启动容器                     ├── Isaac Sim 渲染器（30 Hz，写图像到 SHM）
  │──────────────────────────────>   └── ROS2 内部通信（不出容器）
  │                                  
  │   ② 等待 .geniesim_ready 文件
  │<──────────────────────────────
  │
  │   ③ SHM 双向通信（无 ROS，无网络）
  │<════════════════════════════>  Frame SHM（图像）+ Ctrl SHM（状态/动作）
  │
  └── GenieSimShmClient（纯 stdlib，无 geniesim 依赖）
```

**两个镜像的基本信息：**

| | RLinf Dockerfile | GeneSim Dockerfile |
|--|--|--|
| 基础镜像 | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` | `registry.agibot.com/genie-sim/open_source:latest` |
| 操作系统 | Ubuntu 22.04 | **Ubuntu 24.04** |
| Python | 3.10+ | **3.12** + ROS2 Jazzy |
| 额外软件 | Ray, PyTorch, HuggingFace, 多个 venv | **Isaac Sim 5.1**, ROS2 Jazzy, MuJoCo |
| 镜像大小 | ~20 GB | **~80-100 GB**（含 Isaac Sim cache）|
| 进程模型 | 单进程训练 | 多进程仿真（MuJoCo ×N + Isaac Sim）|

---

## 整合的技术障碍

### 障碍 1：基础镜像不兼容（最大障碍）

Isaac Sim 5.1 强依赖 **Ubuntu 24.04 + NVIDIA Driver 550+**，而 RLinf 当前使用 Ubuntu 22.04。

- 要整合，必须先将 RLinf 基础镜像升级到 Ubuntu 24.04
- 这是一次**全面破坏性变更**：所有现有依赖、venv、构建脚本都需验证
- Isaac Sim 本身不是一个 pip 包——它是一个完整的应用程序，占用约 50-80 GB 磁盘空间

### 障碍 2：进程模型根本不同

| | RLinf 训练容器 | GeneSim 仿真容器 |
|--|--|--|
| 进程数量 | 1 个 Python 进程（+ Ray worker 子进程）| MuJoCo ×N + Isaac Sim + sim_server（同时运行）|
| 生命周期 | 训练完成即退出 | 持久运行（keep_alive 模式），等待下一次训练 |
| 启动时间 | 秒级 | **1-2 分钟**（Isaac Sim 冷启动）|
| 资源竞争 | GPU 用于 VLA 模型推理/训练 | GPU 用于 Isaac Sim 渲染 |

**合并后两者需共享 GPU，导致显存严重冲突**：Isaac Sim 本身需要 8-16 GB 显存，VLA 训练需要 40-80 GB。

### 障碍 3：镜像体积膨胀

- 当前 RLinf 镜像：~20 GB
- 整合后：**~100-120 GB**
- Isaac Sim 的 asset cache 单独可达 30-50 GB（用户首次运行时下载）
- CI/CD push/pull 时间成倍增加

### 障碍 4：User/Permission 差异

GeneSim Dockerfile 明确切换到 `uid=1234`（isaac-sim user）并使用 `setfacl` 处理 bind mount 权限。RLinf 通常以 root 或不同用户运行，合并需要统一权限模型。

### 障碍 5：版本耦合

当前两个镜像可以独立更新：
- Isaac Sim 发布新版本 → 只更新 GeneSim 镜像
- PyTorch/Ray 升级 → 只更新 RLinf 镜像

整合后，任何一方更新都需要重新构建并验证整个 100 GB 镜像。

---

## 替代方案评估

### 方案 A：保持现状（双独立镜像）✅ 推荐继续

**现有设计已经很好：**
- RLinf host 已做到零 geniesim Python 依赖（只用 stdlib SHM）
- `SimContainerManager` 自动管理容器生命周期
- SHM 通信零拷贝，性能极佳
- 两个镜像独立版本管理

**痛点**：用户需要手动构建 GeneSim 镜像、了解容器配置

---

### 方案 B：Docker Compose 编排（推荐改善方向）

不合并镜像，而是提供一个 `docker-compose.yml`，**一条命令拉起两个容器**：

```yaml
# docker-compose.yml（示意）
services:
  geniesim:
    image: geniesim-rlinf:latest
    runtime: nvidia
    ipc: host                    # 关键：共享 /dev/shm
    network_mode: host
    volumes:
      - ${GENIESIM_ROOT}:/geniesim/main
      - ~/docker/isaac-sim:/root/.cache/ov
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    restart: unless-stopped

  rlinf:
    image: rlinf:latest
    runtime: nvidia
    ipc: host                    # 关键：访问同一个 /dev/shm
    network_mode: host
    volumes:
      - ${REPO_PATH}:/workspace/RLinf
      - ${GENIESIM_ROOT}:/geniesim_root
    depends_on:
      - geniesim
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

**优势：**
- 用户只需 `docker-compose up`，无需了解 SHM 细节
- `ipc: host` 确保两个容器共享 `/dev/shm`（SHM 通信的基础）
- 镜像保持独立，可以分别更新
- `depends_on` 确保 GeneSim 先启动

**实施工作量**：约 1-2 天，主要是测试 `ipc: host` 在 compose 下的 SHM 共享是否正常工作

---

### 方案 C：单一镜像（不推荐）

| 评估项 | 结论 |
|--------|------|
| 技术可行性 | 可行，但需要大量工作 |
| 基础镜像升级 | 必须升级到 Ubuntu 24.04（破坏性变更）|
| 构建时间 | 每次重建 6-12 小时（Isaac Sim 编译/下载）|
| 镜像大小 | ~100-120 GB |
| GPU 资源冲突 | 严重（Isaac Sim 渲染 vs VLA 训练共享 GPU）|
| 维护复杂度 | 显著增加 |
| 适用场景 | 几乎没有合理场景 |

---

## 决策建议

```
短期（改善用户体验）：提供 Docker Compose 配置
  → 一条命令拉起两个容器
  → 无需了解 SHM 和容器配置细节
  → 工作量小，风险低

中长期：保持双镜像架构
  → Isaac Sim 版本迭代独立
  → RLinf 训练镜像保持轻量
  → SHM 通信已经是最优方案

不建议：合并为单一镜像
  → 代价大，收益小
  → GPU 资源冲突难以解决
  → 100 GB+ 镜像维护成本高
```

---

## 关键文件

| 文件 | 说明 |
|------|------|
| `RLinf/docker/Dockerfile` | RLinf 训练镜像（Ubuntu 22.04 + CUDA 12.4）|
| `/home/zy/code/main/scripts/dockerfile_geniesim_rlinf` | GeneSim 仿真镜像（Ubuntu 24.04 + Isaac Sim 5.1）|
| `rlinf/envs/geniesim/container_manager.py` | 容器生命周期管理（docker run 参数、readiness 探测）|
| `rlinf/envs/geniesim/shm_client.py` | host 侧 SHM 通信（纯 stdlib，无 geniesim 依赖）|
| `rlinf/envs/geniesim/shm_layout.py` | SHM 内存布局常量（host/container 两侧必须一致）|
| `examples/embodiment/config/env/geniesim_junpu_place_workpiece.yaml` | `container_cfg` 字段（镜像名、路径映射等）|
