# SONIC 微调 Docker 镜像 → MLflow 全流程

### 一、设计思路

| 层 | 内容 | 位置 |
| --- | --- | --- |
| 镜像（不变的） | 系统库 + Python 3.11 + PyTorch 2.7+cu128 + IsaacSim 5.1 + IsaacLab + gear\_sonic | Docker 镜像 ~25-30GB |
| 数据（大且可能变） | robot\_filtered (8.6GB) + smpl\_filtered (31GB) + checkpoint (448MB) | OSS 挂载 |
| 代码（频繁变） | GR00T-WholeBodyControl 项目代码 | OSS 挂载 或 git clone |

镜像只装环境，数据/代码运行时挂载。改了代码不用重新 build 镜像。

---

### 二、构建步骤

```bash
cd /home/balance/GR00T-WholeBodyControl

# 1. 准备构建上下文（复制 IsaacLab 源码到项目目录）
bash docker/prepare_build_context.sh

# 2. 构建镜像（首次需加 --no-cache）
sudo docker build --no-cache -f docker/Dockerfile -t sonic-finetune-env:v1 .

# 3. 本地验证
sudo docker run --gpus all --rm -it \
  -v /home/balance/GR00T-WholeBodyControl:/workspace/GR00T \
  sonic-finetune-env:v1 \
  python -c "import torch; import isaaclab; print('OK', torch.cuda.device_count(), 'GPUs')"
# 输出: OK 1 GPUs ✓ (2026-06-09 验证通过)
```
---

### 三、推送到弹内仓库 → 同步到公有云

```bash
# 1. tag
sudo docker tag sonic-finetune-env:v1 hub.docker.alibaba-inc.com/xlab/sonic-finetune-env:v1

# 2. 登录（用工号 + artlab 临时 token）
sudo docker login hub.docker.alibaba-inc.com -u <你的工号>
#    密码：去 artlab.alibaba-inc.com → hub-docker → 右上角「获取临时授权」→ 复制 token
#    注意：如果 push 报 denied，先 docker logout 再重新 login

# 3. push
sudo docker push hub.docker.alibaba-inc.com/xlab/sonic-finetune-env:v1

# 4. 同步到杭州公有云
#    artlab 界面 → 搜索 xlab/sonic-finetune-env → 点进去 → 「镜像同步」→ 目标选「杭州公有云docker仓库」
```
---

### 四、上传数据与代码到 OSS

OSS bucket: `oss://all-dataset-external/pingheng/Sonic-finetune/` (region: cn-zhangjiakou)

```
oss://all-dataset-external/pingheng/Sonic-finetune/
├── motion_lib/                           # [已上传] 全部 robot PKL
│   ├── robot_filtered/                   #   Bones-SEED 129,785 PKLs（按日期子目录）
│   └── robot_finetune/                   #   Kimodo finetune 144 PKLs（扁平）
├── smpl/                                 # [已上传] 全部 SMPL PKL（扁平，已合并 finetune）
├── smpl_finetune/                        # [已上传] 144 条 finetune SMPL PKLs（备份）
├── checkpoints/last.pt                   # [已上传] 448MB release checkpoint
├── raw/bones_seed_g1_csv/                # 原始 Bones-SEED G1 CSV（按日期子目录）
└── GR00T/                               # [已上传] 训练代码
    ├── gear_sonic/                       #   训练代码 + Hydra 配置 + 机器人资产
    ├── docker/entrypoint.sh              #   MLflow 入口脚本
    └── pyproject.toml                    #   项目配置
```

**设计要点**：
- `motion_lib/` 下 `robot_filtered/` + `robot_finetune/` 两个子目录，训练时递归 glob 自动找到所有 PKL
- `smpl/` 必须扁平（`motion_lib_base.py` 用 `osp.join(smpl_dir, seq + ".pkl")` 匹配），finetune PKL 已合并进去
- `GR00T/` 包含运行训练所需的全部代码，挂载后 `entrypoint.sh` 通过 `PYTHONPATH` 覆盖镜像内的 gear_sonic
---

### 五、MLflow 启动任务

| 配置项 | 值 |
| --- | --- |
| 镜像 | `hub.docker.alibaba-inc.com/xlab/sonic-finetune-env:v1` |
| OSS 挂载 | `oss://all-dataset-external/pingheng/Sonic-finetune/` → `/workspace/data` |
| 启动命令 | `bash /workspace/data/GR00T/docker/entrypoint.sh` |
| 环境变量 | `GPU_COUNT=8`, `NUM_ENVS=4096`, `WANDB_API_KEY=xxx` |

只需一次 OSS 挂载，数据和代码都在同一个 bucket 下。`entrypoint.sh` 默认 `CODE_DIR=${DATA_DIR}/GR00T`。

**entrypoint.sh 关键路径**（均相对于 `DATA_DIR=/workspace/data`）：

| 参数 | 路径 | 说明 |
| --- | --- | --- |
| `checkpoint` | `${DATA_DIR}/checkpoints/last.pt` | release checkpoint |
| `motion_file` | `${DATA_DIR}/motion_lib` | 递归 glob 覆盖 robot_filtered/ + robot_finetune/ |
| `smpl_motion_file` | `${DATA_DIR}/smpl` | 扁平目录，已含全部 SMPL PKL |

---

### 六、已知问题与解决

| 问题 | 解决 |
| --- | --- |
| Docker Hub 超时 | 基础镜像改用 `ubuntu:22.04`（阿里镜像加速有缓存），CUDA 由 PyTorch pip 包自带 |
| curl 报 certificate 错误 | apt 先装 `ca-certificates`，构建时加 `--no-cache` |
| push 报 denied | 先 `docker logout` 再重新 `docker login` |
| SMPL 帧数不对齐 (250 vs 249) | 修复 `convert_kimodo_soma_to_smpl_lib.py` 帧数公式，使用 `int((T-1)/src_fps * tgt_fps) + 1` |

---

### 七、文件清单

```plaintext
GR00T-WholeBodyControl/
├── docker/
│   ├── Dockerfile                 # 镜像定义
│   ├── entrypoint.sh             # MLflow 训练入口脚本
│   ├── prepare_build_context.sh  # 准备构建上下文（复制 IsaacLab 源码）
│   ├── build_mlflow_image.sh     # 一键构建+提示推送
│   ├── requirements_train.txt    # 训练 pip 依赖
│   └── wheels/                   # PyTorch 离线 wheel
├── .dockerignore                  # 排除数据/日志等（构建加速）
└── IsaacLab/                      # (构建时由 prepare_build_context.sh 复制，已加入 .gitignore)
```
