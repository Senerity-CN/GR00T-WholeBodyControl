# Sonic训练指南

## 先决条件

GPU: NVIDIA GPU with CUDA 12.x (L40 recommended)

OS: Ubuntu 22.04+

Python: 3.11 (required by Isaac Lab; sim/teleop/deploy scripts work on 3.10+)

Isaac Lab: 2.3+ (required for simulation environments)

使用 conda 安装 Isaac Lab 和 gear\_sonic (Training) 环境

```plaintext
pip install -e "gear_sonic/[training]"

```

## 模型和数据准备

SONIC model checkpoints and SMPL motion data are hosted on [Hugging Face](https://huggingface.co/nvidia/GEAR-SONIC).

```plaintext
pip install huggingface_hub
python download_from_hf.py --training

```

This downloads:

PyTorch checkpoint (`sonic_release/last.pt`) for finetuning

SMPL motion data (`data/smpl_filtered/`) for the SMPL encoder

Robot Motion Data 位于  [Bones-SEED](https://huggingface.co/datasets/bones-studio/seed) motion capture dataset (142K+ motion sequences retargeted to the Unitree G1).

### Step 1: Download and convert

Download the G1 retargeted CSVs (29 DOF, 120 FPS) from [Bones-SEED on HuggingFace](https://huggingface.co/datasets/bones-studio/seed), then convert:

```plaintext
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
    --input /path/to/bones_seed/g1/csv/ \
    --output data/motion_lib_bones_seed/robot \
    --fps 30 --fps_source 120 --individual --num_workers 16

```

### Step 2: Filter motions

Remove motions the G1 robot cannot perform:

```plaintext
python gear_sonic/data_process/filter_and_copy_bones_data.py \
    --source data/motion_lib_bones_seed/robot \
    --dest data/motion_lib_bones_seed/robot_filtered --workers 16

```

This removes ~8.7% of motions (~130K of 142K remain). See the [Training Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/training.html) for details.

Your data directory should look like:

```plaintext
<repo_root>/
├── data/
│   ├── motion_lib_bones_seed/
│   │   └── robot_filtered/     # Filtered G1 motions (~130K PKLs)
│   └── smpl_filtered/           # SMPL motion data (from Hugging Face)
└── sonic_release/               # Released checkpoint (from Hugging Face)

```

## 验证能否运行

环境验证

```plaintext
python check_environment.py --training

```

Then run a quick smoke test with a small number of environments:

```plaintext
# Interactive (with viewer)
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    num_envs=16 headless=False \
    ++algo.config.num_learning_iterations=5

```

After a minute of initialization you should see training metrics (rewards, errors) printing to the console.

---

## 概述

SONIC 使用一种 **通用 token 架构** 来控制人形机器人（Unitree G1，29 个自由度，不含手），方法是模仿人体动作捕捉数据。  
多个并行编码器可以接收不同格式的动作输入：

*   G1：机器人关节轨迹
    
*   Teleop：VR 三点跟踪目标（头部 + 两只手腕）
    
*   SMPL：参数化人体模型的关节位置
    
*   SOMA：由 BVH 推导出的骨架关节位置（可选的第 4 个编码器）
    

所有编码器都会通过 FSQ（ Finite Scalar Quantization） 投影到一个共享的潜在 token 空间中，而单个解码器则不受输入模态影响，统一输出关节动作。  
训练在 Isaac Lab 仿真环境中使用 PPO，并结合辅助损失进行优化。

| Config | Encoders | Use case |
| --- | --- | --- |
| `sonic_release` | G1, teleop, SMPL | Default — matches the released checkpoint |
| `sonic_bones_seed` | G1, teleop, SMPL, SOMA | Extended training with SOMA skeleton encoder |

建议在微调和评估时使用 `sonic_release`。`sonic_bones_seed` 配置会增加第 4 个 SOMA 编码器。

## 训练

### Basic command

```plaintext
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/robot_filtered> \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=<path/to/smpl_filtered>

```

### Finetuning from the released checkpoint

```plaintext
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    +checkpoint=sonic_release/last.pt \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/robot_filtered> \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=<path/to/smpl_filtered>

```

### Multi-GPU and multi-node training

We recommend training with 64+ GPUs for reasonable convergence times. Single-node (8 GPU) training works but is significantly slower.

```plaintext
# Single node (8 GPUs)
accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    num_envs=4096 headless=True

# Multi-node — use accelerate config for distributed setup
accelerate launch \
    --multi_gpu \
    --num_machines=8 \
    --num_processes=64 \
    --machine_rank=$MACHINE_RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    num_envs=4096 headless=True

```

For multi-node setup, see the [Accelerate distributed training guide](https://huggingface.co/docs/accelerate/usage_guides/deepspeed) and [multi-node launcher docs](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch).

好像起多个gpu训练要配置一下

## 评估

### 查看参考动作

Replay motions to verify data quality before training:

```plaintext
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    ++replay=True num_envs=4 headless=False

```

### 评估一个 checkpoint

Two eval modes: **metrics** (success rate, MPJPE) and **render** (video output).

For the released checkpoint, you must override motion paths since its `config.yaml` has internal training paths. For your own checkpoints trained with `sonic_release`, omit the motion overrides.

```plaintext
# --- Metrics ---
python gear_sonic/eval_agent_trl.py \
    +checkpoint=<path_to_checkpoint.pt> \
    +headless=True \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++num_envs=128 \
    "+manager_env/terminations=tracking/eval" \
    "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=512"

```
```plaintext
# --- Render videos ---
python gear_sonic/eval_agent_trl.py \
    +checkpoint=<path_to_checkpoint.pt> \
    +headless=True \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++num_envs=8 \
    ++manager_env.config.render_results=True \
    "++manager_env.config.save_rendering_dir=/tmp/renders" \
    ++manager_env.config.env_spacing=10.0 \
    "~manager_env/recorders=empty" "+manager_env/recorders=render"

```

For the released checkpoint only, append this override to either command (its embedded config has internal training paths):

```plaintext
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered"

```

Videos are saved as `000000.mp4`, `000001.mp4`, etc. in `save_rendering_dir`.

### 期望的评估指标

Training rewards (W&B `Episode_Reward/`):

| Metric | Converged | Description |
| --- | --- | --- |
| `tracking_vr_5point_local` | \> 0.80 | 5-point tracking quality |
| `tracking_relative_body_pos` | \> 0.44 | Upper-body position tracking |
| `tracking_anchor_pos` | \> 0.14 | Root position tracking |
| `time_out` | \> 0.90 | Episode completion rate |

Eval metrics (from `eval_agent_trl.py`):

| Metric | Converged | Description |
| --- | --- | --- |
| `success_rate` | \> 0.97 | Motions tracked without early termination |
| `mpjpe_l` | < 30 mm | Local per-joint position error |
| `mpjpe_g` | < 200 mm | Global per-joint position error |

A well-converged policy reaches >0.98 success rate and <29 mm mpjpe\_l after 100K iterations.

## ONNX 导出

Export a trained checkpoint to ONNX for C++ deployment:

```plaintext
python gear_sonic/eval_agent_trl.py \
    +checkpoint=<path_to_checkpoint.pt> \
    +headless=True ++num_envs=1 \
    +export_onnx_only=true

```

For the released checkpoint, append the motion path overrides shown in the eval section above.

Output (in `exported/` next to the checkpoint):

| File | Description |
| --- | --- |
| `*_smpl.onnx` | SMPL encoder + decoder (pose estimation input) |
| `*_g1.onnx` | G1 encoder + decoder (robot joint input) |
| `*_teleop.onnx` | Teleop encoder + decoder (VR tracking input) |
| `*_encoder.onnx` | All encoders combined |
| `*_decoder.onnx` | Decoder only |

Use the encoder+decoder pair matching your input modality. See [deployment code reference](https://nvlabs.github.io/GR00T-WholeBodyControl/references/deployment_code.html) for C++ details.

## 使用 SOMA encoder 进行训练

The `sonic_bones_seed` config adds a fourth SOMA encoder for BVH-derived skeleton joint positions.

### SOMA data preparation

```plaintext
# Extract SOMA joints from BVH
python gear_sonic/data_process/extract_soma_joints_from_bvh.py \
    --input /path/to/bones_seed/bvh/ \
    --output data/motion_lib_bones_seed/soma \
    --fps 30 --num_workers 16 --skip_existing

# Filter to match robot data
python gear_sonic/data_process/filter_and_copy_bones_data.py \
    --source data/motion_lib_bones_seed/soma \
    --dest data/motion_lib_bones_seed/soma_filtered \
    --workers 16

```

### Training

Use multi-node training (64+ GPUs recommended):

```plaintext
accelerate launch \
    --multi_gpu --num_machines=8 --num_processes=64 \
    --machine_rank=$MACHINE_RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_bones_seed \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.soma_motion_file=data/motion_lib_bones_seed/soma_filtered

```

Data layout for 4-encoder training:

```plaintext
data/
├── motion_lib_bones_seed/
│   ├── robot_filtered/     # ~130K PKLs (G1 retargeted)
│   └── soma_filtered/      # ~130K PKLs (SOMA skeleton)
└── smpl_filtered/          # ~131K PKLs (SMPL human)
```