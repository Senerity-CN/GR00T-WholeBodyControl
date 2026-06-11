# SONIC 遥操作问题分析与微调方案

### 问题诊断

| 问题 | 根因分析 |
| --- | --- |
| 后退打滑 | Bones-SEED 数据集中后退动作占比少，策略对反向运动的脚底接触学习不充分 |
| 小幅走动跟踪差 | 训练数据以正常步幅为主，微小位移属于 OOD（out of distribution） |
| 伸手传递时肩膀不动 | teleop encoder 的 3-point tracking（头+双手）缺乏肩部中间关节的显式约束，且训练时上肢增强（`cat_upper_body_poses`）可能未覆盖此类场景 |

---

### 方案概览

**Kimodo 文本生成 → SOMA Retargeter → G1 Robot PKL + SMPL PKL → 从 release checkpoint 微调**

走完整管线：Kimodo 生成 BVH → SOMA Retargeter 转为 G1 CSV → 转为 robot motion_lib PKL，同时也生成对应的 SMPL PKL。两种数据按文件名对齐，确保 G1 encoder 和 SMPL encoder 都能消费到新动作。

---

### 当前进度

| 步骤 | 状态 | 说明 |
| --- | --- | --- |
| ~~下载 Bones-SEED G1 数据~~ | ✅ 已完成 | g1.tar.gz 23.5GB |
| ~~转换 Bones-SEED CSV → motion_lib PKL~~ | ✅ 已完成 | 142,220 条 robot PKL |
| ~~过滤不可执行动作~~ | ✅ 已完成 | 129,785 条 robot_filtered PKL |
| ~~解压 smpl_mixed（原 bones_seed_smpl）~~ | ✅ 已完成 | 131,455 条 SMPL PKL |
| ~~Kimodo 生成补充动作~~ | ✅ 已完成 | v1 脚本 75 条 + v2 脚本 ~183 条 ≈ 258 条原始动作 |
| ~~人工筛选~~ | ✅ 已完成 | 逐条检查全部生成动作，保留 144 条质量合格数据 |
| ~~转换 Kimodo NPZ → SMPL PKL~~ | ✅ 已完成 | 144 条 smpl_finetune PKL |
| ~~检查 SMPL 数据格式~~ | ✅ 已完成 | 全部通过：pose_aa(T,72), smpl_joints(T,24,3), transl(T,3), fps=50.0 |
| ~~BVH → SOMA Retargeter → G1 CSV~~ | ✅ 已完成 | 144 条 BVH → Newton IK → 144 个 G1 29-DOF CSV（81s GPU加速） |
| ~~CSV → Robot motion_lib PKL~~ | ✅ 已完成 | 144 条 robot PKL，0 failures |
| ~~配置混合数据目录~~ | ✅ 已完成 | motion_lib_mixed（129,785 + 144 = 129,929），SMPL 合入 smpl_mixed（131,455 + 144 = 131,599） |
| ~~数据完整性验证~~ | ✅ 已完成 | 144/144 robot-SMPL 文件名对齐确认 |
| ~~本地 4090 试跑~~ | ✅ 已完成 | Docker 容器内单 GPU 验证通过，训练循环正常运行 |
| 多 GPU 正式微调 | ⏳ 待做 | 从 release checkpoint 继续（推荐 8+ GPU） |

---

### 数据目录结构（当前状态）

```
GR00T-WholeBodyControl/data/
├── motion_lib_finetune/         # Kimodo → SOMA Retargeter → G1 CSV → robot PKL (144 PKLs，扁平)
├── motion_lib_mixed/            # ★ 微调用混合 robot 目录 (129,929 PKLs)
│   ├── original → /home/balance/GEAR-SONIC/sample_data/robot_filtered  (129,785 PKLs, 按日期子目录)
│   └── finetune → /home/balance/GR00T-WholeBodyControl/data/motion_lib_finetune  (144 PKLs)
├── smpl_mixed → /home/balance/GEAR-SONIC/smpl_filtered
│                                # 131,455 + 144 = 131,599 PKLs（扁平，无子目录）
└── smpl_finetune/               # Kimodo NPZ 转换的 SMPL 数据（转换源，已合入 smpl_mixed）
```

外部数据位置：
- Bones-SEED robot_filtered: `/home/balance/GEAR-SONIC/sample_data/robot_filtered/` (129,785 PKLs，按日期子目录组织)
- Bones-SEED SMPL: `/home/balance/GEAR-SONIC/smpl_filtered/` (131,599 PKLs，扁平)
- Checkpoint: `/home/balance/GEAR-SONIC/sonic_release/last.pt` (448MB)

OSS 备份：`oss://all-dataset-external/pingheng/Sonic-finetune/`
- `smpl_finetune/` — 144 SMPL PKLs
- `motion_lib_finetune/` — 144 robot PKLs

**注意**：
1. SMPL 数据必须放在扁平目录（不能有子目录），因为 `motion_lib_base.py` 的 SMPL 匹配逻辑用 `osp.join(smpl_motion_file, seq + ".pkl")` 构造路径，假设 PKL 直接在顶层。
2. `motion_lib_mixed/` 的 symlink 使用绝对路径，仅本地有效。Docker/MLflow 环境需通过 OSS 挂载重建数据路径。
3. Robot motion 目录支持递归 glob（子目录结构不影响加载）。

---

### 数据格式说明

SONIC 训练同时需要两种数据：

**1. robot_filtered（G1 encoder 消费）** — 主训练数据

```python
{
    "motion_name": {
        "root_trans_offset": np.float32 (T, 3),     # 根节点世界坐标位移
        "pose_aa": np.float32 (T, 30, 3),           # 各 body 的 axis-angle 姿态
        "dof": np.float32 (T, 29),                  # G1 29-DOF 关节角（MuJoCo 顺序，弧度）
        "root_rot": np.float32 (T, 4),              # 根节点旋转四元数（xyzw）
        "smpl_joints": np.float32 (T, 24, 3),       # SMPL 关节位置
        "fps": int,                                  # 帧率（30）
    }
}
```

**2. smpl_motion_file（SMPL encoder 消费）** — 辅助数据，按文件名与 robot_filtered 对齐

```python
{
    "pose_aa": np.float32 (T, 72),           # SMPL 24 关节 axis-angle
    "smpl_joints": np.float32 (T, 24, 3),   # SMPL 24 关节 3D 位置
    "transl": np.float32 (T, 3),            # 根位移
    "fps": float,                            # 帧率（50）
}
```

---

### 关键技术决策：为什么必须走完整管线

**早期设想**：Kimodo NPZ → 直接索引映射 → SMPL PKL，跳过 SOMA Retargeter 避免 IK 噪声

**问题**：SONIC 的 SMPL 数据是**被动查找**的——`motion_lib_base.py:269-276` 按 robot motion 文件名去匹配 SMPL PKL。采样由 `sample_motions()` 驱动，只从 robot motion 中选取。没有对应 robot PKL 的 SMPL 数据永远不会被采样到，**完全不参与训练**。

**当前方案**：走完整管线，同时生成 robot PKL 和 SMPL PKL，确保两条路径都有数据：

```
Kimodo 文本 prompt
    │
    ▼
Kimodo SOMA-RP-v1.1 生成（.npz + .bvh，30fps）
    │
    ├──[NPZ 路径]──→ convert_kimodo_soma_to_smpl_lib.py ──→ SMPL PKL
    │                 SOMA77 关节索引映射 → SMPL24                 ↓
    │                                                    合入 smpl_mixed/
    │
    └──[BVH 路径]──→ SOMA Retargeter (Newton IK) ──→ G1 29-DOF CSV
                                                           │
                                                           ▼
                                          convert_soma_csv_to_motion_lib.py
                                                           │
                                                           ▼
                                              Robot motion_lib PKL
                                                           ↓
                                                  加入 motion_lib_mixed/
```

训练时 `sample_motions()` 采样到 finetune 的 robot motion → 同时通过文件名找到对应 SMPL 数据 → G1 encoder 和 SMPL encoder 都参与该动作的训练 → latent loss 对齐两个 encoder 的表征。

SOMA Retargeter 配置：`/home/balance/soma-retargeter/assets/kimodo_finetune_config.json`

---

### 补充动作详情（Kimodo 生成 → 人工筛选，已完成）

生成脚本（两版）：
- **v1**：`/home/balance/kimodo/generate_finetune_motions.sh` — 75 条，每条 5 秒，单 sample
- **v2**：`/home/balance/kimodo/generate_finetune_motions_v2.sh` — 135 prompts（其中 24 条 ×3 samples），每条 8-10 秒

模型：`Kimodo-SOMA-RP-v1.1`，同时输出 `.npz` + `.bvh`
总生成量：**~258 条**，经逐条人工筛选后保留 **144 条**质量合格数据
转换脚本：`gear_sonic/data_process/convert_kimodo_soma_to_smpl_lib.py`

#### v1 生成类别（75 prompts × 1 sample = 75 条）

| 类别 | prompts | prompt 示例 |
| --- | --- | --- |
| 后退动作 backward_walking/ | 25 | "A person walks backward slowly" |
| 小碎步 small_steps/ | 25 | "A person shuffles in place" |
| 上肢伸展 upper_body_reach/ | 25 | "A person reaches forward with both hands" |

#### v2 补充类别（135 prompts，部分 ×3 samples ≈ 183 条）

| 类别 | prompts | 多 sample 数 | prompt 示例 |
| --- | --- | --- | --- |
| 拖椅子后退 chair_pull/ | 40 | 8 prompts ×3 | "A person bends forward, grabs a chair with both hands, and pulls it backward" |
| 递菜单/递物 menu_handover/ | 30 | 6 prompts ×3 | "A person holds a flat object with both hands and presents it forward" |
| 后退补充 backward_ext/ | 20 | — | "A person walks backward while holding arms forward" |
| 碎步补充 small_steps_ext/ | 20 | — | "A person pivots in place turning to the left" |
| 组合动作 combined/ | 25 | 3 prompts ×3 | "A person walks forward, extends an object, then steps backward" |

#### 人工筛选

对全部 ~258 条生成动作逐一检查，筛除动作质量不佳（穿模、滑步严重、动作不合理等）的片段，最终保留 **144 条**用于微调训练。

---

### 下一步：微调训练

#### 已完成的数据处理步骤（记录备查）

<details>
<summary>Step 1-3 已执行完毕（点击展开查看命令）</summary>

**1. BVH 扁平化 + SOMA Retargeter**

```bash
# 直接展平到 /home/balance/kimodo/outputs/（NPZ+BVH 同级）
# 运行 SOMA Retargeter（headless 批处理，保持 30fps，GPU加速 81s 完成）
cd /home/balance/soma-retargeter
uv run python app/bvh_to_csv_converter.py \
    --config assets/kimodo_finetune_config.json --viewer null
# 输出：/home/balance/kimodo/outputs_retargeted_csv/ 下 144 个 CSV
```

**2. G1 CSV → Robot Motion Lib PKL**

```bash
cd /home/balance/GR00T-WholeBodyControl
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
    --input /home/balance/kimodo/outputs_retargeted_csv \
    --output data/motion_lib_finetune \
    --fps 30 --individual
# 144 条 PKL，0 failures
```

**3. 组织训练数据目录**

```bash
# Robot motion: 混合目录（支持递归 glob，子目录结构无影响）
mkdir -p data/motion_lib_mixed
ln -sf $(realpath data/motion_lib_bones_seed/robot_filtered) data/motion_lib_mixed/original
ln -sf $(realpath data/motion_lib_finetune) data/motion_lib_mixed/finetune

# SMPL: 直接复制 144 个 finetune PKL 到 smpl_mixed 指向的扁平目录
cp data/smpl_finetune/*.pkl /home/balance/GEAR-SONIC/smpl_filtered/
```

</details>

#### 本地 4090 试跑（已验证通过）

Docker 容器内运行，仅用 finetune 小数据集快速验证配置：

```bash
sudo docker run --gpus all --rm -it \
  -v /home/balance/GR00T-WholeBodyControl:/workspace/GR00T \
  -v /home/balance/GEAR-SONIC:/workspace/GEAR-SONIC \
  -w /workspace/GR00T \
  sonic-finetune-env:v1 \
  python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    project_name=SONIC_Finetune_Test \
    exp_var=small_test \
    +checkpoint=/workspace/GEAR-SONIC/sonic_release/last.pt \
    num_envs=64 \
    headless=True \
    use_wandb=false \
    '++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_finetune' \
    '++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_finetune'
```

验证结果：12+ iteration 正常运行，mean_reward=8.37，adaptive sampling 正常工作。

**方式 B：本地 conda 环境直接运行（无 Docker）**

需要先激活已安装 IsaacLab 的 conda 环境：

```bash
conda activate isaaclab-2.3.0-py311
cd /home/balance/GR00T-WholeBodyControl

# 仅 finetune 数据快速验证
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    project_name=SONIC_Finetune_Test \
    exp_var=small_test \
    +checkpoint=/home/balance/GEAR-SONIC/sonic_release/last.pt \
    num_envs=64 \
    headless=True \
    use_wandb=false \
    '++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_finetune' \
    '++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_finetune'

# 本地单 GPU 完整混合数据
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    project_name=SONIC_Finetune \
    exp_var=backward_shuffle_reach \
    +checkpoint=/home/balance/GEAR-SONIC/sonic_release/last.pt \
    num_envs=1024 \
    headless=True \
    use_wandb=false \
    '++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_mixed'
```

**方式 C：本地 Docker 容器运行**

```bash
# 本地单 GPU 完整数据
sudo docker run --gpus all --rm -it \
  -v /home/balance/GR00T-WholeBodyControl:/workspace/GR00T \
  -v /home/balance/GEAR-SONIC:/workspace/GEAR-SONIC \
  -w /workspace/GR00T \
  sonic-finetune-env:v1 \
  python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    project_name=SONIC_Finetune \
    exp_var=backward_shuffle_reach \
    +checkpoint=/workspace/GEAR-SONIC/sonic_release/last.pt \
    num_envs=1024 \
    headless=True \
    use_wandb=false \
    '++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_mixed'
```

#### 多 GPU 正式微调（[Docker 镜像 + MLflow 全流程](SONIC%20微调%20Docker%20镜像%20→%20MLflow%20全流程.md)）

```bash
accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    project_name=SONIC_Finetune \
    exp_var=backward_shuffle_reach \
    +checkpoint=/home/balance/GEAR-SONIC/sonic_release/last.pt \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_mixed \
    use_wandb=true
```

#### 关键参数

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `checkpoint` | `GEAR-SONIC/sonic_release/last.pt` | 从 release 继续（448MB） |
| `motion_file` | `data/motion_lib_mixed` | **需要 override**：原始 129,785 + 新增 144 = 129,929 条 |
| `smpl_motion_file` | `data/smpl_mixed`（默认值） | 无需 override：已包含 131,455 + 144 = 131,599 条 |
| `num_envs` | 1024（单 GPU）/ 4096（多 GPU） | 单 4090 用 1024 避免 OOM |
| `max_grad_norm` | 0.1（已是默认） | 微调保持小梯度 |
| GPU 数量 | 1（本地测试）/ 8+（正式训练） | 文档推荐 64 GPU 才有合理收敛速度 |

#### 训练要点

- **Adaptive Sampling**：配置中 `adp_samp_failure_rate_max_over_mean: 200` 会自动对失败率高的动作增加采样，新增动作初期失败率高会被重点训练
- **上肢增强**：`upper_body_augment_prefixes` 控制哪些动作做上肢姿态增强，可以加入 finetune 数据的前缀
- **SMPL↔G1 Latent Loss**：通过 `g1_smpl_latent` 和 `reencoded_smpl_g1_latent` 辅助损失，对齐 SMPL encoder 和 G1 encoder 在共享 token 空间的表征。现在 robot PKL 和 SMPL PKL 按文件名对齐，两个 encoder 都会处理新动作

#### 肩部跟踪的额外改进思路

如果微调后肩部跟踪仍不理想，可以在 `reward_point_body` 中增加肩部关键点：

```yaml
reward_point_body: ["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
                     "left_shoulder_roll_link", "right_shoulder_roll_link"]
```

这会让奖励函数显式关注肩部位置跟踪精度，但改变了奖励结构，可能需要更多迭代收敛。

---

### 已解决的问题

#### SMPL 帧数与 Robot Motion 不对齐（2026-06-09 修复）

**现象**：训练时报 `AssertionError: smpl_joints=250, global_translation=249`

**根因**：`convert_kimodo_soma_to_smpl_lib.py` 用 `int(T * target_fps / source_fps)` 计算目标帧数（得 250），但 motion_lib 的插值函数用 `duration = (T-1)/fps` 公式（得 249），off-by-one。

**修复**：将 SMPL 转换脚本的帧数公式改为与 motion_lib 一致：
```python
duration = (T - 1) / source_fps
n_target = int(duration * target_fps) + 1
```

修复后重新生成 144 个 SMPL PKL，帧数全部对齐。

---

### 工具链参考

| 工具 | 路径 | 用途 |
| --- | --- | --- |
| Kimodo 文本生成 | `/home/balance/kimodo/` | 文本 → SOMA 运动 |
| 生成脚本 v1 | `/home/balance/kimodo/generate_finetune_motions.sh` | 3 类 75 条基础动作 |
| 生成脚本 v2 | `/home/balance/kimodo/generate_finetune_motions_v2.sh` | 5 类 ~183 条补充动作（含拖椅、递物、组合等） |
| SOMA→SMPL 转换 | `gear_sonic/data_process/convert_kimodo_soma_to_smpl_lib.py` | Kimodo NPZ → SMPL PKL（已修复帧数对齐） |
| CSV→motion_lib 转换 | `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` | G1 CSV → robot motion_lib PKL |
| 动作过滤 | `gear_sonic/data_process/filter_and_copy_bones_data.py` | 按文件名关键词过滤不可执行动作 |
| SOMA Retargeter | `/home/balance/soma-retargeter/` | BVH → G1 29-DOF CSV（Newton IK） |
| Retargeter 配置 | `soma-retargeter/assets/kimodo_finetune_config.json` | Kimodo finetune 批处理配置 |
| Kimodo Demo | `kimodo_demo` → `http://localhost:7860` | 可视化检查生成质量 |
