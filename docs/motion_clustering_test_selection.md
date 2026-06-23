# Motion Data 聚类分析与测试样本选取方案

## 1. 背景与目标

我们有一个训练好的全身运动控制模型（基于 GEAR-SONIC），需要系统性地评估其在不同类型动作上的表现。

**数据来源：**

训练数据来自 [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) 数据集，包含 142,220 条标注人体动作（71,132 原始 + 71,088 镜像），涵盖 8 大类、20 个细分类别、522 位演员。原始数据为 G1 MuJoCo-compatible CSV 格式（120fps），通过 `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` 转换为 motion_lib pkl 格式（30fps）存储在本地。

**当前数据：**

| 数据集 | 路径 | 数量 | 说明 |
|--------|------|------|------|
| Bones Seed（原始） | `data/motion_lib_mixed/original/` | 129,785 个 pkl（去镜像后 ~64,912 独立动作） | 124 个日期子目录，来自 BONES-SEED 官方数据 |
| 微调数据 | `data/motion_lib_mixed/finetune/` | 160 个 pkl，~63 种动作类型 | 我们自采集的目标动作 |

**核心问题：** 从 ~65K 个独立动作中选出一组有代表性的测试样本（~50-60 个），使测试结果具有说服力。

**目标：**
- 找到覆盖动作空间各区域的**典型样本**（cluster 中心）
- 找到模型可能表现较差的**困难样本**（cluster 边界 / 离群点）
- 分析微调数据覆盖了哪些动作类型、哪些没有覆盖，指导后续测试重点

## 2. 数据格式

### 2.1 Motion Lib PKL 格式

每个 pkl 文件是 joblib 序列化的字典，结构为 `{motion_key: {数据字段}}`：

| 字段 | 形状 | 说明 |
|------|------|------|
| `smpl_joints` | (T, 24, 3) | 24 个 SMPL 关节的 3D 坐标序列（转换时填零占位） |
| `pose_aa` | (T, 30, 3) | 30 个关节的轴角表示 |
| `dof` | (T, 29) | 29 个自由度（G1 MuJoCo 顺序） |
| `root_rot` | (T, 4) | 根节点旋转四元数 (xyzw) |
| `root_trans_offset` | (T, 3) | 根节点平移偏移 (meters) |
| `fps` | int | 帧率，30fps（从 120fps 降采样） |

其中 T 为帧数，不同动作长度不同。`_M.pkl` 后缀表示左右镜像版本，聚类时排除以避免重复。

### 2.2 BONES-SEED 元数据

BONES-SEED 提供了一份 parquet 元数据文件（`seed_metadata_v003.parquet`，51 列 × 142,220 行），包含每个动作的：

| 字段 | 示例值 | 用途 |
|------|--------|------|
| `filename` | `walk_forward_amateur_003__A002` | 与 pkl 文件名对应的 motion key |
| `package` | Locomotion / Communication / Dances / ... | 8 大类 |
| `category` | Basic Locomotion Neutral / Gestures / ... | 20 细分类别 |
| `content_type_of_movement` | walking / jogging / gesture / dancing | 运动类型 |
| `content_body_position` | standing / sitting / crouching | 身体姿态 |
| `content_short_description` | reading newspaper sitting | 简短描述 |
| `content_natural_desc_1~4` | 自然语言描述 | 详细描述 |
| `is_mirror` | True/False | 是否镜像 |
| `move_duration_frames` | 366 | 帧数（@120fps） |

> **关键价值：** 这份元数据可以将聚类结果与语义类别对齐——我们不仅知道某个 cluster 在特征空间中是什么样的，还知道它在语义上对应"走路"还是"跳舞"还是"物体交互"。

## 3. 技术方案

整体流程分两条路线，**可以独立使用，也可以结合使用**：

```
路线 A：基于运动特征的无监督聚类（已实现）
    pkl 数据 → 特征提取 → PCA → K-Means → 选样

路线 B：基于 BONES-SEED 元数据的分层采样（推荐补充）
    parquet 元数据 → 按 package/category 分层 → 每层采样 → 选样
```

### 3.1 路线 A：运动特征聚类（已实现）

#### 特征提取

对每个 motion clip 提取**369 维固定长度特征向量**：

| 类别 | 维度 | 内容 | 捕捉信息 |
|------|------|------|----------|
| 姿态特征（静态） | 213 | 关节位置 mean/std + 相对位置 mean | 动作的"平均姿态"（站 vs 蹲 vs 举手） |
| 运动学特征（动态） | 153 | 关节速度 mean/std + 根节点速度 mean/std/max | 运动的快慢和幅度（慢走 vs 快跑 vs 跳跃） |
| 全局特征 | 3 | 时长 + 总位移 + 平均速度 | 动作规模（原地 vs 长距离移动） |

#### 降维与聚类

```
原始特征 (N, 369)
    ↓ StandardScaler 标准化
    ↓ PCA 降维 → (N, 50)，保留 ~95% 方差
    ↓ K-Means 聚类 → K 个 cluster
    ↓ UMAP 降维 → (N, 2)，用于可视化
```

#### 样本选取

```
┌──────────────────────────────────────────────────┐
│              Cluster i                            │
│                                                  │
│      · · ·                                       │
│    ·  · · ·  ·                                   │
│   · · ★ · · ·    ★ = Center（离质心最近）         │
│    ·  · · ·  ·                                   │
│      · · ◆        ◆ = Boundary（离质心最远）      │
│                                                  │
└──────────────────────────────────────────────────┘

    ✖ = Outlier（离所有质心都远的全局离群点）
```

| 策略 | 数量 | 含义 | 测试目的 |
|------|------|------|----------|
| Center | 每 cluster 1 个 | 离质心最近 | 典型动作，模型应该做好 |
| Boundary | 每 cluster 1 个 | 离质心最远 | 该类别中最困难的动作 |
| Outlier | 全局 Top-5 | 离所有质心都远 | 罕见动作，测试泛化极限 |

K=25 时产出：25 center + 25 boundary + 5 outlier = **55 个测试样本**

#### 微调覆盖分析

将 160 个 finetune 数据投影到同一特征空间，分析：
- 哪些 cluster 被 finetune 覆盖 → 模型见过的动作区域
- 哪些 cluster 没有被覆盖 → 纯靠泛化的区域

### 3.2 路线 B：基于元数据的分层采样（推荐补充）

BONES-SEED 的官方分类体系已经很完善，可以直接利用：

#### BONES-SEED 动作类别体系

**8 个 Package（一级分类）：**

| Package | 动作数 | 说明 |
|---------|--------|------|
| Locomotion | 74,488 | 行走、慢跑、跳跃、攀爬、爬行、转弯 |
| Communication | 21,493 | 手势、指向、张望 |
| Interactions | 14,643 | 物体操作、搬运、工具使用 |
| Dances | 11,006 | 各风格舞蹈 |
| Gaming | 8,700 | 游戏动作 |
| Everyday | 5,816 | 家务、坐、读报 |
| Sport | 3,993 | 运动 |
| Other | 2,081 | 特技、武术 |

**20 个 Category（二级分类）：** Basic Locomotion Neutral、Gestures、Object Manipulation、Dancing、Advanced Locomotion、Sports 等。

#### 分层采样策略

```
对每个 category:
    1. 找到该 category 下的所有 motion key
    2. 与本地 pkl 文件做交集（只保留已转换的）
    3. 按比例或固定数量采样：
       - 大类（>5000 条）：采样 5-8 个
       - 中类（1000-5000 条）：采样 3-5 个
       - 小类（<1000 条）：采样 1-3 个
    4. 在类内可以结合路线 A 的聚类结果，选 center + boundary
```

#### 实施要求

元数据已下载到本地：`~/data/bones_seed_metadata/metadata/seed_metadata_v004.parquet`（4.3 MB）。

本地 64,912 个非镜像 pkl 与元数据 100% 匹配（元数据多出 6,220 条是本地未转换的动作）。

```python
import pandas as pd

meta = pd.read_parquet("~/data/bones_seed_metadata/metadata/seed_metadata_v004.parquet")
# 过滤掉镜像
meta = meta[~meta["is_mirror"]]
# 按 category 统计
print(meta["category"].value_counts())
# 用 filename 字段与本地 pkl 文件名匹配
```

### 3.3 两条路线的对比与结合

| 维度 | 路线 A（聚类） | 路线 B（元数据分层） |
|------|---------------|---------------------|
| 依赖 | 只需本地 pkl 数据 | 需要 parquet 元数据文件 |
| 分类依据 | 运动学特征（数据驱动） | 人工标注的语义类别 |
| 优势 | 能发现人工标注可能遗漏的相似性/差异性 | 语义明确，结果可解释 |
| 不足 | cluster 含义需要人工解读 | 受限于标注体系的粒度 |
| 样本选取 | center/boundary/outlier | 按类别分层均匀采样 |

**推荐做法：结合使用**

1. 先跑路线 A 聚类，得到 UMAP 可视化和 cluster 分配
2. 下载 parquet 元数据，将官方 category 标签叠加到聚类结果上
3. 验证聚类与语义标签的一致性（例如：cluster 3 是否全是 locomotion？）
4. 最终选样时，确保每个 BONES-SEED category 至少有 1 个样本，同时兼顾 cluster center/boundary

这样得到的测试集既有数据驱动的代表性，又有语义上的完整覆盖。

## 4. 使用方法

### 4.1 路线 A：运行聚类脚本

脚本已实现在 `scripts/cluster_motions.py`。

```bash
# 快速试跑：采样 5000 个，10 个 cluster
.venv_data_collection/bin/python scripts/cluster_motions.py \
    --data-dir data/motion_lib_mixed/original \
    --finetune-dir data/motion_lib_mixed/finetune \
    --output-dir outputs/cluster_analysis \
    --sample-size 5000 \
    --n-clusters 10

# 正式运行：全量数据，25 个 cluster
.venv_data_collection/bin/python scripts/cluster_motions.py \
    --data-dir data/motion_lib_mixed/original \
    --finetune-dir data/motion_lib_mixed/finetune \
    --output-dir outputs/cluster_analysis \
    --sample-size -1 \
    --n-clusters 25
```

**全部参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | （必填） | 原始动作数据目录 |
| `--finetune-dir` | 无 | 微调数据目录（可选） |
| `--output-dir` | `outputs/cluster_analysis` | 输出目录 |
| `--sample-size` | 5000 | 采样数量，-1 表示全量 |
| `--n-clusters` | 25 | K-Means 聚类数 |
| `--no-exclude-mirror` | - | 不排除 `_M` 镜像文件 |
| `--no-umap` | - | 跳过 UMAP 可视化（更快） |
| `--seed` | 42 | 随机种子 |

### 4.2 路线 B：元数据分层采样

```bash
# 元数据已在: ~/data/bones_seed_metadata/metadata/seed_metadata_v004.parquet
# 运行分层采样（TODO：脚本待实现）
.venv_data_collection/bin/python scripts/stratified_sample_motions.py \
    --metadata ~/data/bones_seed_metadata/metadata/seed_metadata_v004.parquet \
    --data-dir data/motion_lib_mixed/original \
    --output-dir outputs/stratified_analysis
```

### 4.3 输出文件

| 文件 | 用途 |
|------|------|
| `eval_motion_keys.txt` | 纯文本 motion key 列表，每行一个，**直接用于评测** |
| `cluster_test_samples.json` | 每个选中样本的详细信息（motion_key、cluster_id、选取策略、到质心距离） |
| `cluster_analysis.json` | 每个 cluster 的统计（大小、代表性动作名、finetune 覆盖情况） |
| `cluster_visualization.png` | UMAP 2D 散点图，展示聚类结构和 finetune 分布 |
| `cluster_selected_samples.png` | 标注选中的 center/boundary/outlier 样本位置 |

### 4.4 对接评测流程

选中的 motion key 通过现有的 `filter_motion_keys` 机制传入评测：

```bash
# 方式 1：直接传列表
python gear_sonic/eval_agent_trl.py \
    "+checkpoint=<model_path>" \
    "+headless=True" \
    "++eval_callbacks=im_eval" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_mixed/original" \
    "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[key1,key2,key3,...]"

# 方式 2：从文件读取
KEYS=$(paste -sd',' outputs/cluster_analysis/eval_motion_keys.txt)
python gear_sonic/eval_agent_trl.py \
    "+checkpoint=<model_path>" \
    "+headless=True" \
    "++eval_callbacks=im_eval" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_mixed/original" \
    "++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys=[$KEYS]"
```

`filter_motion_keys` 支持精确匹配和正则表达式，定义在 `gear_sonic/utils/motion_lib/motion_lib_base.py:418`。

## 5. 结果解读指南

### 5.1 可视化图解读

- **散点颜色**：不同 cluster 用不同颜色
- **黑色三角 ▲**：finetune 数据的位置 → 集中说明微调数据偏向某些动作区域
- **红色星号 ★**：center 样本 → 应分散在各 cluster 中心
- **橙色菱形 ◆**：boundary 样本 → 应在 cluster 边缘
- **紫色叉号 ✖**：outlier 样本 → 应远离主要聚类

### 5.2 评测结果分析维度

拿到评测指标（成功率、MPJPE 等）后，建议按以下维度分析：

| 分析维度 | 方法 | 回答的问题 |
|----------|------|-----------|
| 按选取策略 | center vs boundary vs outlier | 模型对困难/罕见样本的处理能力如何？ |
| 按 finetune 覆盖 | 覆盖 cluster 内 vs 未覆盖 cluster 内 | 微调是否有效泛化到未见动作？ |
| 按动作类别 | 按 cluster（或 BONES-SEED category） | 模型在哪类动作上最弱？ |
| 按动作特性 | 快/慢、原地/移动、简单/复杂 | 哪些运动学特征与失败相关？ |

### 5.3 结合元数据的深度分析

如果有 parquet 元数据，可以进一步：

```python
import pandas as pd, json

meta = pd.read_parquet("data/seed_metadata_v003.parquet")
samples = json.load(open("outputs/cluster_analysis/cluster_test_samples.json"))

# 给每个选中样本附加语义标签
for s in samples:
    row = meta[meta["filename"] == s["motion_key"]]
    if len(row):
        s["package"] = row.iloc[0]["package"]
        s["category"] = row.iloc[0]["category"]
        s["description"] = row.iloc[0]["content_short_description"]

# 检查：测试集是否覆盖了所有 8 个 package？
packages = set(s.get("package") for s in samples if "package" in s)
print(f"Covered packages: {packages}")
```

## 6. 时间与资源预估

| 步骤 | 采样 5,000 | 全量 ~65,000 |
|------|-----------|-------------|
| 文件扫描 | ~2s | ~5s |
| 加载 + 特征提取 | ~3s | ~40s |
| PCA + K-Means | <1s | ~5s |
| UMAP 可视化 | ~10s | ~3-5min |
| **总计** | **~15s** | **~5-6min** |

内存占用：特征矩阵约 65K × 369 × 4B ≈ 90MB，加上 UMAP 中间结果约 500MB，普通机器可以承受。

## 7. 后续优化方向

1. **实现路线 B 脚本**：基于 parquet 元数据做分层采样，与聚类结果交叉验证
2. **引入时序特征**：当前特征是统计量，可以考虑用 DTW 或时序自编码器捕捉更细粒度的运动模式
3. **自适应 K 值选择**：用 Silhouette Score 或 Gap Statistic 自动确定最优 K
4. **增量更新**：当新增 finetune 数据后，快速将新数据投影到已有聚类空间，更新覆盖分析
