"""
统计所有episode初始状态(第0帧)的 action.motion_token, teleop.left_hand_joints, teleop.right_hand_joints 的均值和方差。
"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

DATA_DIR = Path("outputs/full_merged_dataset/data/chunk-000")
COLUMNS = ["action.motion_token", "teleop.left_hand_joints", "teleop.right_hand_joints"]

episode_files = sorted(DATA_DIR.glob("episode_*.parquet"))
print(f"Found {len(episode_files)} episodes")

initial_values = {col: [] for col in COLUMNS}

for ep_file in episode_files:
    table = pq.read_table(str(ep_file), columns=COLUMNS)
    for col in COLUMNS:
        val = np.array(table.column(col)[0].as_py())
        initial_values[col].append(val)

print("\n" + "=" * 80)
print("初始状态统计 (每个episode的第0帧)")
print("=" * 80)

for col in COLUMNS:
    data = np.stack(initial_values[col])  # shape: (num_episodes, dim)
    mean = data.mean(axis=0)
    var = data.var(axis=0)
    print(f"\n{'─' * 80}")
    print(f"【{col}】 shape: {data.shape}")
    print(f"  均值 (mean): {mean}")
    print(f"  方差 (var):  {var}")
    print(f"  均值范围: [{mean.min():.6f}, {mean.max():.6f}]")
    print(f"  方差范围: [{var.min():.6f}, {var.max():.6f}]")
