#!/usr/bin/env python3
"""
删除指定索引的episode数据

用法:
    python remove_episode.py --data_dir outputs/2026-05-12-17-08-01 --indices 3 4
    
这会删除索引为3和4的episode数据
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def remove_episode(data_dir: str, indices_to_remove: list[int]):
    """删除指定索引的episode数据"""
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"错误: 目录 {data_path} 不存在")
        return
    
    # 找出所有的chunk目录
    data_chunks = sorted(data_path.glob("data/chunk-*"))
    video_chunks = sorted(data_path.glob("videos/chunk-*"))
    
    if not data_chunks:
        print("错误: 没有找到data/chunk-*目录")
        return
    
    # 获取所有chunk目录的名称
    chunk_names = [d.name for d in data_chunks]
    print(f"找到的chunk目录: {chunk_names}")
    
    # 读取episodes.jsonl获取总episode数
    episodes_file = data_path / "meta" / "episodes.jsonl"
    if not episodes_file.exists():
        print(f"错误: 文件 {episodes_file} 不存在")
        return
    
    with open(episodes_file, 'r') as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    
    total_episodes = len(episodes)
    print(f"总共有 {total_episodes} 个episode")
    
    # 验证索引
    valid_indices = []
    for idx in indices_to_remove:
        if idx < 0 or idx >= total_episodes:
            print(f"警告: 索引 {idx} 超出范围 (0-{total_episodes-1})，跳过")
        else:
            valid_indices.append(idx)
    
    if not valid_indices:
        print("没有有效的索引需要删除")
        return
    
    valid_indices.sort()
    print(f"将删除以下索引的episode: {valid_indices}")
    
    # 1. 删除data/chunk-*中的parquet文件
    for chunk_dir in data_chunks:
        for idx in valid_indices:
            parquet_file = chunk_dir / f"episode_{idx:06d}.parquet"
            if parquet_file.exists():
                parquet_file.unlink()
                print(f"已删除: {parquet_file}")
            else:
                print(f"未找到: {parquet_file}")
    
    # 2. 删除videos/chunk-*中的视频文件
    for chunk_dir in video_chunks:
        # 查找视频子目录
        for video_subdir in chunk_dir.iterdir():
            if video_subdir.is_dir():
                for idx in valid_indices:
                    video_file = video_subdir / f"episode_{idx:06d}.mp4"
                    if video_file.exists():
                        video_file.unlink()
                        print(f"已删除: {video_file}")
                    else:
                        print(f"未找到: {video_file}")
    
    # 3. 更新meta/episodes.jsonl
    print("\n更新meta文件...")
    
    # 读取所有jsonl文件
    meta_files = [
        data_path / "meta" / "episodes.jsonl",
        data_path / "meta" / "episodes_stats.jsonl",
        data_path / "meta" / "tasks.jsonl",
    ]
    
    for meta_file in meta_files:
        if not meta_file.exists():
            print(f"警告: {meta_file} 不存在，跳过")
            continue
        
        # 读取所有行
        with open(meta_file, 'r') as f:
            lines = f.readlines()
        
        # 过滤掉要删除的索引
        new_lines = []
        for i, line in enumerate(lines):
            if line.strip() and i not in valid_indices:
                new_lines.append(line)
        
        # 写回文件
        with open(meta_file, 'w') as f:
            f.writelines(new_lines)
        
        print(f"已更新: {meta_file} (从 {len(lines)} 行减少到 {len(new_lines)} 行)")
    
    # 4. 重新索引剩余的episodes
    print("\n重新索引剩余的episodes...")
    
    # 更新所有meta文件中的episode_index（不包括tasks.jsonl，它只有任务定义）
    meta_files_with_index = [
        data_path / "meta" / "episodes.jsonl",
        data_path / "meta" / "episodes_stats.jsonl",
    ]
    
    for meta_file in meta_files_with_index:
        if not meta_file.exists():
            continue
        
        # 读取所有行
        with open(meta_file, 'r') as f:
            lines = [line for line in f if line.strip()]
        
        # 解析JSON并重新分配索引
        new_lines = []
        for new_idx, line in enumerate(lines):
            try:
                data = json.loads(line)
                if 'episode_index' in data:
                    data['episode_index'] = new_idx
                new_lines.append(json.dumps(data) + '\n')
            except json.JSONDecodeError:
                # 如果不是有效的JSON，保持原样
                new_lines.append(line)
        
        # 写回文件
        with open(meta_file, 'w') as f:
            f.writelines(new_lines)
        
        print(f"已重新索引: {meta_file.name} ({len(new_lines)} 个episodes)")
    
    # 重命名数据文件
    for chunk_dir in data_chunks:
        # 获取所有剩余的parquet文件
        parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
        
        # 创建新的索引映射
        old_indices = []
        for pf in parquet_files:
            # 从文件名提取索引
            old_idx = int(pf.stem.split('_')[1])
            old_indices.append(old_idx)
        
        # 重命名文件
        for new_idx, old_idx in enumerate(old_indices):
            old_file = chunk_dir / f"episode_{old_idx:06d}.parquet"
            new_file = chunk_dir / f"episode_{new_idx:06d}.parquet"
            
            if old_file != new_file:
                if old_file.exists():
                    old_file.rename(new_file)
                    print(f"重命名: {old_file.name} -> {new_file.name}")
    
    # 重命名视频文件
    for chunk_dir in video_chunks:
        for video_subdir in chunk_dir.iterdir():
            if video_subdir.is_dir():
                video_files = sorted(video_subdir.glob("episode_*.mp4"))
                
                old_indices = []
                for vf in video_files:
                    old_idx = int(vf.stem.split('_')[1])
                    old_indices.append(old_idx)
                
                for new_idx, old_idx in enumerate(old_indices):
                    old_file = video_subdir / f"episode_{old_idx:06d}.mp4"
                    new_file = video_subdir / f"episode_{new_idx:06d}.mp4"
                    
                    if old_file != new_file:
                        if old_file.exists():
                            old_file.rename(new_file)
                            print(f"重命名: {old_file.name} -> {new_file.name}")
    
    # 更新info.json中的统计信息
    info_file = data_path / "meta" / "info.json"
    if info_file.exists():
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        # 计算剩余的总帧数
        total_frames = sum(ep.get('length', 0) for ep in episodes)
        
        if 'total_episodes' in info:
            info['total_episodes'] = len(episodes)
            print(f"更新info.json中的total_episodes: {info['total_episodes']}")
        
        if 'episodes_count' in info:
            info['episodes_count'] = len(episodes)
            print(f"更新info.json中的episodes_count: {info['episodes_count']}")
        
        if 'total_frames' in info:
            info['total_frames'] = total_frames
            print(f"更新info.json中的total_frames: {info['total_frames']}")
        
        if 'total_videos' in info:
            info['total_videos'] = len(episodes)
            print(f"更新info.json中的total_videos: {info['total_videos']}")
        
        # 更新splits中的episode范围
        if 'splits' in info:
            for split_name, split_range in info['splits'].items():
                # 格式如 "0:6" 表示从episode 0到5
                info['splits'][split_name] = f"0:{len(episodes)}"
                print(f"更新info.json中的splits.{split_name}: {info['splits'][split_name]}")
        
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"已更新: {info_file}")
    
    print(f"\n完成! 剩余 {len(episodes)} 个episode")


def main():
    parser = argparse.ArgumentParser(description='删除指定索引的episode数据')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录路径，例如: outputs/2026-05-12-17-08-01')
    parser.add_argument('--indices', type=int, nargs='+', required=True,
                        help='要删除的episode索引列表，从0开始，例如: 3 4 5')
    
    args = parser.parse_args()
    
    # 确认删除操作
    print(f"数据目录: {args.data_dir}")
    print(f"要删除的索引: {args.indices}")
    print()
    
    response = input("确认删除这些数据? (y/N): ")
    if response.lower() != 'y':
        print("操作已取消")
        return
    
    remove_episode(args.data_dir, args.indices)


if __name__ == "__main__":
    main()

