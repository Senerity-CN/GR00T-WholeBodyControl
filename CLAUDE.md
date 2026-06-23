# GR00T WholeBodyControl

## 项目概述

全身控制（Whole Body Control）项目，用于 G1 机器人的遥操作、SMPL 动作回放与策略部署。

## 虚拟环境

项目包含多个 Python 虚拟环境，按用途隔离依赖：

| 环境 | 路径 | 用途 |
|------|------|------|
| `.venv_data_collection` | `.venv_data_collection/` | 数据采集、SMPL 动作处理、回放脚本 |
| `.venv_teleop` | `.venv_teleop/` | 遥操作（VR → ZMQ 流） |
| `.venv_sim` | `.venv_sim/` | MuJoCo 仿真 |
| `.venv_inference` | `.venv_inference/` | 策略推理 |

## 开发规范

- **所有 Python 相关操作（运行脚本、读取数据、临时代码片段等）默认使用 `.venv_data_collection` 虚拟环境**：`.venv_data_collection/bin/python`
- 安装依赖使用：`uv pip install --python .venv_data_collection/bin/python3 <package>`

