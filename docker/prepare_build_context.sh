#!/bin/bash
# 准备 Docker 构建上下文
# Docker build 的 COPY 只能引用构建上下文内的路径
# 需要把 IsaacLab 源码复制到项目目录下

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== 准备 Docker 构建上下文 ==="

# 1. IsaacLab 源码（editable install 需要）
echo "[1/2] 准备 IsaacLab 源码..."
if [ ! -d "IsaacLab" ]; then
    echo "  复制 /home/balance/IsaacLab/2.3.0/ → IsaacLab/ （约 87MB）"
    cp -r /home/balance/IsaacLab/2.3.0/ IsaacLab/
else
    echo "  IsaacLab/ 已存在，跳过"
fi

# 2. 检查 gear_sonic 和 pyproject.toml
echo "[2/2] 检查 gear_sonic 和 gear_sonic/pyproject.toml..."
if [ -d "gear_sonic" ] && [ -f "gear_sonic/pyproject.toml" ]; then
    echo "  gear_sonic/ 和 gear_sonic/pyproject.toml 就绪"
else
    echo "  ERROR: gear_sonic/ 或 gear_sonic/pyproject.toml 不存在"
    exit 1
fi

echo ""
echo "=== 构建上下文就绪 ==="
echo ""
echo "目录大小："
du -sh IsaacLab/ gear_sonic/ docker/requirements_train.txt 2>/dev/null
echo ""
echo "下一步：运行 docker/build_mlflow_image.sh"
