#!/bin/bash
# SONIC 微调环境镜像构建脚本
# 仅包含运行环境，不含训练数据
# 预估镜像大小：~25-30GB

set -e

IMAGE_NAME="sonic-finetune-env"
IMAGE_TAG="v1"
REGISTRY="hub.docker.alibaba-inc.com/xlab"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== Step 1: 准备构建上下文 ==="
bash docker/prepare_build_context.sh

echo ""
echo "=== Step 2: 构建 Docker 镜像 ==="
sudo docker build \
    -f docker/Dockerfile \
    -t ${IMAGE_NAME}:${IMAGE_TAG} \
    .

echo ""
echo "=== 构建完成 ==="
echo "镜像: ${IMAGE_NAME}:${IMAGE_TAG}"
sudo docker images ${IMAGE_NAME}:${IMAGE_TAG}
echo ""
echo "=== 本地测试 ==="
echo "  sudo docker run --gpus all -it \\"
echo "    -v /home/balance/GR00T-WholeBodyControl:/workspace/GR00T \\"
echo "    -v /home/balance/GEAR-SONIC:/workspace/GEAR-SONIC \\"
echo "    ${IMAGE_NAME}:${IMAGE_TAG} \\"
echo "    python -c \"import torch; import isaaclab; print('OK', torch.cuda.device_count(), 'GPUs')\""
echo ""
echo "=== 推送到弹内仓库 ==="
echo "  sudo docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "  sudo docker login hub.docker.alibaba-inc.com -u <工号>"
echo "  sudo docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "=== MLflow 使用 ==="
echo "镜像: hub.docker.alibaba-inc.com/xlab/${IMAGE_NAME}:${IMAGE_TAG}"
echo "数据挂载: 将 OSS 中的数据挂载到容器 /workspace/data"
