#!/bin/bash
# SONIC Finetune 入口脚本（MLflow 启动用）
#
# OSS bucket oss://all-dataset-external/pingheng/Sonic-finetune/挂载到 DATA_DIR
# 目录结构：
#   motion_lib/          robot PKL（robot_filtered/ + robot_finetune/，递归 glob）
#   smpl/                SMPL PKL（已合并 finetune，扁平）
#   checkpoints/last.pt  release checkpoint
#   GR00T/gear_sonic/    训练代码

set -e

# --- 强制注入自动同意协议的环境变量，防止多卡 DDP 丢失 ---
export ACCEPT_EULA=Y
export OMNIVERSE_AGREE_TO_EULA=yes
export PRIVACY_CONSENT=Y

export WANDB_API_KEY=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

GPU_COUNT=${GPU_COUNT:-8}
NUM_ENVS=${NUM_ENVS:-4096}
EXP_VAR=${EXP_VAR:-"backward_shuffle_reach"}
DATA_DIR=${DATA_DIR:-"/public/zjk_oss/pingheng/Sonic-finetune"}
CODE_DIR=${CODE_DIR:-"${DATA_DIR}/GR00T"}

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}"

echo "========================================"
echo " SONIC Finetune"
echo " GPUs: ${GPU_COUNT}"
echo " num_envs: ${NUM_ENVS}"
echo " exp_var: ${EXP_VAR}"
echo " data_dir: ${DATA_DIR}"
echo " code_dir: ${CODE_DIR}"
echo "========================================"

cd "${CODE_DIR}"

TRAIN_ARGS=(
    +exp=manager/universal_token/all_modes/sonic_release
    project_name=SONIC_Finetune
    exp_var=${EXP_VAR}
    +checkpoint=${DATA_DIR}/checkpoints/last.pt
    num_envs=${NUM_ENVS}
    headless=True
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=${DATA_DIR}/motion_lib
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=${DATA_DIR}/smpl
    use_wandb=true
    +max_iterations=100000
    ++algo.config.learning_rate=2e-5
    ++algo.trl.learning_rate=2e-5
)

# 使用 yes Yes | 确保在非交互式下自动响应任何可能的 EULA 确认提示
if [ "$GPU_COUNT" -gt 1 ]; then
    yes Yes | accelerate launch --num_processes=${GPU_COUNT} gear_sonic/train_agent_trl.py "${TRAIN_ARGS[@]}"
else
    yes Yes | python gear_sonic/train_agent_trl.py "${TRAIN_ARGS[@]}"
fi
