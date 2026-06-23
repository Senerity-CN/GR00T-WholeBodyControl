#!/bin/bash
# Run evaluation in 3 segments. Each segment can be run independently.
# Total: 129,785 motions, split into 3 segments aligned to num_envs=512.
#
# Usage:
#   ./gear_sonic/tools/run_eval_segments.sh [segment_number]
#   segment_number: 1, 2, or 3 (runs only that segment)
#   If no argument given, runs all 3 sequentially.

set -e

CHECKPOINT="${EVAL_CHECKPOINT:-logs_rl/SONIC_Finetune/manager/universal_token/all_modes/sonic_release_only_finetune_data-20260616_155744/last.pt}"
OUTPUT_PREFIX="${EVAL_OUTPUT_PREFIX:-logs_eval/segments_finetune}"
COMMON_ARGS=(
    "+checkpoint=${CHECKPOINT}"
    "+headless=True"
    "++eval_callbacks=im_eval"
    "++run_eval_loop=False"
    "++num_envs=512"
    "+manager_env/terminations=tracking/eval"
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_mixed/original"
    "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=null"
    "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False"
)

run_segment() {
    local seg=$1
    local start=$2
    local end=$3
    local output_dir="${OUTPUT_PREFIX}/seg${seg}"

    echo "========================================"
    echo "Running Segment ${seg}: motions [${start}, ${end})"
    echo "Output: ${output_dir}"
    echo "========================================"

    python gear_sonic/eval_agent_trl.py \
        "${COMMON_ARGS[@]}" \
        "++eval_start_idx=${start}" \
        "++eval_end_idx=${end}" \
        "++eval_output_dir=${output_dir}"
}

# Segment boundaries: 3 segments, aligned to 512
# Segment 1: 0 - 43520      (85 batches)
# Segment 2: 43520 - 87040   (85 batches)
# Segment 3: 87040 - 129785  (84 batches, last batch padded)

if [ -z "$1" ]; then
    echo "Running all 3 segments sequentially..."
    run_segment 1 0 43520
    run_segment 2 43520 87040
    run_segment 3 87040 129785
    echo ""
    echo "All segments complete. Run merge:"
    echo "  python gear_sonic/tools/merge_eval_segments.py \\"
    echo "    --segments logs_eval/segments_finetune/seg1/metrics_eval.json \\"
    echo "               logs_eval/segments_finetune/seg2/metrics_eval.json \\"
    echo "               logs_eval/segments_finetune/seg3/metrics_eval.json \\"
    echo "    --output logs_eval/merged/metrics_eval_finetune.json"
else
    case "$1" in
        1) run_segment 1 0 43520 ;;
        2) run_segment 2 43520 87040 ;;
        3) run_segment 3 87040 129785 ;;
        *) echo "Invalid segment number: $1 (must be 1, 2, or 3)"; exit 1 ;;
    esac
fi
