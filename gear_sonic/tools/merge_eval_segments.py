#!/usr/bin/env python3
"""Merge multiple segmented evaluation results into a single metrics_eval.json.

Usage:
    python gear_sonic/tools/merge_eval_segments.py \
        --segments logs_eval/segments/seg1/metrics_eval.json \
                   logs_eval/segments/seg2/metrics_eval.json \
                   logs_eval/segments/seg3/metrics_eval.json \
        --output logs_eval/merged/metrics_eval.json
"""

import argparse
import json
import os
import sys

import numpy as np


def load_segment(path):
    with open(path) as f:
        return json.load(f)


def merge_segments(segments):
    """Merge multiple segment metrics into a single unified result."""
    all_metrics_dicts = [seg["eval/all_metrics_dict"] for seg in segments]
    failed_metrics_dicts = [seg["eval/failed_metrics_dict"] for seg in segments]

    # Identify per-motion array keys (present in all segments)
    array_keys = set(all_metrics_dicts[0].keys())
    for d in all_metrics_dicts[1:]:
        array_keys &= set(d.keys())

    # Concatenate per-motion arrays
    merged_all = {}
    for key in sorted(array_keys):
        values = []
        for d in all_metrics_dicts:
            v = d[key]
            if isinstance(v, list):
                values.extend(v)
            else:
                values.append(v)
        merged_all[key] = values

    # Concatenate failed metrics
    failed_array_keys = set(failed_metrics_dicts[0].keys())
    for d in failed_metrics_dicts[1:]:
        failed_array_keys &= set(d.keys())

    merged_failed = {}
    for key in sorted(failed_array_keys):
        values = []
        for d in failed_metrics_dicts:
            v = d[key]
            if isinstance(v, list):
                values.extend(v)
            else:
                values.append(v)
        merged_failed[key] = values

    # Recompute aggregate metrics
    num_motions = len(merged_all["terminated"])
    terminated = np.array(merged_all["terminated"], dtype=bool)
    progress = np.array(merged_all["progress"], dtype=float)

    success_rate = 1.0 - terminated.mean()
    progress[~terminated] = 1.0
    progress_rate = progress.mean()

    # Recompute mpjpe metrics (per-motion values are already averaged over frames)
    metrics_eval = {}
    mpjpe_keys = [k for k in merged_all.keys() if "mpjpe" in k]
    for key in mpjpe_keys:
        values = np.array(merged_all[key], dtype=float)
        metrics_eval[f"eval/all/{key}"] = float(values.mean())
        if (~terminated).any():
            metrics_eval[f"eval/success/{key}"] = float(values[~terminated].mean())
        else:
            metrics_eval[f"eval/success/{key}"] = 0.0

    metrics_eval["eval/success/success_rate"] = float(success_rate)
    metrics_eval["eval/success/progress_rate"] = float(progress_rate)

    # Object metrics if present
    if "obj_pos_error" in merged_all:
        obj_pos = np.array(merged_all["obj_pos_error"], dtype=float)
        obj_ori = np.array(merged_all["obj_ori_error"], dtype=float)
        metrics_eval["eval/all/obj_pos_error"] = float(obj_pos.mean())
        metrics_eval["eval/all/obj_ori_error"] = float(obj_ori.mean())
        if (~terminated).any():
            metrics_eval["eval/success/obj_pos_error"] = float(obj_pos[~terminated].mean())
            metrics_eval["eval/success/obj_ori_error"] = float(obj_ori[~terminated].mean())

    metrics_eval["eval/all_metrics_dict"] = merged_all
    metrics_eval["eval/failed_metrics_dict"] = merged_failed

    # Carry over log_keys if present
    for seg in segments:
        if "log_keys" in seg:
            metrics_eval["log_keys"] = seg["log_keys"]
            break

    return metrics_eval


def main():
    parser = argparse.ArgumentParser(description="Merge segmented eval results")
    parser.add_argument(
        "--segments", nargs="+", required=True,
        help="Paths to metrics_eval.json files from each segment"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for merged metrics_eval.json"
    )
    args = parser.parse_args()

    # Validate inputs
    for path in args.segments:
        if not os.path.exists(path):
            print(f"ERROR: Segment file not found: {path}")
            sys.exit(1)

    print(f"Loading {len(args.segments)} segments...")
    segments = [load_segment(p) for p in args.segments]

    # Print segment info
    for i, (path, seg) in enumerate(zip(args.segments, segments)):
        n_motions = len(seg["eval/all_metrics_dict"]["motion_keys"])
        n_terminated = sum(seg["eval/all_metrics_dict"]["terminated"])
        print(f"  Segment {i+1}: {path} — {n_motions} motions, {n_terminated} terminated")

    print("Merging...")
    merged = merge_segments(segments)

    total_motions = len(merged["eval/all_metrics_dict"]["motion_keys"])
    total_terminated = sum(merged["eval/all_metrics_dict"]["terminated"])
    print(f"  Total: {total_motions} motions, {total_terminated} terminated")
    print(f"  Success rate: {merged['eval/success/success_rate']:.4f}")

    # Print mpjpe if available
    mpjpe_keys = [k for k in merged if "mpjpe_l" in k and "eval/all/" in k]
    for k in sorted(mpjpe_keys):
        print(f"  {k}: {merged[k]:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=4)
    print(f"Saved merged result to: {args.output}")


if __name__ == "__main__":
    main()
