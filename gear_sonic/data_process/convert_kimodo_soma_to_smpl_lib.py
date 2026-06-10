#!/usr/bin/env python3
"""Convert Kimodo SOMA NPZ outputs to SONIC smpl_motion_file PKL format.

Kimodo's SOMA-RP model outputs SOMASkeleton77 data as NPZ files.
SONIC's SMPL encoder expects per-sequence PKL files with:
  - pose_aa:     (T, 72)    SMPL 24-joint axis-angle
  - smpl_joints: (T, 24, 3) SMPL 24-joint 3D positions
  - transl:      (T, 3)     root translation
  - fps:         float

This script maps the 77-joint SOMA skeleton to the 24-joint SMPL skeleton,
bypassing SOMA retargeter to avoid introducing retargeting noise.

Usage:
    # Convert a single directory of NPZs
    python gear_sonic/data_process/convert_kimodo_soma_to_smpl_lib.py \
        --input /home/balance/kimodo/outputs/backward_walking \
        --output data/smpl_finetune/backward_walking

    # Convert all three categories
    python gear_sonic/data_process/convert_kimodo_soma_to_smpl_lib.py \
        --input /home/balance/kimodo/outputs/backward_walking \
               /home/balance/kimodo/outputs/small_steps \
               /home/balance/kimodo/outputs/upper_body_reach \
        --output data/smpl_finetune
"""

import argparse
import os
import sys

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

# SOMA SOMASkeleton77 index → SMPL 24-joint index mapping.
# Derived from matching joint names between the two skeletons.
SOMA77_TO_SMPL24 = {
    0: 0,    # Hips → pelvis
    67: 1,   # LeftLeg → left_hip
    72: 2,   # RightLeg → right_hip
    1: 3,    # Spine1 → spine1
    68: 4,   # LeftShin → left_knee
    73: 5,   # RightShin → right_knee
    2: 6,    # Spine2 → spine2
    69: 7,   # LeftFoot → left_ankle
    74: 8,   # RightFoot → right_ankle
    3: 9,    # Chest → spine3
    70: 10,  # LeftToeBase → left_foot
    75: 11,  # RightToeBase → right_foot
    4: 12,   # Neck1 → neck
    11: 13,  # LeftShoulder → left_collar
    39: 14,  # RightShoulder → right_collar
    6: 15,   # Head → head
    12: 16,  # LeftArm → left_shoulder
    40: 17,  # RightArm → right_shoulder
    13: 18,  # LeftForeArm → left_elbow
    41: 19,  # RightForeArm → right_elbow
    14: 20,  # LeftHand → left_wrist
    42: 21,  # RightHand → right_wrist
    14: 22,  # LeftHand → left_hand (duplicate of wrist)
    42: 23,  # RightHand → right_hand (duplicate of wrist)
}

# SOMA77 indices for the 24 SMPL joints, ordered by SMPL joint index
SOMA77_INDICES = [0, 67, 72, 1, 68, 73, 2, 69, 74, 3, 70, 75, 4, 11, 39, 6, 12, 40, 13, 41, 14, 42, 14, 42]


def convert_npz_to_smpl_pkl(npz_path: str, target_fps: float = 30.0) -> dict:
    """Convert a single Kimodo SOMA NPZ to SONIC smpl_motion_file format."""
    data = np.load(npz_path)

    local_rot_mats = data["local_rot_mats"]  # (T, 77, 3, 3)
    posed_joints = data["posed_joints"]      # (T, 77, 3)
    root_positions = data["root_positions"]  # (T, 3)
    T = posed_joints.shape[0]

    # Kimodo generates at 30 fps
    source_fps = 30.0

    # 1. Extract smpl_joints (T, 24, 3) from posed_joints (T, 77, 3)
    smpl_joints = posed_joints[:, SOMA77_INDICES, :]  # (T, 24, 3)

    # 2. Extract pose_aa (T, 72) from local_rot_mats (T, 77, 3, 3)
    #    Convert rotation matrices to axis-angle for the 24 SMPL joints
    rot_mats_24 = local_rot_mats[:, SOMA77_INDICES, :, :]  # (T, 24, 3, 3)
    rot_mats_flat = rot_mats_24.reshape(-1, 3, 3)  # (T*24, 3, 3)
    rotvecs = Rotation.from_matrix(rot_mats_flat).as_rotvec().astype(np.float32)
    pose_aa = rotvecs.reshape(T, 72)  # (T, 24*3)

    # 3. Root translation
    transl = root_positions.astype(np.float32)  # (T, 3)

    # 4. Resample to target_fps if needed
    # Use the same duration-based formula as motion_lib's interpolate_pose:
    #   duration = (T - 1) / source_fps
    #   n_target = floor(duration * target_fps) + 1
    # This ensures SMPL frame count matches robot motion after interpolation.
    if abs(source_fps - target_fps) > 0.5:
        duration = (T - 1) / source_fps
        n_target = int(duration * target_fps) + 1
        indices = np.linspace(0, T - 1, n_target).astype(int)
        pose_aa = pose_aa[indices]
        smpl_joints = smpl_joints[indices]
        transl = transl[indices]

    return {
        "pose_aa": pose_aa.astype(np.float32),
        "smpl_joints": smpl_joints.astype(np.float32),
        "transl": transl.astype(np.float32),
        "fps": target_fps,
        "original_pose_aa": Rotation.from_matrix(
            local_rot_mats[:, SOMA77_INDICES, :, :].reshape(-1, 3, 3)
        ).as_rotvec().astype(np.float32).reshape(T, 72),
        "original_fps": source_fps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kimodo SOMA NPZ to SONIC smpl_motion_file PKL"
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="One or more directories containing Kimodo SOMA NPZ files"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for individual PKL files"
    )
    parser.add_argument(
        "--fps", type=float, default=50.0,
        help="Target FPS (default: 50, matching Bones-SEED SMPL data)"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    total = 0
    failed = 0

    for input_dir in args.input:
        if not os.path.isdir(input_dir):
            print(f"WARNING: {input_dir} is not a directory, skipping")
            continue

        category = os.path.basename(input_dir.rstrip("/"))

        npz_paths = []
        for root, _dirs, files in os.walk(input_dir):
            for f in files:
                if f.endswith(".npz"):
                    npz_paths.append(os.path.join(root, f))
        npz_paths.sort()

        if not npz_paths:
            print(f"  No NPZ files in {input_dir}")
            continue

        print(f"\n[{category}] Converting {len(npz_paths)} motions...")

        for npz_path in npz_paths:
            name = os.path.splitext(os.path.basename(npz_path))[0]
            out_path = os.path.join(args.output, f"{name}.pkl")

            if os.path.exists(out_path):
                print(f"  [SKIP] {name}")
                total += 1
                continue

            try:
                entry = convert_npz_to_smpl_pkl(
                    npz_path,
                    target_fps=args.fps,
                )
                joblib.dump(entry, out_path, compress=True)
                T = entry["smpl_joints"].shape[0]
                print(f"  [OK] {name}: {T} frames @ {args.fps} fps")
                total += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                failed += 1

    print(f"\nDone: {total} converted, {failed} failed")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
