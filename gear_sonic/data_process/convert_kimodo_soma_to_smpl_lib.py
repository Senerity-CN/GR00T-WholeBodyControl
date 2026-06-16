#!/usr/bin/env python3
"""Convert Kimodo SOMA NPZ outputs to SONIC smpl_motion_file PKL format.

Kimodo's SOMA-RP model outputs SOMASkeleton77 data as NPZ files.
SONIC's SMPL encoder expects per-sequence PKL files with:
  - pose_aa:     (T, 72)    SMPL 24-joint axis-angle
  - smpl_joints: (T, 24, 3) SMPL 24-joint 3D positions (from compute_human_joints FK)
  - transl:      (T, 3)     root translation (Y-up convention)
  - fps:         float

The conversion ensures consistency with:
  - Bones-SEED official SMPL data convention
  - Teleop pipeline (pico_manager_thread_server.py:process_smpl_joints)

Key conventions:
  - pose_aa[:, :3] = R_world (world orientation, NO SMPL base rotation)
  - smpl_joints generated via compute_human_joints(body[:63], ytoz(root))
  - transl stays in Y-up (transl[1] = pelvis height)

Usage:
    # Convert a single directory of NPZs
    python gear_sonic/data_process/convert_kimodo_soma_to_smpl_lib.py \
        --input /home/balance/kimodo/outputs/backward_walking \
        --output data/smpl_finetune/backward_walking

    # Convert all categories
    python gear_sonic/data_process/convert_kimodo_soma_to_smpl_lib.py \
        --input /home/balance/kimodo/outputs \
        --output data/smpl_finetune
"""

import argparse
import os
import sys
from typing import Any

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gear_sonic.isaac_utils.rotations import quat_mul
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quaternion_to_angle_axis,
)

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

# NOTE: bones_seed pose_aa[:,:3] does NOT include SMPL base rotation.
# The base rotation (120° around [1,1,1]) is only introduced by the training
# code's remove_smpl_base_rot() pipeline. We store raw R_world here.


def convert_npz_to_smpl_pkl(npz_path: str, target_fps: float = 30.0) -> dict[str, Any]:
    """Convert a single Kimodo SOMA NPZ to SONIC smpl_motion_file format.

    Pipeline (matches bones_seed convention and process_smpl_joints):
      1. pose_aa root = R_world.as_rotvec()  -- world orientation, no base rotation
      2. smpl_joints = compute_human_joints(body[:63], ytoz(root_aa))  -- Z-up FK
      3. transl = root_positions  -- Y-up convention (transl[1] = height)
    """
    data = np.load(npz_path)

    local_rot_mats = data["local_rot_mats"]  # (T, 77, 3, 3)
    posed_joints = data["posed_joints"]      # (T, 77, 3)  (unused now, kept for reference)
    root_positions = data["root_positions"]  # (T, 3)
    T = local_rot_mats.shape[0]

    # Kimodo generates at 30 fps
    source_fps = 30.0

    # === 1. pose_aa: root = R_world (no base rotation), body as local rotations ===
    R_world = Rotation.from_matrix(local_rot_mats[:, 0])        # (T,)
    root_aa = R_world.as_rotvec().astype(np.float32)            # (T, 3)

    # Body joints (SMPL joints 1-23): extract local rotations from SOMA77
    body_rot_mats = local_rot_mats[:, SOMA77_INDICES[1:], :, :]  # (T, 23, 3, 3)
    body_aa = Rotation.from_matrix(
        body_rot_mats.reshape(-1, 3, 3)
    ).as_rotvec().astype(np.float32).reshape(T, 69)              # (T, 69)

    pose_aa = np.concatenate([root_aa, body_aa], axis=1)         # (T, 72)

    # === 2. smpl_joints: use compute_human_joints (same as teleop/bones_seed) ===
    # global_orient for FK = ytoz(root_aa) = Rx90 * R_world
    root_tensor = torch.from_numpy(root_aa).float()
    root_quat = angle_axis_to_quaternion(root_tensor)            # (T, 4)
    rx90_quat = angle_axis_to_quaternion(
        torch.tensor([[np.pi / 2, 0.0, 0.0]])
    )                                                            # (1, 4)
    root_quat_z = quat_mul(
        rx90_quat.expand(T, -1), root_quat, w_last=False
    )                                                            # (T, 4) Z-up
    global_orient_z = quaternion_to_angle_axis(root_quat_z)      # (T, 3)

    body_pose_63 = torch.from_numpy(body_aa[:, :63]).float()     # first 21 body joints
    with torch.no_grad():
        smpl_joints = compute_human_joints(
            body_pose_63, global_orient_z
        ).numpy()                                                # (T, 24, 3)

    # === 3. transl: Y-up root_positions (unchanged) ===
    transl = root_positions.astype(np.float32)                   # (T, 3)

    # === 4. Resample to target_fps ===
    # Use the same duration-based formula as motion_lib's interpolate_pose:
    #   duration = (T - 1) / source_fps
    #   n_target = floor(duration * target_fps) + 1
    # This ensures SMPL frame count matches robot motion after interpolation.
    # Keep original (pre-resample) pose_aa for backup.
    original_pose_aa = pose_aa.copy()

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
        "original_pose_aa": original_pose_aa.astype(np.float32),
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
