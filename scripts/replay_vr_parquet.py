#!/usr/bin/env python3
"""Replay VR body tracking data from a parquet file over ZMQ (Protocol v3).

This script reads raw VR skeletal tracking data (27 bone points × 7D pose)
from a parquet file, converts it to SMPL parameters + VR 3-point pose, and
streams it to the C++ deploy process — reproducing what
pico_manager_thread_server.py does in real-time.

Pipeline::

    parquet observation.data.vr (27 bones × [x,y,z,qx,qy,qz,qw])
        │
        ▼
    VR 27-bone → SMPL 24-joint mapping
        │
        ▼
    compute_from_body_poses() → smpl_pose (21×3) + smpl_joints (24×3) + body_quat (4)
    _process_3pt_pose()       → vr_3pt_pose (3×7)
        │
        ▼
    ZMQ PUB tcp://*:5556  (Protocol v3)
    ├─ topic "command"        : state transitions
    ├─ topic "planner"        : PLANNER mode idle
    ├─ topic "pose"           : smpl_joints + smpl_pose + joint_pos + vr_3pt + ...
    └─ topic "manager_state"  : 2Hz heartbeat
        │
        ▼
    C++ ZMQEndpointInterface (v3 branch)
        │
        ▼
    MuJoCo / Real G1

3-Terminal run topology
-----------------------
Terminal 1 (.venv_teleop)::

    python gear_sonic/scripts/run_sim_loop.py

Terminal 2::

    ./deploy.sh --input-type zmq_manager sim

Terminal 3 (.venv_sim or .venv_teleop)::

    python scripts/replay_vr_parquet.py --parquet <path_to_vr_parquet>

Keyboard controls (press key then ENTER)
----------------------------------------
* ``r`` : OFF -> PLANNER  (power on, robot stands).
* ``s`` : PLANNER -> POSE (start / restart VR replay).
* ``p`` : POSE <-> PLANNER (pause / resume replay).
* ``x`` : -> OFF (power off).
* ``h`` : help.
* ``q`` : quit.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zmq
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as sRot

# -- Make ``gear_sonic`` importable when launched from repo root or scripts/. --
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
)

try:
    from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
except ImportError:
    print("Warning: gear_sonic.isaac_utils.rotations not available, using identity transforms.")
    remove_smpl_base_rot = None
    smpl_root_ytoz_up = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
class StreamMode(IntEnum):
    OFF = 0
    POSE = 1
    PLANNER = 2


class LocomotionMode(IntEnum):
    IDLE = 0


DEFAULT_PORT = 5556
DEFAULT_FPS = 50
NUM_FRAMES_TO_SEND = 5
MANAGER_STATE_PERIOD_S = 0.5

# SMPL parent indices for FK
PARENT_INDICES = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23,
][:24]

# VR bone name → SMPL joint index mapping
VR_TO_SMPL_MAP = {
    "waist": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_hand": 22,
    "right_hand": 23,
}

# VR 3-point rotation offsets (from pico_manager_thread_server.py)
OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),    # Root
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),     # L-Wrist
    sRot.from_euler("xyz", [-90, 0, 180], degrees=True),  # R-Wrist
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),    # Neck
]

# Controller-to-wrist fixed offset (physical mounting property).
# q_wrist = q_controller * Q_CTRL2WRIST
# Measured across 6 sessions / 4820 frames, cross-session deviation = 0.00°.
# Format: [qx, qy, qz, qw] (scalar-last)
Q_CTRL2WRIST_LEFT = sRot.from_quat([+0.4333, -0.3503, +0.6667, +0.4949])
Q_CTRL2WRIST_RIGHT = sRot.from_quat([+0.4231, +0.3791, -0.6810, +0.4621])

# Controller right-multiply correction (local-frame).
# q_ctrl_corrected = q_ctrl_src * Q_R
# Transforms 153640-style controller orientations to match 154545-style.
# Verified: mean 0.4°, max 1.3°.
Q_R_LEFT = sRot.from_quat([+0.3733, -0.2406, +0.4889, -0.7508])
Q_R_RIGHT = sRot.from_quat([+0.4494, +0.1109, -0.5804, -0.6699])


def apply_controller_correction(
    body_poses_np: NDArray[np.float32],
    left_ctrl_quat: NDArray[np.float32] | None,
    right_ctrl_quat: NDArray[np.float32] | None,
) -> NDArray[np.float32]:
    """Replace SMPL wrist/hand orientations using corrected controller orientations.

    Pipeline per side:
      1. correct controller:    q_ctrl' = q_ctrl * Q_R
      2. controller to wrist:   q_wrist = q_ctrl' * Q_CTRL2WRIST
      3. overwrite joints 20/22 or 21/23
    """
    bp = body_poses_np.copy()
    if left_ctrl_quat is not None:
        q_ctrl = sRot.from_quat(left_ctrl_quat) * Q_R_LEFT
        q_wrist = (q_ctrl * Q_CTRL2WRIST_LEFT).as_quat()
        bp[20, 3:] = q_wrist
        bp[22, 3:] = q_wrist
    if right_ctrl_quat is not None:
        q_ctrl = sRot.from_quat(right_ctrl_quat) * Q_R_RIGHT
        q_wrist = (q_ctrl * Q_CTRL2WRIST_RIGHT).as_quat()
        bp[21, 3:] = q_wrist
        bp[23, 3:] = q_wrist
    return bp


# ---------------------------------------------------------------------------
# VR parquet → SMPL body_poses_np (24×7) conversion
# ---------------------------------------------------------------------------
def vr_frame_to_body_poses(row: pd.Series) -> NDArray[np.float32]:
    """Convert one VR parquet row (27 bones) to SMPL-ordered body_poses_np (24, 7).

    VR format per bone: [x, y, z, qx, qy, qz, qw] (scalar-last quaternion)
    Output format:       [x, y, z, qx, qy, qz, qw] (same, matches xrt.get_body_joints_pose())
    """
    body_poses = np.zeros((24, 7), dtype=np.float32)
    for vr_name, smpl_idx in VR_TO_SMPL_MAP.items():
        if vr_name in row.index:
            body_poses[smpl_idx] = row[vr_name]
    return body_poses


# ---------------------------------------------------------------------------
# SMPL processing (from pico_manager_thread_server.py)
# ---------------------------------------------------------------------------
def process_smpl_joints(body_pose, global_orient, transl):
    """Process SMPL parameters to compute local joints."""
    global_orient_quat = angle_axis_to_quaternion(global_orient)
    if smpl_root_ytoz_up is not None:
        global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    global_orient_new = quaternion_to_angle_axis(global_orient_quat)

    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
    )

    if remove_smpl_base_rot is not None:
        global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)

    global_orient_quat_inv = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(global_orient_quat_inv, joints)

    return {
        "smpl_pose": body_pose,
        "smpl_joints_local": smpl_joints_local,
        "global_orient_quat": global_orient_quat,
    }


def compute_from_body_poses(
    parent_indices: list, device: torch.device, body_poses_np: NDArray[np.float32]
) -> dict:
    """Compute SMPL local joints and body orientation from body_poses_np (24, 7)."""
    global_quats = body_poses_np[:, [6, 3, 4, 5]]  # xyzw -> wxyz for scipy

    global_rots = sRot.from_quat(global_quats, scalar_first=True)
    global_rots = global_rots * sRot.from_euler("y", 180, degrees=True)

    local_rots = []
    for i in range(24):
        if parent_indices[i] == -1:
            local_rots.append(global_rots[i])
        else:
            local_rot = global_rots[parent_indices[i]].inv() * global_rots[i]
            local_rots.append(local_rot)

    pose_aa = np.array([rot.as_rotvec() for rot in local_rots])

    body_pose = torch.from_numpy(pose_aa[1:].flatten()).float().to(device).unsqueeze(0)
    global_orient = torch.from_numpy(pose_aa[0]).float().to(device).unsqueeze(0)
    transl = torch.from_numpy(body_poses_np[0, :3]).float().to(device).unsqueeze(0)

    return process_smpl_joints(body_pose, global_orient, transl)


# ---------------------------------------------------------------------------
# VR 3-point pose (from pico_manager_thread_server.py)
# ---------------------------------------------------------------------------
def _compute_rel_transform(pose, world_frame, scalar_first=True):
    """Transform a pose from Unity frame to robot frame."""
    world_frame = world_frame.copy()
    Q = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]])
    pose[:3] = Q @ pose[:3]
    world_frame[:3] = Q @ world_frame[:3]
    rot_base = sRot.from_quat(world_frame[3:], scalar_first=scalar_first).as_matrix()
    rot = sRot.from_quat(pose[3:], scalar_first=scalar_first).as_matrix()
    rel_rot = sRot.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    rel_pos = sRot.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
    return rel_pos, rel_rot.as_quat(scalar_first=True)


def compute_3pt_pose(smpl_pose_np: NDArray[np.float32]) -> NDArray[np.float32]:
    """Extract 3-point VR pose (L-Wrist, R-Wrist, Neck) from full body joint poses.

    Args:
        smpl_pose_np: (24, 7) - [x, y, z, qx, qy, qz, qw] per joint (scalar-last)

    Returns:
        (3, 7) - [x, y, z, qw, qx, qy, qz] for [L-Wrist, R-Wrist, Neck]
                 relative to root, scalar-first quaternion
    """
    smpl_pose_np = smpl_pose_np.copy()

    body_poses = np.zeros((smpl_pose_np.shape[0], 7), dtype=np.float32)
    for i in range(smpl_pose_np.shape[0]):
        pos, orn = _compute_rel_transform(
            smpl_pose_np[i], [0, 0, 0, 0, 0, 0, 1], scalar_first=False
        )
        body_poses[i, :3] = pos
        body_poses[i, 3:] = orn

    positions = np.array([[p[0], p[1], p[2]] for p in body_poses])
    kp_poses = np.zeros((4, 7), dtype=np.float32)

    for i, pose in enumerate(body_poses):
        if i not in [0, 22, 23, 12]:
            continue
        rel_i = [0, 22, 23, 12].index(i)
        quat = np.array([pose[3], pose[4], pose[5], pose[6]])
        rot_quat = (sRot.from_quat(quat, scalar_first=True) * OFFSETS[rel_i]).as_quat(
            scalar_first=False
        )
        kp_poses[rel_i, 3:] = rot_quat
        kp_poses[rel_i, :3] = positions[i]

    root_pos = kp_poses[0, :3].copy()
    root_quat = kp_poses[0, 3:].copy()

    for i in range(1, 4):
        kp_poses[i, :3] = sRot.from_quat(root_quat).inv().apply(kp_poses[i, :3] - root_pos)
        kp_poses[i, 3:] = (
            sRot.from_quat(root_quat).inv() * sRot.from_quat(kp_poses[i, 3:])
        ).as_quat(scalar_first=True)

    return kp_poses[1:]


# ---------------------------------------------------------------------------
# Wrist joint extraction (from pico_manager_thread_server.py)
# ---------------------------------------------------------------------------
def extract_wrist_joints(body_pose_21x3: NDArray[np.float32]) -> NDArray[np.float64]:
    """Extract G1 wrist joint positions from SMPL body pose.

    Returns: (29,) joint_pos array with only wrist joints filled.
    """
    from scipy.spatial.transform import Rotation as R

    joint_pos = np.zeros(29, dtype=np.float64)
    body_pose = body_pose_21x3.reshape(1, 21, 3)

    SMPL_L_ELBOW_IDX = 17
    SMPL_L_WRIST_IDX = 19
    SMPL_R_ELBOW_IDX = 18
    SMPL_R_WRIST_IDX = 20

    G1_L_WRIST_ROLL_IDX = 23
    G1_L_WRIST_PITCH_IDX = 25
    G1_L_WRIST_YAW_IDX = 27
    G1_R_WRIST_ROLL_IDX = 24
    G1_R_WRIST_PITCH_IDX = 26
    G1_R_WRIST_YAW_IDX = 28

    smpl_l_elbow_aa = body_pose[:, SMPL_L_ELBOW_IDX]
    smpl_l_wrist_aa = body_pose[:, SMPL_L_WRIST_IDX]
    smpl_r_elbow_aa = body_pose[:, SMPL_R_ELBOW_IDX]
    smpl_r_wrist_aa = body_pose[:, SMPL_R_WRIST_IDX]

    g1_l_elbow_axis = np.array([0, 1, 0])
    _, g1_l_elbow_q_swing = decompose_rotation_aa(smpl_l_elbow_aa, g1_l_elbow_axis)

    g1_r_elbow_axis = np.array([0, 1, 0])
    _, g1_r_elbow_q_swing = decompose_rotation_aa(smpl_r_elbow_aa, g1_r_elbow_axis)

    l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )
    r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )

    l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
    r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

    g1_l_wrist_roll = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
    g1_l_wrist_pitch = -l_wrist_euler[:, 1]
    g1_l_wrist_yaw = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

    g1_r_wrist_roll = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
    g1_r_wrist_pitch = -r_wrist_euler[:, 1]
    g1_r_wrist_yaw = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

    joint_pos[G1_L_WRIST_ROLL_IDX] = g1_l_wrist_roll[0]
    joint_pos[G1_L_WRIST_PITCH_IDX] = -g1_l_wrist_pitch[0]
    joint_pos[G1_L_WRIST_YAW_IDX] = g1_l_wrist_yaw[0]
    joint_pos[G1_R_WRIST_ROLL_IDX] = g1_r_wrist_roll[0]
    joint_pos[G1_R_WRIST_PITCH_IDX] = g1_r_wrist_pitch[0]
    joint_pos[G1_R_WRIST_YAW_IDX] = g1_r_wrist_yaw[0]

    return joint_pos


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------
def quat_lerp_normalized(q0: NDArray, q1: NDArray, alpha: float) -> NDArray:
    """Linear interpolate two quaternions and renormalize (xyzw order)."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return q


def interp_pose_axis_angle(
    prev_pose: NDArray[np.float32], curr_pose: NDArray[np.float32], alpha: float
) -> NDArray[np.float32]:
    """Interpolate axis-angle joint poses via quaternion lerp."""
    prev_quats = sRot.from_rotvec(prev_pose.reshape(-1, 3)).as_quat()
    curr_quats = sRot.from_rotvec(curr_pose.reshape(-1, 3)).as_quat()
    out_quats = np.empty_like(prev_quats)
    for i in range(prev_quats.shape[0]):
        out_quats[i] = quat_lerp_normalized(prev_quats[i], curr_quats[i], alpha)
    return sRot.from_quat(out_quats).as_rotvec().reshape(prev_pose.shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_vr_parquet(
    parquet_path: str, extract_controllers: bool = False,
) -> list[NDArray[np.float32]] | tuple[list[NDArray[np.float32]], list[tuple]]:
    """Load VR parquet and convert each frame to body_poses_np (24, 7).

    Args:
        parquet_path: Path to the VR observation parquet.
        extract_controllers: Also return per-frame controller quaternions.

    Returns:
        If extract_controllers is False: list of (24, 7) arrays.
        If True: (frames, ctrl_quats) where each ctrl_quats entry is
                 (left_ctrl_quat_xyzw, right_ctrl_quat_xyzw).
    """
    print(f"[Replay] Loading VR parquet: {parquet_path}", flush=True)
    df = pd.read_parquet(parquet_path)

    bone_cols = [c for c in df.columns if c not in ("sdk_timestamp_ns", "timestamp_ns")]
    missing = set(VR_TO_SMPL_MAP.keys()) - set(bone_cols)
    if missing:
        print(f"[Replay] WARNING: Missing VR bones for SMPL mapping: {missing}", flush=True)

    has_lc = "left_controller" in df.columns
    has_rc = "right_controller" in df.columns

    frames = []
    ctrl_quats: list[tuple] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        frames.append(vr_frame_to_body_poses(row))
        if extract_controllers:
            ctrl_quats.append((
                row["left_controller"][3:].copy() if has_lc else None,
                row["right_controller"][3:].copy() if has_rc else None,
            ))

    print(f"[Replay] Loaded {len(frames)} VR frames", flush=True)
    if extract_controllers:
        return frames, ctrl_quats
    return frames


# ---------------------------------------------------------------------------
# ZMQ message builders
# ---------------------------------------------------------------------------
def build_idle_planner_message() -> bytes:
    return build_planner_message(
        mode=int(LocomotionMode.IDLE),
        movement=(0.0, 0.0, 0.0),
        facing=(1.0, 0.0, 0.0),
    )


def build_manager_state_message(stream_mode: StreamMode) -> bytes:
    return pack_pose_message(
        {
            "stream_mode": np.array([int(stream_mode)], dtype=np.int32),
            "toggle_data_collection": np.array([False], dtype=bool),
            "toggle_data_abort": np.array([False], dtype=bool),
        },
        topic="manager_state",
    )


# ---------------------------------------------------------------------------
# Keyboard thread
# ---------------------------------------------------------------------------
HELP_TEXT = """\
Keyboard controls (press key then ENTER):
    r : OFF -> PLANNER  (power on, robot stands)
    s : PLANNER -> POSE (start / restart VR replay)
    p : POSE <-> PLANNER (pause / resume replay)
    x : -> OFF          (power off)
    h : help
    q : quit
"""


class _KeyQueue:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._q: list[str] = []
        self._stop = threading.Event()

    def push(self, key: str) -> None:
        with self._lock:
            self._q.append(key)

    def pop(self) -> str | None:
        with self._lock:
            return self._q.pop(0) if self._q else None

    def stop(self) -> None:
        self._stop.set()

    def stopped(self) -> bool:
        return self._stop.is_set()


def _keyboard_thread(kq: _KeyQueue) -> None:
    print(HELP_TEXT, flush=True)
    while not kq.stopped():
        try:
            line = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            kq.push("q")
            return
        if not line:
            continue
        ch = line[0]
        if ch in ("r", "s", "p", "x", "h", "q"):
            kq.push(ch)
        else:
            print(f"[Keyboard] unknown key '{ch}', press 'h' for help", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay VR body tracking parquet over ZMQ (Protocol v3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="outputs/session_20260527_151431/data/chunk-000/observation.data.vr_000000.parquet",
        help="Path to VR observation parquet file.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="ZMQ PUB port.")
    parser.add_argument(
        "--target_fps", type=int, default=DEFAULT_FPS, help="Target streaming FPS."
    )
    parser.add_argument(
        "--num_frames", type=int, default=NUM_FRAMES_TO_SEND,
        help="Number of frames to buffer before sending.",
    )
    parser.add_argument(
        "--loop", action="store_true", help="Loop replay continuously."
    )
    parser.add_argument(
        "--safety_warmup_sec", type=float, default=1.0,
        help="Pause after bind() for SUBs to connect.",
    )
    parser.add_argument(
        "--correct_controller", action="store_true",
        help="Apply right-multiply correction to controller quats and overwrite "
             "wrist/hand in SMPL (transforms 153640-style to 154545-style).",
    )
    args = parser.parse_args()

    # --- Load data ----------------------------------------------------------
    controller_quats = None
    if args.correct_controller:
        vr_frames, controller_quats = load_vr_parquet(args.parquet, extract_controllers=True)
    else:
        vr_frames = load_vr_parquet(args.parquet)
    n_total_frames = len(vr_frames)

    device = torch.device("cpu")

    # --- Precompute SMPL data for all frames --------------------------------
    print("[Replay] Precomputing SMPL parameters for all frames...", flush=True)
    all_smpl_pose = []
    all_smpl_joints = []
    all_body_quat = []
    all_vr_3pt = []
    all_joint_pos = []

    for i, body_poses_np in enumerate(vr_frames):
        if args.correct_controller and controller_quats is not None:
            body_poses_np = apply_controller_correction(body_poses_np, *controller_quats[i])

        result = compute_from_body_poses(PARENT_INDICES, device, body_poses_np)

        smpl_pose_np = (
            result["smpl_pose"].detach().cpu().numpy()[:, :63].reshape(-1, 21, 3)[0]
        ).astype(np.float32)
        smpl_joints_np = result["smpl_joints_local"].detach().cpu().numpy()[0].astype(np.float32)
        body_quat_np = result["global_orient_quat"].detach().cpu().numpy()[0].astype(np.float32)

        vr_3pt_pose = compute_3pt_pose(body_poses_np)
        joint_pos = extract_wrist_joints(smpl_pose_np)

        all_smpl_pose.append(smpl_pose_np)
        all_smpl_joints.append(smpl_joints_np)
        all_body_quat.append(body_quat_np)
        all_vr_3pt.append(vr_3pt_pose)
        all_joint_pos.append(joint_pos)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_total_frames} frames", flush=True)

    print(f"[Replay] Precomputation done ({n_total_frames} frames)", flush=True)

    # --- Resample to target_fps from VR native rate -------------------------
    # VR data is ~54Hz, target is 50Hz - use nearest-neighbor for simplicity
    # (interpolation is done in the streaming loop like pico_manager does)

    # --- ZMQ PUB ------------------------------------------------------------
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[Replay] ZMQ PUB bound to tcp://*:{args.port}", flush=True)

    sock.send(build_command_message(start=False, stop=False, planner=False))
    sock.send(build_idle_planner_message())

    print(
        f"[Replay] Warmup {args.safety_warmup_sec:.1f}s (lets C++ deploy SUB connect)...",
        flush=True,
    )
    time.sleep(max(0.0, args.safety_warmup_sec))

    # --- Keyboard -----------------------------------------------------------
    kq = _KeyQueue()
    kbd = threading.Thread(target=_keyboard_thread, args=(kq,), daemon=True)
    kbd.start()

    # --- State machine ------------------------------------------------------
    mode = StreamMode.OFF
    sock.send(build_manager_state_message(mode))
    print(f"[Replay] Initial mode: {mode.name}", flush=True)

    pose_idx = 0
    global_frame = 0
    frame_buffer: dict[str, list] = {
        "smpl_pose": [],
        "smpl_joints": [],
        "body_quat_w": [],
        "joint_pos": [],
        "frame_index": [],
    }

    period = 1.0 / float(args.target_fps)
    next_tick = time.time()
    last_manager_state = 0.0

    def reset_pose_state() -> None:
        nonlocal pose_idx, frame_buffer
        pose_idx = 0
        frame_buffer = {k: [] for k in frame_buffer}

    def transition(new_mode: StreamMode) -> None:
        nonlocal mode, last_manager_state
        if new_mode == mode:
            return
        if new_mode == StreamMode.OFF:
            sock.send(build_command_message(start=False, stop=True, planner=True))
        elif new_mode == StreamMode.PLANNER:
            sock.send(build_command_message(start=True, stop=False, planner=True))
            sock.send(build_idle_planner_message())
        elif new_mode == StreamMode.POSE:
            sock.send(build_command_message(start=True, stop=False, planner=False))
            reset_pose_state()
        sock.send(build_manager_state_message(new_mode))
        last_manager_state = time.time()
        print(f"[Replay] Mode: {mode.name} -> {new_mode.name}", flush=True)
        mode = new_mode

    try:
        while True:
            # ---------- handle keyboard ----------
            ch = kq.pop()
            if ch == "h":
                print(HELP_TEXT, flush=True)
            elif ch == "q":
                break
            elif ch == "x":
                transition(StreamMode.OFF)
            elif ch == "r":
                if mode != StreamMode.PLANNER:
                    transition(StreamMode.PLANNER)
            elif ch == "s":
                if mode == StreamMode.POSE:
                    reset_pose_state()
                    print("[Replay] POSE: replay restarted from frame 0", flush=True)
                else:
                    transition(StreamMode.POSE)
            elif ch == "p":
                if mode == StreamMode.POSE:
                    transition(StreamMode.PLANNER)
                elif mode == StreamMode.PLANNER:
                    transition(StreamMode.POSE)

            # ---------- per-tick streaming ----------
            if mode == StreamMode.PLANNER:
                sock.send(build_idle_planner_message())

            elif mode == StreamMode.POSE:
                if pose_idx >= n_total_frames:
                    if args.loop:
                        reset_pose_state()
                        print("[Replay] Looping from frame 0", flush=True)
                    else:
                        print("[Replay] VR frames exhausted -> PLANNER", flush=True)
                        transition(StreamMode.PLANNER)

                if mode == StreamMode.POSE:
                    # Add current frame to buffer
                    frame_buffer["smpl_pose"].append(all_smpl_pose[pose_idx])
                    frame_buffer["smpl_joints"].append(all_smpl_joints[pose_idx])
                    frame_buffer["body_quat_w"].append(all_body_quat[pose_idx])
                    frame_buffer["joint_pos"].append(all_joint_pos[pose_idx])
                    frame_buffer["frame_index"].append(global_frame)

                    # Keep buffer at num_frames size
                    num_frames = args.num_frames
                    for key in frame_buffer:
                        if len(frame_buffer[key]) > num_frames:
                            frame_buffer[key] = frame_buffer[key][-num_frames:]

                    # Send when buffer is full
                    if len(frame_buffer["frame_index"]) >= num_frames:
                        n_buf = len(frame_buffer["frame_index"])
                        vr_3pt = all_vr_3pt[pose_idx]

                        numpy_data = {
                            "smpl_pose": np.stack(frame_buffer["smpl_pose"], axis=0),
                            "smpl_joints": np.stack(frame_buffer["smpl_joints"], axis=0),
                            "body_quat_w": np.stack(frame_buffer["body_quat_w"], axis=0),
                            "joint_pos": np.stack(frame_buffer["joint_pos"], axis=0),
                            "joint_vel": np.zeros((n_buf, 29), dtype=np.float64),
                            "vr_position": vr_3pt[:, :3].flatten().astype(np.float32),
                            "vr_orientation": vr_3pt[:, 3:].flatten().astype(np.float32),
                            "frame_index": np.array(
                                frame_buffer["frame_index"], dtype=np.int64
                            ),
                            "left_hand_joints": np.zeros(7, dtype=np.float32),
                            "right_hand_joints": np.zeros(7, dtype=np.float32),
                            "heading_increment": np.array([0.0], dtype=np.float32),
                            "toggle_data_collection": np.array([False], dtype=bool),
                            "toggle_data_abort": np.array([False], dtype=bool),
                        }

                        packed_message = pack_pose_message(numpy_data, topic="pose", version=3)
                        sock.send(packed_message)

                    pose_idx += 1
                    global_frame += 1

            # ---------- manager_state @ ~2 Hz ----------
            now = time.time()
            if now - last_manager_state >= MANAGER_STATE_PERIOD_S:
                sock.send(build_manager_state_message(mode))
                last_manager_state = now

            # ---------- pace loop ----------
            next_tick += period
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.time()

    except KeyboardInterrupt:
        print("[Replay] Interrupted by user", flush=True)
    finally:
        try:
            sock.send(build_command_message(start=False, stop=True, planner=True))
            sock.send(build_manager_state_message(StreamMode.OFF))
        except Exception:
            pass
        kq.stop()
        sock.close(linger=200)
        ctx.term()
        print("[Replay] Shutdown.", flush=True)


if __name__ == "__main__":
    main()
