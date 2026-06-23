#!/usr/bin/env python3
"""Replay SMPL motion data from a kimodo pkl file over ZMQ (Protocol v3).

This script reads pre-computed SMPL motion parameters (pose_aa, smpl_joints,
transl) from a kimodo joblib pkl file and streams them to the C++ deploy
process — reproducing what pico_manager_thread_server.py does in real-time.

Pipeline::

    kimodo SMPL pkl (pose_aa T×72, smpl_joints T×24×3, transl T×3)
        │
        ▼
    pose_aa → global_orient (3) + body_pose (63)
        │
        ▼
    process_smpl_joints() → smpl_pose (21×3) + smpl_joints_local (24×3) + body_quat (4)
    FK chain → global quats → compute_3pt_pose() → vr_3pt_pose (3×7)
    extract_wrist_joints() → joint_pos (29)
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

    python scripts/replay_smpl.py --pkl data/smpl_finetune/bw_look_02.pkl

Keyboard controls (press key then ENTER)
----------------------------------------
* ``r`` : OFF -> PLANNER  (power on, robot stands).
* ``s`` : PLANNER -> POSE (start / restart replay).
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

import joblib
import numpy as np
import torch
import zmq
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as sRot

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

PARENT_INDICES = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23,
][:24]

OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),
    sRot.from_euler("xyz", [-90, 0, 180], degrees=True),
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),
]


# ---------------------------------------------------------------------------
# SMPL processing (from pico_manager_thread_server.py)
# ---------------------------------------------------------------------------
def process_smpl_joints(body_pose, global_orient, transl):
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


# ---------------------------------------------------------------------------
# FK: local axis-angle → VR-frame body_poses_np (24, 7)
# ---------------------------------------------------------------------------
def smpl_aa_to_body_poses(
    pose_aa_frame: NDArray[np.float32],
    smpl_joints_frame: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Reconstruct VR-like body_poses_np (24, 7) from SMPL local axis-angles.

    The VR→SMPL conversion in compute_from_body_poses right-multiplies a
    180° Y rotation before extracting local rotations.  Here we reverse
    that: FK the locals back to globals, then undo the 180° Y.
    """
    local_aa = pose_aa_frame.reshape(24, 3)
    local_rots = sRot.from_rotvec(local_aa)

    global_rots = [None] * 24
    for i in range(24):
        if PARENT_INDICES[i] == -1:
            global_rots[i] = local_rots[i]
        else:
            global_rots[i] = global_rots[PARENT_INDICES[i]] * local_rots[i]

    r_y180 = sRot.from_euler("y", 180, degrees=True)
    body_poses_np = np.zeros((24, 7), dtype=np.float32)
    for i in range(24):
        body_poses_np[i, :3] = smpl_joints_frame[i]
        vr_rot = global_rots[i] * r_y180
        body_poses_np[i, 3:] = vr_rot.as_quat()  # scalar-last xyzw
    return body_poses_np


# ---------------------------------------------------------------------------
# VR 3-point pose (from pico_manager_thread_server.py)
# ---------------------------------------------------------------------------
def _compute_rel_transform(pose, world_frame, scalar_first=True):
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
    s : PLANNER -> POSE (start / restart replay)
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
# Data loading
# ---------------------------------------------------------------------------
def load_smpl_pkl(pkl_path: str) -> dict:
    """Load kimodo SMPL pkl and precompute all per-frame data for streaming.

    Returns dict with lists of per-frame numpy arrays:
        smpl_pose, smpl_joints, body_quat, vr_3pt, joint_pos
    """
    print(f"[Replay] Loading SMPL pkl: {pkl_path}", flush=True)
    data = joblib.load(pkl_path)

    if not isinstance(data, dict) or "pose_aa" not in data:
        raise ValueError(f"{pkl_path} missing 'pose_aa' key — not a kimodo SMPL pkl.")

    pose_aa = np.asarray(data["pose_aa"], dtype=np.float32)
    smpl_joints = np.asarray(data["smpl_joints"], dtype=np.float32)
    fps = float(data.get("fps", 50.0))
    n_frames = pose_aa.shape[0]

    print(f"[Replay] {n_frames} frames @ {fps} fps", flush=True)

    device = torch.device("cpu")

    all_smpl_pose = []
    all_smpl_joints = []
    all_body_quat = []
    all_vr_3pt = []
    all_joint_pos = []

    print("[Replay] Precomputing SMPL parameters...", flush=True)
    for i in range(n_frames):
        global_orient = torch.from_numpy(pose_aa[i, :3]).float().unsqueeze(0)
        body_pose = torch.from_numpy(pose_aa[i, 3:66]).float().unsqueeze(0)
        transl = torch.from_numpy(smpl_joints[i, 0]).float().unsqueeze(0)

        result = process_smpl_joints(body_pose, global_orient, transl)

        smpl_pose_np = (
            result["smpl_pose"].detach().cpu().numpy()[:, :63].reshape(-1, 21, 3)[0]
        ).astype(np.float32)
        smpl_joints_np = result["smpl_joints_local"].detach().cpu().numpy()[0].astype(np.float32)
        body_quat_np = result["global_orient_quat"].detach().cpu().numpy()[0].astype(np.float32)

        body_poses_np = smpl_aa_to_body_poses(pose_aa[i], smpl_joints[i])
        vr_3pt_pose = compute_3pt_pose(body_poses_np)
        joint_pos = extract_wrist_joints(smpl_pose_np)

        all_smpl_pose.append(smpl_pose_np)
        all_smpl_joints.append(smpl_joints_np)
        all_body_quat.append(body_quat_np)
        all_vr_3pt.append(vr_3pt_pose)
        all_joint_pos.append(joint_pos)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_frames} frames", flush=True)

    print(f"[Replay] Precomputation done ({n_frames} frames)", flush=True)

    return {
        "smpl_pose": all_smpl_pose,
        "smpl_joints": all_smpl_joints,
        "body_quat": all_body_quat,
        "vr_3pt": all_vr_3pt,
        "joint_pos": all_joint_pos,
        "n_frames": n_frames,
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay SMPL motion pkl over ZMQ (Protocol v3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pkl",
        type=str,
        default="data/smpl_finetune/bw_look_02.pkl",
        help="Path to kimodo SMPL motion pkl file.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="ZMQ PUB port.")
    parser.add_argument(
        "--target_fps", type=int, default=None,
        help="Target streaming FPS (default: use fps from pkl).",
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
    args = parser.parse_args()

    # --- Load data ----------------------------------------------------------
    precomputed = load_smpl_pkl(args.pkl)
    n_total_frames = precomputed["n_frames"]
    target_fps = args.target_fps if args.target_fps is not None else int(round(precomputed["fps"]))

    all_smpl_pose = precomputed["smpl_pose"]
    all_smpl_joints = precomputed["smpl_joints"]
    all_body_quat = precomputed["body_quat"]
    all_vr_3pt = precomputed["vr_3pt"]
    all_joint_pos = precomputed["joint_pos"]

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
    print(f"[Replay] Initial mode: {mode.name}  ({n_total_frames} frames @ {target_fps} fps)", flush=True)

    pose_idx = 0
    global_frame = 0
    frame_buffer: dict[str, list] = {
        "smpl_pose": [],
        "smpl_joints": [],
        "body_quat_w": [],
        "joint_pos": [],
        "frame_index": [],
    }

    period = 1.0 / float(target_fps)
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
                        print("[Replay] Frames exhausted -> PLANNER", flush=True)
                        transition(StreamMode.PLANNER)

                if mode == StreamMode.POSE:
                    frame_buffer["smpl_pose"].append(all_smpl_pose[pose_idx])
                    frame_buffer["smpl_joints"].append(all_smpl_joints[pose_idx])
                    frame_buffer["body_quat_w"].append(all_body_quat[pose_idx])
                    frame_buffer["joint_pos"].append(all_joint_pos[pose_idx])
                    frame_buffer["frame_index"].append(global_frame)

                    num_frames = args.num_frames
                    for key in frame_buffer:
                        if len(frame_buffer[key]) > num_frames:
                            frame_buffer[key] = frame_buffer[key][-num_frames:]

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
