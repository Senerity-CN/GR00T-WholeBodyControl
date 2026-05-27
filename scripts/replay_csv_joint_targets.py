#!/usr/bin/env python3
"""Replay G1 joint target angles from a CSV file over ZMQ (Protocol v1).

This script reads joint target positions from a CSV file and streams them
to the C++ deploy process via the ZMQ Protocol v1 (joint-based) pipeline.

CSV format (g1.xml export, 36 columns, no header)
-------------------------------------------------
One row per timestep::

      [0:3]  pelvis position   (x, y, z)         meters
      [3:7]  pelvis quaternion (w, x, y, z)      unit quat, w-first
      [7:36] 29 body joint angles               radians, MuJoCo order

Joint order (columns 7..35) follows g1.xml (MuJoCo order):
    left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee,
    left_ankle_pitch, left_ankle_roll,
    right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee,
    right_ankle_pitch, right_ankle_roll,
    waist_yaw, waist_roll, waist_pitch,
    left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw,
    left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw,
    right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw,
    right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw

NOTE: Joints are reordered to IsaacLab order before ZMQ transmission,
because the C++ deploy stores JointPositions internally in IsaacLab order
(interleaved left-right-waist pattern).  The mapping is defined in
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp.

Pipeline::

    CSV joint targets
        │
        ▼
    ZMQ PUB tcp://*:5556  (Protocol v1)
    ├─ topic "command"        : state transitions
    ├─ topic "planner"        : PLANNER mode idle
    ├─ topic "pose"           : joint_pos + joint_vel + body_quat + frame_index
    └─ topic "manager_state"  : 2Hz heartbeat
        │
        ▼
    C++ ZMQEndpointInterface (v1 branch)
        │
        ▼
    Encoder (mode 0) → Policy → G1 joint targets
        │
        ▼
    MuJoCo / Real G1

3-Terminal run topology
-----------------------
Terminal 1 (.venv_teleop)::

    python gear_sonic/scripts/run_sim_loop.py            # MuJoCo virtual robot

Terminal 2::

    ./deploy.sh --input-type zmq_manager sim              # C++ deploy

Terminal 3 (.venv_data_collection or any env with zmq+numpy+pandas)::

    python scripts/replay_csv_joint_targets.py --csv path/to/joints.csv

Keyboard controls (press key then ENTER)
----------------------------------------
* ``r`` : OFF -> PLANNER  (power on, robot stands).
* ``s`` : PLANNER -> POSE (start / restart joint replay).
* ``p`` : POSE <-> PLANNER (pause / resume replay).
* ``x`` : -> OFF (power off).
* ``h`` : help.
* ``q`` : quit.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zmq
from numpy.typing import NDArray

# -- Make ``gear_sonic`` importable when launched from repo root or scripts/. --
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
    build_planner_message,
    pack_pose_message,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEADER_SIZE = 1280

# G1 body joints: 29 DOF (no hands)
G1_BODY_JOINTS = 29

# Expected CSV columns: 3 (pos) + 4 (quat) + 29 (joints) = 36
G1_CSV_COLS = 3 + 4 + G1_BODY_JOINTS

# Joint order mapping: MuJoCo g1.xml order → IsaacLab order (used internally by
# the C++ deploy encoder/policy). Indexed by IsaacLab position, value is MuJoCo
# index.  i.e.  il_joints[il_idx] = mj_joints[MUJOCO_TO_ISAACLAB[il_idx]]
# Source: gear_sonic_deploy/.../policy_parameters.hpp
MUJOCO_TO_ISAACLAB: list[int] = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
]


class StreamMode(IntEnum):
    OFF = 0
    POSE = 1
    PLANNER = 2


class LocomotionMode(IntEnum):
    IDLE = 0


DEFAULT_PORT = 5556
DEFAULT_FPS = 30  # g1.xml CSV export is typically 30Hz
DEFAULT_OUTPUT_FPS = 50  # match C++ deploy MOTION_FPS by default
MANAGER_STATE_PERIOD_S = 0.5


# ---------------------------------------------------------------------------
# Motion interpolation & velocity utilities
# ---------------------------------------------------------------------------
def _quat_slerp_batch(
    q0: NDArray[np.float32],
    q1: NDArray[np.float32],
    t: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Vectorized SLERP between batches of quaternions (w, x, y, z).

    Args:
        q0, q1: (N, 4) start / end unit quaternions.
        t: (N,) blend factors in [0, 1].
    Returns:
        (N, 4) interpolated unit quaternions, dtype float32.
    """
    q0_d = np.asarray(q0, dtype=np.float64)
    q1_d = np.asarray(q1, dtype=np.float64)
    t_d = np.asarray(t, dtype=np.float64)

    # Shortest path: flip q1 when dot < 0.
    dot = np.sum(q0_d * q1_d, axis=1)
    sign = np.where(dot < 0.0, -1.0, 1.0)
    q1_d = q1_d * sign[:, None]
    dot = np.clip(dot * sign, -1.0, 1.0)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    near_parallel = sin_theta_0 < 1e-6
    safe_sin = np.where(near_parallel, 1.0, sin_theta_0)

    theta = theta_0 * t_d
    s0 = np.cos(theta) - dot * np.sin(theta) / safe_sin
    s1 = np.sin(theta) / safe_sin
    slerp = s0[:, None] * q0_d + s1[:, None] * q1_d

    # Lerp+normalize fallback for nearly-parallel quaternions.
    lerp = q0_d + t_d[:, None] * (q1_d - q0_d)
    lerp = lerp / np.maximum(np.linalg.norm(lerp, axis=1, keepdims=True), 1e-12)

    result = np.where(near_parallel[:, None], lerp, slerp)
    return result.astype(np.float32)


def interpolate_motion(
    body_joints: NDArray[np.float32],
    body_quat: NDArray[np.float32],
    input_fps: float,
    output_fps: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Resample (lerp + slerp) a motion sequence from input_fps to output_fps.

    The output time grid is ``arange(0, duration, 1/output_fps)`` where
    ``duration = (input_frames - 1) / input_fps``.

    Args:
        body_joints: (N, J) joint positions sampled at input_fps.
        body_quat: (N, 4) body quaternion (w, x, y, z) at input_fps.
        input_fps: source frame rate.
        output_fps: target frame rate.
    Returns:
        (M, J) joints, (M, 4) quat at output_fps.
    """
    input_frames = body_joints.shape[0]
    if input_frames < 2 or input_fps == output_fps:
        return body_joints.astype(np.float32), body_quat.astype(np.float32)

    input_dt = 1.0 / float(input_fps)
    output_dt = 1.0 / float(output_fps)
    duration = (input_frames - 1) * input_dt

    # Output time samples (exclusive of duration, matching reference impl).
    times = np.arange(0.0, duration, output_dt, dtype=np.float64)
    if times.size == 0:
        return body_joints.astype(np.float32), body_quat.astype(np.float32)

    phase = np.clip(times / duration, 0.0, 1.0)
    fpos = phase * (input_frames - 1)
    index_0 = np.floor(fpos).astype(np.int64)
    index_1 = np.minimum(index_0 + 1, input_frames - 1)
    blend = (fpos - index_0).astype(np.float64)

    j0 = body_joints[index_0].astype(np.float64)
    j1 = body_joints[index_1].astype(np.float64)
    out_joints = (j0 * (1.0 - blend[:, None]) + j1 * blend[:, None]).astype(np.float32)
    out_quat = _quat_slerp_batch(body_quat[index_0], body_quat[index_1], blend)
    return out_joints, out_quat


def compute_joint_velocities(joints: NDArray[np.float32], dt: float) -> NDArray[np.float32]:
    """Compute joint velocities using central differences (np.gradient)."""
    if joints.shape[0] < 2:
        return np.zeros_like(joints, dtype=np.float32)
    return np.gradient(joints.astype(np.float64), dt, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# ZMQ Protocol v1 message builder
# ---------------------------------------------------------------------------
def build_v1_pose_message(
    joint_pos: NDArray[np.float32],
    joint_vel: NDArray[np.float32],
    body_quat: NDArray[np.float32],
    frame_index: int,
    catch_up: bool = False,
) -> bytes:
    """Build a Protocol v1 pose message for ZMQ.

    Args:
        joint_pos: (num_joints,) joint positions in radians.
        joint_vel: (num_joints,) joint velocities in rad/s.
        body_quat: (4,) body quaternion [w, x, y, z].
        frame_index: monotonically increasing frame counter.
        catch_up: if True, C++ deploy may skip frames to catch up.
    """
    N = 1
    num_joints = joint_pos.shape[0]

    jp = joint_pos.astype(np.float32).reshape(1, num_joints)
    jv = joint_vel.astype(np.float32).reshape(1, num_joints)
    bq = body_quat.astype(np.float32).reshape(1, 4)
    fi = np.array([frame_index], dtype=np.int64)

    fields = [
        {"name": "joint_pos", "dtype": "f32", "shape": [N, num_joints]},
        {"name": "joint_vel", "dtype": "f32", "shape": [N, num_joints]},
        {"name": "body_quat_w", "dtype": "f32", "shape": [N, 4]},
        {"name": "frame_index", "dtype": "i64", "shape": [N]},
        {"name": "catch_up", "dtype": "u8", "shape": [1]},
    ]

    payload = b"".join([
        jp.tobytes(),
        jv.tobytes(),
        bq.tobytes(),
        fi.tobytes(),
        struct.pack("B", 1 if catch_up else 0),
    ])

    # Build header
    header = {
        "v": 1,
        "endian": "le",
        "count": N,
        "fields": fields,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    header_bytes = header_json.ljust(HEADER_SIZE, b"\x00")

    return b"pose" + header_bytes + payload


# ---------------------------------------------------------------------------
# Helper: idle planner / manager_state
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
    s : PLANNER -> POSE (start / restart joint replay)
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
def load_csv_joint_targets(
    csv_path: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Load joint targets from a g1.xml-format CSV (36 columns, no header).

    Layout:
        [0:3]  pelvis position (x, y, z)
        [3:7]  pelvis quaternion (w, x, y, z)
        [7:36] 29 body joint angles (radians, MuJoCo order)

    Returns:
        body_joints: (N, 29) body joint positions **in IsaacLab order**
                     (reordered from MuJoCo via MUJOCO_TO_ISAACLAB).
        body_quat:   (N, 4)  pelvis quaternion [w, x, y, z].
    """
    print(f"[Replay] Loading CSV: {csv_path}", flush=True)

    df = pd.read_csv(csv_path, header=None)
    n_frames = len(df)
    n_cols = len(df.columns)

    # Re-read with header if first cell is non-numeric.
    try:
        float(df.iloc[0, 0])
    except (ValueError, TypeError):
        df = pd.read_csv(csv_path)
        n_frames = len(df)
        n_cols = len(df.columns)
        print("[Replay] CSV has header row", flush=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_numeric = len(numeric_cols)
    print(
        f"[Replay] CSV: {n_frames} rows, {n_cols} columns ({n_numeric} numeric)",
        flush=True,
    )

    if n_numeric != G1_CSV_COLS:
        raise ValueError(
            f"Expected {G1_CSV_COLS} numeric columns (3 pos + 4 quat + 29 joints), "
            f"got {n_numeric}."
        )

    raw_data = df[numeric_cols].to_numpy(dtype=np.float32)
    root_pos = raw_data[:, 0:3]
    body_quat = raw_data[:, 3:7]   # (w, x, y, z)
    body_joints = raw_data[:, 7:36]  # 29 joints, MuJoCo order

    print(
        f"[Replay]   Root pos range: "
        f"x=[{root_pos[:, 0].min():.3f}, {root_pos[:, 0].max():.3f}], "
        f"z=[{root_pos[:, 2].min():.3f}, {root_pos[:, 2].max():.3f}]",
        flush=True,
    )
    print(f"[Replay]   Body quat[0] w={body_quat[0, 0]:.4f}", flush=True)
    print("[Replay]   Joints: 29 DOF (MuJoCo g1.xml order)", flush=True)

    # Reorder joints from MuJoCo → IsaacLab (the order expected by the C++ deploy encoder)
    body_joints = body_joints[:, MUJOCO_TO_ISAACLAB]
    print("[Replay]   Joints reordered: MuJoCo → IsaacLab", flush=True)

    return body_joints, body_quat


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay G1 joint targets from CSV over ZMQ (Protocol v1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file with joint target angles (radians).",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="ZMQ PUB port.")
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Source CSV frame rate in Hz (input rate). Default 30Hz for g1.xml export.",
    )
    parser.add_argument(
        "--output_fps",
        type=int,
        default=DEFAULT_OUTPUT_FPS,
        help="Target sending rate in Hz. CSV is interpolated (lerp+slerp) to this rate "
        "and joint velocities are computed via np.gradient on the upsampled sequence. "
        "Default 50Hz to match deploy MOTION_FPS.",
    )
    parser.add_argument(
        "--ctrl_fps",
        type=int,
        default=50,
        help="C++ deploy internal control rate in Hz (used to compute frame_index). "
        "Set to match the deploy's MOTION_FPS; default 50.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop replay continuously; default: end -> PLANNER.",
    )
    parser.add_argument(
        "--catch_up",
        action="store_true",
        help="Enable catch-up mode (C++ deploy may skip frames).",
    )
    parser.add_argument(
        "--safety_warmup_sec",
        type=float,
        default=1.0,
        help="Pause after bind() for SUBs to connect (ZMQ slow-joiner).",
    )
    args = parser.parse_args()

    # --- Load data ----------------------------------------------------------
    body_joints, csv_body_quat = load_csv_joint_targets(args.csv)
    n_frames_csv = body_joints.shape[0]
    num_body = body_joints.shape[1]

    # --- Interpolate (lerp + slerp) to output_fps ---------------------------
    if args.output_fps != args.fps:
        body_joints, csv_body_quat = interpolate_motion(
            body_joints, csv_body_quat, input_fps=args.fps, output_fps=args.output_fps
        )
        print(
            f"[Replay] Motion interpolated: {n_frames_csv} frames @ {args.fps}Hz "
            f"-> {body_joints.shape[0]} frames @ {args.output_fps}Hz "
            f"(lerp on joints, slerp on body_quat)",
            flush=True,
        )

    n_frames = body_joints.shape[0]

    # Compute joint velocities via central differences on the upsampled sequence.
    output_dt = 1.0 / float(args.output_fps)
    joint_vel = compute_joint_velocities(body_joints, output_dt)
    print(
        f"[Replay] Computed joint velocities via np.gradient "
        f"(dt={output_dt:.4f}s, output_fps={args.output_fps}Hz)",
        flush=True,
    )
    print(
        f"[Replay] vel range: [{joint_vel.min():+.3f}, {joint_vel.max():+.3f}] rad/s",
        flush=True,
    )

    # --- ZMQ PUB ------------------------------------------------------------
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[Replay] ZMQ PUB bound to tcp://*:{args.port}", flush=True)

    # Bootstrap
    sock.send(build_command_message(start=False, stop=False, planner=False))
    sock.send(build_idle_planner_message())

    print(
        f"[Replay] Warmup {args.safety_warmup_sec:.1f}s (lets C++ deploy SUB connect) ...",
        flush=True,
    )
    time.sleep(max(0.0, args.safety_warmup_sec))

    # --- Keyboard -----------------------------------------------------------
    kq = _KeyQueue()
    kbd = threading.Thread(target=_keyboard_thread, args=(kq,), daemon=True)
    kbd.start()

    # --- State --------------------------------------------------------------
    mode = StreamMode.OFF
    sock.send(build_manager_state_message(mode))
    print(f"[Replay] Initial mode: {mode.name}", flush=True)
    print(
        f"[Replay] Data: {n_frames} frames, {num_body} body joints, "
        f"send_FPS={args.output_fps} (input CSV {args.fps}Hz)",
        flush=True,
    )

    pose_idx = 0
    global_frame = 0
    # Per-tick increment of frame_index (in C++ deploy time base).
    # After interpolation we send at output_fps, so the step is ctrl_fps/output_fps.
    # When output_fps == ctrl_fps this is exactly 1.0.
    frame_index_step = float(args.ctrl_fps) / float(args.output_fps)
    global_frame_f = 0.0
    print(
        f"[Replay] frame_index step = {frame_index_step:.4f} "
        f"(send={args.output_fps}Hz -> deploy={args.ctrl_fps}Hz time base)",
        flush=True,
    )

    period = 1.0 / float(args.output_fps)
    next_tick = time.time()
    last_manager_state = 0.0

    print("[Replay] Using body quaternion from CSV (per-frame)", flush=True)

    def reset_pose_state() -> None:
        nonlocal pose_idx, global_frame_f, global_frame
        pose_idx = 0
        global_frame_f = 0.0
        global_frame = 0

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
                if pose_idx >= n_frames:
                    if args.loop:
                        reset_pose_state()
                    else:
                        print("[Replay] Joint data exhausted -> PLANNER", flush=True)
                        transition(StreamMode.PLANNER)

                if mode == StreamMode.POSE:
                    jp = body_joints[pose_idx]
                    jv = joint_vel[pose_idx]
                    bq = csv_body_quat[pose_idx]

                    sock.send(
                        build_v1_pose_message(
                            joint_pos=jp,
                            joint_vel=jv,
                            body_quat=bq,
                            frame_index=global_frame,
                            catch_up=args.catch_up,
                        )
                    )
                    pose_idx += 1
                    global_frame_f += frame_index_step
                    global_frame = int(round(global_frame_f))

                    # Progress
                    if pose_idx % 100 == 0:
                        elapsed_s = pose_idx / float(args.output_fps)
                        print(
                            f"[Replay] Frame {pose_idx}/{n_frames} "
                            f"({100*pose_idx/n_frames:.1f}%, {elapsed_s:.1f}s)",
                            flush=True,
                        )

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
