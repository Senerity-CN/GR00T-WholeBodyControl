#!/usr/bin/env python3
"""Replay LeRobot ``action.motion_token`` over ZMQ (Protocol v4, token-only).

This script is the *latent-action analogue* of ``replay_action_wbc.py``.
Instead of sending joint targets (Protocol v1), it streams 64-dim motion
tokens via Protocol v4 which bypass the SMPL encoder and feed directly
into the g1_dyn decoder in the C++ deploy process.

Pipeline::

    parquet action.motion_token (64)
        │
        ▼
    ZMQ PUB tcp://*:5556  (Protocol v4)
    ├─ topic "command"        : state transitions
    ├─ topic "planner"        : PLANNER mode idle
    ├─ topic "pose"           : POSE mode token_state
    └─ topic "manager_state"  : 2Hz heartbeat
        │
        ▼
    C++ ZMQEndpointInterface (v4 branch)
        │
        ▼
    g1_dyn Decoder (ONNX)  →  G1 joint targets
        │
        ▼
    MuJoCo / Real G1

3-Terminal run topology
-----------------------
Terminal 1 (.venv_teleop)::

    python gear_sonic/scripts/run_sim_loop.py            # MuJoCo virtual robot

Terminal 2::

    ./deploy.sh --input-type zmq_manager sim              # C++ deploy

Terminal 3 (.venv_data_collection)::

    python scripts/replay_action_motion_token.py          # this script (ZMQ PUB)

Keyboard controls (press key then ENTER)
----------------------------------------
* ``r`` : OFF -> PLANNER  (power on, robot stands).
* ``s`` : PLANNER -> POSE (start / restart motion token replay).
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
# State / protocol constants
# ---------------------------------------------------------------------------
class StreamMode(IntEnum):
    """Mirrors ``StreamMode`` in pico_manager_thread_server (subset)."""

    OFF = 0
    POSE = 1
    PLANNER = 2


class LocomotionMode(IntEnum):
    """Mirrors ``LocomotionMode`` in pico_manager_thread_server (subset)."""

    IDLE = 0


DEFAULT_PORT = 5556
DEFAULT_FPS = 50
MANAGER_STATE_PERIOD_S = 0.5

MOTION_TOKEN_DIM = 64

# LeRobot -> MuJoCo per-hand reorder (same as replay_action_wbc.py).
HAND_REMAP_LEROBOT_TO_MUJOCO: NDArray[np.int64] = np.array(
    [4, 5, 6, 2, 3, 0, 1], dtype=np.int64
)


# ---------------------------------------------------------------------------
# Hand remap helper (reused from replay_action_wbc.py logic)
# ---------------------------------------------------------------------------
def remap_hand_to_mujoco(hand_7: NDArray[Any]) -> NDArray[np.float32]:
    """Reorder a 7-DoF LeRobot hand vector into MuJoCo XML order."""
    arr = np.asarray(hand_7, dtype=np.float32)
    if arr.shape[-1] != 7:
        raise ValueError(f"Hand vector must be 7-DoF, got shape {arr.shape}")
    return arr[HAND_REMAP_LEROBOT_TO_MUJOCO]


# ---------------------------------------------------------------------------
# ZMQ message builders
# ---------------------------------------------------------------------------
def build_token_pose_message(
    motion_token: NDArray[np.float32],
    frame_index: int,
    left_hand7: NDArray[np.float32] | None = None,
    right_hand7: NDArray[np.float32] | None = None,
) -> bytes:
    """Pack a Protocol v4 token-only pose message.

    Fields::

        token_state       (1, 64) f32  - motion token (universal token)
        frame_index       (1,)    i64  - monotonically increasing
        left_hand_joints  (1, 7)  f32  - optional, MuJoCo-remapped
        right_hand_joints (1, 7)  f32  - optional, MuJoCo-remapped
    """
    token = np.asarray(motion_token, dtype=np.float32).reshape(1, MOTION_TOKEN_DIM)
    fi = np.array([frame_index], dtype=np.int64)

    payload: dict[str, NDArray[Any]] = {
        "token_state": token,
        "frame_index": fi,
    }

    if left_hand7 is not None:
        payload["left_hand_joints"] = np.asarray(
            left_hand7, dtype=np.float32
        ).reshape(1, 7)
    if right_hand7 is not None:
        payload["right_hand_joints"] = np.asarray(
            right_hand7, dtype=np.float32
        ).reshape(1, 7)

    return pack_pose_message(payload, topic="pose", version=4)


def build_idle_planner_message() -> bytes:
    """One IDLE planner tick - keeps the robot standing in PLANNER mode."""
    return build_planner_message(
        mode=int(LocomotionMode.IDLE),
        movement=(0.0, 0.0, 0.0),
        facing=(1.0, 0.0, 0.0),
    )


def build_manager_state_message(stream_mode: StreamMode) -> bytes:
    """Tell subscribers the current ``stream_mode``."""
    return pack_pose_message(
        {
            "stream_mode": np.array([int(stream_mode)], dtype=np.int32),
            "toggle_data_collection": np.array([False], dtype=bool),
            "toggle_data_abort": np.array([False], dtype=bool),
        },
        topic="manager_state",
    )


# ---------------------------------------------------------------------------
# Keyboard (line-mode) thread
# ---------------------------------------------------------------------------
HELP_TEXT = """\
Keyboard controls (press key then ENTER):
    r : OFF -> PLANNER  (power on, robot stands)
    s : PLANNER -> POSE (start / restart motion token replay)
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
def load_motion_tokens(parquet_path: str) -> NDArray[np.float64]:
    """Load ``action.motion_token`` (Nx64) from a LeRobot parquet file."""
    print(f"[Replay] Loading parquet: {parquet_path}", flush=True)
    df = pd.read_parquet(parquet_path)
    if "action.motion_token" not in df.columns:
        raise KeyError(
            f"'action.motion_token' missing in {parquet_path}. "
            f"Available columns: {df.columns.tolist()}"
        )
    raw = list(df["action.motion_token"].to_numpy())
    tokens = np.asarray(np.stack(raw, axis=0), dtype=np.float64)
    if tokens.ndim != 2 or tokens.shape[1] != MOTION_TOKEN_DIM:
        raise ValueError(
            f"Expected (N, {MOTION_TOKEN_DIM}) action.motion_token, got {tokens.shape}"
        )
    print(f"[Replay] Loaded {tokens.shape[0]} frames of motion tokens (dim={MOTION_TOKEN_DIM})", flush=True)
    return tokens


def load_hand_joints(parquet_path: str) -> NDArray[np.float64] | None:
    """Try to load hand joints from ``action.wbc`` (Nx43).

    Returns (N, 14) array [left7 | right7] in MuJoCo order, or None if unavailable.
    """
    df = pd.read_parquet(parquet_path)
    if "action.wbc" not in df.columns:
        print("[Replay] No 'action.wbc' found - hand joints will not be sent.", flush=True)
        return None
    raw = list(df["action.wbc"].to_numpy())
    actions = np.asarray(np.stack(raw, axis=0), dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 43:
        print(f"[Replay] action.wbc shape {actions.shape} unexpected - skipping hand joints.", flush=True)
        return None

    # Extract and remap hand joints
    n_frames = actions.shape[0]
    hands = np.zeros((n_frames, 14), dtype=np.float64)
    for i in range(n_frames):
        hands[i, :7] = remap_hand_to_mujoco(actions[i, 22:29].astype(np.float32))
        hands[i, 7:] = remap_hand_to_mujoco(actions[i, 36:43].astype(np.float32))
    print(f"[Replay] Loaded hand joints from action.wbc ({n_frames} frames, remapped to MuJoCo order)", flush=True)
    return hands


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay action.motion_token over ZMQ (Protocol v4, token-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="outputs/full_merged_dataset/data/chunk-000/episode_000000.parquet",
        help="Path to LeRobot episode parquet containing 'action.motion_token'.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="ZMQ PUB port.")
    parser.add_argument(
        "--target_fps",
        type=int,
        default=DEFAULT_FPS,
        help="Outer loop rate; pose messages are sent every tick.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the parquet replay continuously; default: end -> PLANNER.",
    )
    parser.add_argument(
        "--no_hands",
        action="store_true",
        help="Do not send hand joints even if action.wbc is available.",
    )
    parser.add_argument(
        "--safety_warmup_sec",
        type=float,
        default=1.0,
        help="Pause after bind() so SUBs (C++ deploy) can connect before any"
        " command is sent (ZMQ PUB/SUB slow-joiner).",
    )
    args = parser.parse_args()

    # --- Load data ----------------------------------------------------------
    tokens = load_motion_tokens(args.parquet)
    n_frames = tokens.shape[0]

    hand_joints: NDArray[np.float64] | None = None
    if not args.no_hands:
        hand_joints = load_hand_joints(args.parquet)

    # --- ZMQ PUB ------------------------------------------------------------
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[Replay] ZMQ PUB bound to tcp://*:{args.port}", flush=True)

    # Bootstrap (mirrors genmo_smpl_to_pose_zmq.py / pico_manager bring-up).
    sock.send(build_command_message(start=False, stop=False, planner=False))
    sock.send(build_idle_planner_message())

    print(
        f"[Replay] Warmup {args.safety_warmup_sec:.1f}s before accepting input"
        f" (lets C++ deploy SUB connect) ...",
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

    pose_idx = 0  # next frame to consume
    global_frame = 0  # monotonic frame counter

    period = 1.0 / float(args.target_fps)
    next_tick = time.time()
    last_manager_state = 0.0

    def reset_pose_state() -> None:
        nonlocal pose_idx
        pose_idx = 0

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
        # Refresh manager_state right away
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
                        print(
                            "[Replay] motion tokens exhausted -> PLANNER", flush=True
                        )
                        transition(StreamMode.PLANNER)

                if mode == StreamMode.POSE:  # may have flipped above
                    token = tokens[pose_idx].astype(np.float32)

                    lh7: NDArray[np.float32] | None = None
                    rh7: NDArray[np.float32] | None = None
                    if hand_joints is not None:
                        lh7 = hand_joints[pose_idx, :7].astype(np.float32)
                        rh7 = hand_joints[pose_idx, 7:].astype(np.float32)

                    sock.send(
                        build_token_pose_message(
                            motion_token=token,
                            frame_index=global_frame,
                            left_hand7=lh7,
                            right_hand7=rh7,
                        )
                    )
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
