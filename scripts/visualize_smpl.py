#!/usr/bin/env python3
"""Replay SMPL (and optionally VR 3-point) data for visualization.

Two input sources are supported:

1. GROOT data-collection episode parquet (``--parquet``):
   Reads ``teleop.smpl_joints`` (root-LOCAL frame), ``teleop.body_quat_w``,
   ``teleop.vr_3pt_position`` and ``teleop.vr_3pt_orientation``. Joints are
   rotated into world frame using ``body_quat_w``. NOTE: parquet does not
   record global translation, so the pelvis stays anchored at the origin.

2. Kimodo SMPL motion pkl (``--smpl-pkl``), e.g.
   ``data/smpl_finetune/bw_fast_01.pkl``:
   joblib dict with ``smpl_joints`` (T, 24, 3) already in WORLD frame,
   ``transl`` (T, 3), ``pose_aa`` (T, 72) and ``fps``. The pelvis trajectory
   is preserved, so the floating base actually moves.

Usage:
    python scripts/visualize_smpl.py \
        --parquet outputs/2026-06-15-17-14-51/data/chunk-000/episode_000000.parquet

    python scripts/visualize_smpl.py \
        --smpl-pkl data/smpl_finetune/bw_fast_01.pkl

Keyboard controls (press key then ENTER):
    p / ENTER : play / pause
    .         : next frame (when paused)
    ,         : previous frame (when paused)
    r         : restart from frame 0
    + / =     : speed up (x2)
    -         : slow down (x0.5)
    q         : quit
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import VR3PtPoseVisualizer


# ---------------------------------------------------------------------------
# Keyboard helpers
# ---------------------------------------------------------------------------
HELP_TEXT = """\
Controls (press key then ENTER):
    p / ENTER : play / pause      ./, : step fwd/back
    r : restart    +/- : speed     q : quit
"""


class _KeyQueue:
    def __init__(self):
        self._lock = threading.Lock()
        self._q: list[str] = []
        self._stop = threading.Event()

    def push(self, key: str):
        with self._lock:
            self._q.append(key)

    def pop(self) -> str | None:
        with self._lock:
            return self._q.pop(0) if self._q else None

    def stop(self):
        self._stop.set()

    def stopped(self) -> bool:
        return self._stop.is_set()


def _keyboard_thread(kq: _KeyQueue):
    print(HELP_TEXT, flush=True)
    while not kq.stopped():
        try:
            line = input().strip()
        except (EOFError, KeyboardInterrupt):
            kq.push("q")
            return
        if not line:
            kq.push("p")
            continue
        ch = line[0]
        if ch in ("p", " ", ".", ",", "r", "+", "=", "-", "h", "q"):
            kq.push(ch)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def build_stale_mask(smpl_arr: np.ndarray) -> np.ndarray:
    """Build a boolean mask of frames to remove (True = remove).

    Removes all-zero rows AND any consecutive frozen (identical-to-next)
    rows that immediately precede a zero row. Frozen runs that do NOT
    lead into a zero row are left untouched.

    Mirrors the logic in gear_sonic/scripts/process_dataset.py.
    """
    n = len(smpl_arr)
    is_zero = np.all(smpl_arr == 0, axis=1)
    remove = is_zero.copy()

    diffs = np.zeros(n)
    diffs[1:] = np.sum(np.abs(smpl_arr[1:] - smpl_arr[:-1]), axis=1)

    for i in range(n):
        if is_zero[i]:
            j = i - 1
            while j >= 0 and diffs[j] == 0.0 and not is_zero[j]:
                remove[j] = True
                j -= 1

    return remove


def load_episode_data(parquet_path: str, remove_stale: bool = True):
    """Load SMPL and VR 3-point data from an episode parquet.

    SMPL joints in the dataset are stored in local frame. This function
    rotates them into world frame using teleop.body_quat_w so the
    skeleton correctly shows turning / orientation changes.

    Args:
        parquet_path: Path to the episode parquet file.
        remove_stale: If True, remove frames where smpl_joints is all-zero
            and frozen lead-in frames (same logic as process_dataset.py).

    Returns:
        smpl_joints_list: list of (24, 3) arrays in world frame
        vr_3pt_list: list of (3, 7) arrays [x,y,z,qw,qx,qy,qz] per point
        has_vr3pt: whether VR 3-point data is non-zero
    """
    from scipy.spatial.transform import Rotation as sRot

    print(f"[Replay] Loading: {parquet_path}", flush=True)
    df = pd.read_parquet(parquet_path)
    n_frames = len(df)
    print(f"[Replay] {n_frames} frames", flush=True)

    # --- Filter stale / zero SMPL frames -----------------------------------
    valid_indices = None
    if remove_stale and "teleop.smpl_joints" in df.columns:
        smpl_arr = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in df["teleop.smpl_joints"]]
        )
        mask = build_stale_mask(smpl_arr)
        n_remove = int(mask.sum())
        n_zero = int(np.all(smpl_arr == 0, axis=1).sum())
        n_frozen = n_remove - n_zero

        if n_remove > 0:
            pct = 100.0 * n_remove / n_frames
            print(
                f"[Replay] Removing {n_remove}/{n_frames} stale frames "
                f"({pct:.1f}%) — {n_zero} zero + {n_frozen} frozen lead-in",
                flush=True,
            )
            valid_indices = np.where(~mask)[0]
            df = df.iloc[valid_indices].copy().reset_index(drop=True)
            n_frames = len(df)
            print(f"[Replay] {n_frames} frames after filtering", flush=True)
        else:
            print("[Replay] No stale frames found", flush=True)

    # --- Extract per-frame data --------------------------------------------
    smpl_joints_list = []
    vr_3pt_list = []
    has_vr3pt = False

    for idx in range(n_frames):
        row = df.iloc[idx]

        # SMPL joints: (72,) -> (24, 3) — these are LOCAL frame positions
        joints_flat = np.asarray(row["teleop.smpl_joints"], dtype=np.float32)
        joints_local = joints_flat.reshape(24, 3)

        # body_quat_w: scalar-first (w, x, y, z) — convert to xyzw for scipy
        body_quat_w = np.asarray(row["teleop.body_quat_w"], dtype=np.float64)
        body_quat_xyzw = body_quat_w[[1, 2, 3, 0]]
        body_rotation = sRot.from_quat(body_quat_xyzw)

        # Rotate local joints into world frame
        joints_world = body_rotation.apply(joints_local).astype(np.float32)
        smpl_joints_list.append(joints_world)

        # VR 3-point position: (9,) -> (3, 3)
        vr_pos = np.asarray(row["teleop.vr_3pt_position"], dtype=np.float32).reshape(3, 3)
        # VR 3-point orientation: (18,) -> (3, 6) rot6d -> convert to quat
        vr_ori_rot6d = np.asarray(row["teleop.vr_3pt_orientation"], dtype=np.float32).reshape(3, 6)

        if np.any(vr_pos != 0) or np.any(vr_ori_rot6d != 0):
            has_vr3pt = True

        # Convert rot6d to quaternion (qw, qx, qy, qz) for the visualizer
        vr_quats = np.zeros((3, 4), dtype=np.float32)
        for i in range(3):
            rot6d = vr_ori_rot6d[i]
            if np.all(rot6d == 0):
                vr_quats[i] = [1.0, 0.0, 0.0, 0.0]
            else:
                rot_mat = np.eye(3, dtype=np.float64)
                rot_mat[:, 0] = rot6d[:3]
                rot_mat[:, 1] = rot6d[3:6]
                rot_mat[:, 2] = np.cross(rot_mat[:, 0], rot_mat[:, 1])
                quat_xyzw = sRot.from_matrix(rot_mat).as_quat()
                vr_quats[i] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        # Combine into (3, 7): [x, y, z, qw, qx, qy, qz]
        vr_3pt = np.hstack([vr_pos, vr_quats]).astype(np.float32)
        vr_3pt_list.append(vr_3pt)

        if (idx + 1) % 200 == 0:
            print(f"  Loaded {idx + 1}/{n_frames}", flush=True)

    print(f"[Replay] Done. VR 3-point data present: {has_vr3pt}", flush=True)
    return smpl_joints_list, vr_3pt_list, has_vr3pt


def load_smpl_pkl_data(pkl_path: str):
    """Load a kimodo SMPL motion pkl (joblib dict).

    Expected keys:
      - ``smpl_joints``: (T, 24, 3) WORLD-frame joint positions
        (``smpl_joints[:, 0]`` already equals ``transl``).
      - ``transl``: (T, 3) global root translation.
      - ``pose_aa``: (T, 72) full-body axis-angle (root_orient is the first 3).
      - ``fps``: scalar playback fps (after upsampling).

    Unlike the parquet loader we do NOT need to rotate by ``body_quat_w`` and
    do NOT need to add ``transl`` separately — both are already baked into
    ``smpl_joints``. The visualizer's first-frame anchoring will subtract the
    initial pelvis position so the trajectory is relative to ``smpl_root_position``
    while preserving frame-to-frame motion (so the pelvis actually moves).

    Args:
        pkl_path: Path to the kimodo SMPL pkl file.

    Returns:
        smpl_joints_list: list of (24, 3) float32 arrays in world frame.
        vr_3pt_list: list of (3, 7) float32 arrays — all zeros (no VR data).
        has_vr3pt: always False.
        fps: float fps recorded in the pkl (use as default playback rate).
    """
    import joblib

    print(f"[Replay] Loading kimodo SMPL pkl: {pkl_path}", flush=True)
    data = joblib.load(pkl_path)

    if not isinstance(data, dict) or "smpl_joints" not in data:
        raise ValueError(
            f"{pkl_path} does not look like a kimodo SMPL pkl: "
            f"missing 'smpl_joints' key."
        )

    smpl_joints = np.asarray(data["smpl_joints"], dtype=np.float32)
    if smpl_joints.ndim != 3 or smpl_joints.shape[1:] != (24, 3):
        raise ValueError(
            f"smpl_joints has unexpected shape {smpl_joints.shape}, expected (T, 24, 3)"
        )

    fps = float(data.get("fps", 50.0))
    n_frames = smpl_joints.shape[0]
    transl = np.asarray(data.get("transl", smpl_joints[:, 0]), dtype=np.float32)

    pelvis_xyz_range = transl.max(0) - transl.min(0)
    print(
        f"[Replay] {n_frames} frames @ {fps} fps  "
        f"pelvis range (m): x={pelvis_xyz_range[0]:.2f}, "
        f"y={pelvis_xyz_range[1]:.2f}, z={pelvis_xyz_range[2]:.2f}",
        flush=True,
    )

    # Per-frame copies for the visualizer (it expects (24, 3) per frame).
    smpl_joints_list = [smpl_joints[i].copy() for i in range(n_frames)]

    # No VR 3-point in kimodo data — fill zeros so the per-frame loop is uniform.
    zero_vr = np.zeros((3, 7), dtype=np.float32)
    zero_vr[:, 3] = 1.0  # qw=1 default quaternion (unused since has_vr3pt=False)
    vr_3pt_list = [zero_vr.copy() for _ in range(n_frames)]

    return smpl_joints_list, vr_3pt_list, False, fps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Replay SMPL + VR 3-point data from episode parquet or kimodo SMPL pkl.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--parquet", type=str, default=None,
        help="Path to episode parquet file (GROOT data collection format).",
    )
    src.add_argument(
        "--smpl-pkl", type=str, default=None,
        help="Path to kimodo SMPL motion pkl, e.g. data/smpl_finetune/bw_fast_01.pkl.",
    )
    parser.add_argument(
        "--fps", type=int, default=None,
        help="Playback FPS. Default: 50 for parquet, value of 'fps' field for kimodo pkl.",
    )
    parser.add_argument(
        "--no-remove-stale", action="store_true",
        help="(parquet only) Do NOT remove zero/stale SMPL frames.",
    )
    args = parser.parse_args()

    # Default to parquet path if neither flag was given (preserves old behavior).
    if args.parquet is None and args.smpl_pkl is None:
        args.parquet = "outputs/2026-06-15-17-14-51/data/chunk-000/episode_000000.parquet"

    if args.smpl_pkl is not None:
        smpl_joints_list, vr_3pt_list, has_vr3pt, src_fps = load_smpl_pkl_data(args.smpl_pkl)
        playback_fps = args.fps if args.fps is not None else int(round(src_fps))
    else:
        smpl_joints_list, vr_3pt_list, has_vr3pt = load_episode_data(
            args.parquet, remove_stale=not args.no_remove_stale,
        )
        playback_fps = args.fps if args.fps is not None else 50

    n_frames = len(smpl_joints_list)

    # Create visualizer with SMPL skeleton enabled
    visualizer = VR3PtPoseVisualizer(
        axis_length=0.08,
        ball_radius=0.015,
        enable_smpl_vis=True,
    )
    visualizer.create_realtime_plotter(interactive=True)

    # If VR 3-point data exists, also show VR pose markers
    if has_vr3pt:
        print("[Replay] VR 3-point data found — showing VR markers", flush=True)
    else:
        print("[Replay] VR 3-point data is all zeros — only SMPL skeleton shown", flush=True)

    # Keyboard
    kq = _KeyQueue()
    threading.Thread(target=_keyboard_thread, args=(kq,), daemon=True).start()

    # Playback loop
    playing = True
    frame_idx = 0
    speed = 1.0
    period = 1.0 / playback_fps
    next_tick = time.time()
    print(f"[Replay] Playing at {playback_fps} FPS ({n_frames} frames)", flush=True)

    try:
        while visualizer.is_open:
            ch = kq.pop()
            if ch == "q":
                break
            elif ch == "h":
                print(HELP_TEXT, flush=True)
            elif ch in ("p", " "):
                playing = not playing
                next_tick = time.time()  # reset tick after pause
                print(f"[Replay] {'PLAY' if playing else 'PAUSED'} frame {frame_idx}/{n_frames}", flush=True)
            elif ch == "." and not playing and frame_idx < n_frames - 1:
                frame_idx += 1
            elif ch == "," and not playing and frame_idx > 0:
                frame_idx -= 1
            elif ch == "r":
                frame_idx = 0
                next_tick = time.time()
                visualizer.reset_smpl_anchor()
                print("[Replay] Restart", flush=True)
            elif ch in ("+", "="):
                speed = min(speed * 2, 16.0)
                next_tick = time.time()
                print(f"[Replay] Speed: {speed:.1f}x", flush=True)
            elif ch == "-":
                speed = max(speed / 2, 0.125)
                next_tick = time.time()
                print(f"[Replay] Speed: {speed:.1f}x", flush=True)

            # Update SMPL skeleton
            visualizer.update_smpl_joints(smpl_joints_list[frame_idx])

            # Update VR 3-point markers if data is available
            if has_vr3pt:
                visualizer.update_vr_poses(vr_3pt_list[frame_idx])

            visualizer.render()

            if playing:
                frame_idx += 1
                if frame_idx >= n_frames:
                    frame_idx = 0
                    visualizer.reset_smpl_anchor()

            # Wall-clock synchronized pacing (compensates for render time)
            next_tick += period / speed
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        kq.stop()
        visualizer.close()
        print("[Replay] Done.", flush=True)


if __name__ == "__main__":
    main()
