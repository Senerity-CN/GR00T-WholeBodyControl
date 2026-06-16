#!/usr/bin/env python3
"""Replay motion_lib PKL data in MuJoCo viewer for visual verification.

Loads a motion_lib PKL file (containing root_trans_offset, root_rot, dof)
and plays it back on the G1 29-DOF MuJoCo model with interactive controls.

Usage:
    # Play finetune motion
    .venv_teleop/bin/python scripts/visualize_motion_lib.py \
        data/motion_lib_mixed/finetune/bw_fast_01.pkl

    # Play bones-seed motion
    .venv_teleop/bin/python scripts/visualize_motion_lib.py \
        data/motion_lib_mixed/original/210531/jump_and_land_heavy_001__A001_M.pkl

    # Side-by-side comparison (two windows)
    .venv_teleop/bin/python scripts/visualize_motion_lib.py \
        data/motion_lib_mixed/finetune/bw_fast_01.pkl \
        data/motion_lib_mixed/original/210531/jump_and_land_heavy_001__A001_M.pkl

Keyboard controls (in the MuJoCo viewer window):
    SPACE   : play / pause
    RIGHT   : next frame (when paused)
    LEFT    : previous frame (when paused)
    R       : restart from frame 0
    BACKSPACE: toggle speed (1x / 0.5x / 0.25x)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import mujoco  # type: ignore[import]
import mujoco.viewer  # type: ignore[import]
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
G1_XML = str(_REPO_ROOT / "gear_sonic" / "data" / "robots" / "g1" / "g1_29dof.xml")


def load_motion_lib(pkl_path: str) -> dict[str, Any]:
    """Load a motion_lib PKL and return the inner dict."""
    data = joblib.load(pkl_path)
    # motion_lib PKLs have a single top-level key wrapping the actual data
    if isinstance(data, dict) and len(data) == 1:
        return list(data.values())[0]
    return data


def play_motion(pkl_path: str, speed: float = 1.0):
    """Play a single motion_lib PKL in MuJoCo viewer."""
    motion = load_motion_lib(pkl_path)

    root_trans = motion["root_trans_offset"]  # (T, 3) xyz position
    root_rot = motion["root_rot"]  # (T, 4) quaternion xyzw
    dof = motion["dof"]  # (T, 29) joint angles
    fps = motion.get("fps", 30)
    T = root_trans.shape[0]

    name = Path(pkl_path).stem
    print(f"Playing: {name}")
    print(f"  Frames: {T}, FPS: {fps}, Duration: {(T-1)/fps:.2f}s")
    print(f"  root_trans range: x=[{root_trans[:,0].min():.3f}, {root_trans[:,0].max():.3f}], "
          f"y=[{root_trans[:,1].min():.3f}, {root_trans[:,1].max():.3f}], "
          f"z=[{root_trans[:,2].min():.3f}, {root_trans[:,2].max():.3f}]")
    print(f"  dof range: [{dof.min():.3f}, {dof.max():.3f}]")
    print()
    print("Controls: SPACE=play/pause, LEFT/RIGHT=step, R=restart, BACKSPACE=speed")
    print()

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)

    # State
    frame_idx = 0
    playing = True
    speed_mult = speed
    last_frame_time = time.time()

    def set_pose(idx: int):
        """Set robot pose for frame idx."""
        idx = max(0, min(idx, T - 1))
        # Position (x, y, z)
        data.qpos[0:3] = root_trans[idx]
        # Quaternion: convert xyzw -> wxyz for MuJoCo
        q_xyzw = root_rot[idx]
        data.qpos[3:7] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        # Joint angles
        data.qpos[7:36] = dof[idx]
        # Forward kinematics (no simulation)
        mujoco.mj_forward(model, data)

    def key_callback(keycode):
        nonlocal frame_idx, playing, speed_mult
        # SPACE = 32
        if keycode == 32:
            playing = not playing
        # RIGHT = 262
        elif keycode == 262:
            if not playing:
                frame_idx = min(frame_idx + 1, T - 1)
                set_pose(frame_idx)
        # LEFT = 263
        elif keycode == 263:
            if not playing:
                frame_idx = max(frame_idx - 1, 0)
                set_pose(frame_idx)
        # R = 82
        elif keycode == 82:
            frame_idx = 0
            set_pose(frame_idx)
        # BACKSPACE = 259
        elif keycode == 259:
            if speed_mult >= 1.0:
                speed_mult = 0.5
            elif speed_mult >= 0.5:
                speed_mult = 0.25
            else:
                speed_mult = 1.0
            print(f"  Speed: {speed_mult}x")

    # Set initial pose
    set_pose(0)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Set camera
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135
        viewer.cam.lookat[:] = root_trans[0]

        while viewer.is_running():
            if playing:
                now = time.time()
                dt = 1.0 / (fps * speed_mult)
                if now - last_frame_time >= dt:
                    frame_idx += 1
                    if frame_idx >= T:
                        frame_idx = 0  # loop
                    set_pose(frame_idx)
                    last_frame_time = now
                    # Update camera to follow robot
                    viewer.cam.lookat[:] = data.qpos[0:3]

            viewer.sync()
            time.sleep(0.001)


def main():
    parser = argparse.ArgumentParser(
        description="Replay motion_lib PKL in MuJoCo viewer"
    )
    parser.add_argument(
        "pkl_files", nargs="+",
        help="One or more motion_lib PKL files to visualize"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0)"
    )
    args = parser.parse_args()

    if len(args.pkl_files) == 1:
        play_motion(args.pkl_files[0], speed=args.speed)
    else:
        # For multiple files, play sequentially
        for pkl_path in args.pkl_files:
            print(f"\n{'='*60}")
            print(f"  {Path(pkl_path).stem}")
            print(f"{'='*60}")
            play_motion(pkl_path, speed=args.speed)


if __name__ == "__main__":
    main()
