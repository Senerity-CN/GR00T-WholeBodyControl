#!/usr/bin/env python3
"""Visualize VR parquet body tracking data with controller orientation frames.

Displays raw parquet data in the original Y-up (Unity) coordinate system:
  - 24 SMPL joint positions as connected spheres
  - Wrist/hand coordinate frames (green/cyan balls + RGB axes)
  - Controller coordinate frames (orange/tomato balls + RGB axes)

No coordinate transforms are applied — positions and quaternions are
rendered exactly as stored in the parquet file.

Usage
-----
Basic playback::

    python scripts/visualize_vr_smpl.py \\
        --parquet outputs/Ego-Centric-Internal/session_20260528_153640/data/chunk-000/observation.data.vr_000000.parquet

Custom FPS::

    python scripts/visualize_vr_smpl.py --parquet <path> --fps 25

With controller correction (transforms 153640-style to 154545-style,
replaces wrist/hand orientations with corrected controller-derived values)::

    python scripts/visualize_vr_smpl.py --parquet <path> --correct_controller

Correction pipeline (--correct_controller):
    q_ctrl_raw  ->  * Q_R  (inter-session right-multiply correction)
                ->  * Q_CTRL2WRIST  (controller -> wrist mounting offset)
                ->  overwrite SMPL joints 20-23 (wrist + hand)

Keyboard controls (press key then ENTER)
----------------------------------------
    p / ENTER : play / pause
    .         : next frame (when paused)
    ,         : previous frame (when paused)
    r         : restart from frame 0
    + / =     : speed up (x2)
    -         : slow down (x0.5)
    q         : quit

Visual legend
-------------
    lightgreen / lime   : left wrist / left hand  (body tracking orientation)
    lightblue / cyan    : right wrist / right hand (body tracking orientation)
    orange              : left controller  (raw or corrected controller IMU)
    tomato              : right controller (raw or corrected controller IMU)
    RGB arrows          : X=red, Y=green, Z=blue local coordinate axes
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial.transform import Rotation as sRot

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import VR3PtPoseVisualizer

# ---------------------------------------------------------------------------
PARENT_INDICES = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23,
][:24]

VR_TO_SMPL_MAP = {
    "waist": 0, "left_hip": 1, "right_hip": 2, "spine1": 3,
    "left_knee": 4, "right_knee": 5, "spine2": 6, "left_ankle": 7,
    "right_ankle": 8, "spine3": 9, "left_foot": 10, "right_foot": 11,
    "neck": 12, "left_collar": 13, "right_collar": 14, "head": 15,
    "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18,
    "right_elbow": 19, "left_wrist": 20, "right_wrist": 21,
    "left_hand": 22, "right_hand": 23,
}

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
    body_poses_np: np.ndarray,
    left_ctrl_quat: np.ndarray | None,
    right_ctrl_quat: np.ndarray | None,
) -> np.ndarray:
    """Replace SMPL wrist/hand orientations using corrected controller orientations.

    Pipeline per side:
      1. correct controller:    q_ctrl' = q_ctrl * Q_R
      2. controller to wrist:   q_wrist = q_ctrl' * Q_CTRL2WRIST
      3. overwrite joints 20/22 or 21/23

    Positions remain from body tracking.
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


# Joints whose coordinate frames we want to display
FRAME_JOINTS = {
    "left_wrist": 20,
    "right_wrist": 21,
    "left_hand": 22,
    "right_hand": 23,
}
FRAME_JOINT_COLORS = {
    "left_wrist": "lightgreen",
    "right_wrist": "lightblue",
    "left_hand": "lime",
    "right_hand": "cyan",
}

# Controller frames: displayed at the corresponding hand joint position
# Maps name → SMPL joint index used for position
CONTROLLER_FRAMES = {
    "left_controller": 22,   # placed at left_hand FK position
    "right_controller": 23,  # placed at right_hand FK position
}
CONTROLLER_COLORS = {
    "left_controller": "orange",
    "right_controller": "tomato",
}


# ---------------------------------------------------------------------------
# VR parquet -> SMPL joints + VR 3-point pose
# ---------------------------------------------------------------------------
def vr_frame_to_body_poses(row: pd.Series) -> np.ndarray:
    body_poses = np.zeros((24, 7), dtype=np.float32)
    for vr_name, smpl_idx in VR_TO_SMPL_MAP.items():
        if vr_name in row.index:
            body_poses[smpl_idx] = row[vr_name]
    return body_poses


def compute_from_body_poses(body_poses_np: np.ndarray) -> np.ndarray:
    """Extract raw joint positions from body_poses_np (24,7). No coordinate transforms."""
    return body_poses_np[:, :3].copy()


# ---------------------------------------------------------------------------
# Keyboard thread
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
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize SMPL body data from VR parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--parquet", type=str,
        default="outputs/Ego-Centric-Internal/session_20260528_153640/data/chunk-000/observation.data.vr_000000.parquet",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--correct_controller", action="store_true",
        help="Apply right-multiply correction to controller quats and overwrite "
             "wrist/hand in SMPL (transforms 153640-style to 154545-style).",
    )
    args = parser.parse_args()

    # --- Load and precompute ------------------------------------------------
    print(f"[Vis] Loading: {args.parquet}", flush=True)
    df = pd.read_parquet(args.parquet)
    n_frames = len(df)

    has_left_ctrl = "left_controller" in df.columns
    has_right_ctrl = "right_controller" in df.columns

    print(f"[Vis] {n_frames} frames, loading...", flush=True)

    all_joints = []
    all_body_poses = []
    all_ctrl_quats = []
    for idx in range(n_frames):
        row = df.iloc[idx]
        body_poses_np = vr_frame_to_body_poses(row)

        l_q = row["left_controller"][3:].copy() if has_left_ctrl else None
        r_q = row["right_controller"][3:].copy() if has_right_ctrl else None

        if args.correct_controller:
            body_poses_np = apply_controller_correction(body_poses_np, l_q, r_q)
            if l_q is not None:
                l_q = (sRot.from_quat(l_q) * Q_R_LEFT).as_quat().astype(np.float32)
            if r_q is not None:
                r_q = (sRot.from_quat(r_q) * Q_R_RIGHT).as_quat().astype(np.float32)

        all_joints.append(compute_from_body_poses(body_poses_np))
        all_body_poses.append(body_poses_np)
        all_ctrl_quats.append((l_q, r_q))
        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{n_frames}", flush=True)
    print(f"[Vis] Done ({n_frames} frames)", flush=True)

    # --- Visualizer ---------------------------------------------------------
    visualizer = VR3PtPoseVisualizer(
        axis_length=0.08, ball_radius=0.015, enable_smpl_vis=True,
    )
    visualizer.create_realtime_plotter(interactive=True)

    # --- Coordinate frame actors (wrist/hand + controllers) -----------------
    import vtk as _vtk

    _frame_axis_length = 0.06
    _frame_axis_colors = ["red", "green", "blue"]
    _frame_actors: dict[str, dict] = {}

    def _create_frame_actor(name: str, ball_color: str):
        arrows = []
        arrow_transforms = []
        for color in _frame_axis_colors:
            arrow = pv.Arrow(
                start=(0, 0, 0), direction=(1, 0, 0),
                scale=_frame_axis_length,
                tip_length=0.3, tip_radius=0.15, shaft_radius=0.05,
                tip_resolution=6, shaft_resolution=6,
            )
            actor = visualizer.plotter.add_mesh(arrow, color=color, smooth_shading=True)
            t = _vtk.vtkTransform()
            actor.SetUserTransform(t)
            arrows.append(actor)
            arrow_transforms.append(t)
        ball = pv.Sphere(radius=0.012, center=(0, 0, 0), theta_resolution=8, phi_resolution=8)
        ball_actor = visualizer.plotter.add_mesh(ball, color=ball_color, smooth_shading=True)
        ball_t = _vtk.vtkTransform()
        ball_actor.SetUserTransform(ball_t)
        _frame_actors[name] = {
            "arrows": arrows, "arrow_transforms": arrow_transforms,
            "ball_actor": ball_actor, "ball_transform": ball_t,
        }

    for name in FRAME_JOINTS:
        _create_frame_actor(name, FRAME_JOINT_COLORS[name])
    for name in CONTROLLER_FRAMES:
        _create_frame_actor(name, CONTROLLER_COLORS[name])

    _x_axis = np.array([1.0, 0.0, 0.0])
    _diag_flip = np.diag([-1.0, 1.0, -1.0])

    def _set_frame_pose(name: str, vis_pos: np.ndarray, rot_matrix: np.ndarray):
        """Set position and orientation for a named frame actor."""
        actors = _frame_actors[name]
        axis_dirs = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        for j, local_dir in enumerate(axis_dirs):
            world_dir = rot_matrix @ local_dir
            v = np.cross(_x_axis, world_dir)
            c = float(np.dot(_x_axis, world_dir))
            v_norm = float(np.linalg.norm(v))
            if v_norm > 1e-6:
                s = v_norm
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                arrow_rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-9))
            elif c < 0:
                arrow_rot = _diag_flip
            else:
                arrow_rot = np.eye(3)

            mat = _vtk.vtkMatrix4x4()
            mat.Identity()
            for ri in range(3):
                for ci in range(3):
                    mat.SetElement(ri, ci, arrow_rot[ri, ci])
            mat.SetElement(0, 3, float(vis_pos[0]))
            mat.SetElement(1, 3, float(vis_pos[1]))
            mat.SetElement(2, 3, float(vis_pos[2]))
            actors["arrow_transforms"][j].SetMatrix(mat)

        actors["ball_transform"].Identity()
        actors["ball_transform"].Translate(float(vis_pos[0]), float(vis_pos[1]), float(vis_pos[2]))

    def _quat_xyzw_to_rot(quat_xyzw: np.ndarray) -> np.ndarray:
        """Convert a parquet quaternion (xyzw) directly to rotation matrix. No transforms."""
        return sRot.from_quat(quat_xyzw).as_matrix()

    def _update_frame_actors(body_poses_np: np.ndarray, smpl_joints: np.ndarray,
                             offset: np.ndarray,
                             ctrl_quats: tuple | None = None):
        """Update wrist/hand and controller coordinate frame actors."""
        for joint_name, smpl_idx in FRAME_JOINTS.items():
            vis_pos = smpl_joints[smpl_idx] + offset
            _set_frame_pose(joint_name, vis_pos, _quat_xyzw_to_rot(body_poses_np[smpl_idx, 3:]))

        if ctrl_quats is not None:
            for ctrl_name, pos_joint_idx in CONTROLLER_FRAMES.items():
                vis_pos = smpl_joints[pos_joint_idx] + offset
                q = ctrl_quats[0] if "left" in ctrl_name else ctrl_quats[1]
                if q is None:
                    continue
                _set_frame_pose(ctrl_name, vis_pos, _quat_xyzw_to_rot(q))

    # --- Keyboard -----------------------------------------------------------
    kq = _KeyQueue()
    threading.Thread(target=_keyboard_thread, args=(kq,), daemon=True).start()

    # --- Playback -----------------------------------------------------------
    playing = True
    frame_idx = 0
    speed = 1.0
    period = 1.0 / args.fps
    print(f"[Vis] Playing at {args.fps} FPS", flush=True)

    try:
        while visualizer.is_open:
            ch = kq.pop()
            if ch == "q":
                break
            elif ch == "h":
                print(HELP_TEXT, flush=True)
            elif ch in ("p", " "):
                playing = not playing
                print(f"[Vis] {'PLAY' if playing else 'PAUSED'} frame {frame_idx}/{n_frames}", flush=True)
            elif ch == "." and not playing and frame_idx < n_frames - 1:
                frame_idx += 1
            elif ch == "," and not playing and frame_idx > 0:
                frame_idx -= 1
            elif ch == "r":
                frame_idx = 0
                visualizer.reset_smpl_anchor()
                print("[Vis] Restart", flush=True)
            elif ch in ("+", "="):
                speed = min(speed * 2, 16.0)
                print(f"[Vis] Speed: {speed:.1f}x", flush=True)
            elif ch == "-":
                speed = max(speed / 2, 0.125)
                print(f"[Vis] Speed: {speed:.1f}x", flush=True)

            visualizer.update_smpl_joints(all_joints[frame_idx])
            _smpl_offset = visualizer.smpl_root_position - (
                visualizer._smpl_initial_root if visualizer._smpl_initial_root is not None
                else all_joints[frame_idx][0]
            )
            _update_frame_actors(
                all_body_poses[frame_idx], all_joints[frame_idx],
                _smpl_offset,
                ctrl_quats=all_ctrl_quats[frame_idx],
            )
            visualizer.render()

            if playing:
                frame_idx += 1
                if frame_idx >= n_frames:
                    frame_idx = 0
                    visualizer.reset_smpl_anchor()

            time.sleep(period / speed)
    except KeyboardInterrupt:
        pass
    finally:
        kq.stop()
        visualizer.close()
        print("[Vis] Done.", flush=True)


if __name__ == "__main__":
    main()
