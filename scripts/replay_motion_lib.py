#!/usr/bin/env python3
"""Replay a motion_lib PKL over ZMQ (Protocol v1, joint targets).

motion_lib PKL 是**机器人空间**数据（已 retarget 完成的 G1 关节角），不是 SMPL：

    root_trans_offset  (T, 3)      pelvis 世界坐标 xyz
    root_rot           (T, 4)      pelvis 四元数 (x, y, z, w)  scipy 约定
    dof                (T, 29)     29 个身体关节角，**MuJoCo g1.xml 顺序**，弧度
    pose_aa            (T, 30, 3)  root_rot 的 axis-angle + DOF_AXIS * dof（冗余表示）
    smpl_joints        (T, 24, 3)  全零占位符，不可用
    fps                float
    hand_dof_pos       (T, 14)     可选。Dex3 双手关节角，**IsaacLab 顺序**（见下）
    hand_action_left   (T,)        可选。训练用的离散抓取动作标签，本脚本不使用
    hand_action_right  (T,)        可选。同上

因此这里走的是 ``replay_csv_joint_targets.py`` 的 Protocol v1 路径（joint targets
直接进 encoder/policy），**不是** ``replay_smpl.py`` 的 Protocol v3 SMPL 路径 ——
后者需要真实的 SMPL ``pose_aa`` (T, 72) 与 ``smpl_joints``，而 motion_lib 里没有。

motion_lib 字段与 replay_csv_joint_targets 的 36 列 CSV 一一对应::

    CSV [0:3]  pelvis pos    <->  root_trans_offset
    CSV [3:7]  quat (w,x,y,z) <->  root_rot (x,y,z,w) 换序
    CSV [7:36] 29 dof MuJoCo  <->  dof

Pipeline::

    motion_lib PKL (dof 29 MuJoCo order + root_rot)
        │  MuJoCo -> IsaacLab 重排 + lerp/slerp 升采样到 output_fps
        ▼
    ZMQ PUB tcp://*:5556  (Protocol v1)
    ├─ topic "command"        : state transitions
    ├─ topic "planner"        : PLANNER mode idle
    ├─ topic "pose"           : joint_pos + joint_vel + body_quat + frame_index
    └─ topic "manager_state"  : 2Hz heartbeat
        │
        ▼
    C++ ZMQEndpointInterface (v1 branch) -> Encoder (mode 0) -> Policy
        │
        ▼
    MuJoCo / Real G1

手部数据
--------
``left_hand_joints`` / ``right_hand_joints`` 是 pose 消息的**可选字段**，所有协议
版本（含 v1）都支持，每只手 7 个 Dex3 关节值。C++ 侧按字段名解码，接受 shape
``[7]`` 或 ``[N, 7]``、dtype ``f32``/``f64``；解码后直接
``left_hand_joint_.SetData()`` → ``dex3_hands_.setAllJointsCommand()`` →
``motor_cmd()[i]``，**元素下标即 Dex3 电机下标，中途没有任何重排**。
参见 ``zmq_endpoint_interface.hpp`` 的 "Optional Fields (all versions)"。

因此需要做一次 IsaacLab → Dex3 的重排。``hand_dof_pos`` 的 14 个通道是
IsaacLab 顺序（按运动树深度分层、每层先左手后右手）::

     0 left_hand_index_0     7 left_hand_middle_1
     1 left_hand_middle_0    8 left_hand_thumb_1
     2 left_hand_thumb_0     9 right_hand_index_1
     3 right_hand_index_0   10 right_hand_middle_1
     4 right_hand_middle_0  11 right_hand_thumb_1
     5 right_hand_thumb_0   12 left_hand_thumb_2
     6 left_hand_index_1    13 right_hand_thumb_2

该布局是拿 40 个 GRAIL pkl 共 1e4 帧逐通道对 ``g1_supplemental_info.py`` /
``dex3_hands.hpp`` 的关节限位做拟合确定的：本布局 0 违例，而 ``G1_HAND_JOINTS``
顺序、左右交错顺序、Dex3 顺序分别有 7 / 6 / 7 个通道超限。右手数值已经是 Dex3
硬件的镜像符号约定（与 ``MAX/MIN_LIMITS_RIGHT`` 完全吻合），无需再翻符号。

Dex3 每只手的电机顺序取 ``thumb_0, thumb_1, thumb_2, middle_0, middle_1,
index_0, index_1`` —— 与 ``scripts/replay_observation_state.py`` 里记录的约定
一致。**注意** ``gear_sonic_deploy/g1/g1_29dof_with_hand.xml`` 的右手 MJCF 里
index 和 middle 是反的（左手 thumb/middle/index，右手 thumb/index/middle），
而 deploy C++ 全程只用下标、没有带名字的映射可以判定谁对。若实机右手的中指 /
食指动作对调，用 ``--right_hand_finger_order index_middle`` 翻过来。

Ramp-in（按 s 时的缓入）
-----------------------
PLANNER 模式下机器人保持站立姿态（deploy 的 ``default_angles``），而 motion 的
第 0 帧通常离站立姿态很远。直接切进 POSE 会让 tracking policy 在一个控制周期内
去追一个大跳变的参考，冲击很大。因此本脚本在 motion 前面**合成一段 ramp**
（``--ramp_sec``，默认 2s，0 关闭）：

身体关节 / 根四元数 / 手部共用一条 quintic smoothstep 权重
``w(s) = 6s⁵-15s⁴+10s³``（两端速度和加速度都为 0，单调、不过冲）：

* 身体 29 关节：站立姿态 → 第 0 帧，``joint_vel`` 取 ``w`` 的解析导数。
* 根四元数：从「直立 + 与第 0 帧同 heading」slerp 到第 0 帧。
* 手部：从 open（全 0，即 ``dex3_hands_.open()`` 的姿态）插到第 0 帧。

（试过用 cubic Hermite 额外匹配第 0 帧的 ``joint_vel``，但它的末速度基函数与
ramp 时长成正比，``--ramp_sec 2`` 时过冲 1.36 rad、4s 时 2.87 rad —— ramp 越长
越糟，所以放弃。代价是交接点的参考 ``joint_vel`` 有一拍阶跃，而这个阶跃原本在
没有 ramp 时的第一帧就已经存在了。）

ramp 帧是流里的正常帧（``frame_index`` 仍从 0 开始）。根四元数的起点之所以取
第 0 帧的 heading 而不是单位四元数：deploy 在 ``current_frame_ == 0`` 时会把
**收到的第一帧**的四元数记进 ``init_ref_data_root_rot_array_``，只取它的 yaw 来
对齐机器人朝向（``UpdateHeadingState`` / ``ComputeApplyDeltaHeading``）。保持
heading 一致，ramp 才不会把整段 motion 转个方向。

``--loop`` 回绕时 ramp 会以 motion **最后一帧**为起点重建，所以循环接缝同样是缓入的。

其它注意事项
------------
* Protocol v1 的 pose 消息不发送根节点平移。``root_trans_offset`` 仅用于打印
  统计，水平位移由 policy 自行产生。
* ``joint_vel`` 只对 29 个身体关节计算；手部只发位置（Dex3 侧自带 PD + 限速）。
* 纯运动学检查（不经过 policy）请用 ``scripts/visualize_motion_lib.py``。

3-Terminal run topology
-----------------------
Terminal 1 (.venv_teleop)::

    python gear_sonic/scripts/run_sim_loop.py            # MuJoCo virtual robot

Terminal 2::

    ./deploy.sh --input-type zmq_manager sim              # C++ deploy

Terminal 3 (.venv_data_collection)::

    python scripts/replay_motion_lib.py \
        --pkl /home/balance/GRAIL_data/data/pickup_table/robot/pickup_table__alcohol_0__000.pkl

    # 列出 pkl 里的 motion key（多段 motion 的 pkl）
    python scripts/replay_motion_lib.py --pkl foo.pkl --list_keys

    # 指定 motion、循环播放、覆盖源帧率
    python scripts/replay_motion_lib.py --pkl foo.pkl --motion_key bw_look_02 --loop --fps 30

    # 缓入时间加长到 4s（冲击还是大就调这个）／完全关掉缓入
    python scripts/replay_motion_lib.py --pkl foo.pkl --ramp_sec 4.0
    python scripts/replay_motion_lib.py --pkl foo.pkl --ramp_sec 0

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
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
import zmq

# -- Make ``gear_sonic`` and sibling scripts importable. --
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_FILE.parent))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
)

# Protocol v1 building blocks are shared with the CSV replay script — same wire
# format, same joint-order mapping, same interpolation semantics.
from replay_csv_joint_targets import (  # noqa: E402
    DEFAULT_OUTPUT_FPS,
    DEFAULT_PORT,
    G1_BODY_JOINTS,
    HEADER_SIZE,
    HELP_TEXT,
    MANAGER_STATE_PERIOD_S,
    MUJOCO_TO_ISAACLAB,
    StreamMode,
    _KeyQueue,
    _keyboard_thread,
    _quat_slerp_batch,
    build_idle_planner_message,
    build_manager_state_message,
    build_v1_pose_message,
    compute_joint_velocities,
    interpolate_motion,
)


# ---------------------------------------------------------------------------
# Hand constants
# ---------------------------------------------------------------------------
G1_HAND_DOF = 14  # motion_lib hand_dof_pos width (7 per Dex3 hand)
DEX3_MOTOR_MAX = 7  # motors per Dex3 hand

# hand_dof_pos channel index for each named joint (IsaacLab order — see module
# docstring for how this layout was established).
_HAND_DOF_CHANNEL = {
    "left_index_0": 0,
    "left_middle_0": 1,
    "left_thumb_0": 2,
    "right_index_0": 3,
    "right_middle_0": 4,
    "right_thumb_0": 5,
    "left_index_1": 6,
    "left_middle_1": 7,
    "left_thumb_1": 8,
    "right_index_1": 9,
    "right_middle_1": 10,
    "right_thumb_1": 11,
    "left_thumb_2": 12,
    "right_thumb_2": 13,
}

# Dex3 motor order per hand, as two candidate finger orderings (see docstring).
_DEX3_FINGER_ORDER = {
    "middle_index": ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"],
    "index_middle": ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"],
}


def hand_dof_to_dex3_indices(side: str, finger_order: str) -> list[int]:
    """hand_dof_pos channel indices that produce Dex3 motor order for one hand.

    Args:
        side: ``"left"`` or ``"right"``.
        finger_order: key into ``_DEX3_FINGER_ORDER``.
    Returns:
        7 indices into the 14-wide ``hand_dof_pos`` vector.
    """
    return [_HAND_DOF_CHANNEL[f"{side}_{j}"] for j in _DEX3_FINGER_ORDER[finger_order]]


# ---------------------------------------------------------------------------
# Standing pose (ramp-in start) & ramp construction
# ---------------------------------------------------------------------------
DEFAULT_RAMP_SEC = 2.0

# The standing pose the deploy holds in PLANNER / IDLE mode: verbatim copy of
# ``default_angles`` in
# ``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp``.
# MuJoCo / hardware joint order — reorder with MUJOCO_TO_ISAACLAB before sending.
G1_DEFAULT_ANGLES_MUJOCO = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
        0.0, 0.0, 0.0,                          # waist yaw / roll / pitch
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
    ],
    dtype=np.float32,
)

# Dex3 "open" pose — ``dex3_hands_.open()`` commands q = 0 on every motor, so
# this is what the hands hold before the first pose message arrives.
DEX3_OPEN_POSE = np.zeros(DEX3_MOTOR_MAX, dtype=np.float32)


def heading_quat(q_wxyz: NDArray[np.float32]) -> NDArray[np.float32]:
    """Yaw-only (upright) quaternion with the same heading as ``q_wxyz``.

    Mirrors ``calc_heading_quat_d`` in ``math_utils.hpp``: heading is
    ``atan2`` of the quaternion-rotated +X axis, then rebuilt about +Z.

    This is the right ramp-in start orientation: the standing robot is upright,
    and keeping frame 0's heading means the deploy's yaw alignment (which reads
    the *first streamed frame*'s quaternion at ``current_frame_ == 0``) still
    latches onto the motion's own heading.
    """
    w, x, y, z = (float(v) for v in q_wxyz)
    # First column of the rotation matrix = R @ [1, 0, 0].
    fwd_x = 1.0 - 2.0 * (y * y + z * z)
    fwd_y = 2.0 * (x * y + w * z)
    half = 0.5 * np.arctan2(fwd_y, fwd_x)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def ramp_weights(
    n_frames: int, dt: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Quintic-smoothstep blend weights and their time derivative.

    ``w(s) = 6s^5 - 15s^4 + 10s^3`` — monotone on [0, 1] with **zero velocity and
    zero acceleration at both ends**, so the ramp neither jerks the actuators at
    the start nor overshoots the target pose.

    Sampled at ``s = i / n_frames`` for ``i = 0 .. n_frames-1``: the start pose is
    included, the end pose is not (the motion block supplies it).

    A cubic Hermite that also matched the motion's frame-0 velocity was tried and
    rejected: its end-velocity basis function scales with the ramp duration, so on
    ``pickup_table__alcohol_0__000`` it overshot the target pose by 1.36 rad at
    ``--ramp_sec 2`` and 2.87 rad at 4 — worse the gentler you ask it to be.
    The consequence of not matching it is a one-tick step in the *reference*
    ``joint_vel`` at the hand-off (0 -> the motion's own frame-0 velocity), which
    the unramped script already emitted on its very first frame anyway.

    Args:
        n_frames: number of ramp frames.
        dt: seconds per frame (1 / output_fps).
    Returns:
        ``(w, dwdt)``, both (n_frames,). ``dwdt`` is in 1/s.
    """
    s = np.arange(n_frames, dtype=np.float64) / float(n_frames)
    s2 = s * s
    s3 = s2 * s
    w = s3 * (6.0 * s2 - 15.0 * s + 10.0)
    dwdt = (30.0 * s2 * (s2 - 2.0 * s + 1.0)) / (n_frames * dt)
    return w, dwdt


# ---------------------------------------------------------------------------
# ZMQ Protocol v1 message builder (with optional hand joints)
# ---------------------------------------------------------------------------
def build_v1_pose_message_with_hands(
    joint_pos: NDArray[np.float32],
    joint_vel: NDArray[np.float32],
    body_quat: NDArray[np.float32],
    frame_index: int,
    left_hand_joints: NDArray[np.float32],
    right_hand_joints: NDArray[np.float32],
    catch_up: bool = False,
) -> bytes:
    """Protocol v1 pose message plus the optional 7-DOF Dex3 hand fields.

    Byte-for-byte identical to ``build_v1_pose_message`` for the shared fields;
    ``left_hand_joints`` / ``right_hand_joints`` are appended as f32[1, 7].
    The C++ side matches fields by name, so ordering within the header is free.
    """
    N = 1
    num_joints = joint_pos.shape[0]

    jp = joint_pos.astype(np.float32).reshape(1, num_joints)
    jv = joint_vel.astype(np.float32).reshape(1, num_joints)
    bq = body_quat.astype(np.float32).reshape(1, 4)
    fi = np.array([frame_index], dtype=np.int64)
    lh = left_hand_joints.astype(np.float32).reshape(1, DEX3_MOTOR_MAX)
    rh = right_hand_joints.astype(np.float32).reshape(1, DEX3_MOTOR_MAX)

    fields = [
        {"name": "joint_pos", "dtype": "f32", "shape": [N, num_joints]},
        {"name": "joint_vel", "dtype": "f32", "shape": [N, num_joints]},
        {"name": "body_quat_w", "dtype": "f32", "shape": [N, 4]},
        {"name": "frame_index", "dtype": "i64", "shape": [N]},
        {"name": "catch_up", "dtype": "u8", "shape": [1]},
        {"name": "left_hand_joints", "dtype": "f32", "shape": [N, DEX3_MOTOR_MAX]},
        {"name": "right_hand_joints", "dtype": "f32", "shape": [N, DEX3_MOTOR_MAX]},
    ]

    payload = b"".join([
        jp.tobytes(),
        jv.tobytes(),
        bq.tobytes(),
        fi.tobytes(),
        struct.pack("B", 1 if catch_up else 0),
        lh.tobytes(),
        rh.tobytes(),
    ])

    header = {"v": 1, "endian": "le", "count": N, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")

    return b"pose" + header_json.ljust(HEADER_SIZE, b"\x00") + payload


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def unwrap_motion_lib(
    pkl_path: str, motion_key: str | None = None
) -> tuple[str, dict[str, Any]]:
    """Load a motion_lib PKL and return ``(motion_key, entry)``.

    motion_lib PKLs are ``{motion_name: {root_trans_offset, root_rot, dof, ...}}``.
    A PKL may hold several motions; ``motion_key`` selects one (default: the
    first / only one).
    """
    data = joblib.load(pkl_path)
    if not isinstance(data, dict):
        raise ValueError(f"{pkl_path}: expected a dict at top level, got {type(data)}")

    # Already unwrapped (a bare motion entry).
    if "dof" in data and "root_rot" in data:
        return Path(pkl_path).stem, data

    keys = list(data.keys())
    if not keys:
        raise ValueError(f"{pkl_path}: empty PKL")

    if motion_key is None:
        motion_key = keys[0]
        if len(keys) > 1:
            print(
                f"[Replay] PKL holds {len(keys)} motions, using the first "
                f"('{motion_key}'); pass --motion_key to pick another, "
                f"--list_keys to list them all",
                flush=True,
            )
    elif motion_key not in data:
        raise KeyError(
            f"{pkl_path}: motion key '{motion_key}' not found. "
            f"Available: {keys[:10]}{' ...' if len(keys) > 10 else ''}"
        )

    entry = data[motion_key]
    if not isinstance(entry, dict):
        raise ValueError(
            f"{pkl_path}['{motion_key}']: expected a dict, got {type(entry)}"
        )
    return motion_key, entry


def load_motion_lib_joint_targets(
    pkl_path: str, motion_key: str | None = None, with_hands: bool = True
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32] | None, float, str]:
    """Load joint targets from a motion_lib PKL.

    Returns:
        body_joints: (N, 29) joint positions **in IsaacLab order**
                     (reordered from MuJoCo via MUJOCO_TO_ISAACLAB).
        body_quat:   (N, 4)  pelvis quaternion [w, x, y, z].
        hand_dof:    (N, 14) raw ``hand_dof_pos`` in IsaacLab order, or None when
                     absent from the PKL / disabled via ``with_hands=False``.
        fps:         source frame rate stored in the PKL.
        motion_key:  the motion actually selected.
    """
    print(f"[Replay] Loading motion_lib PKL: {pkl_path}", flush=True)
    motion_key, entry = unwrap_motion_lib(pkl_path, motion_key)
    print(f"[Replay] Motion: '{motion_key}'", flush=True)

    for required in ("dof", "root_rot"):
        if required not in entry:
            raise KeyError(
                f"{pkl_path}['{motion_key}'] missing '{required}' — "
                f"not a motion_lib PKL. Keys: {list(entry.keys())}"
            )

    dof = np.asarray(entry["dof"], dtype=np.float32)  # (T, 29) MuJoCo order
    root_rot_xyzw = np.asarray(entry["root_rot"], dtype=np.float32)  # (T, 4) xyzw
    fps = float(entry.get("fps", 30.0))
    n_frames = dof.shape[0]

    if dof.ndim != 2 or dof.shape[1] != G1_BODY_JOINTS:
        raise ValueError(
            f"Expected dof of shape (T, {G1_BODY_JOINTS}), got {dof.shape}"
        )
    if root_rot_xyzw.shape != (n_frames, 4):
        raise ValueError(
            f"Expected root_rot of shape ({n_frames}, 4), got {root_rot_xyzw.shape}"
        )

    print(
        f"[Replay]   {n_frames} frames @ {fps}Hz "
        f"(duration {(n_frames - 1) / fps:.2f}s)",
        flush=True,
    )
    if "root_trans_offset" in entry:
        rt = np.asarray(entry["root_trans_offset"], dtype=np.float32)
        print(
            f"[Replay]   Root pos range: "
            f"x=[{rt[:, 0].min():.3f}, {rt[:, 0].max():.3f}], "
            f"y=[{rt[:, 1].min():.3f}, {rt[:, 1].max():.3f}], "
            f"z=[{rt[:, 2].min():.3f}, {rt[:, 2].max():.3f}] "
            f"(NOT sent — Protocol v1 has no root translation channel)",
            flush=True,
        )
    print(
        f"[Replay]   dof range: [{dof.min():+.3f}, {dof.max():+.3f}] rad",
        flush=True,
    )

    # Hand DOFs: optional pose-message fields, streamed when present.
    hand_dof: NDArray[np.float32] | None = None
    if "hand_dof_pos" in entry:
        raw_hand = np.asarray(entry["hand_dof_pos"], dtype=np.float32)
        if raw_hand.shape != (n_frames, G1_HAND_DOF):
            print(
                f"[Replay]   WARNING: ignoring hand_dof_pos with unexpected shape "
                f"{raw_hand.shape}, expected ({n_frames}, {G1_HAND_DOF})",
                flush=True,
            )
        elif not with_hands:
            print("[Replay]   hand_dof_pos present but --no_hands given: ignored", flush=True)
        else:
            hand_dof = raw_hand
            print(
                f"[Replay]   hand_dof_pos: {G1_HAND_DOF} DOF, range "
                f"[{hand_dof.min():+.3f}, {hand_dof.max():+.3f}] rad",
                flush=True,
            )
    else:
        print("[Replay]   No hand_dof_pos in this motion — body only", flush=True)

    # hand_action_* are discrete grasp labels used by the RL env, not joint targets.
    labels = [k for k in ("hand_action_left", "hand_action_right") if k in entry]
    if labels:
        print(
            f"[Replay]   Ignoring {', '.join(labels)} (training-time grasp labels, "
            f"not joint targets)",
            flush=True,
        )

    # root_rot is scipy xyzw; the v1 pose message wants body_quat_w as wxyz.
    body_quat = root_rot_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    print(f"[Replay]   Root quat: xyzw -> wxyz (w={body_quat[0, 0]:+.4f})", flush=True)

    # Reorder MuJoCo → IsaacLab (the order the C++ deploy encoder expects).
    body_joints = dof[:, MUJOCO_TO_ISAACLAB]
    print("[Replay]   Joints reordered: MuJoCo -> IsaacLab", flush=True)

    return body_joints, body_quat, hand_dof, fps, motion_key


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a motion_lib PKL over ZMQ (Protocol v1, joint targets).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pkl",
        type=str,
        required=True,
        help="Path to a motion_lib PKL (root_trans_offset / root_rot / dof).",
    )
    parser.add_argument(
        "--motion_key",
        type=str,
        default=None,
        help="Motion name inside the PKL. Default: the first (usually only) one.",
    )
    parser.add_argument(
        "--list_keys",
        action="store_true",
        help="Print the motion keys in the PKL and exit.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="ZMQ PUB port.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Source frame rate in Hz. Default: the 'fps' field stored in the PKL.",
    )
    parser.add_argument(
        "--output_fps",
        type=int,
        default=DEFAULT_OUTPUT_FPS,
        help="Target sending rate in Hz. The motion is interpolated (lerp on joints, "
        "slerp on root quat) to this rate and joint velocities are computed via "
        "np.gradient on the upsampled sequence. Default 50Hz to match deploy MOTION_FPS.",
    )
    parser.add_argument(
        "--ctrl_fps",
        type=int,
        default=50,
        help="C++ deploy internal control rate in Hz (used to compute frame_index). "
        "Set to match the deploy's MOTION_FPS; default 50.",
    )
    parser.add_argument(
        "--no_hands",
        action="store_true",
        help="Do not stream hand joints even when the PKL has hand_dof_pos.",
    )
    parser.add_argument(
        "--right_hand_finger_order",
        choices=sorted(_DEX3_FINGER_ORDER),
        default="middle_index",
        help="Right-hand Dex3 motor order after the 3 thumb motors. Use "
        "'index_middle' if the real right hand swaps middle/index (see module docstring).",
    )
    parser.add_argument(
        "--ramp_sec",
        type=float,
        default=DEFAULT_RAMP_SEC,
        help="Ease into the motion over this many seconds instead of snapping to "
        "frame 0 (which slams the robot out of its standing pose). Ramps joints "
        "from the deploy's default standing angles, the root quaternion from "
        "upright, and the hands from open. 0 disables.",
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

    if args.list_keys:
        data = joblib.load(args.pkl)
        if isinstance(data, dict) and "dof" in data:
            print(f"{args.pkl}: bare motion entry (no motion keys)")
        else:
            print(f"{args.pkl}: {len(data)} motion(s)")
            for k, v in data.items():
                n = v["dof"].shape[0] if isinstance(v, dict) and "dof" in v else "?"
                fps = v.get("fps", "?") if isinstance(v, dict) else "?"
                print(f"  {k}  ({n} frames @ {fps}Hz)")
        return

    # --- Load data ----------------------------------------------------------
    body_joints, body_quat, hand_dof, pkl_fps, motion_key = load_motion_lib_joint_targets(
        args.pkl, args.motion_key, with_hands=not args.no_hands
    )
    input_fps = args.fps if args.fps is not None else pkl_fps
    if args.fps is not None and args.fps != pkl_fps:
        print(
            f"[Replay] Source fps overridden: PKL says {pkl_fps}Hz, using {args.fps}Hz",
            flush=True,
        )
    n_frames_src = body_joints.shape[0]
    num_body = body_joints.shape[1]

    # --- Interpolate (lerp + slerp) to output_fps ---------------------------
    # Body and hand DOFs are resampled as one array so they stay on the same
    # time grid; only the body half feeds joint_pos / joint_vel.
    if float(args.output_fps) != float(input_fps):
        stacked = body_joints if hand_dof is None else np.concatenate([body_joints, hand_dof], axis=1)
        stacked, body_quat = interpolate_motion(
            stacked, body_quat, input_fps=input_fps, output_fps=args.output_fps
        )
        body_joints = stacked[:, :G1_BODY_JOINTS]
        hand_dof = None if hand_dof is None else stacked[:, G1_BODY_JOINTS:]
        print(
            f"[Replay] Motion interpolated: {n_frames_src} frames @ {input_fps}Hz "
            f"-> {body_joints.shape[0]} frames @ {args.output_fps}Hz "
            f"(lerp on joints{'' if hand_dof is None else ' + hands'}, slerp on root quat)",
            flush=True,
        )

    n_frames = body_joints.shape[0]

    # --- Hand DOFs: IsaacLab -> Dex3 motor order ----------------------------
    left_hand: NDArray[np.float32] | None = None
    right_hand: NDArray[np.float32] | None = None
    if hand_dof is not None:
        left_idx = hand_dof_to_dex3_indices("left", "middle_index")
        right_idx = hand_dof_to_dex3_indices("right", args.right_hand_finger_order)
        left_hand = hand_dof[:, left_idx]
        right_hand = hand_dof[:, right_idx]
        print(
            f"[Replay] Hands: hand_dof_pos -> Dex3 motor order "
            f"(left channels {left_idx}, right channels {right_idx}, "
            f"right finger order '{args.right_hand_finger_order}')",
            flush=True,
        )

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

    # --- Ramp-in prefix -----------------------------------------------------
    # Entering POSE hands the robot a reference that is frame 0 of the motion,
    # which is generally nowhere near the standing pose it holds in PLANNER —
    # the tracking policy then slams towards it in one control step. Prepend
    # `n_ramp` synthesised frames that ease standing -> frame 0 so the reference
    # is continuous. The prefix is part of the streamed sequence, so
    # frame_index still starts at 0 and the deploy's yaw alignment (which reads
    # the first frame's quaternion) sees the motion's own heading.
    n_ramp = max(0, int(round(float(args.ramp_sec) * float(args.output_fps))))
    stand_isaaclab = G1_DEFAULT_ANGLES_MUJOCO[MUJOCO_TO_ISAACLAB]

    n_stream = n_ramp + n_frames
    stream_pos = np.empty((n_stream, num_body), dtype=np.float32)
    stream_vel = np.empty((n_stream, num_body), dtype=np.float32)
    stream_quat = np.empty((n_stream, 4), dtype=np.float32)
    stream_pos[n_ramp:] = body_joints
    stream_vel[n_ramp:] = joint_vel
    stream_quat[n_ramp:] = body_quat

    stream_lh: NDArray[np.float32] | None = None
    stream_rh: NDArray[np.float32] | None = None
    if left_hand is not None and right_hand is not None:
        stream_lh = np.empty((n_stream, DEX3_MOTOR_MAX), dtype=np.float32)
        stream_rh = np.empty((n_stream, DEX3_MOTOR_MAX), dtype=np.float32)
        stream_lh[n_ramp:] = left_hand
        stream_rh[n_ramp:] = right_hand

    def fill_ramp(from_end: bool) -> None:
        """(Re)build the ramp prefix rows in place.

        Args:
            from_end: ramp from the motion's *last* frame (``--loop`` wrap-around)
                instead of from the standing pose (fresh POSE entry).
        """
        if n_ramp == 0:
            return
        if from_end:
            pos_from = body_joints[-1]
            quat_from = body_quat[-1]
            lh_from = DEX3_OPEN_POSE if left_hand is None else left_hand[-1]
            rh_from = DEX3_OPEN_POSE if right_hand is None else right_hand[-1]
        else:
            pos_from = stand_isaaclab
            quat_from = heading_quat(body_quat[0])  # upright, motion's heading
            lh_from = DEX3_OPEN_POSE
            rh_from = DEX3_OPEN_POSE

        w, dwdt = ramp_weights(n_ramp, output_dt)
        wc = w[:, None]
        # Row n_ramp is motion frame 0, already copied in above.
        delta = stream_pos[n_ramp] - pos_from
        stream_pos[:n_ramp] = pos_from + wc * delta
        stream_vel[:n_ramp] = dwdt[:, None] * delta
        stream_quat[:n_ramp] = _quat_slerp_batch(
            np.tile(quat_from, (n_ramp, 1)), np.tile(body_quat[0], (n_ramp, 1)), w
        )
        if stream_lh is not None and stream_rh is not None:
            stream_lh[:n_ramp] = lh_from + wc * (stream_lh[n_ramp] - lh_from)
            stream_rh[:n_ramp] = rh_from + wc * (stream_rh[n_ramp] - rh_from)

    fill_ramp(from_end=False)
    if n_ramp > 0:
        max_step = float(np.abs(body_joints[0] - stand_isaaclab).max())
        print(
            f"[Replay] Ramp-in: {n_ramp} frames ({n_ramp / float(args.output_fps):.2f}s) "
            f"standing -> frame 0 (quintic smoothstep on joints, slerp from upright on "
            f"root quat{'' if stream_lh is None else ', open -> frame 0 on hands'}); "
            f"largest joint gap bridged {max_step:.3f} rad, "
            f"peak ramp speed {np.abs(stream_vel[:n_ramp]).max():.3f} rad/s",
            flush=True,
        )
    else:
        print(
            "[Replay] Ramp-in disabled (--ramp_sec 0): frame 0 is sent immediately",
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
        f"[Replay] Data: '{motion_key}', {n_stream} frames "
        f"({n_ramp} ramp + {n_frames} motion), {num_body} body joints"
        f"{'' if left_hand is None else f' + 2x{DEX3_MOTOR_MAX} hand joints'}, "
        f"send_FPS={args.output_fps} (source {input_fps}Hz)",
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

    print("[Replay] Using root quaternion from PKL (per-frame)", flush=True)

    def reset_pose_state(ramp_from_end: bool = False) -> None:
        """Rewind to the start of the stream and rebuild the ramp prefix.

        Args:
            ramp_from_end: True at a ``--loop`` wrap-around, where the robot is
                sitting at the motion's last frame rather than standing.
        """
        nonlocal pose_idx, global_frame_f, global_frame
        fill_ramp(from_end=ramp_from_end)
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
                if pose_idx >= n_stream:
                    if args.loop:
                        reset_pose_state(ramp_from_end=True)
                    else:
                        print("[Replay] Motion exhausted -> PLANNER", flush=True)
                        transition(StreamMode.PLANNER)

                if mode == StreamMode.POSE:
                    if stream_lh is None or stream_rh is None:
                        msg = build_v1_pose_message(
                            joint_pos=stream_pos[pose_idx],
                            joint_vel=stream_vel[pose_idx],
                            body_quat=stream_quat[pose_idx],
                            frame_index=global_frame,
                            catch_up=args.catch_up,
                        )
                    else:
                        msg = build_v1_pose_message_with_hands(
                            joint_pos=stream_pos[pose_idx],
                            joint_vel=stream_vel[pose_idx],
                            body_quat=stream_quat[pose_idx],
                            frame_index=global_frame,
                            left_hand_joints=stream_lh[pose_idx],
                            right_hand_joints=stream_rh[pose_idx],
                            catch_up=args.catch_up,
                        )
                    sock.send(msg)
                    if pose_idx == n_ramp and n_ramp > 0:
                        print("[Replay] Ramp-in done -> motion frame 0", flush=True)
                    pose_idx += 1
                    global_frame_f += frame_index_step
                    global_frame = int(round(global_frame_f))

                    # Progress
                    if pose_idx % 100 == 0:
                        elapsed_s = pose_idx / float(args.output_fps)
                        print(
                            f"[Replay] Frame {pose_idx}/{n_stream} "
                            f"({100 * pose_idx / n_stream:.1f}%, {elapsed_s:.1f}s)",
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
