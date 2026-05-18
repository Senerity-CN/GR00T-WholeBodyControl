#!/usr/bin/env python
"""
Replay VLA motion_token data in MuJoCo using ONNX decoder.

This script reads recorded motion_token from a LeRobot parquet dataset,
builds obs_dict from parquet observation data (NOT from MuJoCo),
and feeds it to the ONNX decoder to produce 29-DOF joint actions for visualization.

Key design: obs_dict is built entirely from parquet data to match training distribution.
MuJoCo is used ONLY for visualization.

Data transformations (matching C++ g1_deploy_onnx_ref):
- Joint positions in obs_dict: reorder to IsaacLab order, subtract default_angles
- Joint velocities in obs_dict: reorder to IsaacLab order
- Last actions in obs_dict: reverse action.wbc scaling/offset, reorder to IsaacLab
- ONNX output → MuJoCo qpos: scale, reorder IsaacLab→MuJoCo, add default_angles

Controls:
- SPACE: Start/pause replay
- R: Restart replay from beginning
- ESC: Exit
"""

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.spatial.transform import Rotation

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# =============================================================================
# Policy parameters (from policy_parameters.hpp)
# =============================================================================

# Joint reordering: isaaclab_to_mujoco[isaaclab_idx] = mujoco_idx
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int32)

# mujoco_to_isaaclab[mujoco_idx] = isaaclab_idx (inverse of above)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32)

# Default joint angles in MuJoCo/published order (radians)
DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,       # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,       # right leg
    0.0, 0.0, 0.0,                                # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,          # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,         # right arm
], dtype=np.float64)

# Motor parameters (from policy_parameters.hpp)
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
_FREQ = 10.0 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0

# Stiffness: armature * freq^2
_S5020 = _ARMATURE_5020 * _FREQ * _FREQ
_S7520_14 = _ARMATURE_7520_14 * _FREQ * _FREQ
_S7520_22 = _ARMATURE_7520_22 * _FREQ * _FREQ
_S4010 = _ARMATURE_4010 * _FREQ * _FREQ

# Damping: 2 * damping_ratio * armature * freq
_D5020 = 2.0 * _DAMPING_RATIO * _ARMATURE_5020 * _FREQ
_D7520_14 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_14 * _FREQ
_D7520_22 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_22 * _FREQ
_D4010 = 2.0 * _DAMPING_RATIO * _ARMATURE_4010 * _FREQ

# Action scale: 0.25 * effort_limit / stiffness (MuJoCo order, 29 body joints)
ACTION_SCALE = np.array([
    0.25*139.0/_S7520_22, 0.25*139.0/_S7520_22, 0.25*88.0/_S7520_14,  # left hip p/r/y
    0.25*139.0/_S7520_22, 0.25*25.0/_S5020, 0.25*25.0/_S5020,        # left knee, ankle p/r
    0.25*139.0/_S7520_22, 0.25*139.0/_S7520_22, 0.25*88.0/_S7520_14,  # right hip p/r/y
    0.25*139.0/_S7520_22, 0.25*25.0/_S5020, 0.25*25.0/_S5020,        # right knee, ankle p/r
    0.25*88.0/_S7520_14, 0.25*25.0/_S5020, 0.25*25.0/_S5020,         # waist y/r/p
    0.25*25.0/_S5020, 0.25*25.0/_S5020, 0.25*25.0/_S5020,            # left shoulder p/r/y
    0.25*25.0/_S5020, 0.25*25.0/_S5020, 0.25*5.0/_S4010,             # left elbow, wrist r/p
    0.25*5.0/_S4010,                                                   # left wrist yaw
    0.25*25.0/_S5020, 0.25*25.0/_S5020, 0.25*25.0/_S5020,            # right shoulder p/r/y
    0.25*25.0/_S5020, 0.25*25.0/_S5020, 0.25*5.0/_S4010,             # right elbow, wrist r/p
    0.25*5.0/_S4010,                                                   # right wrist yaw
], dtype=np.float64)

# PD control gains for 29 body joints (MuJoCo order, from policy_parameters.hpp)
# KP = stiffness values
KP_BODY = np.array([
    _S7520_22, _S7520_22, _S7520_14,        # left hip p/r/y
    _S7520_22, 2*_S5020, 2*_S5020,          # left knee, ankle p/r
    _S7520_22, _S7520_22, _S7520_14,        # right hip p/r/y
    _S7520_22, 2*_S5020, 2*_S5020,          # right knee, ankle p/r
    _S7520_14, 2*_S5020, 2*_S5020,          # waist y/r/p
    _S5020, _S5020, _S5020,                 # left shoulder p/r/y
    _S5020, _S5020, _S4010,                 # left elbow, wrist r/p
    _S4010,                                  # left wrist yaw
    _S5020, _S5020, _S5020,                 # right shoulder p/r/y
    _S5020, _S5020, _S4010,                 # right elbow, wrist r/p
    _S4010,                                  # right wrist yaw
], dtype=np.float64)

# KD = damping values
KD_BODY = np.array([
    _D7520_22, _D7520_22, _D7520_14,        # left hip p/r/y
    _D7520_22, 2*_D5020, 2*_D5020,          # left knee, ankle p/r
    _D7520_22, _D7520_22, _D7520_14,        # right hip p/r/y
    _D7520_22, 2*_D5020, 2*_D5020,          # right knee, ankle p/r
    _D7520_14, 2*_D5020, 2*_D5020,          # waist y/r/p
    _D5020, _D5020, _D5020,                 # left shoulder p/r/y
    _D5020, _D5020, _D4010,                 # left elbow, wrist r/p
    _D4010,                                  # left wrist yaw
    _D5020, _D5020, _D5020,                 # right shoulder p/r/y
    _D5020, _D5020, _D4010,                 # right elbow, wrist r/p
    _D4010,                                  # right wrist yaw
], dtype=np.float64)

# Hand PD gains (moderate values for holding position)
_KP_HAND = 5.0
_KD_HAND = 0.5

# Simulation parameters
SIM_DT = 0.002          # MuJoCo physics timestep (s)
CONTROL_DECIMATION = 10 # Policy runs every N sim steps → 50Hz control


def quaternion_to_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to gravity direction in body frame.
    
    Matches C++ GetGravityOrientation / quat_rotate_d(quat_conjugate_d(q), [0,0,-1]).
    """
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    gx = 2.0 * (-z * x + w * y)
    gy = -2.0 * (z * y + w * x)
    gz = 1.0 - 2.0 * (w * w + z * z)
    return np.array([gx, gy, gz], dtype=np.float32)


def published_to_internal_pos(obs_state_29: np.ndarray) -> np.ndarray:
    """Convert published joint positions to internal model format.
    
    internal_body_q[j] = obs_state[MUJOCO_TO_ISAACLAB[j]] - DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB[j]]
    """
    reordered = obs_state_29[MUJOCO_TO_ISAACLAB]
    return (reordered - DEFAULT_ANGLES[MUJOCO_TO_ISAACLAB]).astype(np.float32)


def published_to_internal_vel(vel_29: np.ndarray) -> np.ndarray:
    """Convert published joint velocities to internal model format (reorder only)."""
    return vel_29[MUJOCO_TO_ISAACLAB].astype(np.float32)


def published_to_internal_action(action_wbc_29: np.ndarray) -> np.ndarray:
    """Convert published action.wbc back to raw model output (internal last_action).
    
    internal_last_action[j] = (action.wbc[B[j]] - D[B[j]]) / scale[B[j]]
    where B = MUJOCO_TO_ISAACLAB
    """
    idx = MUJOCO_TO_ISAACLAB
    raw = (action_wbc_29[idx] - DEFAULT_ANGLES[idx]) / ACTION_SCALE[idx]
    return raw.astype(np.float32)


def onnx_output_to_mujoco_qpos(onnx_action_29: np.ndarray) -> np.ndarray:
    """Convert raw ONNX output to MuJoCo joint target positions.
    
    q_target[i] = DEFAULT_ANGLES[i] + onnx_output[ISAACLAB_TO_MUJOCO[i]] * ACTION_SCALE[i]
    """
    q_target = DEFAULT_ANGLES + onnx_action_29[ISAACLAB_TO_MUJOCO] * ACTION_SCALE
    return q_target.astype(np.float32)


def compute_pd_torques_43dof(
    q_target_body_29: np.ndarray,
    q_target_hand_14: np.ndarray,
    data: "mujoco.MjData",
) -> np.ndarray:
    """Compute PD control torques for all 43 actuators.
    
    Actuator layout (43 total):
      [0:22]  body: left_leg(6) + right_leg(6) + waist(3) + left_arm(7)
      [22:29] left_hand (7)
      [29:36] body: right_arm (7)
      [36:43] right_hand (7)
    
    PD formula: tau = kp * (q_target - q) + kd * (0 - dq)
    """
    torques = np.zeros(43, dtype=np.float64)
    
    # Body joints: target[0:22] -> actuator[0:22], target[22:29] -> actuator[29:36]
    # Current positions: qpos[7:29] for first 22, qpos[36:43] for right arm
    # Current velocities: qvel[6:28] for first 22, qvel[35:42] for right arm
    
    # First 22 body joints (legs + waist + left arm)
    q_curr_22 = data.qpos[7:29]
    dq_curr_22 = data.qvel[6:28]
    torques[0:22] = KP_BODY[:22] * (q_target_body_29[:22] - q_curr_22) + KD_BODY[:22] * (0 - dq_curr_22)
    
    # Right arm (7 body joints)
    q_curr_rarm = data.qpos[36:43]
    dq_curr_rarm = data.qvel[35:42]
    torques[29:36] = KP_BODY[22:29] * (q_target_body_29[22:29] - q_curr_rarm) + KD_BODY[22:29] * (0 - dq_curr_rarm)
    
    # Left hand (actuator 22:29, qpos 29:36, qvel 28:35)
    q_curr_lh = data.qpos[29:36]
    dq_curr_lh = data.qvel[28:35]
    torques[22:29] = _KP_HAND * (q_target_hand_14[:7] - q_curr_lh) + _KD_HAND * (0 - dq_curr_lh)
    
    # Right hand (actuator 36:43, qpos 43:50, qvel 42:49)
    q_curr_rh = data.qpos[43:50]
    dq_curr_rh = data.qvel[42:49]
    torques[36:43] = _KP_HAND * (q_target_hand_14[7:14] - q_curr_rh) + _KD_HAND * (0 - dq_curr_rh)
    
    return torques


def build_obs_dict_from_parquet(
    df: pd.DataFrame,
    step_idx: int,
    motion_token: np.ndarray,
    frame_dt: float,
) -> np.ndarray:
    """Build 994-dim obs_dict entirely from parquet data.
    
    Order matches observation_config.yaml (C++ processes in config declaration order):
      [0:64]    token_state
      [64:94]   his_base_angular_velocity_10frame_step1
      [94:384]  his_body_joint_positions_10frame_step1  (IsaacLab order, defaults subtracted)
      [384:674] his_body_joint_velocities_10frame_step1 (IsaacLab order)
      [674:964] his_last_actions_10frame_step1          (raw model output, IsaacLab order)
      [964:994] his_gravity_dir_10frame_step1
    
    Args:
        df: parquet DataFrame
        step_idx: current step index
        motion_token: 64-dim token from parquet
        frame_dt: time interval between frames
    Returns:
        994-dim obs_dict (float32)
    """
    obs_dict = np.zeros(994, dtype=np.float32)
    
    # 1. token_state [0:64]
    obs_dict[0:64] = motion_token
    
    # Build 10-frame history window (oldest to newest, matching C++ newest_first=false)
    has_root_orientation = 'observation.root_orientation' in df.columns
    
    for i in range(10):
        frame_idx = max(0, step_idx - 9 + i)
        
        # Extract 29 body joints from 43-dim observation.state
        # Layout: [legs+waist+left_arm (22) | left_hand (7) | right_arm (7) | right_hand (7)]
        obs_full = np.array(df.iloc[frame_idx]['observation.state'], dtype=np.float64)
        obs_state = np.concatenate([obs_full[:22], obs_full[29:36]])  # 29 body joints
        
        # --- Base angular velocity (3 dim per frame) [64:94] ---
        base_ang_vel = np.zeros(3, dtype=np.float32)
        if has_root_orientation and frame_idx > 0:
            quat_curr = np.array(df.iloc[frame_idx]['observation.root_orientation'], dtype=np.float64)
            quat_prev = np.array(df.iloc[frame_idx - 1]['observation.root_orientation'], dtype=np.float64)
            rot_curr = Rotation.from_quat([quat_curr[1], quat_curr[2], quat_curr[3], quat_curr[0]])
            rot_prev = Rotation.from_quat([quat_prev[1], quat_prev[2], quat_prev[3], quat_prev[0]])
            rot_diff = rot_prev.inv() * rot_curr
            base_ang_vel = (rot_diff.as_rotvec() / frame_dt).astype(np.float32)
        obs_dict[64 + i * 3: 64 + (i + 1) * 3] = base_ang_vel
        
        # --- Joint positions (29 dim per frame) [94:384] ---
        # Convert: reorder to IsaacLab order, subtract default_angles
        joint_pos_internal = published_to_internal_pos(obs_state)
        obs_dict[94 + i * 29: 94 + (i + 1) * 29] = joint_pos_internal
        
        # --- Joint velocities (29 dim per frame) [384:674] ---
        if frame_idx > 0:
            prev_full = np.array(df.iloc[frame_idx - 1]['observation.state'], dtype=np.float64)
            prev_obs = np.concatenate([prev_full[:22], prev_full[29:36]])  # 29 body joints
            vel_published = (obs_state - prev_obs) / frame_dt
            joint_vel_internal = published_to_internal_vel(vel_published)
        else:
            joint_vel_internal = np.zeros(29, dtype=np.float32)
        obs_dict[384 + i * 29: 384 + (i + 1) * 29] = joint_vel_internal
        
        # --- Last actions (29 dim per frame) [674:964] ---
        # Convert action.wbc back to raw model output format
        if frame_idx > 0:
            prev_wbc_full = np.array(df.iloc[frame_idx - 1]['action.wbc'], dtype=np.float64)
            # Extract 29 body joints (skip hands)
            prev_wbc = np.concatenate([prev_wbc_full[:22], prev_wbc_full[29:36]])
            last_action_internal = published_to_internal_action(prev_wbc)
            obs_dict[674 + i * 29: 674 + (i + 1) * 29] = last_action_internal
        # else: zeros (no previous action)
        
        # --- Gravity direction (3 dim per frame) [964:994] ---
        if has_root_orientation:
            root_quat = np.array(df.iloc[frame_idx]['observation.root_orientation'], dtype=np.float64)
            obs_dict[964 + i * 3: 964 + (i + 1) * 3] = quaternion_to_gravity(root_quat)
        else:
            obs_dict[964 + i * 3: 964 + (i + 1) * 3] = np.array([0, 0, -1], dtype=np.float32)
    
    return obs_dict


def main(
    parquet_path: str = "outputs/2026-05-09-12-51-02/data/chunk-000/episode_000000.parquet",
    onnx_path: str | None = None,
    replay_speed: float = 1.0,
    use_wbc_action: bool = True,
    use_physics: bool = False,
):
    """
    Replay VLA motion_token or WBC actions in MuJoCo.
    
    Args:
        parquet_path: Path to the LeRobot parquet file
        onnx_path: Path to the control policy ONNX model (only needed if use_wbc_action=False)
        replay_speed: Speed multiplier (1.0 = real-time, 0.5 = half speed)
        use_wbc_action: If True, use action.wbc directly; if False, decode motion_token via ONNX
        use_physics: If True, use PD control + mj_step; if False, kinematic (mj_forward)
    """
    
    # Load the parquet data
    print(f"Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} steps")
    
    # Check available action columns
    has_wbc_action = 'action.wbc' in df.columns
    has_motion_token = 'action.motion_token' in df.columns
    
    if use_wbc_action:
        if not has_wbc_action:
            print("ERROR: 'action.wbc' column not found in dataset!")
            print(f"Available columns: {df.columns.tolist()}")
            return
        print(f"\nMode: Direct WBC action replay (action.wbc)")
    else:
        if not has_motion_token:
            print("ERROR: 'action.motion_token' column not found in dataset!")
            print(f"Available columns: {df.columns.tolist()}")
            return
        if onnx_path is None:
            print("ERROR: --onnx-path is required when --use-motion-token")
            return
        print(f"\nMode: VLA motion_token decoding via ONNX")
    
    # Load ONNX model (only needed for motion_token mode)
    session = None
    input_name = None
    output_name = None
    if not use_wbc_action:
        print(f"\nLoading ONNX model from: {onnx_path}")
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape
        print(f"  Input: {input_name}, shape: {input_shape}")
        print(f"  Output: {output_name}, shape: {output_shape}")
    
    # Load G1 MuJoCo model (43DOF scene)
    model_path = Path("gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml")
    
    if not model_path.exists():
        print(f"ERROR: Could not find G1 MuJoCo model!")
        print(f"Searched: {model_path}")
        return
    
    print(f"\nLoading MuJoCo model from: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    # Disable gravity for pure kinematic replay (keep gravity for physics mode)
    if not use_physics:
        model.opt.gravity[:] = 0.0
        print("Mode: Kinematic (mj_forward, no gravity)")
    else:
        model.opt.timestep = SIM_DT
        print(f"Mode: Physics (PD control + mj_step, dt={SIM_DT}, decimation={CONTROL_DECIMATION})")
        print(f"  Gravity: {model.opt.gravity}")
    
    # Initialize robot to first frame's state
    print(f"\nInitializing robot to first frame...")
    first_obs = np.array(df.iloc[0]['observation.state'], dtype=np.float32)
    num_obs_joints = len(first_obs)  # 43 for full model (includes hands)
    num_model_joints = model.nq - 7  # 43 for scene_43dof
    
    print(f"  first_obs shape: {first_obs.shape}")
    print(f"  MuJoCo model nq: {model.nq}, body joints: {num_model_joints}")
    print(f"  Dataset columns: {list(df.columns)}")
    
    # Set initial base orientation and joint positions
    if 'observation.root_orientation' in df.columns:
        root_quat = np.array(df.iloc[0]['observation.root_orientation'])
        data.qpos[3:7] = root_quat
        print(f"  Set base orientation: {root_quat}")
    
    # Initialize joints from observation.state (43 dims for full model)
    num_to_set = min(num_obs_joints, num_model_joints)
    data.qpos[7:7 + num_to_set] = first_obs[:num_to_set]
    print(f"  Set {num_to_set} joints from observation.state")
    
    mujoco.mj_forward(model, data)
    
    # Calculate frame rate from timestamps
    frame_dt = 0.02  # Default 50 FPS
    data_fps = 50.0
    total_duration = len(df) * frame_dt
    
    if 'timestamp' in df.columns and len(df) > 1:
        timestamps = df['timestamp'].values
        frame_intervals = np.diff(timestamps.astype(float))
        frame_dt = float(np.median(frame_intervals))
        data_fps = 1.0 / frame_dt if frame_dt > 0 else 50.0
        total_duration = timestamps[-1] - timestamps[0]
        print(f"\nData statistics:")
        print(f"  - Frame rate: {data_fps:.1f} FPS")
        print(f"  - Frame interval: {frame_dt:.4f}s")
        print(f"  - Total duration: {total_duration:.2f}s")
        print(f"  - Total frames: {len(df)}")
    else:
        print(f"\nUsing default frame rate: {data_fps:.1f} FPS")
    
    print(f"\nReady! Press SPACE to start replay, ESC to exit.")
    
    # Interactive replay loop
    running = False
    step_idx = 0
    should_exit = False
    replay_start_time = None
    
    def key_callback(keycode):
        nonlocal running, step_idx, should_exit, replay_start_time
        if keycode == 32:  # SPACE
            running = not running
            if running:
                replay_start_time = None
                print(f"\n▶ Replaying... (press SPACE to pause)")
            else:
                print(f"\n⏸ Paused at step {step_idx}/{len(df)} (press SPACE to resume)")
        elif keycode == 82 or keycode == 114:  # R or r
            step_idx = 0
            running = False
            replay_start_time = None
            # Reset to first frame state
            num_to_set_rst = min(num_obs_joints, num_model_joints)
            data.qpos[7:7 + num_to_set_rst] = first_obs[:num_to_set_rst]
            if 'observation.root_orientation' in df.columns:
                root_quat = np.array(df.iloc[0]['observation.root_orientation'])
                data.qpos[3:7] = root_quat
            mujoco.mj_forward(model, data)
            print(f"\n↺ Reset to initial pose. Press SPACE to start replay.")
        elif keycode == 256:  # ESC
            should_exit = True
            print(f"\n⏹ Exiting...")
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        print("\nMuJoCo viewer started!")
        print("Controls: SPACE=Start/Pause, R=Reset, ESC=Exit\n")
        
        try:
            while viewer.is_running() and not should_exit:
                current_time = time.time()
                
                if running and step_idx < len(df):
                    if replay_start_time is None:
                        replay_start_time = current_time
                    
                    if use_wbc_action:
                        # Direct WBC action replay (43 dim: body + hands)
                        action = np.array(df.iloc[step_idx]['action.wbc'], dtype=np.float32)
                    else:
                        # Motion token decoding via ONNX
                        motion_token = np.array(df.iloc[step_idx]['action.motion_token'], dtype=np.float32)
                        
                        # Build obs_dict entirely from parquet (correct order per config)
                        obs_dict = build_obs_dict_from_parquet(df, step_idx, motion_token, frame_dt)
                        
                        # Run ONNX decoder
                        onnx_raw = session.run([output_name], {input_name: obs_dict.reshape(1, -1)})[0][0]
                        
                        # Convert ONNX output to MuJoCo joint targets
                        action = onnx_output_to_mujoco_qpos(onnx_raw)
                        
                        # DEBUG: Compare with action.wbc for first few steps
                        if step_idx < 5 and has_wbc_action:
                            wbc_action = np.array(df.iloc[step_idx]['action.wbc'], dtype=np.float32)
                            # Compare first 22 joints (legs+waist+left arm) + right arm (indices 29:36 in wbc)
                            wbc_body_29 = np.concatenate([wbc_action[:22], wbc_action[29:36]])
                            diff = np.abs(action - wbc_body_29)
                            print(f"\n[Step {step_idx}] ONNX vs WBC comparison (29 body joints):")
                            print(f"  ONNX->qpos[:5]: {action[:5]}")
                            print(f"  WBC body[:5]:   {wbc_body_29[:5]}")
                            print(f"  MaxDiff: {diff.max():.6f}  MeanDiff: {diff.mean():.6f}")
                            print(f"  Raw ONNX[:5]:   {onnx_raw[:5]}")
                    
                    # Apply action to MuJoCo
                    if use_physics:
                        # === Physics mode: PD control + mj_step ===
                        # Get 29 body joint targets and 14 hand joint targets
                        if use_wbc_action:
                            # action is 43 dim: body[0:22] + left_hand[22:29] + right_arm[29:36] + right_hand[36:43]
                            body_target_29 = np.concatenate([action[:22], action[29:36]])
                            hand_target_14 = np.concatenate([action[22:29], action[36:43]])
                        else:
                            # ONNX gives 29 body joints, no hand targets (hold current)
                            body_target_29 = action
                            hand_target_14 = np.concatenate([data.qpos[29:36], data.qpos[43:50]])
                        
                        # Run decimation steps of physics simulation
                        for _ in range(CONTROL_DECIMATION):
                            torques = compute_pd_torques_43dof(body_target_29, hand_target_14, data)
                            data.ctrl[:43] = torques
                            mujoco.mj_step(model, data)
                    else:
                        # === Kinematic mode: direct qpos assignment ===
                        if use_wbc_action:
                            # WBC action is 43 dim, maps directly to qpos[7:50]
                            num_set = min(len(action), num_model_joints)
                            data.qpos[7:7 + num_set] = action[:num_set]
                        else:
                            # ONNX output is 29 body joints (no hands)
                            data.qpos[7:29] = action[:22]   # left leg + right leg + waist + left arm
                            data.qpos[36:43] = action[22:29] # right arm
                        
                        # Update base orientation from parquet
                        if 'observation.root_orientation' in df.columns:
                            data.qpos[3:7] = np.array(df.iloc[step_idx]['observation.root_orientation'])
                        
                        mujoco.mj_forward(model, data)
                    
                    # Time synchronization
                    expected_time = replay_start_time + (step_idx * frame_dt / replay_speed)
                    time_error = expected_time - current_time
                    if time_error > 0.001:
                        time.sleep(time_error)
                    
                    step_idx += 1
                    
                    # Progress reporting
                    if step_idx % 50 == 0 or step_idx == len(df):
                        progress = 100 * step_idx / len(df)
                        elapsed = current_time - replay_start_time
                        print(f"  Step {step_idx}/{len(df)} ({progress:.1f}%) - {elapsed:.1f}s", end='\r')
                
                viewer.sync()
            
            if should_exit:
                print(f"\n\n✓ Exit requested.")
            elif replay_start_time is not None:
                actual_duration = time.time() - replay_start_time
                print(f"\n\n✓ Replay completed! {step_idx} frames in {actual_duration:.2f}s")
            
        except KeyboardInterrupt:
            print(f"\n\n⚠ Interrupted at step {step_idx}/{len(df)}")
        except Exception as e:
            print(f"\n\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("Viewer closed.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Replay VLA motion_token in MuJoCo")
    parser.add_argument("--parquet-path", type=str,
        default="outputs/2026-05-09-12-51-02/data/chunk-000/episode_000000.parquet")
    parser.add_argument("--onnx-path", type=str, default=None,
        help="Path to ONNX decoder model (required for --use-motion-token)")
    parser.add_argument("--replay-speed", type=float, default=1.0,
        help="Speed multiplier (1.0 = real-time)")
    parser.add_argument("--use-wbc-action", action="store_true", default=True,
        help="Use action.wbc directly (default)")
    parser.add_argument("--use-motion-token", action="store_true", default=False,
        help="Decode motion_token via ONNX (requires --onnx-path)")
    parser.add_argument("--physics", action="store_true", default=False,
        help="Use PD control + mj_step instead of kinematic replay")
    
    args = parser.parse_args()
    
    use_wbc = args.use_wbc_action and not args.use_motion_token
    
    if args.use_motion_token and not args.onnx_path:
        print("ERROR: --use-motion-token requires --onnx-path")
        sys.exit(1)
    
    main(args.parquet_path, args.onnx_path, args.replay_speed, use_wbc, args.physics)
