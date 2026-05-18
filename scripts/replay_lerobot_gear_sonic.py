#!/usr/bin/env python
"""
Simple interactive LeRobot episode replay in MuJoCo.

Controls:
- SPACE: Start/pause replay
- ESC: Exit
"""

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def remap_observation_to_mujoco(observation_state):
    """
    Remap observation.state to MuJoCo joint order.
    
    observation.state 和 MuJoCo 的位置顺序相同：
      - 左腿: 0-5 (6 joints)
      - 右腿: 6-11 (6 joints)
      - 腰部: 12-14 (3 joints)
      - 左臂: 15-21 (7 joints)
      - 左手: 22-28 (7 joints)
      - 右臂: 29-35 (7 joints)
      - 右手: 36-42 (7 joints)
    
    唯一区别是手部关节内部顺序：
      - observation (joint_utils): index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2
      - MuJoCo:                    thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
    
    Args:
        observation_state: numpy array of 43 joint positions
    
    Returns:
        numpy array of 43 joint positions in MuJoCo order
    """
    if len(observation_state) < 43:
        return observation_state
    
    mujoco_joints = np.zeros(43)
    
    # 左腿、右腿、腰部、左臂：位置相同，直接复制
    mujoco_joints[0:22] = observation_state[0:22]
    
    # 左手[22:29]：位置相同，但内部关节顺序需要重映射
    # observation: [index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2]
    # MuJoCo:     [thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1]
    obs_left_hand = observation_state[22:29]
    mujoco_joints[22] = obs_left_hand[4]   # thumb_0
    mujoco_joints[23] = obs_left_hand[5]   # thumb_1
    mujoco_joints[24] = obs_left_hand[6]   # thumb_2
    mujoco_joints[25] = obs_left_hand[2]   # middle_0
    mujoco_joints[26] = obs_left_hand[3]   # middle_1
    mujoco_joints[27] = obs_left_hand[0]   # index_0
    mujoco_joints[28] = obs_left_hand[1]   # index_1
    
    # 右臂[29:36]：位置相同，直接复制
    mujoco_joints[29:36] = observation_state[29:36]
    
    # 右手[36:43]：位置相同，但内部关节顺序需要重映射
    # observation: [index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2]
    # MuJoCo:     [thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1]
    obs_right_hand = observation_state[36:43]
    mujoco_joints[36] = obs_right_hand[4]   # thumb_0
    mujoco_joints[37] = obs_right_hand[5]   # thumb_1
    mujoco_joints[38] = obs_right_hand[6]   # thumb_2
    mujoco_joints[39] = obs_right_hand[2]   # middle_0
    mujoco_joints[40] = obs_right_hand[3]   # middle_1
    mujoco_joints[41] = obs_right_hand[0]   # index_0
    mujoco_joints[42] = obs_right_hand[1]   # index_1
    
    return mujoco_joints


def main(
    parquet_path: str = "outputs/2026-05-07-16-59-58/data/chunk-000/episode_000000.parquet",
    replay_speed: float = 1.0,
    use_observation: bool = False,
):
    """
    Replay a LeRobot episode in MuJoCo with interactive controls.
    
    Args:
        parquet_path: Path to the LeRobot parquet file
        replay_speed: Speed multiplier (1.0 = real-time, 0.5 = half speed)
        use_observation: If True, use observation.state instead of action.wbc
    """
    
    # Load the parquet data
    print(f"Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} steps")
    
    # Identify action columns based on mode
    if use_observation:
        # Use observation.state for replay
        data_col = 'observation.state'
        if data_col not in df.columns:
            print(f"ERROR: Column '{data_col}' not found!")
            print(f"Available columns: {df.columns.tolist()}")
            return
        print(f"Using observation column: {data_col}")
        print("Mode: Replaying actual robot states (what the robot experienced)")
    else:
        # Use action.wbc or fallback columns
        data_col = None
        for col_name in ['action.wbc', 'action', 'action.joint_position']:
            if col_name in df.columns:
                data_col = col_name
                break
        
        if data_col is None:
            print("ERROR: No action column found!")
            print(f"Available columns: {df.columns.tolist()}")
            return
        print(f"Using action column: {data_col}")
        print("Mode: Replaying target commands (what was commanded)")
    
    # Load G1 MuJoCo model (with hand support)
    # Use the same scene as run_sim_loop.py
    model_path = Path("gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml")
    
    if not model_path.exists():
        print(f"ERROR: Could not find G1 MuJoCo model!")
        print(f"Searched: {model_path}")
        return
    
    print(f"Loading MuJoCo model from: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    # Disable gravity for pure data replay (no physics constraints)
    model.opt.gravity[0] = 0.0
    model.opt.gravity[1] = 0.0
    model.opt.gravity[2] = 0.0
    print("Gravity disabled for pure data replay")
    
    # Get data dimension from first frame
    first_data = np.array(df.iloc[0][data_col])
    data_dim = len(first_data)
    num_joints = model.nu
    
    print(f"\nModel info:")
    print(f"  - Number of actuators: {num_joints}")
    print(f"  - Data dimension: {data_dim}")
    print(f"  - Simulation timestep: {model.opt.timestep}")
    print(f"  - Number of qpos: {len(data.qpos)}")
    print(f"  - Expected joint qpos: {len(data.qpos) - 7} (after floating base)")
    
    # Initialize robot to first frame's state
    print(f"\nInitializing robot to first frame...")
    first_obs = np.array(df.iloc[0]['observation.state'])
    first_data = np.array(df.iloc[0][data_col])
    
    # Remap observation.state to MuJoCo order if using observation
    if use_observation:
        first_obs = remap_observation_to_mujoco(first_obs)
        print(f"  Remapped observation.state to MuJoCo joint order")
    
    # Set initial base position and orientation
    if hasattr(data, 'qpos') and len(data.qpos) >= 7:
        # Keep default base position (from XML: pos="0 0 0.793")
        # Only update orientation if available
        if 'observation.root_orientation' in df.columns:
            root_quat = np.array(df.iloc[0]['observation.root_orientation'])
            data.qpos[3:7] = root_quat
            print(f"  Set base orientation to first frame")
        
        # Initialize joints based on mode
        joint_offset = 7
        if use_observation:
            # For observation replay: use observation.state (remapped to MuJoCo order)
            num_model_joints = len(data.qpos) - joint_offset
            num_to_set = min(len(first_obs), num_model_joints)
            data.qpos[joint_offset:joint_offset+num_to_set] = first_obs[:num_to_set]
            print(f"  Set {num_to_set} joints from observation.state (remapped)")
        else:
            # For action replay: use action command
            num_model_joints = len(data.qpos) - joint_offset
            num_to_set = min(len(first_data), num_model_joints)
            data.qpos[joint_offset:joint_offset+num_to_set] = first_data[:num_to_set]
            print(f"  Set {num_to_set} joints from action command")
    
    # Pure kinematic replay - no physics needed
    print("Configuring for pure kinematic replay...")
    # Disable gravity for pure data replay (no physics constraints)
    model.opt.gravity[0] = 0.0
    model.opt.gravity[1] = 0.0
    model.opt.gravity[2] = 0.0
    print("Gravity disabled for pure kinematic replay")
    
    print(f"Ready! Press SPACE to start replay, ESC to exit.")
    
    # Interactive replay loop with strict time alignment
    running = False
    step_idx = 0
    
    # Calculate actual frame rate from data timestamps
    frame_dt = 0.02  # Default 50 FPS
    data_fps = 50.0
    total_duration = len(df) * frame_dt
    
    if 'timestamp' in df.columns and len(df) > 1:
        timestamps = df['timestamp'].values
        # Use median frame interval to handle potential irregularities
        frame_intervals = np.diff(timestamps.astype(float))
        frame_dt = float(np.median(frame_intervals))  # Time between frames
        data_fps = 1.0 / frame_dt if frame_dt > 0 else 50.0
        total_duration = timestamps[-1] - timestamps[0]
        print(f"\nData statistics:")
        print(f"  - Frame rate: {data_fps:.1f} FPS")
        print(f"  - Frame interval: {frame_dt:.4f}s")
        print(f"  - Total duration: {total_duration:.2f}s")
        print(f"  - Total frames: {len(df)}")
    else:
        print(f"\nUsing default frame rate: {data_fps:.1f} FPS")
    
    def key_callback(keycode):
        nonlocal running, step_idx
        if keycode == 32:  # SPACE
            running = not running
            if running:
                print(f"\n▶ Replaying... (press SPACE to pause)")
            else:
                print(f"\n⏸ Paused at step {step_idx}/{len(df)} (press SPACE to resume)")
        elif keycode == 82 or keycode == 114:  # R or r
            # Reset and restart replay
            step_idx = 0
            running = True
            # Reset to first frame
            first_data = np.array(df.iloc[0][data_col])
            if len(data.qpos) > 7:
                data.qpos[7:7+min(len(first_data), len(data.qpos)-7)] = first_data[:min(len(first_data), len(data.qpos)-7)]
            mujoco.mj_forward(model, data)
            print(f"\n↺ Restarted replay from beginning")
    
    # Launch viewer with key callback and proper scene configuration
    with mujoco.viewer.launch_passive(
        model, 
        data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False
    ) as viewer:
        print("\nMuJoCo viewer started!")
        print("Controls:")
        print("  - SPACE: Start/pause replay")
        print("  - R: Restart replay from beginning")
        print("  - ESC: Exit")
        print()
        
        try:
            # Set initial camera view similar to run_sim_loop.py
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -30
            viewer.cam.distance = 2.0
            viewer.cam.lookat = np.array([0, 0, 0.5])
            viewer.cam.trackbodyid = model.body("pelvis").id
            
            replay_start_time: float | None = None
            
            while viewer.is_running():
                current_time = time.time()
                
                # If running, replay actions with strict time alignment
                if running and step_idx < len(df):
                    # Track replay timing for real-time synchronization
                    if replay_start_time is None:
                        replay_start_time = current_time
                    
                    # Get data from dataframe (action or observation based on mode)
                    frame_data = np.array(df.iloc[step_idx][data_col])
                    
                    # Update base position and orientation
                    if 'observation.root_orientation' in df.columns:
                        root_quat = np.array(df.iloc[step_idx]['observation.root_orientation'])
                        data.qpos[3:7] = root_quat
                    
                    # Directly set joint positions (pure kinematic replay)
                    if use_observation:
                        # Remap observation.state to MuJoCo order and set joints
                        remapped_data = remap_observation_to_mujoco(frame_data)
                        num_model_joints = len(data.qpos) - 7
                        num_to_set = min(len(remapped_data), num_model_joints)
                        data.qpos[7:7+num_to_set] = remapped_data[:num_to_set]
                    else:
                        # action.wbc is already in the correct order
                        if len(frame_data) >= num_joints:
                            data.qpos[7:7+num_joints] = frame_data[:num_joints]
                    
                    # Update kinematics only (no physics simulation)
                    mujoco.mj_forward(model, data)
                    
                    # Calculate expected time for this frame
                    expected_time = replay_start_time + (step_idx * frame_dt / replay_speed)
                    actual_elapsed = current_time - replay_start_time
                    time_error = float(expected_time - current_time)
                    
                    # Sleep to maintain real-time synchronization
                    if time_error > 0.001:  # Only sleep if ahead by more than 1ms
                        time.sleep(time_error)
                    
                    step_idx += 1
                    
                    # Progress reporting
                    if step_idx % 100 == 0 or step_idx == len(df):
                        progress = 100 * step_idx / len(df)
                        elapsed_time = current_time - replay_start_time
                        print(f"  Step {step_idx}/{len(df)} ({progress:.1f}%) - Elapsed: {elapsed_time:.1f}s", end='\r')
                
                # Update viewer (this is non-blocking)
                viewer.sync()
            
            # Final summary
            if replay_start_time is not None:
                actual_duration = time.time() - replay_start_time
                print(f"\n\n✓ Replay completed!")
                print(f"  - Played {step_idx} frames")
                print(f"  - Expected duration: {total_duration:.2f}s")
                print(f"  - Actual duration: {actual_duration:.2f}s")
                print(f"  - Drift: {abs(actual_duration - total_duration):.2f}s")
            else:
                print(f"\n\n✓ Replay completed! Played {step_idx} steps.")
            
        except KeyboardInterrupt:
            print(f"\n\nReplay interrupted at step {step_idx}/{len(df)}")
    
    print("\nViewer closed.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Replay LeRobot episode in MuJoCo")
    parser.add_argument(
        "--parquet-path",
        type=str,
        default="outputs/2026-05-09-12-51-02/data/chunk-000/episode_000000.parquet",
        help="Path to the LeRobot parquet file"
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="Speed multiplier (1.0 = real-time, 0.5 = half speed)"
    )
    parser.add_argument(
        "--use-observation",
        action="store_true",
        default=False,
        help="Use observation.state instead of action.wbc for replay"
    )
    
    args = parser.parse_args()
    main(args.parquet_path, args.replay_speed, args.use_observation)
