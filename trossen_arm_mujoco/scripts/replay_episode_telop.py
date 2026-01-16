import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mujoco import viewer as mj_viewer

from trossen_arm_mujoco.assets.food_task.single_arm_env import SingleArmTask
from trossen_arm_mujoco.utils import make_sim_env


def load_arm_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load joint positions and timestamps from a CSV file.

    :param csv_path: Path to the CSV file.
    :return: Tuple of (positions array, timestamps array in seconds).
    """
    df = pd.read_csv(csv_path)
    position_cols = [f"position_{i}" for i in range(7)]
    positions = df[position_cols].to_numpy()

    # Convert timestamps from nanoseconds to seconds (relative to start)
    timestamps_ns = df["timestamp"].to_numpy()
    timestamps = (timestamps_ns - timestamps_ns[0]) / 1e9

    return positions, timestamps


def replay_episode_sim(
    data_dir: str,
    arm: str = "right",
    role: str = "follower",
    speed: float = 1.0,
    cam_list: list[str] = None,
):
    """
    Replay recorded joint positions in MuJoCo simulation with real-time playback.

    :param data_dir: Directory containing the arm_data folder.
    :param arm: Which arm to use ('left' or 'right').
    :param role: Which role to replay ('leader' or 'follower').
    :param speed: Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed).
    :param cam_list: List of cameras for observation.
    """
    if cam_list is None:
        cam_list = ["cam_high", "cam"]

    # Find the CSV file
    arm_data_dir = os.path.join(data_dir, "arm_data")
    csv_files = list(Path(arm_data_dir).glob(f"{arm}_{role}_*.csv"))
    if not csv_files:
        print(f"No CSV file found for {arm}_{role} in {arm_data_dir}")
        return
    csv_path = str(csv_files[0])
    print(f"Loading data from: {csv_path}")

    # Load joint positions and timestamps
    positions, timestamps = load_arm_data(csv_path)
    duration = timestamps[-1]
    print(f"Loaded {len(positions)} timesteps")
    print(f"Recording duration: {duration:.2f} seconds")

    # Setup simulation environment
    env = make_sim_env(
        SingleArmTask,
        xml_file="wxai/telop_scene.xml",
        task_name="single_arm",
        onscreen_render=True,
        cam_list=cam_list,
    )
    env.reset()

    print(f"Replaying {arm} {role} arm in simulation...")
    print(f"Playback speed: {speed}x (will take {duration/speed:.2f} seconds)")
    print(f"Press Ctrl+C to stop")

    frame_idx = 0
    start_time = time.time()
    logged_times = set()
    log_at_seconds = [5, 10, 15, 20]

    # Find the end effector site/body name (spoon tip or gripper)
    ee_site_name = "spoon_tip"  # or use "link_6" for the wrist

    print(f"\nWill log end effector pose at t = {log_at_seconds} seconds\n")

    with mj_viewer.launch_passive(env.physics.model.ptr, env.physics.data.ptr) as viewer:
        while viewer.is_running():
            # Calculate elapsed time adjusted for playback speed
            elapsed = (time.time() - start_time) * speed

            if elapsed <= duration:
                # Find the frame corresponding to current time
                while frame_idx < len(timestamps) - 1 and timestamps[frame_idx + 1] <= elapsed:
                    frame_idx += 1

                # Get joint positions (7 values: 6 arm + 1 gripper)
                joint_pos = positions[frame_idx]

                # Convert to 8-value action (duplicate gripper for both gripper joints)
                # [0:6] = arm joints, [6:8] = gripper joints (coupled)
                action = np.zeros(8)
                action[:6] = joint_pos[:6]  # arm joints
                action[6] = joint_pos[6]     # gripper joint 1
                action[7] = joint_pos[6]     # gripper joint 2 (same as joint 1)

                env.step(action)

                # Log end effector position at specified times
                for t in log_at_seconds:
                    if t not in logged_times and elapsed >= t:
                        try:
                            ee_pos = env.physics.named.data.site_xpos[ee_site_name]
                            print(f"t={t:2d}s: End effector XYZ = [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
                        except KeyError:
                            # Fallback to link_6 body position
                            ee_pos = env.physics.named.data.xpos["link_6"]
                            print(f"t={t:2d}s: End effector XYZ = [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
                        logged_times.add(t)

            else:
                # Replay finished, close viewer
                print(f"\nReplay complete! Total time: {duration:.2f}s")
                break

            viewer.sync()


def main(args):
    data_dir = args.data_dir
    if not Path(data_dir).exists():
        print(f"Data directory not found: {data_dir}")
        return

    replay_episode_sim(
        data_dir=data_dir,
        arm=args.arm,
        role=args.role,
        speed=args.speed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay CSV episode data in MuJoCo simulation."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing arm_data folder with CSV files.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        choices=["left", "right"],
        help="Which arm to replay (default: right).",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="follower",
        choices=["leader", "follower"],
        help="Which role to replay (default: follower).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0 = real-time).",
    )

    args = parser.parse_args()
    main(args)
