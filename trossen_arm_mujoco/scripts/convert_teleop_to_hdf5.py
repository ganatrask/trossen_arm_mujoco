# Copyright 2025 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Convert real teleop CSV recordings to HDF5 datasets by replaying in simulation.

This script:
1. Reads joint positions from teleop CSV files
2. Replays them in the MuJoCo simulation
3. Captures simulated camera images and observations
4. Saves everything as HDF5 files for training

Usage:
    # Convert a single episode
    python -m trossen_arm_mujoco.scripts.convert_teleop_to_hdf5 \
        --data_dir /path/to/dual_arm_recording_XXXXX \
        --output_dir teleop_hdf5_dataset \
        --arm right \
        --role follower

    # Convert all episodes from a root directory
    python -m trossen_arm_mujoco.scripts.convert_teleop_to_hdf5 \
        --data_root /home/shyam/projects/cc/data_from_raven \
        --output_dir teleop_hdf5_dataset \
        --arm right \
        --role follower

    # Convert only first N episodes
    python -m trossen_arm_mujoco.scripts.convert_teleop_to_hdf5 \
        --data_root /home/shyam/projects/cc/data_from_raven \
        --output_dir teleop_hdf5_dataset \
        --num_episodes 10

HDF5 structure:
    episode_X.hdf5
    ├── observations/
    │   ├── images/
    │   │   ├── cam_high  (timesteps, 480, 640, 3) uint8
    │   │   └── cam       (timesteps, 480, 640, 3) uint8
    │   ├── qpos          (timesteps, 8) float64
    │   └── qvel          (timesteps, 8) float64
    ├── action            (timesteps, 8) float64
    └── attrs: sim=True, source="teleop_replay", original_episode=<name>
"""

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from trossen_arm_mujoco.assets.food_task.single_arm_env import FoodTransferTask
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


def convert_episode(
    csv_path: str,
    output_path: str,
    original_episode_name: str,
    cam_list: list[str],
    onscreen_render: bool = False,
    realtime: bool = False,
) -> dict:
    """
    Convert a single teleop CSV episode to HDF5 by replaying in simulation.

    :param csv_path: Path to the CSV file with joint positions.
    :param output_path: Path to save the HDF5 file.
    :param original_episode_name: Name of the original episode directory.
    :param cam_list: List of camera names.
    :param onscreen_render: Whether to show viewer during replay.
    :param realtime: Whether to replay at original timing (slower but matches real dynamics).
    :return: Dictionary with episode results.
    """
    # Load trajectory (use original timesteps, no resampling)
    positions, timestamps = load_arm_data(csv_path)
    duration = timestamps[-1]
    print(f"  Loaded: {len(positions)} timesteps, {duration:.2f}s")

    # Setup simulation environment
    env = make_sim_env(
        FoodTransferTask,
        xml_file="wxai/telop_scene.xml",
        task_name="food_transfer",
        onscreen_render=False,
        cam_list=cam_list,
    )
    ts = env.reset()

    # Data collection
    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/action": [],
    }
    for cam_name in cam_list:
        data_dict[f"/observations/images/{cam_name}"] = []

    episode_rewards = []
    max_reward_possible = env.task.max_reward

    # For realtime playback, track wall-clock time
    if realtime:
        start_time = time.time()
        print(f"  Realtime mode: will take ~{duration:.1f}s")

    # Replay trajectory and collect data (use original positions directly)
    for i in tqdm(range(len(positions)), desc="  Replaying", leave=False):
        # Realtime: wait until we reach the timestamp for this frame
        if realtime and i > 0:
            target_time = timestamps[i]
            elapsed = time.time() - start_time
            if elapsed < target_time:
                time.sleep(target_time - elapsed)

        joint_pos = positions[i]

        # Convert to 8-value action
        action = np.zeros(8)
        action[:6] = joint_pos[:6]
        action[6] = joint_pos[6]
        action[7] = joint_pos[6]

        # Collect observation BEFORE taking action
        data_dict["/observations/qpos"].append(ts.observation["qpos"].copy())
        data_dict["/observations/qvel"].append(ts.observation["qvel"].copy())
        data_dict["/action"].append(action.copy())
        for cam_name in cam_list:
            data_dict[f"/observations/images/{cam_name}"].append(
                ts.observation["images"][cam_name].copy()
            )

        # Take action
        ts = env.step(action)
        reward = ts.reward if ts.reward is not None else 0
        episode_rewards.append(reward)

    # Compute results
    max_reward = max(episode_rewards) if episode_rewards else 0
    success = max_reward == max_reward_possible

    # Get dimensions
    max_timesteps = len(data_dict["/action"])
    sample_img = data_dict[f"/observations/images/{cam_list[0]}"][0]
    img_height, img_width = sample_img.shape[:2]
    qpos_dim = data_dict["/observations/qpos"][0].shape[0]
    qvel_dim = data_dict["/observations/qvel"][0].shape[0]
    action_dim = data_dict["/action"][0].shape[0]

    # Save to HDF5
    t0 = time.time()
    with h5py.File(output_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        # Attributes
        root.attrs["sim"] = True
        root.attrs["source"] = "teleop_replay"
        root.attrs["original_episode"] = original_episode_name
        root.attrs["original_duration"] = duration

        # Observations group
        obs = root.create_group("observations")

        # Images
        image = obs.create_group("images")
        for cam_name in cam_list:
            _ = image.create_dataset(
                cam_name,
                (max_timesteps, img_height, img_width, 3),
                dtype="uint8",
                chunks=(1, img_height, img_width, 3),
            )

        # Qpos and qvel
        _ = obs.create_dataset("qpos", (max_timesteps, qpos_dim))
        _ = obs.create_dataset("qvel", (max_timesteps, qvel_dim))

        # Actions
        _ = root.create_dataset("action", (max_timesteps, action_dim))

        # Write data
        for name, array in data_dict.items():
            root[name][...] = array

    save_time = time.time() - t0

    # Cleanup
    del env

    return {
        "max_reward": max_reward,
        "success": success,
        "timesteps": max_timesteps,
        "duration": duration,
        "save_time": save_time,
    }


def main(args):
    """
    Convert teleop CSV recordings to HDF5 datasets.
    """
    # Collect episode directories
    if args.data_root:
        root_path = Path(args.data_root)
        if not root_path.exists():
            print(f"Data root not found: {args.data_root}")
            return
        episode_dirs = sorted([
            d for d in root_path.iterdir()
            if d.is_dir() and d.name.startswith("dual_arm_recording")
        ])
        if not episode_dirs:
            print(f"No episode directories found in {args.data_root}")
            return
        if args.num_episodes is not None:
            episode_dirs = episode_dirs[:args.num_episodes]
    else:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"Data directory not found: {args.data_dir}")
            return
        episode_dirs = [data_dir]

    # Setup output directory
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Camera list
    cam_list = args.cam_names.split(",") if args.cam_names else ["cam_high", "cam"]

    print(f"Converting {len(episode_dirs)} episodes to HDF5")
    print(f"  Arm: {args.arm}, Role: {args.role}")
    print(f"  Cameras: {cam_list}")
    print(f"  Output: {output_dir}")
    print()

    # Process each episode
    all_results = []
    success_count = 0

    for i, episode_dir in enumerate(episode_dirs):
        print(f"Episode {i+1}/{len(episode_dirs)}: {episode_dir.name}")

        # Find CSV file
        arm_data_dir = episode_dir / "arm_data"
        csv_files = list(arm_data_dir.glob(f"{args.arm}_{args.role}_*.csv"))
        if not csv_files:
            print(f"  SKIPPED: No CSV file found for {args.arm}_{args.role}")
            all_results.append({
                "episode_name": episode_dir.name,
                "episode_idx": i,
                "error": "CSV not found",
                "success": False,
                "max_reward": 0,
            })
            continue

        csv_path = str(csv_files[0])
        output_path = os.path.join(output_dir, f"episode_{i}.hdf5")

        # Convert episode
        result = convert_episode(
            csv_path=csv_path,
            output_path=output_path,
            original_episode_name=episode_dir.name,
            cam_list=cam_list,
            onscreen_render=args.onscreen_render,
            realtime=args.realtime,
        )

        result["episode_name"] = episode_dir.name
        result["episode_idx"] = i
        all_results.append(result)

        status = "SUCCESS" if result["success"] else "FAILED"
        print(f"  {status}: reward={result['max_reward']}, "
              f"{result['timesteps']} steps, saved in {result['save_time']:.1f}s")

        if result["success"]:
            success_count += 1

    # Print summary
    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Total episodes: {len(episode_dirs)}")
    print(f"Converted: {len([r for r in all_results if 'error' not in r])}")
    print(f"Successful: {success_count} ({100 * success_count / len(episode_dirs):.1f}%)")
    print(f"Output: {output_dir}")
    print()

    # Episode results
    print("Episode Results:")
    print("-" * 70)
    for r in all_results:
        if "error" in r:
            print(f"  {r['episode_idx']:3d}. {r['episode_name']}: SKIPPED ({r['error']})")
        else:
            status = "SUCCESS" if r["success"] else "FAILED"
            print(f"  {r['episode_idx']:3d}. {r['episode_name']}: "
                  f"reward={r['max_reward']} [{status}] ({r['timesteps']} steps)")
    print("-" * 70)

    # Reward distribution
    reward_counts = {}
    for r in all_results:
        if "error" not in r:
            reward = r.get("max_reward", 0)
            reward_counts[reward] = reward_counts.get(reward, 0) + 1

    print("\nReward Distribution:")
    for reward in sorted(reward_counts.keys()):
        count = reward_counts[reward]
        label = {0: "No reach", 1: "Container only", 2: "Full success"}.get(reward, f"Reward {reward}")
        print(f"  Reward {reward} ({label}): {count} episodes")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert teleop CSV recordings to HDF5 by replaying in simulation."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Single episode directory containing arm_data folder.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory containing multiple episode folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save HDF5 files.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to convert (default: all).",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        choices=["left", "right"],
        help="Which arm to use (default: right).",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="follower",
        choices=["leader", "follower"],
        help="Which role to use (default: follower).",
    )
    parser.add_argument(
        "--cam_names",
        type=str,
        default=None,
        help="Comma-separated camera names (default: cam_high,cam).",
    )
    parser.add_argument(
        "--onscreen_render",
        action="store_true",
        help="Show viewer during replay (slower).",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Replay at original timing (slower, useful for verification).",
    )

    args = parser.parse_args()

    if not args.data_dir and not args.data_root:
        parser.error("Either --data_dir or --data_root must be provided")

    main(args)
