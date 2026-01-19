import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mujoco import viewer as mj_viewer

from trossen_arm_mujoco.assets.food_task.single_arm_env import FoodTransferTask
from trossen_arm_mujoco.utils import make_sim_env

# Evaluation target positions (from telop_scene.xml)
CONTAINER_POS = np.array([-0.63, -0.15, 0.04])
RAMEKIN_POSITIONS = {
    "ramekin_1": np.array([-0.22, -0.26, 0.04]),
    "ramekin_2": np.array([-0.36, -0.26, 0.04]),
    "ramekin_3": np.array([-0.36, -0.12, 0.04]),
    "ramekin_4": np.array([-0.22, -0.12, 0.04]),
}
REACH_THRESHOLD = 0.06  # 6cm threshold for "reached"
DWELL_TIME = 2.0  # seconds the spoon must stay near target to count as "reached"


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
) -> dict:
    """
    Replay recorded joint positions in MuJoCo simulation with real-time playback.

    :param data_dir: Directory containing the arm_data folder.
    :param arm: Which arm to use ('left' or 'right').
    :param role: Which role to replay ('leader' or 'follower').
    :param speed: Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed).
    :param cam_list: List of cameras for observation.
    :return: Dictionary with episode results including max_reward and success status.
    """
    if cam_list is None:
        cam_list = ["cam_high", "cam"]

    # Find the CSV file
    arm_data_dir = os.path.join(data_dir, "arm_data")
    csv_files = list(Path(arm_data_dir).glob(f"{arm}_{role}_*.csv"))
    if not csv_files:
        print(f"No CSV file found for {arm}_{role} in {arm_data_dir}")
        return {"max_reward": 0, "success": False, "error": "CSV not found"}
    csv_path = str(csv_files[0])
    print(f"Loading data from: {csv_path}")

    # Load joint positions and timestamps
    positions, timestamps = load_arm_data(csv_path)
    duration = timestamps[-1]
    print(f"Loaded {len(positions)} timesteps")
    print(f"Recording duration: {duration:.2f} seconds")

    # Setup simulation environment with FoodTransferTask for reward tracking
    env = make_sim_env(
        FoodTransferTask,
        xml_file="wxai/telop_scene.xml",
        task_name="food_transfer",
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

    # Find the end effector body name
    # Note: "spoon" body is the actual spoon, "link_6" is the wrist
    ee_body_name = "spoon"

    # Evaluation tracking
    eval_results = {
        "reached_container": False,
        "container_reach_time": None,
        "reached_ramekins": set(),
        "ramekin_reach_times": {},
        "min_container_dist": float("inf"),
        "min_ramekin_dists": {name: float("inf") for name in RAMEKIN_POSITIONS.keys()},
    }

    # Dwell time tracking (when did we first enter the threshold zone)
    dwell_tracking = {
        "container_enter_time": None,
        "ramekin_enter_times": {name: None for name in RAMEKIN_POSITIONS.keys()},
    }

    # Print actual object positions from simulation
    print("\n[DEBUG] Actual object positions in simulation:")
    print(f"  Container: {env.physics.named.data.xpos['container']}")
    for name in RAMEKIN_POSITIONS.keys():
        print(f"  {name}: {env.physics.named.data.xpos[name]}")

    print(f"\nWill log end effector pose at t = {log_at_seconds} seconds\n")

    # Reward tracking
    episode_rewards = []
    max_reward_possible = env.task.max_reward  # Should be 2 for FoodTransferTask

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

                ts = env.step(action)
                # ts.reward can be None in some cases, default to 0
                reward = ts.reward if ts.reward is not None else 0
                episode_rewards.append(reward)

                # Evaluation: check if spoon reached container or ramekins
                ee_pos = env.physics.named.data.xpos[ee_body_name]

                # Get actual container position from simulation
                container_actual_pos = env.physics.named.data.xpos["container"]

                # Check container reach (using XY distance to actual position)
                container_dist = np.linalg.norm(ee_pos[:2] - container_actual_pos[:2])
                eval_results["min_container_dist"] = min(eval_results["min_container_dist"], container_dist)

                if container_dist < REACH_THRESHOLD:
                    # Inside threshold zone
                    if dwell_tracking["container_enter_time"] is None:
                        dwell_tracking["container_enter_time"] = elapsed
                    # Check if we've dwelled long enough
                    dwell_duration = elapsed - dwell_tracking["container_enter_time"]
                    if dwell_duration >= DWELL_TIME and not eval_results["reached_container"]:
                        eval_results["reached_container"] = True
                        eval_results["container_reach_time"] = dwell_tracking["container_enter_time"]
                        print(f"[EVAL] Reached CONTAINER at t={elapsed:.2f}s (dwelled {dwell_duration:.1f}s)")
                else:
                    # Left threshold zone, reset dwell timer
                    dwell_tracking["container_enter_time"] = None

                # Check ramekin reach (using actual positions from simulation)
                for ramekin_name in RAMEKIN_POSITIONS.keys():
                    ramekin_actual_pos = env.physics.named.data.xpos[ramekin_name]
                    ramekin_dist = np.linalg.norm(ee_pos[:2] - ramekin_actual_pos[:2])
                    eval_results["min_ramekin_dists"][ramekin_name] = min(
                        eval_results["min_ramekin_dists"][ramekin_name], ramekin_dist
                    )

                    if ramekin_dist < REACH_THRESHOLD:
                        # Inside threshold zone
                        if dwell_tracking["ramekin_enter_times"][ramekin_name] is None:
                            dwell_tracking["ramekin_enter_times"][ramekin_name] = elapsed
                        # Check if we've dwelled long enough
                        dwell_duration = elapsed - dwell_tracking["ramekin_enter_times"][ramekin_name]
                        if dwell_duration >= DWELL_TIME and ramekin_name not in eval_results["reached_ramekins"]:
                            eval_results["reached_ramekins"].add(ramekin_name)
                            eval_results["ramekin_reach_times"][ramekin_name] = dwell_tracking["ramekin_enter_times"][ramekin_name]
                            print(f"[EVAL] Reached {ramekin_name.upper()} at t={elapsed:.2f}s (dwelled {dwell_duration:.1f}s)")
                    else:
                        # Left threshold zone, reset dwell timer
                        dwell_tracking["ramekin_enter_times"][ramekin_name] = None

                # Log end effector position at specified times
                for t in log_at_seconds:
                    if t not in logged_times and elapsed >= t:
                        print(f"t={t:2d}s: Spoon XYZ = [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
                        logged_times.add(t)

            else:
                # Replay finished, compute final results
                max_reward = max(episode_rewards) if episode_rewards else 0
                success = max_reward == max_reward_possible

                # Print evaluation summary
                print(f"\nReplay complete! Total time: {duration:.2f}s")
                print("\n" + "=" * 40)
                print("EVALUATION RESULTS")
                print("=" * 40)
                print(f"Max Reward: {max_reward}/{max_reward_possible} {'✓ SUCCESS' if success else '✗ FAILED'}")

                if eval_results["reached_container"]:
                    print(f"Container: REACHED at t={eval_results['container_reach_time']:.2f}s")
                else:
                    print("Container: NOT REACHED")

                if eval_results["reached_ramekins"]:
                    print(f"Ramekins reached: {', '.join(sorted(eval_results['reached_ramekins']))}")
                    for name, t in sorted(eval_results["ramekin_reach_times"].items()):
                        print(f"  - {name}: t={t:.2f}s")
                else:
                    print("Ramekins: NONE REACHED")

                print("\n--- Minimum distances (threshold={:.2f}m) ---".format(REACH_THRESHOLD))
                print(f"  Container: {eval_results['min_container_dist']:.3f}m")
                for name, dist in sorted(eval_results["min_ramekin_dists"].items()):
                    status = "REACHED" if name in eval_results["reached_ramekins"] else "missed"
                    print(f"  {name}: {dist:.3f}m ({status})")
                print("=" * 40)

                return {
                    "max_reward": max_reward,
                    "success": success,
                    "reached_container": eval_results["reached_container"],
                    "reached_ramekins": list(eval_results["reached_ramekins"]),
                }

            viewer.sync()

    # Fallback return if viewer closed early
    max_reward = max(episode_rewards) if episode_rewards else 0
    return {
        "max_reward": max_reward,
        "success": max_reward == max_reward_possible,
        "reached_container": eval_results["reached_container"],
        "reached_ramekins": list(eval_results["reached_ramekins"]),
    }


def main(args):
    # Collect all episode directories
    if args.data_root:
        # Find all episode directories in the root
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
        # Limit to num_episodes if specified
        if args.num_episodes is not None:
            episode_dirs = episode_dirs[:args.num_episodes]
        print(f"Found {len(episode_dirs)} episodes to replay")
    else:
        # Single episode mode
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"Data directory not found: {args.data_dir}")
            return
        episode_dirs = [data_dir]

    # Track results for all episodes
    all_results = []
    total_episodes = len(episode_dirs)

    # Replay each episode
    for i, episode_dir in enumerate(episode_dirs):
        print(f"\n{'='*60}")
        print(f"EPISODE {i+1}/{total_episodes}: {episode_dir.name}")
        print(f"{'='*60}\n")

        result = replay_episode_sim(
            data_dir=str(episode_dir),
            arm=args.arm,
            role=args.role,
            speed=args.speed,
        )
        result["episode_name"] = episode_dir.name
        result["episode_idx"] = i + 1
        all_results.append(result)

        if i < total_episodes - 1:
            print(f"\nStarting next episode in 2 seconds...")
            time.sleep(2)

    # Print final summary
    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    successful = [r for r in all_results if r.get("success", False)]

    print(f"Replayed: {len(all_results)}/{total_episodes} episodes")
    print(f"Success:  {len(successful)}/{len(all_results)} ({100*len(successful)/len(all_results):.1f}%)")
    print()

    # List episodes with their rewards
    print("Episode Results:")
    print("-" * 70)
    for r in all_results:
        status = "SUCCESS" if r.get("success") else "FAILED"
        reward = r.get("max_reward", 0)
        container = "container" if r.get("reached_container") else ""
        ramekins = ", ".join(r.get("reached_ramekins", []))
        reached = ", ".join(filter(None, [container, ramekins])) or "none"
        print(f"  {r['episode_idx']:2d}. {r['episode_name']}: reward={reward} [{status}] (reached: {reached})")

    print("-" * 70)

    # Summary by reward level
    reward_counts = {}
    for r in all_results:
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
        description="Replay CSV episode data in MuJoCo simulation."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing arm_data folder with CSV files (single episode).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory containing multiple episode folders. Replays all episodes.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to replay (default: all). Only used with --data_root.",
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

    if not args.data_dir and not args.data_root:
        parser.error("Either --data_dir or --data_root must be provided")

    main(args)