import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mujoco import viewer as mj_viewer

from trossen_arm_mujoco.assets.food_task.single_arm_env import SingleArmTask
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
                # Replay finished, print evaluation summary
                print(f"\nReplay complete! Total time: {duration:.2f}s")
                print("\n" + "=" * 40)
                print("EVALUATION RESULTS")
                print("=" * 40)
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