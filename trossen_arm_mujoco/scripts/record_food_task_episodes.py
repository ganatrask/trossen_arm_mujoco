
"""
Record demonstration episodes for the food transfer task in simulation.

This script runs scripted policies (TeleopPolicy, TeleopPolicy2, etc.) in the
FoodTransferTask environment and saves the episodes as HDF5 files with:
- Camera images (cam_high, cam)
- Joint positions (qpos) and velocities (qvel)
- Actions (joint commands)

Usage:
    python -m trossen_arm_mujoco.scripts.record_food_task_episodes \
        --data_dir food_task_demos \
        --num_episodes 50 \
        --policy teleop

HDF5 structure:
    episode_X.hdf5
    ├── observations/
    │   ├── images/
    │   │   ├── cam_high  (max_timesteps, 480, 640, 3) uint8
    │   │   └── cam       (max_timesteps, 480, 640, 3) uint8
    │   ├── qpos          (max_timesteps, 8) float64
    │   └── qvel          (max_timesteps, 8) float64
    ├── action            (max_timesteps, 8) float64
    └── attrs: sim=True, policy=<policy_name>
"""

import argparse
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from trossen_arm_mujoco.assets.food_task.single_arm_env import FoodTransferTask
from trossen_arm_mujoco.assets.food_task.scripted_policy_single_arm import (
    TeleopPolicy,
    TeleopPolicy2,
    BowlToPlatePolicy,
)
from trossen_arm_mujoco.utils import (
    make_sim_env,
    plot_observation_images,
    set_observation_images,
)


# Policy registry
POLICY_REGISTRY = {
    "teleop": TeleopPolicy,
    "teleop_2": TeleopPolicy2,
    "bowl_to_plate": BowlToPlatePolicy,
}


def record_episode(
    env,
    policy,
    episode_len: int,
    cam_list: list[str],
    onscreen_render: bool = False,
) -> tuple[list, list, bool]:
    """
    Record a single episode using the given policy.

    :param env: The simulation environment.
    :param policy: The policy to execute.
    :param episode_len: Maximum episode length.
    :param cam_list: List of camera names.
    :param onscreen_render: Whether to show matplotlib camera views.
    :return: Tuple of (episode timesteps, actions, success).
    """
    ts = env.reset()
    episode = [ts]
    actions = []

    # Pass physics to policy for IK solving
    policy.set_physics(env.physics)

    # Setup plotting if needed
    plt_imgs = None
    if onscreen_render:
        plt_imgs = plot_observation_images(ts.observation, cam_list)
        plt.pause(0.02)
        plt.show(block=False)

    # Run episode
    for step in tqdm(range(episode_len), desc="Recording"):
        action = policy(ts)
        actions.append(action.copy())
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render and plt_imgs is not None:
            plt_imgs = set_observation_images(ts.observation, plt_imgs, cam_list)
            plt.pause(0.001)

    if onscreen_render:
        plt.close()

    # Check success
    rewards = [ts.reward for ts in episode[1:] if ts.reward is not None]
    max_reward = max(rewards) if rewards else 0
    success = max_reward == env.task.max_reward

    return episode, actions, success


def save_episode_hdf5(
    episode: list,
    actions: list,
    cam_list: list[str],
    save_path: str,
    policy_name: str,
):
    """
    Save an episode to HDF5 format.

    :param episode: List of timesteps.
    :param actions: List of actions.
    :param cam_list: List of camera names.
    :param save_path: Path to save the HDF5 file.
    :param policy_name: Name of the policy used.
    """
    # Prepare data dictionary
    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/action": [],
    }
    for cam_name in cam_list:
        data_dict[f"/observations/images/{cam_name}"] = []

    # Truncate to match actions length
    # episode has len(actions) + 1 timesteps (initial + after each action)
    max_timesteps = len(actions)

    for i in range(max_timesteps):
        ts = episode[i]  # Use observation before action was taken
        action = actions[i]

        data_dict["/observations/qpos"].append(ts.observation["qpos"])
        data_dict["/observations/qvel"].append(ts.observation["qvel"])
        data_dict["/action"].append(action)

        for cam_name in cam_list:
            data_dict[f"/observations/images/{cam_name}"].append(
                ts.observation["images"][cam_name]
            )

    # Get image dimensions from first image
    sample_img = data_dict[f"/observations/images/{cam_list[0]}"][0]
    img_height, img_width = sample_img.shape[:2]

    # Get qpos/qvel dimensions
    qpos_dim = data_dict["/observations/qpos"][0].shape[0]
    qvel_dim = data_dict["/observations/qvel"][0].shape[0]
    action_dim = data_dict["/action"][0].shape[0]

    # Save to HDF5
    t0 = time.time()
    with h5py.File(save_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        # Attributes
        root.attrs["sim"] = True
        root.attrs["policy"] = policy_name

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

    print(f"  Saved {save_path} ({time.time() - t0:.1f}s)")


def main(args):
    """
    Record demonstration episodes for the food transfer task.
    """
    # Setup paths
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.getcwd(), data_dir)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    # Get policy class
    if args.policy not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy: {args.policy}. Available: {list(POLICY_REGISTRY.keys())}")
    policy_cls = POLICY_REGISTRY[args.policy]

    # Camera list
    cam_list = args.cam_names.split(",") if args.cam_names else ["cam_high", "cam"]

    print(f"Recording {args.num_episodes} episodes")
    print(f"  Policy: {args.policy}")
    print(f"  Episode length: {args.episode_len} steps")
    print(f"  Cameras: {cam_list}")
    print(f"  Output directory: {data_dir}")
    print()

    success_count = 0
    results = []

    for episode_idx in range(args.num_episodes):
        print(f"Episode {episode_idx + 1}/{args.num_episodes}")

        # Create fresh environment and policy for each episode
        env = make_sim_env(
            FoodTransferTask,
            xml_file="wxai/teleop_scene.xml",
            task_name="food_transfer",
            onscreen_render=False,
            cam_list=cam_list,
        )

        policy = policy_cls(inject_noise=args.inject_noise)

        # Record episode
        episode, actions, success = record_episode(
            env=env,
            policy=policy,
            episode_len=args.episode_len,
            cam_list=cam_list,
            onscreen_render=args.onscreen_render,
        )

        # Log result
        status = "SUCCESS" if success else "FAILED"
        rewards = [ts.reward for ts in episode[1:] if ts.reward is not None]
        max_reward = max(rewards) if rewards else 0
        print(f"  Episode {episode_idx}: {status} (max_reward={max_reward})")

        if success:
            success_count += 1

        results.append({
            "episode_idx": episode_idx,
            "success": success,
            "max_reward": max_reward,
        })

        # Save episode
        save_path = os.path.join(data_dir, f"episode_{episode_idx}.hdf5")
        save_episode_hdf5(
            episode=episode,
            actions=actions,
            cam_list=cam_list,
            save_path=save_path,
            policy_name=args.policy,
        )

        # Cleanup
        del env
        del policy
        del episode
        del actions

    # Print summary
    print()
    print("=" * 60)
    print("RECORDING COMPLETE")
    print("=" * 60)
    print(f"Total episodes: {args.num_episodes}")
    print(f"Successful: {success_count} ({100 * success_count / args.num_episodes:.1f}%)")
    print(f"Saved to: {data_dir}")
    print()

    # List results
    print("Episode Results:")
    for r in results:
        status = "SUCCESS" if r["success"] else "FAILED"
        print(f"  {r['episode_idx']:3d}: reward={r['max_reward']} [{status}]")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record food transfer task episodes in simulation."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory to save episode HDF5 files.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of episodes to record (default: 50).",
    )
    parser.add_argument(
        "--episode_len",
        type=int,
        default=1100,
        help="Episode length in timesteps (default: 1100 = 22s at 50Hz).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="teleop",
        choices=list(POLICY_REGISTRY.keys()),
        help="Policy to use for recording (default: teleop).",
    )
    parser.add_argument(
        "--inject_noise",
        action="store_true",
        help="Inject noise into actions for robustness.",
    )
    parser.add_argument(
        "--onscreen_render",
        action="store_true",
        help="Show camera views during recording (slower).",
    )
    parser.add_argument(
        "--cam_names",
        type=str,
        default=None,
        help="Comma-separated camera names (default: cam_high,cam).",
    )

    args = parser.parse_args()
    main(args)
