#!/usr/bin/env python3
"""
Record food transfer IK demonstrations to HDF5 for training.

This script runs the food transfer IK task and records:
- Camera images (main_view overhead, cam wrist)
- Joint positions (qpos) and velocities (qvel)
- Actions (joint commands)

Usage:
    # Record 10 episodes cycling through all bowls
    python -m trossen_arm_mujoco.scripts.record_food_transfer_ik \
        --output_dir food_transfer_ik_dataset \
        --num_episodes 10

    # Record 5 episodes targeting specific bowl
    python -m trossen_arm_mujoco.scripts.record_food_transfer_ik \
        --output_dir food_transfer_ik_dataset \
        --num_episodes 5 \
        --target bowl_2

    # Record with noise injection for robustness
    python -m trossen_arm_mujoco.scripts.record_food_transfer_ik \
        --output_dir food_transfer_ik_dataset \
        --num_episodes 50 \
        --inject_noise

    # Record with domain randomization (pose variation + variable bowl count)
    python -m trossen_arm_mujoco.scripts.record_food_transfer_ik \
        --output_dir dr_dataset \
        --num_episodes 100 \
        --enable_dr \
        --dr_position_noise 0.03 \
        --dr_rotation_noise 0.1 \
        --dr_min_bowls 1 \
        --dr_max_bowls 8 \
        --dr_seed 42

    # Continue existing DR dataset (auto-increment seed)
    python -m trossen_arm_mujoco.scripts.record_food_transfer_ik \
        --output_dir dr_dataset \
        --num_episodes 50 \
        --enable_dr \
        --continue_dataset

    # Check dataset status
    python -m trossen_arm_mujoco.scripts.record_food_transfer_ik \
        --output_dir dr_dataset \
        --status

HDF5 structure:
    episode_X.hdf5
    ├── observations/
    │   ├── images/
    │   │   ├── main_view  (timesteps, 480, 640, 3) uint8 - overhead view
    │   │   └── cam        (timesteps, 480, 640, 3) uint8 - wrist camera
    │   ├── qpos           (timesteps, 8) float64
    │   └── qvel           (timesteps, 8) float64
    ├── action             (timesteps, 8) float64
    ├── success            bool - True if reward reached max (2/2)
    ├── env_state/         (dynamic - includes all bowls found in scene)
    │   ├── source_container  (7,) float64 - [x,y,z,qw,qx,qy,qz] container pose
    │   ├── target_container  (7,) float64 - [x,y,z,qw,qx,qy,qz] target bowl pose
    │   ├── bowl_1            (7,) float64 - [x,y,z,qw,qx,qy,qz] (if exists)
    │   ├── bowl_2            (7,) float64 - [x,y,z,qw,qx,qy,qz] (if exists)
    │   └── ...               (up to bowl_8 with DR enabled)
    └── attrs:
        ├── sim=True
        ├── source="food_transfer_ik"
        ├── target=<name>
        ├── dr_enabled (bool)
        ├── dr_seed (int, if DR enabled)
        ├── dr_config (JSON, if DR enabled)
        ├── num_bowls (int, if DR enabled)
        ├── active_bowls (JSON list, if DR enabled)
        └── scene_xml (str, if DR enabled)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import h5py
import mujoco
import numpy as np
from tqdm import tqdm

from trossen_arm_mujoco.food_transfer_base import (
    ALL_BOWLS,
    DT,
    FoodTransferBase,
    TaskPhase,
)
from trossen_arm_mujoco.domain_randomization import (
    DomainRandomizationConfig,
    SceneConfiguration,
)

MANIFEST_FILENAME = "manifest.json"


def load_manifest(output_dir: str) -> Optional[Dict[str, Any]]:
    """Load manifest from output directory if it exists."""
    manifest_path = Path(output_dir) / MANIFEST_FILENAME
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)
    return None


def save_manifest(output_dir: str, manifest: Dict[str, Any]) -> None:
    """Save manifest to output directory."""
    manifest_path = Path(output_dir) / MANIFEST_FILENAME
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def create_manifest(
    dr_config: Optional[DomainRandomizationConfig],
    base_seed: int,
) -> Dict[str, Any]:
    """Create a new manifest for tracking dataset generation."""
    return {
        "dataset_name": "",
        "created": datetime.now().isoformat(),
        "total_episodes_attempted": 0,
        "successful_episodes": 0,
        "failed_episodes": 0,
        "last_seed": base_seed - 1,  # Will be incremented to base_seed on first use
        "seed_history": [],
        "failed_seeds": [],
        "dr_enabled": dr_config.enabled if dr_config else False,
        "config": dr_config.to_dict() if dr_config and dr_config.enabled else None,
    }


def print_dataset_status(output_dir: str) -> None:
    """Print status of existing dataset."""
    manifest = load_manifest(output_dir)
    if manifest is None:
        print(f"No manifest found in {output_dir}")
        return

    print("=" * 60)
    print("DATASET STATUS")
    print("=" * 60)
    print(f"Directory: {output_dir}")
    print(f"Created: {manifest.get('created', 'unknown')}")
    print(f"DR enabled: {manifest.get('dr_enabled', False)}")
    print()
    print(f"Total attempted: {manifest['total_episodes_attempted']}")
    print(f"Successful: {manifest['successful_episodes']} "
          f"({100*manifest['successful_episodes']/max(1, manifest['total_episodes_attempted']):.1f}%)")
    print(f"Failed: {manifest['failed_episodes']}")
    print()
    print(f"Last seed used: {manifest['last_seed']}")
    print(f"Next seed for continuation: {manifest['last_seed'] + 1}")
    print("=" * 60)


def get_next_episode_index(output_dir: str) -> int:
    """Find the next available episode index in the output directory."""
    output_path = Path(output_dir)
    existing_episodes = list(output_path.glob("episode_*.hdf5"))
    if not existing_episodes:
        return 0
    indices = []
    for ep_file in existing_episodes:
        try:
            idx = int(ep_file.stem.split("_")[1])
            indices.append(idx)
        except (ValueError, IndexError):
            continue
    return max(indices) + 1 if indices else 0


class FoodTransferRecorder(FoodTransferBase):
    """
    Records food transfer task demonstrations to HDF5.

    Extends FoodTransferBase with camera rendering and HDF5 saving.
    """

    # Available cameras in the scene
    # Note: cam_high and cam_front in teleop_scene.xml have bad positions
    # Use main_view (overhead) and cam (wrist camera on robot)
    DEFAULT_CAMERAS = ["main_view", "cam"]

    def __init__(
        self,
        scene_xml: str = "wxai/teleop_scene.xml",
        cam_list: Optional[List[str]] = None,
        img_width: int = 640,
        img_height: int = 480,
        inject_noise: bool = False,
        noise_scale: float = 0.02,
        dr_config: Optional[DomainRandomizationConfig] = None,
    ):
        """
        Initialize the recorder.

        Args:
            scene_xml: Path to scene XML file, relative to assets/ directory
            cam_list: Camera names to record. Defaults to ["main_view", "cam"]
            img_width: Image width in pixels
            img_height: Image height in pixels
            inject_noise: Whether to add noise to actions
            noise_scale: Scale of noise in radians
            dr_config: Optional domain randomization configuration
        """
        super().__init__(scene_xml=scene_xml, dr_config=dr_config)

        self.cam_list = cam_list or self.DEFAULT_CAMERAS
        self.img_width = img_width
        self.img_height = img_height
        self.inject_noise = inject_noise
        self.noise_scale = noise_scale

        # Setup rendering
        self.renderer = mujoco.Renderer(self.model, self.img_height, self.img_width)

        # Validate camera names exist in model
        self.valid_cameras = []
        for cam_name in self.cam_list:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id >= 0:
                self.valid_cameras.append(cam_name)
            else:
                print(f"Warning: Camera '{cam_name}' not found in model")

    def render_cameras(self) -> Dict[str, np.ndarray]:
        """Render all cameras and return images."""
        images = {}
        for cam_name in self.valid_cameras:
            self.renderer.update_scene(self.data, camera=cam_name)
            images[cam_name] = self.renderer.render().copy()
        return images

    def get_body_pose(self, body_name: str) -> np.ndarray:
        """
        Get pose (position + quaternion) of a body from simulation.

        Args:
            body_name: Name of the body in MuJoCo model

        Returns:
            7D array [x, y, z, qw, qx, qy, qz]
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model")

        pos = self.data.xpos[body_id].copy()
        quat = self.data.xquat[body_id].copy()  # MuJoCo uses [w, x, y, z] format
        return np.concatenate([pos, quat])

    def get_env_state(self) -> Dict[str, np.ndarray]:
        """
        Get current environment state (container and all bowl poses).

        Dynamically includes all bowls that exist in the scene.
        If scene has 4 bowls, env_state will have bowl_1 through bowl_4.
        If scene has only 2 bowls, env_state will only have those 2.

        Returns:
            Dictionary with:
            - source_container: container pose [x,y,z,qw,qx,qy,qz]
            - target_container: target bowl pose
            - bowl_1, bowl_2, ... : all bowl poses found in scene
        """
        env_state = {
            "source_container": self.get_body_pose("container"),
            "target_container": self.get_body_pose(self.target),
        }
        # Add all bowls that exist in the scene (from base class _bowl_ids)
        for bowl_name in self._bowl_ids.keys():
            env_state[bowl_name] = self.get_body_pose(bowl_name)
        return env_state

    def record_episode(
        self, target: str, speed: float = 1.0
    ) -> Tuple[Dict, int, bool, Dict[str, np.ndarray]]:
        """
        Record a single episode.

        Args:
            target: Which bowl to target
            speed: Speed multiplier

        Returns:
            Tuple of (data_dict, num_timesteps, is_success, env_state)
        """
        # Set target and reset simulation
        self.target = target
        self.reset()

        # Capture environment state at start (container/bowl poses)
        env_state = self.get_env_state()

        # Initialize data collection
        data_dict = {
            "/observations/qpos": [],
            "/observations/qvel": [],
            "/action": [],
        }
        for cam_name in self.valid_cameras:
            data_dict[f"/observations/images/{cam_name}"] = []

        # Time scale
        time_scale = 1.0 / speed

        # Track collisions during episode
        collision_occurred = False
        collision_step = None
        collision_phase = None

        # Run through phases
        phase = TaskPhase.HOME
        current_joints = self.data.qpos[:6].copy()

        while phase != TaskPhase.DONE:
            # Get target for phase
            target_joints = self.solve_for_phase(phase, current_joints)
            duration = self.get_phase_duration(phase) * time_scale

            # Animate movement
            start_joints = current_joints.copy()
            steps = int(duration / DT)

            for step in range(max(steps, 1)):
                t = step / max(steps, 1)
                q = self.interpolate_joints(start_joints, target_joints, t)

                # Add noise if enabled
                if self.inject_noise:
                    noise = np.random.randn(6) * self.noise_scale
                    q = q + noise

                # Capture observation BEFORE applying action
                mujoco.mj_forward(self.model, self.data)
                images = self.render_cameras()
                qpos = self.data.qpos[:8].copy()  # 6 arm + 2 gripper
                qvel = self.data.qvel[:8].copy()

                # Create action (8 values: 6 arm + 2 gripper)
                action = np.zeros(8)
                action[:6] = q
                action[6] = 0.0  # gripper left
                action[7] = 0.0  # gripper right

                # Store data
                data_dict["/observations/qpos"].append(qpos)
                data_dict["/observations/qvel"].append(qvel)
                data_dict["/action"].append(action)
                for cam_name in self.valid_cameras:
                    data_dict[f"/observations/images/{cam_name}"].append(images[cam_name])

                # Apply action to simulation
                self.data.qpos[:6] = q
                self.data.ctrl[:6] = q
                mujoco.mj_step(self.model, self.data)

                # Check for collisions after physics step
                if not collision_occurred and self.has_collision():
                    collision_occurred = True
                    collision_step = len(data_dict["/action"])
                    collision_phase = phase.name
                    collision_info = self.get_collision_info()
                    print(f"  [COLLISION] Detected at step {collision_step} during {collision_phase}:")
                    for pair in collision_info.get("collision_pairs", []):
                        print(f"    - {pair[0]} <-> {pair[1]}")

                # Update reward tracking (call get_reward to update internal state)
                _ = self.get_reward()

            # Update current joints
            current_joints = target_joints.copy()

            # Next phase
            phase = self.next_phase(phase)

        # Check final reward for success (2 = reached bowl = success)
        final_reward = self.get_reward()

        # Episode is only successful if:
        # 1. Reward reached max (completed task)
        # 2. No collisions occurred during episode
        is_success = (final_reward == self.max_reward) and not collision_occurred

        if collision_occurred:
            print(f"  Episode marked as FAILED due to collision at step {collision_step} ({collision_phase})")

        return data_dict, len(data_dict["/action"]), is_success, env_state

    def save_episode_hdf5(
        self,
        data_dict: Dict,
        save_path: str,
        target: str,
        is_success: bool,
        env_state: Dict[str, np.ndarray],
        scene_config: Optional[SceneConfiguration] = None,
        episode_seed: Optional[int] = None,
    ) -> float:
        """
        Save episode data to HDF5 file.

        Args:
            data_dict: Dictionary of recorded data
            save_path: Path to save HDF5 file
            target: Target bowl name (for metadata)
            is_success: Whether episode was successful (reward == max_reward)
            env_state: Environment state dict with container and all bowl poses.
                       Keys: source_container, target_container, bowl_1, bowl_2, etc.
                       Values: 7D pose arrays [x, y, z, qw, qx, qy, qz]
            scene_config: Optional scene configuration from domain randomization
            episode_seed: Optional seed used for this episode

        Returns:
            Time taken to save (seconds)
        """
        max_timesteps = len(data_dict["/action"])

        # Get dimensions
        sample_img = data_dict[f"/observations/images/{self.valid_cameras[0]}"][0]
        img_height, img_width = sample_img.shape[:2]
        qpos_dim = data_dict["/observations/qpos"][0].shape[0]
        qvel_dim = data_dict["/observations/qvel"][0].shape[0]
        action_dim = data_dict["/action"][0].shape[0]

        t0 = time.time()
        with h5py.File(save_path, "w", rdcc_nbytes=1024**2 * 2) as root:
            # Attributes - ensure all strings are native Python str, not numpy str
            root.attrs["sim"] = True
            root.attrs["source"] = "food_transfer_ik"
            root.attrs["target"] = str(target)  # Convert to native str for h5py

            # Domain randomization attributes
            if self.dr_config is not None and self.dr_config.enabled:
                root.attrs["dr_enabled"] = True
                if episode_seed is not None:
                    root.attrs["dr_seed"] = int(episode_seed)
                if scene_config is not None:
                    root.attrs["dr_config"] = scene_config.to_json()
                    root.attrs["num_bowls"] = int(scene_config.num_bowls)
                    # active_bowls might contain numpy strings, convert them
                    active_bowls = [str(b) for b in scene_config.active_bowls]
                    root.attrs["active_bowls"] = json.dumps(active_bowls)
                    root.attrs["scene_xml"] = str(scene_config.scene_xml)
            else:
                root.attrs["dr_enabled"] = False

            # Success indicator (scalar boolean)
            root.create_dataset("success", data=is_success)

            # Environment state group (dynamic - includes all bowls found in scene)
            env_grp = root.create_group("env_state")
            # Write all env_state entries dynamically
            # This includes: source_container, target_container, and all bowl_* poses
            for key, pose in env_state.items():
                env_grp.create_dataset(key, data=pose)

            # Observations group
            obs = root.create_group("observations")

            # Images
            image_grp = obs.create_group("images")
            for cam_name in self.valid_cameras:
                image_grp.create_dataset(
                    cam_name,
                    (max_timesteps, img_height, img_width, 3),
                    dtype="uint8",
                    chunks=(1, img_height, img_width, 3),
                )

            # Qpos and qvel
            obs.create_dataset("qpos", (max_timesteps, qpos_dim))
            obs.create_dataset("qvel", (max_timesteps, qvel_dim))

            # Actions
            root.create_dataset("action", (max_timesteps, action_dim))

            # Write data
            for name, array in data_dict.items():
                root[name][...] = np.array(array)

        return time.time() - t0


def main(args):
    """Record food transfer IK demonstrations."""
    # Setup output directory
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Parse cameras
    cam_list = args.cam_names.split(",") if args.cam_names else None

    # Setup domain randomization config
    dr_config = None
    if args.enable_dr:
        dr_config = DomainRandomizationConfig.from_cli_args(
            enable_dr=True,
            position_noise=args.dr_position_noise,
            rotation_noise=args.dr_rotation_noise,
            container_rotation=args.dr_container_rotation,
            min_bowls=args.dr_min_bowls,
            max_bowls=args.dr_max_bowls,
            seed=args.dr_seed,
        )

    # Handle manifest for dataset continuation
    manifest = None
    starting_seed = args.dr_seed if args.dr_seed is not None else 42
    starting_episode_idx = 0

    if args.continue_dataset:
        manifest = load_manifest(output_dir)
        if manifest is not None:
            # Continue from existing manifest
            starting_seed = manifest["last_seed"] + 1
            starting_episode_idx = get_next_episode_index(output_dir)
            print(f"Continuing from existing dataset:")
            print(f"  Next seed: {starting_seed}")
            print(f"  Next episode index: {starting_episode_idx}")
            print()
        else:
            print("No existing manifest found, starting fresh.")
            manifest = create_manifest(dr_config, starting_seed) if args.enable_dr else None
    elif args.enable_dr:
        # Check if manifest exists but --continue_dataset not specified
        existing_manifest = load_manifest(output_dir)
        if existing_manifest is not None:
            print("WARNING: Existing manifest found but --continue_dataset not specified.")
            print("         Will overwrite existing data. Use --continue_dataset to append.")
            print()
        manifest = create_manifest(dr_config, starting_seed)
        starting_episode_idx = get_next_episode_index(output_dir) if existing_manifest else 0

    # Create recorder (scene_xml will be overridden by DR if enabled)
    recorder = FoodTransferRecorder(
        scene_xml=args.scene,
        cam_list=cam_list,
        inject_noise=args.inject_noise,
        noise_scale=args.noise_scale,
        dr_config=dr_config,
    )

    # Determine bowl targets (ignored when DR is enabled - DR controls target)
    cycle_all = (args.target == "all")

    print("=" * 60)
    print("Food Transfer IK Dataset Recording")
    print("=" * 60)
    print(f"Scene: {args.scene}")
    print(f"Output directory: {output_dir}")
    print(f"Number of episodes: {args.num_episodes}")
    if args.enable_dr:
        print(f"Domain Randomization: ENABLED")
        print(f"  Position noise: +/-{args.dr_position_noise*100:.1f}cm")
        print(f"  Rotation noise: +/-{args.dr_rotation_noise:.2f} rad")
        print(f"  Container rotation: +/-{args.dr_container_rotation:.2f} rad")
        print(f"  Bowl count: {args.dr_min_bowls} to {args.dr_max_bowls}")
        print(f"  Starting seed: {starting_seed}")
    else:
        print(f"Domain Randomization: DISABLED")
        print(f"Target: {'all bowls (cycling)' if cycle_all else args.target}")
    print(f"Cameras: {recorder.valid_cameras}")
    print(f"Noise injection: {args.inject_noise}")
    if args.inject_noise:
        print(f"Noise scale: {args.noise_scale} rad")
    print()

    results = []
    current_seed = starting_seed

    for i in tqdm(range(args.num_episodes), desc="Recording episodes"):
        episode_idx = starting_episode_idx + i
        episode_seed = None
        scene_config = None

        if args.enable_dr:
            # Use current seed for this episode
            episode_seed = current_seed
            current_seed += 1

            # Reset with specific seed - this sets up the randomized scene
            scene_config = recorder.reset_with_seed(episode_seed)
            target = str(scene_config.target_bowl)  # Ensure native str
        else:
            # Determine target bowl without DR
            if cycle_all:
                bowl_idx = i % len(ALL_BOWLS)
                target = ALL_BOWLS[bowl_idx]
            else:
                target = args.target
            recorder.target = target

        # Record episode
        try:
            data_dict, num_timesteps, is_success, env_state = recorder.record_episode(
                target=target,
                speed=args.speed,
            )
        except Exception as e:
            print(f"\nError recording episode {episode_idx}: {e}")
            is_success = False
            # Update manifest for failed episode
            if manifest is not None:
                manifest["total_episodes_attempted"] += 1
                manifest["failed_episodes"] += 1
                manifest["last_seed"] = episode_seed if episode_seed else manifest["last_seed"]
                if episode_seed is not None:
                    manifest["seed_history"].append(episode_seed)
                    manifest["failed_seeds"].append(episode_seed)
                save_manifest(output_dir, manifest)
            continue

        # Save episode
        save_path = os.path.join(output_dir, f"episode_{episode_idx}.hdf5")
        save_time = recorder.save_episode_hdf5(
            data_dict=data_dict,
            save_path=save_path,
            target=target,
            is_success=is_success,
            env_state=env_state,
            scene_config=scene_config,
            episode_seed=episode_seed,
        )

        # Update manifest
        if manifest is not None:
            manifest["total_episodes_attempted"] += 1
            if is_success:
                manifest["successful_episodes"] += 1
            else:
                manifest["failed_episodes"] += 1
                if episode_seed is not None:
                    manifest["failed_seeds"].append(episode_seed)
            manifest["last_seed"] = episode_seed if episode_seed else manifest["last_seed"]
            if episode_seed is not None:
                manifest["seed_history"].append(episode_seed)
            save_manifest(output_dir, manifest)

        results.append({
            "episode_idx": episode_idx,
            "target": target,
            "timesteps": num_timesteps,
            "save_time": save_time,
            "success": is_success,
            "seed": episode_seed,
        })

        if args.verbose:
            status = "SUCCESS" if is_success else "FAILED"
            seed_info = f", seed={episode_seed}" if episode_seed else ""
            print(f"  Episode {episode_idx}: {target}, "
                  f"{num_timesteps} steps, {status}{seed_info}, saved in {save_time:.1f}s")

    # Print summary
    print()
    print("=" * 60)
    print("RECORDING COMPLETE")
    print("=" * 60)
    print(f"Total episodes: {len(results)}")
    print(f"Output directory: {output_dir}")
    if args.enable_dr:
        print(f"Seeds used: {starting_seed} to {current_seed - 1}")
    print()

    # Bowl distribution and success rate
    bowl_counts = {}
    total_timesteps = 0
    success_count = 0
    for r in results:
        bowl = r["target"]
        bowl_counts[bowl] = bowl_counts.get(bowl, 0) + 1
        total_timesteps += r["timesteps"]
        if r["success"]:
            success_count += 1

    print("Target distribution:")
    for bowl, count in sorted(bowl_counts.items()):
        print(f"  {bowl}: {count} episodes")

    if results:
        print(f"\nSuccess rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Average timesteps per episode: {total_timesteps / len(results):.0f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record food transfer IK demonstrations to HDF5."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save episode HDF5 files.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to record (default: 10).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all", "bowl_1", "bowl_2", "bowl_3", "bowl_4",
                 "bowl_5", "bowl_6", "bowl_7", "bowl_8"],
        help="Target bowl. 'all' cycles through available bowls (default: all).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--inject_noise",
        action="store_true",
        help="Inject noise into actions for robustness.",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.02,
        help="Noise scale in radians (default: 0.02).",
    )
    parser.add_argument(
        "--cam_names",
        type=str,
        default=None,
        help="Comma-separated camera names (default: main_view,cam).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress for each episode.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="wxai/teleop_scene.xml",
        help="Scene XML file path relative to assets/ (default: wxai/teleop_scene.xml)",
    )

    # Domain randomization arguments
    dr_group = parser.add_argument_group("Domain Randomization")
    dr_group.add_argument(
        "--enable_dr",
        action="store_true",
        help="Enable domain randomization.",
    )
    dr_group.add_argument(
        "--dr_position_noise",
        type=float,
        default=0.03,
        help="Position noise in meters for bowls (default: 0.03 = ±3cm).",
    )
    dr_group.add_argument(
        "--dr_rotation_noise",
        type=float,
        default=0.1,
        help="Rotation noise in radians for bowls (default: 0.1 rad).",
    )
    dr_group.add_argument(
        "--dr_container_rotation",
        type=float,
        default=0.15,
        help="Container yaw rotation noise in radians (default: 0.15 rad).",
    )
    dr_group.add_argument(
        "--dr_min_bowls",
        type=int,
        default=1,
        help="Minimum number of bowls per episode (default: 1).",
    )
    dr_group.add_argument(
        "--dr_max_bowls",
        type=int,
        default=8,
        help="Maximum number of bowls per episode (default: 8).",
    )
    dr_group.add_argument(
        "--dr_seed",
        type=int,
        default=None,
        help="Random seed for domain randomization (default: None = auto from manifest or 42).",
    )

    # Dataset continuation arguments
    continuation_group = parser.add_argument_group("Dataset Continuation")
    continuation_group.add_argument(
        "--continue_dataset",
        action="store_true",
        help="Continue from existing dataset manifest (auto-increment seed).",
    )
    continuation_group.add_argument(
        "--status",
        action="store_true",
        help="Print dataset status and exit.",
    )

    args = parser.parse_args()

    # Handle --status flag
    if args.status:
        output_dir = args.output_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        print_dataset_status(output_dir)
        sys.exit(0)

    main(args)
