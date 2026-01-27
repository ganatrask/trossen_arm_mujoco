#!/usr/bin/env python3
"""
Automated dataset generation pipeline with HuggingFace upload.

This script:
1. Records N episodes to HDF5 in batches
2. Converts episodes to MP4 videos
3. Uploads batches incrementally to HuggingFace
4. Handles failures gracefully with retry at end
5. Supports resume from interruption

Usage:
    # Full pipeline with DR and cleanup
    python -m trossen_arm_mujoco.scripts.generate_dataset_hf \\
        --output_dir ./my_dataset \\
        --hf_repo_id "username/food-transfer-sim" \\
        --num_episodes 10000 \\
        --batch_size 100 \\
        --enable_dr \\
        --workers 8 \\
        --cleanup

    # Resume interrupted run
    python -m trossen_arm_mujoco.scripts.generate_dataset_hf \\
        --output_dir ./my_dataset \\
        --resume

    # Check status
    python -m trossen_arm_mujoco.scripts.generate_dataset_hf \\
        --output_dir ./my_dataset \\
        --status

    # Dry run (no upload)
    python -m trossen_arm_mujoco.scripts.generate_dataset_hf \\
        --output_dir ./my_dataset \\
        --hf_repo_id "username/food-transfer-sim" \\
        --num_episodes 100 \\
        --dry_run
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
from tqdm import tqdm

from trossen_arm_mujoco.domain_randomization import (
    DomainRandomizationConfig,
    get_optimal_workers,
)
from trossen_arm_mujoco.food_transfer_base import ALL_BOWLS, DT
from trossen_arm_mujoco.hf_utils.batch_processor import BatchProcessor
from trossen_arm_mujoco.hf_utils.dataset_card import generate_dataset_card
from trossen_arm_mujoco.hf_utils.uploader import HuggingFaceUploader

# Import the recording function from record_food_transfer_ik
from trossen_arm_mujoco.scripts.record_food_transfer_ik import (
    FoodTransferRecorder,
    _record_single_episode,
)


PIPELINE_STATE_FILENAME = "pipeline_state.json"


@dataclass
class PipelineConfig:
    """Configuration for the dataset generation pipeline."""

    output_dir: str
    hf_repo_id: str
    num_episodes: int
    batch_size: int = 100
    workers: int = 0  # 0 = auto
    cleanup: bool = False
    keep_hdf5: bool = False
    keep_videos: bool = False
    dry_run: bool = False
    video_fps: int = 50

    # Domain randomization
    enable_dr: bool = False
    dr_position_noise: float = 0.03
    dr_rotation_noise: float = 0.1
    dr_container_rotation: float = 0.15
    dr_min_bowls: int = 1
    dr_max_bowls: int = 8
    dr_min_spacing: float = 0.12
    dr_randomize_container: bool = False
    dr_90_degree_rotation: bool = False
    dr_seed: int = 42

    # Visual domain randomization
    enable_visual_dr: bool = False
    dr_no_table_texture: bool = False
    dr_num_table_textures: int = 100
    dr_no_floor_texture: bool = False
    dr_num_floor_textures: int = 100
    dr_no_container_color: bool = False
    dr_randomize_bowl_color: bool = False
    dr_no_lighting: bool = False
    dr_light_position_noise: float = 0.3
    dr_light_intensity_min: float = 0.5
    dr_light_intensity_max: float = 1.2

    # Recording settings
    scene: str = "wxai/teleop_scene.xml"
    inject_noise: bool = False
    noise_scale: float = 0.02
    speed: float = 1.0

    # Batch/episode numbering
    start_batch: int = 0  # Starting batch number for naming (batch_000, batch_001, ...)
    start_episode: int = 0  # Starting episode number for naming (episode_0, episode_1, ...)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PipelineState:
    """Tracks pipeline progress for resume functionality."""

    total_recorded: int = 0
    total_converted: int = 0
    total_uploaded: int = 0
    current_batch: int = 0
    current_seed: int = 42
    failed_episodes: List[int] = field(default_factory=list)
    failed_seeds: List[int] = field(default_factory=list)
    uploaded_batches: List[int] = field(default_factory=list)
    pending_batches: List[int] = field(default_factory=list)
    failed_uploads: List[int] = field(default_factory=list)
    started_at: str = ""
    last_updated: str = ""
    config: Optional[Dict[str, Any]] = None

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        self.last_updated = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional["PipelineState"]:
        """Load state from JSON file."""
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        # Handle config separately
        config = data.pop("config", None)
        state = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        state.config = config
        return state


class DatasetPipeline:
    """
    Main orchestrator for dataset generation with HuggingFace upload.

    Handles:
    - Batch recording of episodes
    - Video conversion
    - Incremental upload to HuggingFace
    - Resume from interruption
    - Cleanup after successful upload
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir).resolve()
        self.state_path = self.output_dir / PIPELINE_STATE_FILENAME

        # Initialize components
        self.batch_processor = BatchProcessor(
            str(self.output_dir),
            batch_size=config.batch_size,
        )
        self.uploader = HuggingFaceUploader(
            repo_id=config.hf_repo_id,
            dry_run=config.dry_run,
        )

        # Load or create state
        self.state = PipelineState.load(self.state_path)
        if self.state is None:
            self.state = PipelineState(
                started_at=datetime.now().isoformat(),
                current_seed=config.dr_seed,
                config=config.to_dict(),
            )

        # Setup DR config
        self.dr_config = self._create_dr_config()

        # Determine workers
        self.workers = config.workers if config.workers > 0 else get_optimal_workers()

    def _create_dr_config(self) -> Optional[DomainRandomizationConfig]:
        """Create domain randomization config from pipeline config."""
        if not self.config.enable_dr and not self.config.enable_visual_dr:
            return None

        return DomainRandomizationConfig.from_cli_args(
            enable_dr=self.config.enable_dr or self.config.enable_visual_dr,
            position_noise=self.config.dr_position_noise,
            rotation_noise=self.config.dr_rotation_noise,
            container_rotation=self.config.dr_container_rotation,
            min_bowls=self.config.dr_min_bowls,
            max_bowls=self.config.dr_max_bowls,
            seed=self.state.current_seed,
            min_spacing=self.config.dr_min_spacing,
            randomize_container_position=self.config.dr_randomize_container,
            allow_90_degree_rotation=self.config.dr_90_degree_rotation,
            enable_visual_dr=self.config.enable_visual_dr,
            randomize_table_texture=not self.config.dr_no_table_texture,
            num_table_textures=self.config.dr_num_table_textures,
            randomize_floor_texture=not self.config.dr_no_floor_texture,
            num_floor_textures=self.config.dr_num_floor_textures,
            randomize_container_color=not self.config.dr_no_container_color,
            randomize_bowl_color=self.config.dr_randomize_bowl_color,
            randomize_lighting=not self.config.dr_no_lighting,
            light_position_noise=self.config.dr_light_position_noise,
            light_intensity_min=self.config.dr_light_intensity_min,
            light_intensity_max=self.config.dr_light_intensity_max,
        )

    def run(self) -> None:
        """Run the full pipeline."""
        print("=" * 70)
        print("DATASET GENERATION PIPELINE")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")
        print(f"HuggingFace repo: {self.config.hf_repo_id}")
        print(f"Target episodes: {self.config.num_episodes}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Workers: {self.workers}")
        print(f"Dry run: {self.config.dry_run}")
        print(f"Cleanup after upload: {self.config.cleanup}")
        if self.config.enable_dr:
            print(f"Domain randomization: ENABLED")
        if self.config.enable_visual_dr:
            print(f"Visual DR: ENABLED")
        print("=" * 70)
        print()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure HF repo exists
        print("Ensuring HuggingFace repository exists...")
        if not self.uploader.ensure_repo_exists():
            print("ERROR: Failed to create/verify HuggingFace repository")
            return

        # Calculate batches
        num_batches = (self.config.num_episodes + self.config.batch_size - 1) // self.config.batch_size
        start_batch = self.state.current_batch

        print(f"Total batches: {num_batches}")
        print(f"Starting from batch: {start_batch}")
        if self.config.start_batch > 0:
            print(f"HF batch naming starts at: batch_{self.config.start_batch:03d}")
        if self.config.start_episode > 0:
            print(f"Episode naming starts at: episode_{self.config.start_episode}")
        print()

        # Main loop
        for batch_num in range(start_batch, num_batches):
            hf_batch_num = batch_num + self.config.start_batch
            print(f"\n{'='*70}")
            print(f"BATCH {batch_num + 1}/{num_batches} (batch_{hf_batch_num:03d} in HF)")
            print(f"{'='*70}")

            # Calculate episode range for this batch
            # Local indices (0-based within this run)
            local_start_ep = batch_num * self.config.batch_size
            local_end_ep = min(local_start_ep + self.config.batch_size, self.config.num_episodes)
            num_episodes_in_batch = local_end_ep - local_start_ep
            # Global indices (offset by start_episode for file naming)
            start_ep = local_start_ep + self.config.start_episode
            end_ep = local_end_ep + self.config.start_episode

            # Record batch
            print(f"\n[1/3] Recording episodes {start_ep} to {end_ep - 1}...")
            recorded, failed = self._record_batch(batch_num, start_ep, num_episodes_in_batch)
            self.state.total_recorded += recorded
            print(f"Recorded: {recorded}, Failed: {failed}")

            # Convert to videos
            print(f"\n[2/3] Converting to videos...")
            converted, conv_failed, failed_eps = self.batch_processor.convert_batch_to_videos(
                batch_num,
                fps=self.config.video_fps,
                workers=self.workers,
            )
            self.state.total_converted += converted
            print(f"Converted: {converted}, Failed: {conv_failed}")

            # Upload batch
            print(f"\n[3/3] Uploading to HuggingFace...")
            success = self._upload_batch(batch_num)

            if success:
                self.state.uploaded_batches.append(batch_num)
                self.state.total_uploaded += recorded
                print(f"Upload successful!")

                # Cleanup if enabled
                if self.config.cleanup:
                    self._cleanup_batch(batch_num)
            else:
                self.state.failed_uploads.append(batch_num)
                print(f"Upload FAILED - batch saved for retry")

            # Update state
            self.state.current_batch = batch_num + 1
            self.state.save(self.state_path)

            # Print batch stats
            stats = self.batch_processor.get_batch_stats(batch_num)
            print(f"\nBatch stats: {stats['hdf5_count']} HDF5, {stats['video_count']} videos, "
                  f"{stats['total_size_mb']:.1f} MB")

        # Retry failed uploads
        if self.state.failed_uploads:
            print(f"\n{'='*70}")
            print("RETRYING FAILED UPLOADS")
            print(f"{'='*70}")
            self._retry_failed_uploads()

        # Upload dataset card
        print(f"\n{'='*70}")
        print("UPLOADING DATASET CARD")
        print(f"{'='*70}")
        self._upload_dataset_card()

        # Upload manifest
        self._upload_manifest()

        # Final summary
        self._print_summary()

    def _record_batch(
        self,
        batch_num: int,
        start_ep: int,
        num_episodes: int,
    ) -> tuple[int, int]:
        """Record episodes for a batch."""
        batch_dir = self.batch_processor.ensure_batch_dir(batch_num)

        # Create recorder kwargs
        recorder_kwargs = {
            "scene_xml": self.config.scene,
            "cam_list": None,
            "inject_noise": self.config.inject_noise,
            "noise_scale": self.config.noise_scale,
            "dr_config": self.dr_config,
        }

        # Build episode arguments
        # DR is enabled if either geometric or visual DR is requested
        dr_enabled = self.config.enable_dr or self.config.enable_visual_dr
        episode_args = []
        for i in range(num_episodes):
            episode_idx = start_ep + i
            episode_seed = self.state.current_seed
            self.state.current_seed += 1

            if dr_enabled:
                target = None  # DR will determine target
            else:
                target = ALL_BOWLS[i % len(ALL_BOWLS)]

            episode_args.append((
                episode_idx,
                target,
                episode_seed,
                recorder_kwargs,
                self.config.speed,
                dr_enabled,  # Pass combined DR flag
                ALL_BOWLS,
            ))

        # Record in parallel
        recorded = 0
        failed = 0

        if self.workers > 1:
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                futures = {
                    executor.submit(_record_single_episode, arg): arg[0]
                    for arg in episode_args
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Recording episodes",
                ):
                    result = future.result()
                    episode_idx = result["episode_idx"]

                    if result["success"]:
                        # Save to batch directory
                        save_path = batch_dir / f"episode_{episode_idx}.hdf5"
                        self._save_episode(result, save_path)
                        recorded += 1
                    else:
                        failed += 1
                        self.state.failed_episodes.append(episode_idx)
                        self.state.failed_seeds.append(result.get("episode_seed", -1))
                        print(f"Episode {episode_idx} failed: {result.get('error', 'Unknown')}")
        else:
            for arg in tqdm(episode_args, desc="Recording episodes"):
                result = _record_single_episode(arg)
                episode_idx = result["episode_idx"]

                if result["success"]:
                    save_path = batch_dir / f"episode_{episode_idx}.hdf5"
                    self._save_episode(result, save_path)
                    recorded += 1
                else:
                    failed += 1
                    self.state.failed_episodes.append(episode_idx)
                    self.state.failed_seeds.append(result.get("episode_seed", -1))

        return recorded, failed

    def _save_episode(self, result: Dict[str, Any], save_path: Path) -> None:
        """Save episode result to HDF5."""
        data_dict = result["data_dict"]
        valid_cameras = result["valid_cameras"]
        max_timesteps = len(data_dict["/action"])

        sample_img = data_dict[f"/observations/images/{valid_cameras[0]}"][0]
        img_height, img_width = sample_img.shape[:2]
        qpos_dim = data_dict["/observations/qpos"][0].shape[0]
        qvel_dim = data_dict["/observations/qvel"][0].shape[0]
        action_dim = data_dict["/action"][0].shape[0]

        with h5py.File(save_path, "w", rdcc_nbytes=1024**2 * 2) as root:
            root.attrs["sim"] = True
            root.attrs["source"] = "food_transfer_ik"
            root.attrs["target"] = str(result["target"])

            if self.dr_config is not None and self.dr_config.enabled:
                root.attrs["dr_enabled"] = True
                if result["episode_seed"] is not None:
                    root.attrs["dr_seed"] = int(result["episode_seed"])
                scene_config = result.get("scene_config")
                if scene_config is not None:
                    root.attrs["dr_config"] = scene_config.to_json()
                    root.attrs["num_bowls"] = int(scene_config.num_bowls)
                    active_bowls = [str(b) for b in scene_config.active_bowls]
                    root.attrs["active_bowls"] = json.dumps(active_bowls)
                    root.attrs["scene_xml"] = str(scene_config.scene_xml)
            else:
                root.attrs["dr_enabled"] = False

            root.create_dataset("success", data=result["is_success"])

            env_grp = root.create_group("env_state")
            for key, pose in result["env_state"].items():
                env_grp.create_dataset(key, data=pose)

            obs = root.create_group("observations")
            image_grp = obs.create_group("images")

            for cam_name in valid_cameras:
                image_grp.create_dataset(
                    cam_name,
                    (max_timesteps, img_height, img_width, 3),
                    dtype="uint8",
                    chunks=(1, img_height, img_width, 3),
                    compression="gzip",
                    compression_opts=4,
                )

            obs.create_dataset("qpos", (max_timesteps, qpos_dim), compression="lzf")
            obs.create_dataset("qvel", (max_timesteps, qvel_dim), compression="lzf")
            root.create_dataset("action", (max_timesteps, action_dim), compression="lzf")

            for name, array in data_dict.items():
                root[name][...] = np.array(array)

    def _upload_batch(self, batch_num: int) -> bool:
        """Upload a batch to HuggingFace."""
        batch_dir = self.batch_processor.get_batch_dir(batch_num)
        # Use start_batch offset for naming in HF repo
        hf_batch_num = batch_num + self.config.start_batch
        path_in_repo = f"data/batch_{hf_batch_num:03d}"

        return self.uploader.upload_batch(
            str(batch_dir),
            path_in_repo,
            max_retries=3,
        )

    def _cleanup_batch(self, batch_num: int) -> None:
        """Clean up local files after successful upload."""
        delete_hdf5 = not self.config.keep_hdf5
        delete_videos = not self.config.keep_videos

        hdf5_deleted, videos_deleted = self.batch_processor.cleanup_batch(
            batch_num,
            delete_hdf5=delete_hdf5,
            delete_videos=delete_videos,
        )
        print(f"Cleanup: deleted {hdf5_deleted} HDF5, {videos_deleted} videos")

    def _retry_failed_uploads(self) -> None:
        """Retry any failed batch uploads."""
        still_failed = []

        for batch_num in self.state.failed_uploads:
            print(f"Retrying batch {batch_num}...")
            success = self._upload_batch(batch_num)

            if success:
                self.state.uploaded_batches.append(batch_num)
                print(f"Batch {batch_num} uploaded successfully on retry")
                if self.config.cleanup:
                    self._cleanup_batch(batch_num)
            else:
                still_failed.append(batch_num)
                print(f"Batch {batch_num} still failing")

        self.state.failed_uploads = still_failed
        self.state.save(self.state_path)

    def _upload_dataset_card(self) -> None:
        """Generate and upload dataset card."""
        dr_config = None
        visual_dr_config = None

        if self.config.enable_dr:
            dr_config = {
                "position_noise": self.config.dr_position_noise,
                "rotation_noise": self.config.dr_rotation_noise,
                "container_rotation": self.config.dr_container_rotation,
                "min_bowls": self.config.dr_min_bowls,
                "max_bowls": self.config.dr_max_bowls,
                "min_spacing": self.config.dr_min_spacing,
                "randomize_container_position": self.config.dr_randomize_container,
                "allow_90_degree_rotation": self.config.dr_90_degree_rotation,
            }

        if self.config.enable_visual_dr:
            visual_dr_config = {
                "randomize_table_texture": not self.config.dr_no_table_texture,
                "num_table_textures": self.config.dr_num_table_textures,
                "randomize_floor_texture": not self.config.dr_no_floor_texture,
                "num_floor_textures": self.config.dr_num_floor_textures,
                "randomize_container_color": not self.config.dr_no_container_color,
                "randomize_bowl_color": self.config.dr_randomize_bowl_color,
                "randomize_lighting": not self.config.dr_no_lighting,
                "light_position_noise": self.config.dr_light_position_noise,
                "light_intensity_min": self.config.dr_light_intensity_min,
                "light_intensity_max": self.config.dr_light_intensity_max,
            }

        card_content = generate_dataset_card(
            repo_id=self.config.hf_repo_id,
            num_episodes=self.state.total_recorded,
            dr_config=dr_config,
            visual_dr_config=visual_dr_config,
            additional_info={
                "Total batches": len(self.state.uploaded_batches),
                "Failed episodes": len(self.state.failed_episodes),
            },
        )

        success = self.uploader.update_readme(card_content)
        if success:
            print("Dataset card uploaded successfully")
        else:
            print("Failed to upload dataset card")

    def _upload_manifest(self) -> None:
        """Upload pipeline state as manifest.json."""
        manifest = {
            "hf_repo_id": self.config.hf_repo_id,
            "recording": {
                "total_attempted": self.state.total_recorded + len(self.state.failed_episodes),
                "successful": self.state.total_recorded,
                "failed_seeds": self.state.failed_seeds,
            },
            "video_conversion": {
                "converted": self.state.total_converted,
                "failed": [],
            },
            "upload": {
                "uploaded_batches": self.state.uploaded_batches,
                "pending_batches": self.state.pending_batches,
                "failed_uploads": self.state.failed_uploads,
            },
            "config": self.config.to_dict(),
            "generated_at": datetime.now().isoformat(),
        }

        # Save locally
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Upload to HF
        success = self.uploader.upload_file(str(manifest_path), "manifest.json")
        if success:
            print("Manifest uploaded successfully")
        else:
            print("Failed to upload manifest")

    def _print_summary(self) -> None:
        """Print final summary."""
        print()
        print("=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Total recorded: {self.state.total_recorded}")
        print(f"Total converted: {self.state.total_converted}")
        print(f"Total uploaded: {self.state.total_uploaded}")
        print(f"Batches uploaded: {len(self.state.uploaded_batches)}")
        print(f"Failed episodes: {len(self.state.failed_episodes)}")
        print(f"Failed uploads: {len(self.state.failed_uploads)}")
        print()
        if not self.config.dry_run:
            print(f"Dataset URL: https://huggingface.co/datasets/{self.config.hf_repo_id}")
        print("=" * 70)


def print_status(output_dir: str) -> None:
    """Print status of existing pipeline."""
    state_path = Path(output_dir) / PIPELINE_STATE_FILENAME
    state = PipelineState.load(state_path)

    if state is None:
        print(f"No pipeline state found in {output_dir}")
        return

    print("=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)
    print(f"Directory: {output_dir}")
    print(f"Started: {state.started_at}")
    print(f"Last updated: {state.last_updated}")
    print()
    print(f"Total recorded: {state.total_recorded}")
    print(f"Total converted: {state.total_converted}")
    print(f"Total uploaded: {state.total_uploaded}")
    print(f"Current batch: {state.current_batch}")
    print(f"Current seed: {state.current_seed}")
    print()
    print(f"Uploaded batches: {len(state.uploaded_batches)}")
    print(f"Failed episodes: {len(state.failed_episodes)}")
    print(f"Failed uploads: {len(state.failed_uploads)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset with HuggingFace upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for local dataset storage",
    )

    # Pipeline control
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="HuggingFace repo ID (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="Number of episodes to generate (default: 1000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Episodes per batch (default: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0=auto)",
    )

    # Upload/cleanup
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete local files after successful upload",
    )
    parser.add_argument(
        "--keep_hdf5",
        action="store_true",
        help="Keep HDF5 files when cleaning up",
    )
    parser.add_argument(
        "--keep_videos",
        action="store_true",
        help="Keep videos when cleaning up",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't actually upload (for testing)",
    )

    # Resume/status
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print status and exit",
    )

    # Recording settings
    parser.add_argument(
        "--scene",
        type=str,
        default="wxai/teleop_scene.xml",
        help="Scene XML file",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Recording speed multiplier",
    )
    parser.add_argument(
        "--inject_noise",
        action="store_true",
        help="Inject noise into actions",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.02,
        help="Noise scale in radians",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=50,
        help="Video FPS (default: 50)",
    )

    # Domain randomization
    dr_group = parser.add_argument_group("Domain Randomization")
    dr_group.add_argument("--enable_dr", action="store_true", help="Enable geometric DR")
    dr_group.add_argument("--dr_position_noise", type=float, default=0.03)
    dr_group.add_argument("--dr_rotation_noise", type=float, default=0.1)
    dr_group.add_argument("--dr_container_rotation", type=float, default=0.15)
    dr_group.add_argument("--dr_min_bowls", type=int, default=1)
    dr_group.add_argument("--dr_max_bowls", type=int, default=8)
    dr_group.add_argument("--dr_min_spacing", type=float, default=0.12)
    dr_group.add_argument("--dr_randomize_container", action="store_true")
    dr_group.add_argument("--dr_90_degree_rotation", action="store_true")
    dr_group.add_argument("--dr_seed", type=int, default=42)

    # Visual domain randomization
    visual_dr_group = parser.add_argument_group("Visual Domain Randomization")
    visual_dr_group.add_argument("--enable_visual_dr", action="store_true")
    visual_dr_group.add_argument("--dr_no_table_texture", action="store_true")
    visual_dr_group.add_argument("--dr_num_table_textures", type=int, default=100)
    visual_dr_group.add_argument("--dr_no_floor_texture", action="store_true")
    visual_dr_group.add_argument("--dr_num_floor_textures", type=int, default=100)
    visual_dr_group.add_argument("--dr_no_container_color", action="store_true")
    visual_dr_group.add_argument("--dr_randomize_bowl_color", action="store_true")
    visual_dr_group.add_argument("--dr_no_lighting", action="store_true")
    visual_dr_group.add_argument("--dr_light_position_noise", type=float, default=0.3)
    visual_dr_group.add_argument("--dr_light_intensity_min", type=float, default=0.5)
    visual_dr_group.add_argument("--dr_light_intensity_max", type=float, default=1.2)

    # Batch/episode numbering
    parser.add_argument(
        "--start_batch",
        type=int,
        default=0,
        help="Starting batch number for naming (batch_000, batch_001, ...). "
             "Use this when extending an existing dataset to avoid overwriting batches.",
    )
    parser.add_argument(
        "--start_episode",
        type=int,
        default=0,
        help="Starting episode number for naming (episode_0, episode_1, ...). "
             "Use this when extending an existing dataset to continue numbering.",
    )

    args = parser.parse_args()

    # Handle --status
    if args.status:
        print_status(args.output_dir)
        return

    # Handle --resume
    if args.resume:
        state_path = Path(args.output_dir) / PIPELINE_STATE_FILENAME
        state = PipelineState.load(state_path)
        if state is None:
            print(f"No state found to resume from in {args.output_dir}")
            return
        if state.config is None:
            print("State has no config - cannot resume")
            return

        config = PipelineConfig.from_dict(state.config)
        print(f"Resuming from batch {state.current_batch}")
    else:
        # Validate required args
        if not args.hf_repo_id:
            parser.error("--hf_repo_id is required unless using --resume or --status")

        config = PipelineConfig(
            output_dir=args.output_dir,
            hf_repo_id=args.hf_repo_id,
            num_episodes=args.num_episodes,
            batch_size=args.batch_size,
            workers=args.workers,
            cleanup=args.cleanup,
            keep_hdf5=args.keep_hdf5,
            keep_videos=args.keep_videos,
            dry_run=args.dry_run,
            video_fps=args.video_fps,
            enable_dr=args.enable_dr,
            dr_position_noise=args.dr_position_noise,
            dr_rotation_noise=args.dr_rotation_noise,
            dr_container_rotation=args.dr_container_rotation,
            dr_min_bowls=args.dr_min_bowls,
            dr_max_bowls=args.dr_max_bowls,
            dr_min_spacing=args.dr_min_spacing,
            dr_randomize_container=args.dr_randomize_container,
            dr_90_degree_rotation=args.dr_90_degree_rotation,
            dr_seed=args.dr_seed,
            enable_visual_dr=args.enable_visual_dr,
            dr_no_table_texture=args.dr_no_table_texture,
            dr_num_table_textures=args.dr_num_table_textures,
            dr_no_floor_texture=args.dr_no_floor_texture,
            dr_num_floor_textures=args.dr_num_floor_textures,
            dr_no_container_color=args.dr_no_container_color,
            dr_randomize_bowl_color=args.dr_randomize_bowl_color,
            dr_no_lighting=args.dr_no_lighting,
            dr_light_position_noise=args.dr_light_position_noise,
            dr_light_intensity_min=args.dr_light_intensity_min,
            dr_light_intensity_max=args.dr_light_intensity_max,
            scene=args.scene,
            inject_noise=args.inject_noise,
            noise_scale=args.noise_scale,
            speed=args.speed,
            start_batch=args.start_batch,
            start_episode=args.start_episode,
        )

    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Run pipeline
    pipeline = DatasetPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
