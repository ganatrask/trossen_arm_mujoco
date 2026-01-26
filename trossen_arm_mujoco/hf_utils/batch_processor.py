"""
Batch processor for HDF5 to video conversion with parallel support.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def _convert_single_episode(args: Tuple[str, str, int]) -> Tuple[str, bool, Optional[str]]:
    """
    Worker function to convert a single HDF5 episode to MP4.

    Args:
        args: Tuple of (hdf5_path, output_path, fps)

    Returns:
        Tuple of (episode_name, success, error_message)
    """
    hdf5_path, output_path, fps = args
    episode_name = Path(hdf5_path).stem

    try:
        # Load images from HDF5
        with h5py.File(hdf5_path, "r") as root:
            if "/observations/images" not in root:
                return episode_name, False, "No images found in HDF5"

            image_dict = {}
            for cam_name in root["/observations/images/"].keys():
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][()]

        if not image_dict:
            return episode_name, False, "Empty image dictionary"

        # Get dimensions
        cam_names = list(image_dict.keys())
        h, w, _ = image_dict[cam_names[0]][0].shape
        w_total = w * len(cam_names)

        # Initialize video writer
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w_total, h),
        )

        num_frames = len(image_dict[cam_names[0]])

        for frame_idx in range(num_frames):
            # Convert RGB to BGR and concatenate horizontally
            frame_row = [
                image_dict[cam_name][frame_idx][:, :, [2, 1, 0]]
                for cam_name in cam_names
            ]
            concatenated_frame = np.concatenate(frame_row, axis=1)
            out.write(concatenated_frame)

        out.release()
        return episode_name, True, None

    except Exception as e:
        return episode_name, False, str(e)


class BatchProcessor:
    """
    Handles batch organization and video conversion for dataset generation.

    Manages batch directories and provides parallel video conversion.
    """

    def __init__(
        self,
        output_dir: str,
        batch_size: int = 100,
    ):
        """
        Initialize the batch processor.

        Args:
            output_dir: Base output directory for the dataset
            batch_size: Number of episodes per batch
        """
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.data_dir = self.output_dir / "data"

    def get_batch_num(self, episode_idx: int) -> int:
        """Get batch number for a given episode index."""
        return episode_idx // self.batch_size

    def get_batch_dir(self, batch_num: int) -> Path:
        """Get the directory path for a specific batch."""
        return self.data_dir / f"batch_{batch_num:03d}"

    def get_episode_path(self, episode_idx: int) -> Path:
        """Get the HDF5 file path for an episode."""
        batch_num = self.get_batch_num(episode_idx)
        batch_dir = self.get_batch_dir(batch_num)
        return batch_dir / f"episode_{episode_idx}.hdf5"

    def get_video_dir(self, batch_num: int) -> Path:
        """Get the videos subdirectory for a batch."""
        return self.get_batch_dir(batch_num) / "videos"

    def get_video_path(self, episode_idx: int) -> Path:
        """Get the MP4 file path for an episode (in videos/ subdirectory)."""
        batch_num = self.get_batch_num(episode_idx)
        video_dir = self.get_video_dir(batch_num)
        return video_dir / f"episode_{episode_idx}.mp4"

    def ensure_batch_dir(self, batch_num: int) -> Path:
        """Create batch directory if it doesn't exist."""
        batch_dir = self.get_batch_dir(batch_num)
        batch_dir.mkdir(parents=True, exist_ok=True)
        return batch_dir

    def ensure_video_dir(self, batch_num: int) -> Path:
        """Create videos subdirectory if it doesn't exist."""
        video_dir = self.get_video_dir(batch_num)
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir

    def get_batch_episodes(self, batch_num: int) -> List[int]:
        """
        Get list of episode indices that belong to a batch.

        Args:
            batch_num: Batch number

        Returns:
            List of episode indices
        """
        start_idx = batch_num * self.batch_size
        end_idx = start_idx + self.batch_size
        return list(range(start_idx, end_idx))

    def get_existing_hdf5_in_batch(self, batch_num: int) -> List[Path]:
        """Get list of existing HDF5 files in a batch directory."""
        batch_dir = self.get_batch_dir(batch_num)
        if not batch_dir.exists():
            return []
        return sorted(batch_dir.glob("episode_*.hdf5"))

    def get_existing_videos_in_batch(self, batch_num: int) -> List[Path]:
        """Get list of existing MP4 files in a batch's videos/ subdirectory."""
        video_dir = self.get_video_dir(batch_num)
        if not video_dir.exists():
            return []
        return sorted(video_dir.glob("episode_*.mp4"))

    def convert_episode_to_video(
        self,
        hdf5_path: str,
        output_path: str,
        fps: int = 50,
    ) -> bool:
        """
        Convert a single HDF5 episode to MP4 video.

        Args:
            hdf5_path: Path to input HDF5 file
            output_path: Path to output MP4 file
            fps: Frames per second for output video

        Returns:
            True if conversion succeeded
        """
        _, success, error = _convert_single_episode((hdf5_path, output_path, fps))
        if not success:
            print(f"Video conversion failed: {error}")
        return success

    def convert_batch_to_videos(
        self,
        batch_num: int,
        fps: int = 50,
        workers: int = 4,
        skip_existing: bool = True,
    ) -> Tuple[int, int, List[str]]:
        """
        Convert all HDF5 episodes in a batch to MP4 videos using parallel processing.

        Args:
            batch_num: Batch number to convert
            fps: Frames per second for output videos
            workers: Number of parallel workers
            skip_existing: Skip episodes that already have videos

        Returns:
            Tuple of (successful_count, failed_count, list_of_failed_episodes)
        """
        batch_dir = self.get_batch_dir(batch_num)
        if not batch_dir.exists():
            print(f"Batch directory does not exist: {batch_dir}")
            return 0, 0, []

        hdf5_files = self.get_existing_hdf5_in_batch(batch_num)
        if not hdf5_files:
            print(f"No HDF5 files found in batch {batch_num}")
            return 0, 0, []

        # Ensure videos subdirectory exists
        video_dir = self.ensure_video_dir(batch_num)

        # Build conversion tasks
        tasks = []
        for hdf5_path in hdf5_files:
            # Video goes in videos/ subdirectory
            video_path = video_dir / f"{hdf5_path.stem}.mp4"
            if skip_existing and video_path.exists():
                continue
            tasks.append((str(hdf5_path), str(video_path), fps))

        if not tasks:
            print(f"All videos already exist for batch {batch_num}")
            return len(hdf5_files), 0, []

        successful = 0
        failed = 0
        failed_episodes = []

        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_convert_single_episode, task): task[0]
                    for task in tasks
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Converting batch {batch_num} to videos",
                ):
                    episode_name, success, error = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        failed_episodes.append(episode_name)
                        print(f"Failed to convert {episode_name}: {error}")
        else:
            for task in tqdm(tasks, desc=f"Converting batch {batch_num} to videos"):
                episode_name, success, error = _convert_single_episode(task)
                if success:
                    successful += 1
                else:
                    failed += 1
                    failed_episodes.append(episode_name)
                    print(f"Failed to convert {episode_name}: {error}")

        return successful, failed, failed_episodes

    def cleanup_batch(
        self,
        batch_num: int,
        delete_hdf5: bool = True,
        delete_videos: bool = True,
    ) -> Tuple[int, int]:
        """
        Delete local files from a batch after successful upload.

        Args:
            batch_num: Batch number to clean up
            delete_hdf5: Whether to delete HDF5 files
            delete_videos: Whether to delete MP4 files

        Returns:
            Tuple of (hdf5_deleted_count, videos_deleted_count)
        """
        batch_dir = self.get_batch_dir(batch_num)
        if not batch_dir.exists():
            return 0, 0

        hdf5_deleted = 0
        videos_deleted = 0

        if delete_hdf5:
            for f in batch_dir.glob("*.hdf5"):
                f.unlink()
                hdf5_deleted += 1

        if delete_videos:
            video_dir = self.get_video_dir(batch_num)
            if video_dir.exists():
                for f in video_dir.glob("*.mp4"):
                    f.unlink()
                    videos_deleted += 1

        return hdf5_deleted, videos_deleted

    def get_batch_stats(self, batch_num: int) -> dict:
        """
        Get statistics for a batch.

        Returns:
            Dict with hdf5_count, video_count, total_size_mb
        """
        batch_dir = self.get_batch_dir(batch_num)
        if not batch_dir.exists():
            return {"hdf5_count": 0, "video_count": 0, "total_size_mb": 0}

        hdf5_files = list(batch_dir.glob("*.hdf5"))
        video_dir = self.get_video_dir(batch_num)
        video_files = list(video_dir.glob("*.mp4")) if video_dir.exists() else []

        total_size = sum(f.stat().st_size for f in hdf5_files + video_files)

        return {
            "hdf5_count": len(hdf5_files),
            "video_count": len(video_files),
            "total_size_mb": total_size / (1024 * 1024),
        }
