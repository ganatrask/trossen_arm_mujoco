#!/usr/bin/env python3
"""
Generate preview images of domain randomization configurations.

Creates 3x3 grid images showing all DR samples for visual verification.
For example:
  - 100 samples -> 12 images (9 samples per image, last image has 1)
  - 1000 samples -> 112 images (9 samples per image, last image has 1)

Usage:
    # Generate 100 DR samples with full randomization
    python scripts/generate_dr_preview.py --num_samples 100 --output_dir dr_preview

    # Generate 1000 samples with container position and rotation
    python scripts/generate_dr_preview.py --num_samples 1000 --output_dir dr_preview \
        --randomize_container --allow_90_rotation

    # Vary bowl count from 1-8
    python scripts/generate_dr_preview.py --num_samples 100 --min_bowls 1 --max_bowls 8
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trossen_arm_mujoco.domain_randomization.config import DomainRandomizationConfig
from trossen_arm_mujoco.domain_randomization.scene_sampler import SceneSampler

ASSETS_DIR = project_root / "trossen_arm_mujoco" / "assets"


def load_scene(scene_xml: str) -> tuple:
    """Load a MuJoCo scene and return model, data."""
    xml_path = ASSETS_DIR / scene_xml
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def apply_scene_config(model, data, scene_config):
    """Apply a scene configuration to the MuJoCo model."""
    # Apply container pose
    container_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "container")
    if container_id >= 0:
        model.body_pos[container_id] = scene_config.container_pose.position
        model.body_quat[container_id] = scene_config.container_pose.quaternion

    # Apply bowl poses
    for bowl_name, pose in scene_config.bowl_poses.items():
        bowl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bowl_name)
        if bowl_id >= 0:
            model.body_pos[bowl_id] = pose.position
            model.body_quat[bowl_id] = pose.quaternion

    mujoco.mj_forward(model, data)


def render_scene(model, data, camera_name="main_view", width=640, height=480):
    """Render the scene from a specific camera."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera_name)
    img = renderer.render()
    renderer.close()
    return img


def generate_dr_previews(
    num_samples: int,
    output_dir: Path,
    min_bowls: int = 1,
    max_bowls: int = 8,
    randomize_container: bool = False,
    allow_90_rotation: bool = False,
    position_noise: float = 0.03,
    seed: int = 42,
    grid_size: int = 3,
):
    """Generate DR preview images in grid format.

    Args:
        num_samples: Total number of DR samples to generate
        output_dir: Directory to save preview images
        min_bowls: Minimum number of bowls per sample
        max_bowls: Maximum number of bowls per sample
        randomize_container: Enable container position randomization
        allow_90_rotation: Allow 0° or 90° container rotation
        position_noise: Position noise in meters
        seed: Base random seed
        grid_size: Number of samples per row/column (default 3x3)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate number of images needed
    samples_per_image = grid_size * grid_size
    num_images = (num_samples + samples_per_image - 1) // samples_per_image

    print(f"Generating {num_samples} DR samples...")
    print(f"  - Bowl count: {min_bowls}-{max_bowls}")
    print(f"  - Container position randomization: {randomize_container}")
    print(f"  - 90° rotation: {allow_90_rotation}")
    print(f"  - Output: {num_images} images ({grid_size}x{grid_size} grid)")
    print(f"  - Output directory: {output_dir}")
    print()

    # Create DR config
    dr_config = DomainRandomizationConfig.from_cli_args(
        enable_dr=True,
        position_noise=position_noise,
        rotation_noise=0.1,
        container_rotation=0.15,
        min_bowls=min_bowls,
        max_bowls=max_bowls,
        seed=seed,
        min_spacing=0.12,
        randomize_container_position=randomize_container,
        allow_90_degree_rotation=allow_90_rotation,
    )

    all_bowls = ["bowl_1", "bowl_2", "bowl_3", "bowl_4",
                 "bowl_5", "bowl_6", "bowl_7", "bowl_8"]

    # Track statistics
    stats = {
        "total": num_samples,
        "success": 0,
        "failed": 0,
        "bowl_counts": {i: 0 for i in range(1, 9)},
    }

    sample_idx = 0
    for img_idx in tqdm(range(num_images), desc="Generating images"):
        # Calculate samples for this image
        start_sample = img_idx * samples_per_image
        end_sample = min(start_sample + samples_per_image, num_samples)
        samples_in_image = end_sample - start_sample

        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(5 * grid_size, 5 * grid_size))
        axes = axes.flatten()

        for i in range(samples_per_image):
            ax = axes[i]

            if i < samples_in_image:
                current_sample = start_sample + i
                sampler = SceneSampler(dr_config)
                sampler.set_seed(seed + current_sample)

                try:
                    scene_config = sampler.sample()

                    # Load and configure scene
                    model, data = load_scene("wxai/teleop_scene_8bowl.xml")
                    apply_scene_config(model, data, scene_config)

                    # Hide inactive bowls
                    for bowl_name in all_bowls:
                        if bowl_name not in scene_config.bowl_poses:
                            bowl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bowl_name)
                            if bowl_id >= 0:
                                model.body_pos[bowl_id] = np.array([0.0, 0.0, -1.0])

                    mujoco.mj_forward(model, data)
                    img = render_scene(model, data, camera_name="main_view", width=640, height=480)

                    ax.imshow(img)
                    ax.set_title(f"#{current_sample + 1} | {scene_config.num_bowls} bowls | {scene_config.target_bowl}",
                                fontsize=9)
                    ax.axis('off')

                    stats["success"] += 1
                    stats["bowl_counts"][scene_config.num_bowls] += 1

                except RuntimeError as e:
                    ax.text(0.5, 0.5, f"FAILED\n#{current_sample + 1}\n{str(e)[:30]}",
                           ha='center', va='center', fontsize=8, color='red')
                    ax.set_facecolor('lightgray')
                    ax.axis('off')
                    stats["failed"] += 1
            else:
                # Empty cell
                ax.axis('off')

        # Add title with info
        title = f"DR Preview - Samples {start_sample + 1}-{end_sample} of {num_samples}"
        if randomize_container:
            title += " | Container Randomized"
        if allow_90_rotation:
            title += " | 90° Rotation"
        plt.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save image
        output_path = output_dir / f"dr_preview_{img_idx + 1:04d}.png"
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

    # Print statistics
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Total samples: {stats['total']}")
    print(f"Successful: {stats['success']} ({100 * stats['success'] / stats['total']:.1f}%)")
    print(f"Failed: {stats['failed']} ({100 * stats['failed'] / stats['total']:.1f}%)")
    print(f"\nBowl count distribution:")
    for bowls, count in sorted(stats['bowl_counts'].items()):
        if count > 0:
            print(f"  {bowls} bowls: {count} ({100 * count / stats['success']:.1f}%)")
    print(f"\nImages saved to: {output_dir}")
    print(f"Total images: {num_images}")

    # Save statistics to file
    stats_path = output_dir / "stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"DR Preview Statistics\n")
        f.write(f"=" * 40 + "\n")
        f.write(f"Total samples: {stats['total']}\n")
        f.write(f"Successful: {stats['success']}\n")
        f.write(f"Failed: {stats['failed']}\n")
        f.write(f"\nConfiguration:\n")
        f.write(f"  min_bowls: {min_bowls}\n")
        f.write(f"  max_bowls: {max_bowls}\n")
        f.write(f"  randomize_container: {randomize_container}\n")
        f.write(f"  allow_90_rotation: {allow_90_rotation}\n")
        f.write(f"  position_noise: {position_noise}\n")
        f.write(f"  seed: {seed}\n")
        f.write(f"\nBowl count distribution:\n")
        for bowls, count in sorted(stats['bowl_counts'].items()):
            if count > 0:
                f.write(f"  {bowls} bowls: {count}\n")

    print(f"Statistics saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DR preview images for visual verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 samples with default settings
  python scripts/generate_dr_preview.py --num_samples 100

  # Full randomization with container position and rotation
  python scripts/generate_dr_preview.py --num_samples 1000 \\
      --randomize_container --allow_90_rotation

  # Fixed 4 bowls
  python scripts/generate_dr_preview.py --num_samples 50 \\
      --min_bowls 4 --max_bowls 4
        """
    )

    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of DR samples to generate (default: 100)")
    parser.add_argument("--output_dir", type=str, default="dr_preview",
                       help="Output directory for preview images (default: dr_preview)")
    parser.add_argument("--min_bowls", type=int, default=1,
                       help="Minimum number of bowls (default: 1)")
    parser.add_argument("--max_bowls", type=int, default=8,
                       help="Maximum number of bowls (default: 8)")
    parser.add_argument("--randomize_container", action="store_true",
                       help="Enable container position randomization (5 slots)")
    parser.add_argument("--allow_90_rotation", action="store_true",
                       help="Allow 0° or 90° container rotation")
    parser.add_argument("--position_noise", type=float, default=0.03,
                       help="Position noise in meters (default: 0.03)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed (default: 42)")
    parser.add_argument("--grid_size", type=int, default=3,
                       help="Grid size (NxN samples per image, default: 3)")

    args = parser.parse_args()

    generate_dr_previews(
        num_samples=args.num_samples,
        output_dir=Path(args.output_dir),
        min_bowls=args.min_bowls,
        max_bowls=args.max_bowls,
        randomize_container=args.randomize_container,
        allow_90_rotation=args.allow_90_rotation,
        position_noise=args.position_noise,
        seed=args.seed,
        grid_size=args.grid_size,
    )


if __name__ == "__main__":
    main()
