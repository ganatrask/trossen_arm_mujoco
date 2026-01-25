#!/usr/bin/env python3
"""
Unified domain randomization visualization tool.

This script combines all DR visualization capabilities into a single tool
with multiple modes for different use cases.

Modes:
    preview     - Batch generation of DR samples with grid output (for dataset verification)
    compare     - Side-by-side nominal vs randomized comparison
    samples     - Multiple random samples for a fixed configuration
    variations  - Container position and rotation variations
    test        - Success rate testing for different configurations

Usage:
    # Batch preview (generates grid images to disk)
    python scripts/visualize_dr.py preview --num_samples 100 --output_dir dr_preview

    # Compare nominal vs randomized for all bowl counts
    python scripts/visualize_dr.py compare

    # Show 9 random samples with 4 bowls
    python scripts/visualize_dr.py samples --num_bowls 4 --render_3d

    # Show container position/rotation variations
    python scripts/visualize_dr.py variations

    # Test sampling success rates
    python scripts/visualize_dr.py test

    # Save any visualization to file
    python scripts/visualize_dr.py samples --num_bowls 6 --save output.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trossen_arm_mujoco.domain_randomization.config import (
    DomainRandomizationConfig,
    NOMINAL_POSITIONS,
    CONTAINER_SLOT_POSITIONS,
    IDENTITY_QUAT,
    ObjectPose,
    SceneConfiguration,
)
from trossen_arm_mujoco.domain_randomization.scene_sampler import SceneSampler
from trossen_arm_mujoco.domain_randomization.viz_utils import (
    ALL_BOWL_NAMES,
    load_scene,
    apply_scene_config,
    hide_inactive_bowls,
    render_scene,
    create_2d_layout_plot,
    yaw_to_quaternion,
)


# =============================================================================
# Mode: preview - Batch generation with grid output
# =============================================================================

def run_preview_mode(args):
    """Generate DR preview images in grid format.

    Creates multiple grid images showing DR samples for visual verification.
    For example: 100 samples -> 12 images (9 samples per image).
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate number of images needed
    samples_per_image = args.grid_size * args.grid_size
    num_images = (args.num_samples + samples_per_image - 1) // samples_per_image

    print(f"Generating {args.num_samples} DR samples...")
    print(f"  - Bowl count: {args.min_bowls}-{args.max_bowls}")
    print(f"  - Container position randomization: {args.randomize_container}")
    print(f"  - 90° rotation: {args.allow_90_rotation}")
    print(f"  - Output: {num_images} images ({args.grid_size}x{args.grid_size} grid)")
    print(f"  - Output directory: {output_dir}")
    print()

    # Create DR config
    dr_config = DomainRandomizationConfig.from_cli_args(
        enable_dr=True,
        position_noise=args.position_noise,
        rotation_noise=0.1,
        container_rotation=0.15,
        min_bowls=args.min_bowls,
        max_bowls=args.max_bowls,
        seed=args.seed,
        min_spacing=args.min_spacing,
        randomize_container_position=args.randomize_container,
        allow_90_degree_rotation=args.allow_90_rotation,
    )

    # Track statistics
    stats = {
        "total": args.num_samples,
        "success": 0,
        "failed": 0,
        "bowl_counts": {i: 0 for i in range(1, 9)},
    }

    for img_idx in tqdm(range(num_images), desc="Generating images"):
        # Calculate samples for this image
        start_sample = img_idx * samples_per_image
        end_sample = min(start_sample + samples_per_image, args.num_samples)
        samples_in_image = end_sample - start_sample

        # Create figure
        fig, axes = plt.subplots(
            args.grid_size, args.grid_size,
            figsize=(5 * args.grid_size, 5 * args.grid_size)
        )
        axes = axes.flatten()

        for i in range(samples_per_image):
            ax = axes[i]

            if i < samples_in_image:
                current_sample = start_sample + i
                sampler = SceneSampler(dr_config)
                sampler.set_seed(args.seed + current_sample)

                try:
                    scene_config = sampler.sample()

                    # Load and configure scene
                    model, data = load_scene("wxai/teleop_scene_8bowl.xml")
                    apply_scene_config(model, data, scene_config)
                    hide_inactive_bowls(model, scene_config.active_bowls)

                    mujoco.mj_forward(model, data)
                    img = render_scene(model, data, camera_name="main_view")

                    ax.imshow(img)
                    ax.set_title(
                        f"#{current_sample + 1} | {scene_config.num_bowls} bowls | "
                        f"{scene_config.target_bowl}",
                        fontsize=9
                    )
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
        title = f"DR Preview - Samples {start_sample + 1}-{end_sample} of {args.num_samples}"
        if args.randomize_container:
            title += " | Container Randomized"
        if args.allow_90_rotation:
            title += " | 90deg Rotation"
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
            print(f"  {bowls} bowls: {count} ({100 * count / max(1, stats['success']):.1f}%)")
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
        f.write(f"  min_bowls: {args.min_bowls}\n")
        f.write(f"  max_bowls: {args.max_bowls}\n")
        f.write(f"  randomize_container: {args.randomize_container}\n")
        f.write(f"  allow_90_rotation: {args.allow_90_rotation}\n")
        f.write(f"  position_noise: {args.position_noise}\n")
        f.write(f"  min_spacing: {args.min_spacing}\n")
        f.write(f"  seed: {args.seed}\n")
        f.write(f"\nBowl count distribution:\n")
        for bowls, count in sorted(stats['bowl_counts'].items()):
            if count > 0:
                f.write(f"  {bowls} bowls: {count}\n")

    print(f"Statistics saved to: {stats_path}")


# =============================================================================
# Mode: compare - Side-by-side nominal vs randomized
# =============================================================================

def run_compare_mode(args):
    """Generate visualization comparing nominal vs randomized configurations.

    Shows 8 configurations: 1/4/6/8 bowls × nominal/randomized,
    with both 2D layout and 3D render for each.
    """
    fig = plt.figure(figsize=(20, 16))

    # Configuration variants to show
    configs = [
        # (num_bowls, use_randomization, seed, title)
        (1, False, args.seed, "1 Bowl - Nominal"),
        (1, True, args.seed, "1 Bowl - Randomized"),
        (4, False, args.seed, "4 Bowls - Nominal"),
        (4, True, args.seed, "4 Bowls - Randomized"),
        (6, False, args.seed, "6 Bowls - Nominal"),
        (6, True, args.seed, "6 Bowls - Randomized"),
        (8, False, args.seed, "8 Bowls - Nominal"),
        (8, True, args.seed, "8 Bowls - Randomized"),
    ]

    # Create 4x4 grid: 2D layout on left, 3D render on right for each config
    for idx, (num_bowls, use_random, seed, title) in enumerate(configs):
        print(f"Generating: {title}")

        # Create DR config
        dr_config = DomainRandomizationConfig.from_cli_args(
            enable_dr=use_random,
            position_noise=args.position_noise,
            rotation_noise=0.1,
            container_rotation=0.15,
            min_bowls=num_bowls,
            max_bowls=num_bowls,
            seed=seed,
            min_spacing=args.min_spacing,
        )

        # Sample scene configuration
        sampler = SceneSampler(dr_config)
        try:
            scene_config = sampler.sample(target_bowl="bowl_1")
        except RuntimeError as e:
            print(f"  Failed to sample: {e}")
            ax = fig.add_subplot(4, 4, idx * 2 + 1)
            ax.text(0.5, 0.5, f"FAILED\n{e}", ha='center', va='center', fontsize=10)
            ax.set_title(title + " (FAILED)")
            continue

        # 2D layout plot
        ax_2d = fig.add_subplot(4, 4, idx * 2 + 1)
        create_2d_layout_plot(scene_config, f"{title}\n(2D Layout)", ax_2d)

        # 3D render
        try:
            model, data = load_scene(scene_config.scene_xml)
            apply_scene_config(model, data, scene_config)
            img = render_scene(model, data)

            ax_3d = fig.add_subplot(4, 4, idx * 2 + 2)
            ax_3d.imshow(img)
            ax_3d.set_title(f"{title}\n(3D Render)")
            ax_3d.axis('off')
        except Exception as e:
            print(f"  Render failed: {e}")
            ax_3d = fig.add_subplot(4, 4, idx * 2 + 2)
            ax_3d.text(0.5, 0.5, f"Render failed:\n{e}", ha='center', va='center')
            ax_3d.set_title(title + " (Render Failed)")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved to: {args.save}")
    else:
        output_path = project_root / "dr_config_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
        plt.show()


# =============================================================================
# Mode: samples - Multiple random samples
# =============================================================================

def run_samples_mode(args):
    """Show multiple random samples for the same bowl count.

    Displays a grid of samples with either 2D layouts or 3D renders.
    """
    # Determine grid size
    rows = int(np.ceil(np.sqrt(args.num_samples)))
    cols = int(np.ceil(args.num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if args.num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    dr_config = DomainRandomizationConfig.from_cli_args(
        enable_dr=True,
        position_noise=args.position_noise,
        rotation_noise=0.1,
        container_rotation=0.15,
        min_bowls=args.num_bowls,
        max_bowls=args.num_bowls,
        seed=None,  # Random
        min_spacing=args.min_spacing,
        randomize_container_position=args.randomize_container,
        allow_90_degree_rotation=args.allow_90_rotation,
    )

    for i in range(args.num_samples):
        sampler = SceneSampler(dr_config)
        sample_seed = i * 100
        sampler.set_seed(sample_seed)

        try:
            scene_config = sampler.sample(target_bowl="bowl_1")

            if args.render_3d:
                # Render 3D view
                model, data = load_scene("wxai/teleop_scene_8bowl.xml")
                apply_scene_config(model, data, scene_config)
                hide_inactive_bowls(model, scene_config.active_bowls)

                mujoco.mj_forward(model, data)
                img = render_scene(model, data, camera_name="main_view")

                axes[i].imshow(img)
                axes[i].set_title(f"Sample {i+1} (seed={sample_seed})", fontsize=10)
                axes[i].axis('off')
            else:
                # 2D layout
                create_2d_layout_plot(
                    scene_config,
                    f"Sample {i+1} (seed={sample_seed})",
                    axes[i]
                )
        except RuntimeError as e:
            axes[i].text(0.5, 0.5, f"Failed:\n{e}", ha='center', va='center', fontsize=8)
            axes[i].set_title(f"Sample {i+1} - FAILED")

    # Hide extra axes if num_samples doesn't fill the grid
    for i in range(args.num_samples, len(axes)):
        axes[i].axis('off')

    container_mode = "Container Randomized" if args.randomize_container else "Container Fixed"
    rotation_mode = " + 90deg Rotation" if args.allow_90_rotation else ""
    view_mode = "3D" if args.render_3d else "2D"
    plt.suptitle(
        f"Multiple Randomized Samples with {args.num_bowls} Bowls "
        f"({container_mode}{rotation_mode}) - {view_mode}",
        fontsize=14
    )
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved to: {args.save}")
    else:
        suffix = "_container_rand" if args.randomize_container else ""
        suffix += "_90rot" if args.allow_90_rotation else ""
        suffix += "_3d" if args.render_3d else ""
        output_path = project_root / f"dr_samples_{args.num_bowls}bowls{suffix}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
        plt.show()


# =============================================================================
# Mode: variations - Container position and rotation variations
# =============================================================================

def run_variations_mode(args):
    """Show all major scene variations in a single image with 3D renders.

    Displays preset variations with container at different positions
    and 0°/90° rotations.
    """
    # Define all variations to show
    variations = [
        # Row 1: Container at default position with 0-degree and 90-degree rotation
        {"title": "Left (0deg) - 4 Bowls Bottom", "container_slot": 0,
         "bowls": ["bowl_1", "bowl_2", "bowl_3", "bowl_4"], "seed": 10, "base_rotation": 0},
        {"title": "Left (90deg) - 4 Bowls Bottom", "container_slot": 0,
         "bowls": ["bowl_1", "bowl_2", "bowl_3", "bowl_4"], "seed": 15, "base_rotation": np.pi/2},
        {"title": "Left (0deg) - 8 Bowls All", "container_slot": 0,
         "bowls": ALL_BOWL_NAMES.copy(), "seed": 40, "base_rotation": 0},

        # Row 2: Container at bottom bowl area
        {"title": "Bottom (0deg) - 4 Bowls Top", "container_slot": 1,
         "bowls": ["bowl_5", "bowl_6", "bowl_7", "bowl_8"], "seed": 50, "base_rotation": 0},
        {"title": "Bottom (90deg) - 4 Bowls Top", "container_slot": 1,
         "bowls": ["bowl_5", "bowl_6", "bowl_7", "bowl_8"], "seed": 55, "base_rotation": np.pi/2},
        {"title": "Bottom (0deg) - 2 Bowls Top", "container_slot": 1,
         "bowls": ["bowl_5", "bowl_7"], "seed": 60, "base_rotation": 0},

        # Row 3: Container at top bowl area
        {"title": "Top (0deg) - 4 Bowls Bottom", "container_slot": 2,
         "bowls": ["bowl_1", "bowl_2", "bowl_3", "bowl_4"], "seed": 90, "base_rotation": 0},
        {"title": "Top (90deg) - 4 Bowls Bottom", "container_slot": 2,
         "bowls": ["bowl_1", "bowl_2", "bowl_3", "bowl_4"], "seed": 95, "base_rotation": np.pi/2},
        {"title": "Top (90deg) - 2 Bowls Bottom", "container_slot": 2,
         "bowls": ["bowl_2", "bowl_4"], "seed": 100, "base_rotation": np.pi/2},
    ]

    # Create figure with 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Rotation noise settings (matching config defaults)
    container_rotation_noise = 0.15
    bowl_rotation_noise = 0.1

    for idx, var in enumerate(variations):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Load fresh model for each variation
        model, data = load_scene("wxai/teleop_scene_8bowl.xml")

        # Create scene configuration with rotation
        np.random.seed(var["seed"])
        container_pos = CONTAINER_SLOT_POSITIONS[var["container_slot"]].copy()
        container_pos[:2] += np.random.uniform(-0.02, 0.02, 2)

        # Apply base rotation (0 or 90 deg) plus small noise
        base_rotation = var.get("base_rotation", 0)
        container_yaw = base_rotation + np.random.uniform(
            -container_rotation_noise, container_rotation_noise
        )
        container_pose = ObjectPose(
            position=container_pos,
            quaternion=yaw_to_quaternion(container_yaw)
        )

        bowl_poses = {}
        for bowl_name in var["bowls"]:
            bowl_pos = NOMINAL_POSITIONS[bowl_name].copy()
            bowl_pos[:2] += np.random.uniform(-0.02, 0.02, 2)
            bowl_yaw = np.random.uniform(-bowl_rotation_noise, bowl_rotation_noise)
            bowl_poses[bowl_name] = ObjectPose(
                position=bowl_pos,
                quaternion=yaw_to_quaternion(bowl_yaw)
            )

        scene_config = SceneConfiguration(
            container_pose=container_pose,
            bowl_poses=bowl_poses,
            active_bowls=var["bowls"],
            target_bowl=var["bowls"][0],
            num_bowls=len(var["bowls"]),
            scene_xml="wxai/teleop_scene_8bowl.xml",
            seed=var["seed"],
        )

        # Apply scene config
        apply_scene_config(model, data, scene_config)
        hide_inactive_bowls(model, var["bowls"])

        mujoco.mj_forward(model, data)

        # Render 3D view
        img = render_scene(model, data, camera_name="main_view")

        ax.imshow(img)
        ax.set_title(var["title"], fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle("All Major Scene Variations (3D Renders)", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved to: {args.save}")
    else:
        output_path = project_root / "dr_all_variations_3d.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")
        plt.show()


# =============================================================================
# Mode: test - Success rate testing
# =============================================================================

def run_test_mode(args):
    """Test success rate for different configurations.

    Runs multiple sampling attempts for various bowl counts and spacing
    settings to identify configurations that may cause failures.
    """
    print("\n" + "=" * 60)
    print("Testing sampling success rates for different configurations")
    print("=" * 60)

    test_configs = [
        # (num_bowls, min_spacing, description)
        (1, 0.12, "1 bowl, 12cm spacing"),
        (4, 0.12, "4 bowls, 12cm spacing"),
        (4, 0.10, "4 bowls, 10cm spacing"),
        (6, 0.12, "6 bowls, 12cm spacing"),
        (6, 0.10, "6 bowls, 10cm spacing"),
        (8, 0.12, "8 bowls, 12cm spacing"),
        (8, 0.10, "8 bowls, 10cm spacing"),
        (8, 0.08, "8 bowls, 8cm spacing"),
    ]

    num_trials = args.num_trials

    results = []
    for num_bowls, min_spacing, desc in test_configs:
        dr_config = DomainRandomizationConfig.from_cli_args(
            enable_dr=True,
            position_noise=args.position_noise,
            rotation_noise=0.1,
            container_rotation=0.15,
            min_bowls=num_bowls,
            max_bowls=num_bowls,
            seed=None,
            min_spacing=min_spacing,
        )

        successes = 0
        for trial in range(num_trials):
            sampler = SceneSampler(dr_config)
            sampler.set_seed(trial)
            try:
                sampler.sample(target_bowl="bowl_1")
                successes += 1
            except RuntimeError:
                pass

        success_rate = successes / num_trials * 100
        status = "OK" if success_rate > 90 else "WARN" if success_rate > 50 else "FAIL"
        print(f"[{status}] {desc}: {successes}/{num_trials} ({success_rate:.0f}%)")

        results.append({
            "config": desc,
            "successes": successes,
            "trials": num_trials,
            "rate": success_rate,
            "status": status,
        })

    print("\n" + "=" * 60)

    # Save results if requested
    if args.save:
        with open(args.save, 'w') as f:
            f.write("DR Sampling Success Rate Test Results\n")
            f.write("=" * 50 + "\n\n")
            for r in results:
                f.write(f"[{r['status']}] {r['config']}: "
                       f"{r['successes']}/{r['trials']} ({r['rate']:.0f}%)\n")
        print(f"Results saved to: {args.save}")


# =============================================================================
# Main entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified domain randomization visualization tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch preview (generates grid images to disk)
  python scripts/visualize_dr.py preview --num_samples 100 --output_dir dr_preview

  # Compare nominal vs randomized for all bowl counts
  python scripts/visualize_dr.py compare

  # Show 9 random samples with 4 bowls (3D renders)
  python scripts/visualize_dr.py samples --num_bowls 4 --render_3d

  # Show container position/rotation variations
  python scripts/visualize_dr.py variations

  # Test sampling success rates
  python scripts/visualize_dr.py test

  # Save any visualization to file
  python scripts/visualize_dr.py samples --num_bowls 6 --save output.png
        """
    )

    subparsers = parser.add_subparsers(dest="mode", help="Visualization mode")

    # Common arguments for all modes
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    common.add_argument("--position_noise", type=float, default=0.03,
                       help="Position noise in meters (default: 0.03)")
    common.add_argument("--min_spacing", type=float, default=0.12,
                       help="Minimum object spacing in meters (default: 0.12)")
    common.add_argument("--save", type=str, default=None,
                       help="Save output to file instead of displaying")

    # Preview mode
    preview_parser = subparsers.add_parser(
        "preview", parents=[common],
        help="Batch generation with grid output"
    )
    preview_parser.add_argument("--num_samples", type=int, default=100,
                               help="Number of samples to generate (default: 100)")
    preview_parser.add_argument("--output_dir", type=str, default="dr_preview",
                               help="Output directory (default: dr_preview)")
    preview_parser.add_argument("--min_bowls", type=int, default=1,
                               help="Minimum bowl count (default: 1)")
    preview_parser.add_argument("--max_bowls", type=int, default=8,
                               help="Maximum bowl count (default: 8)")
    preview_parser.add_argument("--randomize_container", action="store_true",
                               help="Enable container position randomization")
    preview_parser.add_argument("--allow_90_rotation", action="store_true",
                               help="Allow 0/90 degree container rotation")
    preview_parser.add_argument("--grid_size", type=int, default=3,
                               help="Grid size NxN per image (default: 3)")

    # Compare mode
    compare_parser = subparsers.add_parser(
        "compare", parents=[common],
        help="Side-by-side nominal vs randomized comparison"
    )

    # Samples mode
    samples_parser = subparsers.add_parser(
        "samples", parents=[common],
        help="Multiple random samples for fixed configuration"
    )
    samples_parser.add_argument("--num_bowls", type=int, default=4,
                               help="Number of bowls (default: 4)")
    samples_parser.add_argument("--num_samples", type=int, default=9,
                               help="Number of samples (default: 9)")
    samples_parser.add_argument("--randomize_container", action="store_true",
                               help="Enable container position randomization")
    samples_parser.add_argument("--allow_90_rotation", action="store_true",
                               help="Allow 0/90 degree container rotation")
    samples_parser.add_argument("--render_3d", action="store_true",
                               help="Render 3D views instead of 2D layouts")

    # Variations mode
    variations_parser = subparsers.add_parser(
        "variations", parents=[common],
        help="Container position and rotation variations"
    )

    # Test mode
    test_parser = subparsers.add_parser(
        "test", parents=[common],
        help="Success rate testing"
    )
    test_parser.add_argument("--num_trials", type=int, default=50,
                            help="Number of trials per configuration (default: 50)")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    # Run the appropriate mode
    if args.mode == "preview":
        run_preview_mode(args)
    elif args.mode == "compare":
        run_compare_mode(args)
    elif args.mode == "samples":
        run_samples_mode(args)
    elif args.mode == "variations":
        run_variations_mode(args)
    elif args.mode == "test":
        run_test_mode(args)


if __name__ == "__main__":
    main()
