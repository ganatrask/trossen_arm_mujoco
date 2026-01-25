#!/usr/bin/env python3
"""
Visualization script to show domain randomization bowl configurations.

Creates a grid of images showing different configurations:
- Different bowl counts (1, 4, 6, 8)
- Nominal positions vs randomized positions
- Shows workspace bounds and spacing constraints
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trossen_arm_mujoco.domain_randomization.config import (
    DomainRandomizationConfig,
    NOMINAL_POSITIONS,
)
from trossen_arm_mujoco.domain_randomization.scene_sampler import SceneSampler

ASSETS_DIR = project_root / "trossen_arm_mujoco" / "assets"


def load_scene(scene_xml: str) -> tuple:
    """Load a MuJoCo scene and return model, data."""
    xml_path = ASSETS_DIR / scene_xml
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def apply_scene_config(model, data, scene_config):
    """Apply a scene configuration to the MuJoCo model.

    For static bodies (no joints), we modify model.body_pos and model.body_quat.
    """
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

    # Recompute kinematics with new positions
    mujoco.mj_forward(model, data)


def render_scene(model, data, camera_name="main_view", width=640, height=480):
    """Render the scene from a specific camera."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera_name)
    img = renderer.render()
    renderer.close()
    return img


def create_2d_layout_plot(scene_config, title, ax, show_workspace=True):
    """Create a 2D top-down view showing object positions."""
    ax.set_aspect('equal')
    ax.set_xlim(-0.75, 0.15)
    ax.set_ylim(-0.35, 0.35)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Draw workspace bounds
    if show_workspace:
        workspace_x = [-0.58, 0.1]
        workspace_y = [-0.28, 0.28]
        rect = plt.Rectangle(
            (workspace_x[0], workspace_y[0]),
            workspace_x[1] - workspace_x[0],
            workspace_y[1] - workspace_y[0],
            fill=False, edgecolor='green', linestyle='--', linewidth=2,
            label='Workspace bounds'
        )
        ax.add_patch(rect)

    # Draw robot base
    robot_circle = plt.Circle((0, 0), 0.05, color='gray', alpha=0.5, label='Robot base')
    ax.add_patch(robot_circle)

    # Draw container (rectangle) - accounting for 90-degree rotation
    # Container local size: 0.1625 x 0.1325, after 90-deg Z rotation: axes swap
    # World frame: half_x=0.14, half_y=0.17
    container_pos = scene_config.container_pose.position
    container_half_x = 0.14
    container_half_y = 0.17
    container_rect = plt.Rectangle(
        (container_pos[0] - container_half_x, container_pos[1] - container_half_y),
        container_half_x * 2, container_half_y * 2,
        fill=True, facecolor='lightblue', edgecolor='blue', linewidth=2,
        label='Container'
    )
    ax.add_patch(container_rect)

    # Draw bowls (circles, ~10cm diameter)
    bowl_colors = plt.cm.tab10(np.linspace(0, 1, 8))
    for i, (bowl_name, pose) in enumerate(scene_config.bowl_poses.items()):
        pos = pose.position
        circle = plt.Circle(
            (pos[0], pos[1]), 0.05,  # 10cm diameter = 5cm radius
            fill=True, facecolor=bowl_colors[i], edgecolor='black',
            linewidth=1.5, alpha=0.7
        )
        ax.add_patch(circle)
        ax.annotate(bowl_name.replace('bowl_', 'B'), (pos[0], pos[1]),
                   ha='center', va='center', fontsize=8, fontweight='bold')

    # Draw min spacing circles (dashed)
    min_spacing = 0.12
    for bowl_name, pose in scene_config.bowl_poses.items():
        pos = pose.position
        spacing_circle = plt.Circle(
            (pos[0], pos[1]), min_spacing / 2,
            fill=False, edgecolor='red', linestyle=':', linewidth=0.5, alpha=0.3
        )
        ax.add_patch(spacing_circle)

    ax.legend(loc='upper right', fontsize=6)


def visualize_all_configs():
    """Generate visualization of all domain randomization configurations."""
    fig = plt.figure(figsize=(20, 16))

    # Configuration variants to show
    configs = [
        # (num_bowls, use_randomization, seed, title)
        (1, False, 42, "1 Bowl - Nominal"),
        (1, True, 42, "1 Bowl - Randomized"),
        (4, False, 42, "4 Bowls - Nominal"),
        (4, True, 42, "4 Bowls - Randomized"),
        (6, False, 42, "6 Bowls - Nominal"),
        (6, True, 42, "6 Bowls - Randomized"),
        (8, False, 42, "8 Bowls - Nominal"),
        (8, True, 42, "8 Bowls - Randomized"),
    ]

    # Create 4x4 grid: 2D layout on left, 3D render on right for each config
    for idx, (num_bowls, use_random, seed, title) in enumerate(configs):
        print(f"Generating: {title}")

        # Create DR config
        dr_config = DomainRandomizationConfig.from_cli_args(
            enable_dr=use_random,
            position_noise=0.03,
            rotation_noise=0.1,
            container_rotation=0.15,
            min_bowls=num_bowls,
            max_bowls=num_bowls,
            seed=seed,
            min_spacing=0.12,
        )

        # Sample scene configuration
        sampler = SceneSampler(dr_config)
        try:
            scene_config = sampler.sample(target_bowl="bowl_1")
        except RuntimeError as e:
            print(f"  Failed to sample: {e}")
            # Create a placeholder plot
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
    output_path = project_root / "dr_config_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.show()


def visualize_multiple_samples(num_bowls=4, num_samples=9, randomize_container=False,
                               allow_90_rotation=False, render_3d=False):
    """Show multiple random samples for the same bowl count.

    Args:
        num_bowls: Number of bowls per sample
        num_samples: Number of samples to generate (default 9 for 3x3 grid)
        randomize_container: Enable container position randomization
        allow_90_rotation: Allow 0° or 90° container rotation
        render_3d: If True, render 3D views instead of 2D layouts
    """
    # Determine grid size
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    dr_config = DomainRandomizationConfig.from_cli_args(
        enable_dr=True,
        position_noise=0.03,
        rotation_noise=0.1,
        container_rotation=0.15,
        min_bowls=num_bowls,
        max_bowls=num_bowls,
        seed=None,  # Random
        min_spacing=0.12,
        randomize_container_position=randomize_container,
        allow_90_degree_rotation=allow_90_rotation,
    )

    all_bowls = ["bowl_1", "bowl_2", "bowl_3", "bowl_4",
                 "bowl_5", "bowl_6", "bowl_7", "bowl_8"]

    for i in range(num_samples):
        sampler = SceneSampler(dr_config)
        sampler.set_seed(i * 100)  # Different seed for each sample

        try:
            scene_config = sampler.sample(target_bowl="bowl_1")

            if render_3d:
                # Render 3D view
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

                axes[i].imshow(img)
                axes[i].set_title(f"Sample {i+1} (seed={i*100})", fontsize=10)
                axes[i].axis('off')
            else:
                # 2D layout
                create_2d_layout_plot(
                    scene_config,
                    f"Sample {i+1} (seed={i*100})",
                    axes[i]
                )
        except RuntimeError as e:
            axes[i].text(0.5, 0.5, f"Failed:\n{e}", ha='center', va='center', fontsize=8)
            axes[i].set_title(f"Sample {i+1} - FAILED")

    # Hide extra axes if num_samples doesn't fill the grid
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    container_mode = "Container Randomized" if randomize_container else "Container Fixed"
    rotation_mode = " + 90° Rotation" if allow_90_rotation else ""
    view_mode = "3D" if render_3d else "2D"
    plt.suptitle(f"Multiple Randomized Samples with {num_bowls} Bowls ({container_mode}{rotation_mode}) - {view_mode}", fontsize=14)
    plt.tight_layout()
    suffix = "_container_rand" if randomize_container else ""
    suffix += "_90rot" if allow_90_rotation else ""
    suffix += "_3d" if render_3d else ""
    output_path = project_root / f"dr_samples_{num_bowls}bowls{suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.show()


def test_sampling_success_rate():
    """Test success rate for different configurations."""
    print("\n" + "="*60)
    print("Testing sampling success rates for different configurations")
    print("="*60)

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

    num_trials = 50

    for num_bowls, min_spacing, desc in test_configs:
        dr_config = DomainRandomizationConfig.from_cli_args(
            enable_dr=True,
            position_noise=0.03,
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


def visualize_all_variations():
    """Show all major scene variations in a single image with 3D renders.

    Uses the actual SceneSampler to generate configurations with proper
    position AND rotation randomization.
    """
    from trossen_arm_mujoco.domain_randomization.config import (
        CONTAINER_SLOT_POSITIONS,
        NOMINAL_POSITIONS,
        ObjectPose,
        SceneConfiguration,
        IDENTITY_QUAT,
        ContainerPoseConfig,
    )

    def yaw_to_quaternion(yaw: float) -> np.ndarray:
        """Convert yaw angle to quaternion [w, x, y, z]."""
        return np.array([
            np.cos(yaw / 2),
            0.0,
            0.0,
            np.sin(yaw / 2),
        ])

    # Define all variations to show - now with 0 and 90-degree rotation
    # base_rotation: 0 = default orientation, pi/2 = 90-degree rotation
    variations = [
        # Row 1: Container at default position with 0-degree and 90-degree rotation
        {"title": "Left (0°) - 4 Bowls Bottom", "container_slot": 0, "bowls": ["bowl_1", "bowl_2", "bowl_3", "bowl_4"], "seed": 10, "base_rotation": 0},
        {"title": "Left (90°) - 4 Bowls Bottom", "container_slot": 0, "bowls": ["bowl_1", "bowl_2", "bowl_3", "bowl_4"], "seed": 15, "base_rotation": np.pi/2},
        {"title": "Left (0°) - 8 Bowls All", "container_slot": 0, "bowls": ["bowl_1", "bowl_2", "bowl_3", "bowl_4", "bowl_5", "bowl_6", "bowl_7", "bowl_8"], "seed": 40, "base_rotation": 0},

        # Row 2: Container at bottom bowl area - with 0 and 90-degree rotation
        {"title": "Bottom (0°) - 4 Bowls Top", "container_slot": 1, "bowls": ["bowl_5", "bowl_6", "bowl_7", "bowl_8"], "seed": 50, "base_rotation": 0},
        {"title": "Bottom (90°) - 4 Bowls Top", "container_slot": 1, "bowls": ["bowl_5", "bowl_6", "bowl_7", "bowl_8"], "seed": 55, "base_rotation": np.pi/2},
        {"title": "Bottom (0°) - 2 Bowls Top", "container_slot": 1, "bowls": ["bowl_5", "bowl_7"], "seed": 60, "base_rotation": 0},

        # Row 3: Container at top bowl area - with 0 and 90-degree rotation
        {"title": "Top (0°) - 4 Bowls Bottom", "container_slot": 2, "bowls": ["bowl_1", "bowl_2", "bowl_3", "bowl_4"], "seed": 90, "base_rotation": 0},
        {"title": "Top (90°) - 4 Bowls Bottom", "container_slot": 2, "bowls": ["bowl_1", "bowl_2", "bowl_3", "bowl_4"], "seed": 95, "base_rotation": np.pi/2},
        {"title": "Top (90°) - 2 Bowls Bottom", "container_slot": 2, "bowls": ["bowl_2", "bowl_4"], "seed": 100, "base_rotation": np.pi/2},
    ]

    # Create figure with 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # All bowls in scene for reference (we'll position inactive ones off-screen)
    all_bowls = ["bowl_1", "bowl_2", "bowl_3", "bowl_4", "bowl_5", "bowl_6", "bowl_7", "bowl_8"]

    # Rotation noise settings (matching config defaults)
    container_rotation_noise = 0.15  # ±0.15 rad (~9 degrees)
    bowl_rotation_noise = 0.1  # ±0.1 rad (~6 degrees)

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
        container_yaw = base_rotation + np.random.uniform(-container_rotation_noise, container_rotation_noise)
        container_pose = ObjectPose(
            position=container_pos,
            quaternion=yaw_to_quaternion(container_yaw)
        )

        bowl_poses = {}
        for bowl_name in var["bowls"]:
            bowl_pos = NOMINAL_POSITIONS[bowl_name].copy()
            bowl_pos[:2] += np.random.uniform(-0.02, 0.02, 2)
            # Apply random yaw rotation to bowls
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

        # Move inactive bowls off-screen (below the table)
        for bowl_name in all_bowls:
            if bowl_name not in var["bowls"]:
                bowl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bowl_name)
                if bowl_id >= 0:
                    model.body_pos[bowl_id] = np.array([0.0, 0.0, -1.0])  # Hidden below

        mujoco.mj_forward(model, data)

        # Render 3D view
        img = render_scene(model, data, camera_name="main_view", width=640, height=480)

        ax.imshow(img)
        ax.set_title(var["title"], fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle("All Major Scene Variations (3D Renders)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = project_root / "dr_all_variations_3d.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize domain randomization configurations")
    parser.add_argument("--mode", choices=["all", "samples", "test", "variations"], default="all",
                       help="Visualization mode: all=bowl configs, samples=random samples, test=success rates, variations=3D preset variations")
    parser.add_argument("--num_bowls", type=int, default=4,
                       help="Number of bowls for 'samples' mode (1-8)")
    parser.add_argument("--num_samples", type=int, default=9,
                       help="Number of samples to generate in 'samples' mode")
    parser.add_argument("--randomize_container", action="store_true",
                       help="Enable container position randomization (5 slots)")
    parser.add_argument("--allow_90_rotation", action="store_true",
                       help="Allow 0° or 90° container rotation")
    parser.add_argument("--render_3d", action="store_true",
                       help="Render 3D views instead of 2D layouts (for 'samples' mode)")
    args = parser.parse_args()

    if args.mode == "all":
        visualize_all_configs()
    elif args.mode == "samples":
        visualize_multiple_samples(
            num_bowls=args.num_bowls,
            num_samples=args.num_samples,
            randomize_container=args.randomize_container,
            allow_90_rotation=args.allow_90_rotation,
            render_3d=args.render_3d,
        )
    elif args.mode == "test":
        test_sampling_success_rate()
    elif args.mode == "variations":
        visualize_all_variations()
