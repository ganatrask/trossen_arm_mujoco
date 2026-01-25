"""
Shared visualization utilities for domain randomization.

This module provides common functions used by visualization scripts
to avoid code duplication.
"""

import mujoco
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .config import (
    NOMINAL_POSITIONS,
    IDENTITY_QUAT,
    SceneConfiguration,
)

# Path to assets directory
ASSETS_DIR = Path(__file__).parent.parent / "assets"

# All bowl names (1-8)
ALL_BOWL_NAMES = [f"bowl_{i}" for i in range(1, 9)]

# Position to hide inactive objects (far below table)
HIDDEN_OBJECT_POSITION = np.array([0.0, 0.0, -10.0])

# Default camera settings
DEFAULT_CAMERA = "main_view"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


def get_available_ram_gb() -> float:
    """Get available RAM in GB.

    Returns:
        Available RAM in GB (falls back to 4GB if psutil unavailable)
    """
    if HAS_PSUTIL:
        try:
            return psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            return 4.0
    return 4.0  # Conservative fallback


def get_optimal_batch_size(
    num_samples: int,
    image_width: int = DEFAULT_WIDTH,
    image_height: int = DEFAULT_HEIGHT,
    ram_fraction: float = 0.25,
    min_batch: int = 10,
    max_batch: int = 1000,
) -> int:
    """Calculate optimal batch size for rendering based on available RAM.

    Args:
        num_samples: Total number of samples to render
        image_width: Image width in pixels (default: 640)
        image_height: Image height in pixels (default: 480)
        ram_fraction: Fraction of available RAM to use (default: 0.25)
        min_batch: Minimum batch size (default: 10)
        max_batch: Maximum batch size (default: 1000)

    Returns:
        Optimal batch size capped by num_samples
    """
    image_size_mb = (image_width * image_height * 3) / (1024 * 1024)
    available_ram_gb = get_available_ram_gb()
    batch_size = int((available_ram_gb * 1024 * ram_fraction) / image_size_mb)

    batch_size = max(min_batch, batch_size)
    batch_size = min(max_batch, batch_size)
    batch_size = min(num_samples, batch_size)

    return batch_size


def get_optimal_workers(min_workers: int = 1, max_workers: Optional[int] = None) -> int:
    """Calculate optimal number of worker processes.

    Args:
        min_workers: Minimum number of workers (default: 1)
        max_workers: Maximum number of workers (default: None, no limit)

    Returns:
        Optimal number of worker processes
    """
    cpu_count = os.cpu_count() or 4

    if HAS_PSUTIL:
        try:
            ram_bytes = psutil.virtual_memory().total
            ram_gb = ram_bytes / (1024 ** 3)
        except Exception:
            ram_gb = 8
    else:
        ram_gb = 8

    cpu_based = max(1, cpu_count - 2)
    ram_based = max(1, int(ram_gb // 4))

    workers = min(cpu_based, ram_based)

    # Apply bounds
    workers = max(workers, min_workers)
    if max_workers is not None:
        workers = min(workers, max_workers)

    return workers


def get_body_id(model, name: str) -> int:
    """Get MuJoCo body ID by name.

    Args:
        model: MuJoCo model
        name: Body name

    Returns:
        Body ID, or -1 if not found
    """
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def load_scene(scene_xml: str) -> Tuple:
    """Load a MuJoCo scene and return model, data.

    Args:
        scene_xml: Path to scene XML relative to assets directory

    Returns:
        Tuple of (model, data)
    """
    xml_path = ASSETS_DIR / scene_xml
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def apply_scene_config(model, data, scene_config: SceneConfiguration) -> None:
    """Apply a scene configuration to the MuJoCo model.

    For static bodies (no joints), we modify model.body_pos and model.body_quat.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        scene_config: Scene configuration to apply
    """
    # Apply container pose
    container_id = get_body_id(model, "container")
    if container_id >= 0:
        model.body_pos[container_id] = scene_config.container_pose.position
        model.body_quat[container_id] = scene_config.container_pose.quaternion

    # Apply bowl poses
    for bowl_name, pose in scene_config.bowl_poses.items():
        bowl_id = get_body_id(model, bowl_name)
        if bowl_id >= 0:
            model.body_pos[bowl_id] = pose.position
            model.body_quat[bowl_id] = pose.quaternion

    # Recompute kinematics with new positions
    mujoco.mj_forward(model, data)


def hide_inactive_bowls(
    model,
    active_bowls: List[str],
    all_bowls: Optional[List[str]] = None,
) -> None:
    """Hide bowls that are not active by moving them below the table.

    Args:
        model: MuJoCo model
        active_bowls: List of bowl names that should remain visible
        all_bowls: List of all bowl names to check (default: ALL_BOWL_NAMES)
    """
    if all_bowls is None:
        all_bowls = ALL_BOWL_NAMES

    for bowl_name in all_bowls:
        if bowl_name not in active_bowls:
            bowl_id = get_body_id(model, bowl_name)
            if bowl_id >= 0:
                model.body_pos[bowl_id] = HIDDEN_OBJECT_POSITION.copy()


def render_scene(
    model,
    data,
    camera_name: str = DEFAULT_CAMERA,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> np.ndarray:
    """Render the scene from a specific camera.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        camera_name: Name of camera to render from
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        RGB image as numpy array (height, width, 3)
    """
    renderer = mujoco.Renderer(model, height=height, width=width)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera_name)
    img = renderer.render()
    renderer.close()
    return img


def yaw_to_quaternion(yaw: float) -> np.ndarray:
    """Convert yaw angle to quaternion [w, x, y, z].

    Args:
        yaw: Yaw angle in radians

    Returns:
        Quaternion as numpy array [w, x, y, z]
    """
    return np.array([
        np.cos(yaw / 2),
        0.0,
        0.0,
        np.sin(yaw / 2),
    ])


def create_2d_layout_plot(
    scene_config: SceneConfiguration,
    title: str,
    ax,
    show_workspace: bool = True,
    show_spacing: bool = True,
    min_spacing: float = 0.12,
) -> None:
    """Create a 2D top-down view showing object positions.

    Args:
        scene_config: Scene configuration to visualize
        title: Plot title
        ax: Matplotlib axes to draw on
        show_workspace: Whether to show workspace bounds
        show_spacing: Whether to show minimum spacing circles
        min_spacing: Minimum spacing value for circles
    """
    import matplotlib.pyplot as plt

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

    # Draw container (rectangle) - accounting for rotation
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
    if show_spacing:
        for bowl_name, pose in scene_config.bowl_poses.items():
            pos = pose.position
            spacing_circle = plt.Circle(
                (pos[0], pos[1]), min_spacing / 2,
                fill=False, edgecolor='red', linestyle=':', linewidth=0.5, alpha=0.3
            )
            ax.add_patch(spacing_circle)

    ax.legend(loc='upper right', fontsize=6)
