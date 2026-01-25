"""
Domain Randomization package for food transfer task.

This package provides domain randomization for sim-to-real transfer,
including object pose randomization, variable bowl counts, and
collision-aware scene sampling.
"""

from .config import (
    DomainRandomizationConfig,
    ObjectPoseConfig,
    CollisionConfig,
    SceneVariantConfig,
    ObjectPose,
    SceneConfiguration,
)
from .scene_sampler import SceneSampler
from .scene_loader import SceneLoader
from .viz_utils import (
    ALL_BOWL_NAMES,
    HIDDEN_OBJECT_POSITION,
    load_scene,
    apply_scene_config,
    hide_inactive_bowls,
    render_scene,
    yaw_to_quaternion,
    create_2d_layout_plot,
    get_optimal_workers,
    get_optimal_batch_size,
    get_available_ram_gb,
)

__all__ = [
    "DomainRandomizationConfig",
    "ObjectPoseConfig",
    "CollisionConfig",
    "SceneVariantConfig",
    "ObjectPose",
    "SceneConfiguration",
    "SceneSampler",
    "SceneLoader",
    # Visualization utilities
    "ALL_BOWL_NAMES",
    "HIDDEN_OBJECT_POSITION",
    "load_scene",
    "apply_scene_config",
    "hide_inactive_bowls",
    "render_scene",
    "yaw_to_quaternion",
    "create_2d_layout_plot",
    "get_optimal_workers",
    "get_optimal_batch_size",
    "get_available_ram_gb",
]
