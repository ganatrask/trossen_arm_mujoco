"""
Domain Randomization package for food transfer task.

This package provides domain randomization for sim-to-real transfer,
including object pose randomization, variable bowl counts, collision-aware
scene sampling, and visual randomization (textures, colors, lighting).
"""

from .config import (
    DomainRandomizationConfig,
    ObjectPoseConfig,
    CollisionConfig,
    SceneVariantConfig,
    ObjectPose,
    SceneConfiguration,
    # Visual DR configs
    TextureConfig,
    ColorConfig,
    LightingConfig,
    VisualRandomizationConfig,
    VisualConfiguration,
)
from .scene_sampler import SceneSampler
from .scene_loader import SceneLoader
from .visual_sampler import VisualSampler
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
    # Visual DR
    "TextureConfig",
    "ColorConfig",
    "LightingConfig",
    "VisualRandomizationConfig",
    "VisualConfiguration",
    "VisualSampler",
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
