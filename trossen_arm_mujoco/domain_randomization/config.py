"""
Configuration dataclasses for domain randomization.

This module defines all configuration parameters for domain randomization
including object pose noise, collision constraints, scene variants, and
visual randomization (textures, colors, lighting).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json


@dataclass
class ObjectPose:
    """
    Represents a 7-DOF pose (position + quaternion).

    Attributes:
        position: [x, y, z] in meters
        quaternion: [w, x, y, z] quaternion (scalar-first)
    """
    position: np.ndarray
    quaternion: np.ndarray

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.quaternion = np.asarray(self.quaternion, dtype=np.float64)

    def to_array(self) -> np.ndarray:
        """Return 7D pose array [x, y, z, qw, qx, qy, qz]."""
        return np.concatenate([self.position, self.quaternion])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ObjectPose":
        """Create ObjectPose from 7D array."""
        return cls(position=arr[:3], quaternion=arr[3:])

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "position": self.position.tolist(),
            "quaternion": self.quaternion.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ObjectPose":
        """Create from dict."""
        return cls(
            position=np.array(d["position"]),
            quaternion=np.array(d["quaternion"]),
        )


# =============================================================================
# Visual Domain Randomization Configuration
# =============================================================================

@dataclass
class TextureConfig:
    """
    Configuration for texture randomization.

    Attributes:
        randomize_table: Whether to randomize table/counter texture
        num_table_textures: Number of table textures available (default: 100)
        randomize_floor: Whether to randomize floor texture
        num_floor_textures: Number of floor textures available (default: 100)
    """
    randomize_table: bool = True
    num_table_textures: int = 100  # counter_1.png through counter_100.png
    randomize_floor: bool = True
    num_floor_textures: int = 100  # floor_1.png through floor_100.png


@dataclass
class ColorConfig:
    """
    Configuration for object color randomization.

    Attributes:
        randomize_container: Whether to randomize container color
        container_hue_range: Hue shift range for container (0-1)
        container_saturation_range: Saturation multiplier range
        container_value_range: Value/brightness multiplier range
        randomize_bowls: Whether to randomize bowl material tint
        bowl_tint_range: RGB tint multiplier range for bowls
    """
    randomize_container: bool = True
    # Container color: base is dark gray (0.3, 0.3, 0.35)
    # We'll vary around this with RGB noise
    container_rgb_noise: float = 0.15  # ±0.15 on each RGB channel

    randomize_bowls: bool = False  # Off by default - bowls use texture
    bowl_tint_range: Tuple[float, float] = (0.85, 1.15)  # RGB multiplier range


@dataclass
class LightingConfig:
    """
    Configuration for lighting randomization.

    Attributes:
        randomize_position: Whether to randomize light positions
        position_noise: Position noise in meters (applied to XYZ)
        randomize_intensity: Whether to randomize light intensity
        intensity_range: Multiplier range for diffuse intensity (min, max)
        randomize_color: Whether to randomize light color temperature
        color_temp_range: Color temperature shift range (warm < 1.0 < cool)
    """
    randomize_position: bool = True
    position_noise: float = 0.3  # ±30cm noise on light positions

    randomize_intensity: bool = True
    intensity_range: Tuple[float, float] = (0.5, 1.2)  # 50-120% of base

    randomize_color: bool = True
    color_temp_range: Tuple[float, float] = (0.85, 1.15)  # Warm to cool shift


@dataclass
class VisualRandomizationConfig:
    """
    Master configuration for visual domain randomization.

    Combines texture, color, and lighting randomization settings.
    """
    enabled: bool = False
    seed: Optional[int] = None

    texture: TextureConfig = field(default_factory=TextureConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            "enabled": self.enabled,
            "seed": self.seed,
            "texture": {
                "randomize_table": self.texture.randomize_table,
                "num_table_textures": self.texture.num_table_textures,
                "randomize_floor": self.texture.randomize_floor,
                "num_floor_textures": self.texture.num_floor_textures,
            },
            "color": {
                "randomize_container": self.color.randomize_container,
                "container_rgb_noise": self.color.container_rgb_noise,
                "randomize_bowls": self.color.randomize_bowls,
                "bowl_tint_range": list(self.color.bowl_tint_range),
            },
            "lighting": {
                "randomize_position": self.lighting.randomize_position,
                "position_noise": self.lighting.position_noise,
                "randomize_intensity": self.lighting.randomize_intensity,
                "intensity_range": list(self.lighting.intensity_range),
                "randomize_color": self.lighting.randomize_color,
                "color_temp_range": list(self.lighting.color_temp_range),
            },
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "VisualRandomizationConfig":
        """Create from dict."""
        texture = TextureConfig(
            randomize_table=d.get("texture", {}).get("randomize_table", True),
            num_table_textures=d.get("texture", {}).get("num_table_textures", 100),
            randomize_floor=d.get("texture", {}).get("randomize_floor", True),
            num_floor_textures=d.get("texture", {}).get("num_floor_textures", 100),
        )
        color = ColorConfig(
            randomize_container=d.get("color", {}).get("randomize_container", True),
            container_rgb_noise=d.get("color", {}).get("container_rgb_noise", 0.15),
            randomize_bowls=d.get("color", {}).get("randomize_bowls", False),
            bowl_tint_range=tuple(d.get("color", {}).get("bowl_tint_range", (0.85, 1.15))),
        )
        lighting = LightingConfig(
            randomize_position=d.get("lighting", {}).get("randomize_position", True),
            position_noise=d.get("lighting", {}).get("position_noise", 0.3),
            randomize_intensity=d.get("lighting", {}).get("randomize_intensity", True),
            intensity_range=tuple(d.get("lighting", {}).get("intensity_range", (0.5, 1.2))),
            randomize_color=d.get("lighting", {}).get("randomize_color", True),
            color_temp_range=tuple(d.get("lighting", {}).get("color_temp_range", (0.85, 1.15))),
        )
        return cls(
            enabled=d.get("enabled", False),
            seed=d.get("seed"),
            texture=texture,
            color=color,
            lighting=lighting,
        )


@dataclass
class VisualConfiguration:
    """
    Complete visual configuration for one episode.

    Stores the sampled visual parameters for reproducibility.
    """
    # Texture indices
    table_texture_index: int  # Which counter texture (0-99)
    floor_texture_index: int  # Which floor texture (0-99)

    # Colors (RGBA tuples)
    container_color: Tuple[float, float, float, float]
    bowl_tints: Dict[str, Tuple[float, float, float, float]]  # bowl_name -> RGBA

    # Lighting state
    top_light_pos: Tuple[float, float, float]
    top_light_diffuse: Tuple[float, float, float]
    side_light_pos: Tuple[float, float, float]
    side_light_diffuse: Tuple[float, float, float]

    # Headlight (ambient) settings
    headlight_diffuse: Tuple[float, float, float]
    headlight_ambient: Tuple[float, float, float]

    seed: int

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict for HDF5 storage."""
        return {
            "table_texture_index": int(self.table_texture_index),
            "floor_texture_index": int(self.floor_texture_index),
            "container_color": list(self.container_color),
            "bowl_tints": {k: list(v) for k, v in self.bowl_tints.items()},
            "top_light_pos": list(self.top_light_pos),
            "top_light_diffuse": list(self.top_light_diffuse),
            "side_light_pos": list(self.side_light_pos),
            "side_light_diffuse": list(self.side_light_diffuse),
            "headlight_diffuse": list(self.headlight_diffuse),
            "headlight_ambient": list(self.headlight_ambient),
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "VisualConfiguration":
        """Create from dict."""
        return cls(
            table_texture_index=d["table_texture_index"],
            floor_texture_index=d["floor_texture_index"],
            container_color=tuple(d["container_color"]),
            bowl_tints={k: tuple(v) for k, v in d["bowl_tints"].items()},
            top_light_pos=tuple(d["top_light_pos"]),
            top_light_diffuse=tuple(d["top_light_diffuse"]),
            side_light_pos=tuple(d["side_light_pos"]),
            side_light_diffuse=tuple(d["side_light_diffuse"]),
            headlight_diffuse=tuple(d["headlight_diffuse"]),
            headlight_ambient=tuple(d["headlight_ambient"]),
            seed=d["seed"],
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "VisualConfiguration":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(s))


# =============================================================================
# Scene Configuration
# =============================================================================

@dataclass
class SceneConfiguration:
    """
    Complete scene configuration for one episode.

    Contains poses for all objects, metadata, and optional visual configuration.
    """
    container_pose: ObjectPose
    bowl_poses: Dict[str, ObjectPose]  # bowl_name -> pose
    active_bowls: List[str]  # Which bowls are active in this episode
    target_bowl: str  # Which bowl is the target
    num_bowls: int  # Total number of active bowls
    scene_xml: str  # Which scene XML to use
    seed: int  # Random seed for reproducibility
    visual_config: Optional[VisualConfiguration] = None  # Visual DR settings

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict for HDF5 storage."""
        # Ensure active_bowls are plain strings (not numpy strings)
        active_bowls = [str(b) for b in self.active_bowls]
        d = {
            "container_pose": self.container_pose.to_dict(),
            "bowl_poses": {str(k): v.to_dict() for k, v in self.bowl_poses.items()},
            "active_bowls": active_bowls,
            "target_bowl": str(self.target_bowl),
            "num_bowls": int(self.num_bowls),  # Ensure native Python int
            "scene_xml": str(self.scene_xml),
            "seed": int(self.seed),  # Ensure native Python int
        }
        # Add visual config if present
        if self.visual_config is not None:
            d["visual_config"] = self.visual_config.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "SceneConfiguration":
        """Create from dict."""
        visual_config = None
        if "visual_config" in d and d["visual_config"] is not None:
            visual_config = VisualConfiguration.from_dict(d["visual_config"])
        return cls(
            container_pose=ObjectPose.from_dict(d["container_pose"]),
            bowl_poses={k: ObjectPose.from_dict(v) for k, v in d["bowl_poses"].items()},
            active_bowls=d["active_bowls"],
            target_bowl=d["target_bowl"],
            num_bowls=d["num_bowls"],
            scene_xml=d["scene_xml"],
            seed=d["seed"],
            visual_config=visual_config,
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "SceneConfiguration":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(s))


@dataclass
class ObjectPoseConfig:
    """
    Configuration for single object pose randomization.

    Attributes:
        position_noise_xy: Position noise in X and Y directions (meters)
        position_noise_z: Position noise in Z direction (meters)
        rotation_noise_yaw: Yaw rotation noise (radians)
        keep_upright: If True, only apply yaw rotation (no tilt)
    """
    position_noise_xy: float = 0.03  # ±3cm in XY
    position_noise_z: float = 0.005  # ±5mm in Z
    rotation_noise_yaw: float = 0.1  # ±0.1 rad (~6 degrees)
    keep_upright: bool = True  # Only yaw rotation, no tilt


@dataclass
class ContainerPoseConfig:
    """
    Configuration for container pose randomization.

    When randomize_position=False (default):
        Container stays near workspace edge with small noise.
        Asymmetric X noise ensures reachability.

    When randomize_position=True:
        Container can be placed at any of the predefined slot positions
        (similar to bowl positions), enabling much more diverse scenes.

    Rotation modes:
        - allow_90_degree_rotation=False: Small yaw noise around default orientation
        - allow_90_degree_rotation=True: Container can be rotated 0 or 90 degrees,
          plus small noise. This significantly changes the container footprint.
    """
    # Small noise mode (when randomize_position=False)
    position_noise_x_min: float = 0.03  # Minimum +3cm toward robot (required for reachability)
    position_noise_x_max: float = 0.06  # Up to +6cm toward robot
    position_noise_y: float = 0.02  # ±2cm in Y
    position_noise_z: float = 0.005  # ±5mm in Z
    rotation_noise_yaw: float = 0.15  # ±0.15 rad (~9 degrees) noise around base orientation

    # Large position randomization (when randomize_position=True)
    randomize_position: bool = False  # If True, container can be at any slot position
    position_noise_xy: float = 0.03  # ±3cm noise when at slot positions

    # 90-degree rotation option
    allow_90_degree_rotation: bool = False  # If True, container can be 0 or 90 deg rotated


@dataclass
class CollisionConfig:
    """
    Configuration for collision-aware placement.

    Attributes:
        min_object_spacing: Minimum distance between object centers (meters)
                           Ramekin bowls are ~10cm diameter, so this should be >= 0.10
                           to prevent overlap. Using 0.12 for safety margin.
        max_sample_attempts: Maximum rejection sampling attempts
        workspace_bounds: Dictionary of (min, max) bounds for x, y, z
    """
    min_object_spacing: float = 0.12  # 12cm between object centers (bowls are ~10cm diameter)
    max_sample_attempts: int = 100  # Rejection sampling limit
    workspace_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "x": (-0.58, 0.1),  # Conservative (actual IK limit is -0.6)
        "y": (-0.28, 0.28),  # Conservative (actual limit is -0.3 to 0.3)
        "z": (0.0, 0.4),
    })


@dataclass
class SceneVariantConfig:
    """
    Configuration for scene variant selection.

    Controls which scene XMLs are used based on bowl count.
    """
    min_bowls: int = 1
    max_bowls: int = 8

    def get_scene_xml(self, num_bowls: int) -> str:
        """Return appropriate scene XML for bowl count."""
        if num_bowls <= 1:
            return "wxai/teleop_scene_1bowl.xml"
        elif num_bowls <= 4:
            return "wxai/teleop_scene.xml"  # Original 4-bowl scene
        elif num_bowls <= 6:
            return "wxai/teleop_scene_6bowl.xml"
        else:
            return "wxai/teleop_scene_8bowl.xml"

    def get_available_bowls(self, scene_xml: str) -> List[str]:
        """Return list of bowl names available in the given scene."""
        if "1bowl" in scene_xml:
            return ["bowl_1"]
        elif "6bowl" in scene_xml:
            return ["bowl_1", "bowl_2", "bowl_3", "bowl_4", "bowl_5", "bowl_6"]
        elif "8bowl" in scene_xml:
            return ["bowl_1", "bowl_2", "bowl_3", "bowl_4",
                    "bowl_5", "bowl_6", "bowl_7", "bowl_8"]
        else:
            # Default 4-bowl scene
            return ["bowl_1", "bowl_2", "bowl_3", "bowl_4"]


@dataclass
class DomainRandomizationConfig:
    """
    Master configuration for domain randomization.

    This is the main configuration object that controls all aspects
    of domain randomization for the food transfer task, including
    both geometric (pose) and visual (texture/color/lighting) randomization.
    """
    enabled: bool = True
    seed: Optional[int] = None

    # Object pose randomization
    container_pose: ContainerPoseConfig = field(default_factory=ContainerPoseConfig)
    bowl_pose: ObjectPoseConfig = field(default_factory=ObjectPoseConfig)

    # Collision configuration
    collision: CollisionConfig = field(default_factory=CollisionConfig)

    # Scene variant configuration
    scene_variant: SceneVariantConfig = field(default_factory=SceneVariantConfig)

    # Which objects to randomize
    randomize_container: bool = True
    randomize_bowls: bool = True

    # Visual domain randomization
    visual: VisualRandomizationConfig = field(default_factory=VisualRandomizationConfig)

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        d = {
            "enabled": self.enabled,
            "seed": self.seed,
            "container_pose": {
                "position_noise_x_min": self.container_pose.position_noise_x_min,
                "position_noise_x_max": self.container_pose.position_noise_x_max,
                "position_noise_y": self.container_pose.position_noise_y,
                "position_noise_z": self.container_pose.position_noise_z,
                "rotation_noise_yaw": self.container_pose.rotation_noise_yaw,
            },
            "bowl_pose": {
                "position_noise_xy": self.bowl_pose.position_noise_xy,
                "position_noise_z": self.bowl_pose.position_noise_z,
                "rotation_noise_yaw": self.bowl_pose.rotation_noise_yaw,
                "keep_upright": self.bowl_pose.keep_upright,
            },
            "collision": {
                "min_object_spacing": self.collision.min_object_spacing,
                "max_sample_attempts": self.collision.max_sample_attempts,
            },
            "scene_variant": {
                "min_bowls": self.scene_variant.min_bowls,
                "max_bowls": self.scene_variant.max_bowls,
            },
            "randomize_container": self.randomize_container,
            "randomize_bowls": self.randomize_bowls,
            "visual": self.visual.to_dict(),
        }
        return d

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_cli_args(
        cls,
        enable_dr: bool = False,
        position_noise: float = 0.03,
        rotation_noise: float = 0.1,
        container_rotation: float = 0.15,
        min_bowls: int = 1,
        max_bowls: int = 8,
        seed: Optional[int] = None,
        min_spacing: float = 0.12,
        randomize_container_position: bool = False,
        allow_90_degree_rotation: bool = False,
        # Visual DR arguments
        enable_visual_dr: bool = False,
        randomize_table_texture: bool = True,
        num_table_textures: int = 100,
        randomize_floor_texture: bool = True,
        num_floor_textures: int = 100,
        randomize_container_color: bool = True,
        randomize_bowl_color: bool = False,
        randomize_lighting: bool = True,
        light_position_noise: float = 0.3,
        light_intensity_min: float = 0.5,
        light_intensity_max: float = 1.2,
    ) -> "DomainRandomizationConfig":
        """Create config from CLI arguments."""
        # Container X noise: min=0.03 (required for reachability), max=0.03+position_noise
        container_x_min = 0.03  # Required minimum to reach IK workspace
        container_x_max = container_x_min + position_noise

        # Visual randomization config
        visual = VisualRandomizationConfig(
            enabled=enable_visual_dr,
            seed=seed,
            texture=TextureConfig(
                randomize_table=randomize_table_texture,
                num_table_textures=num_table_textures,
                randomize_floor=randomize_floor_texture,
                num_floor_textures=num_floor_textures,
            ),
            color=ColorConfig(
                randomize_container=randomize_container_color,
                randomize_bowls=randomize_bowl_color,
            ),
            lighting=LightingConfig(
                randomize_position=randomize_lighting,
                position_noise=light_position_noise,
                randomize_intensity=randomize_lighting,
                intensity_range=(light_intensity_min, light_intensity_max),
                randomize_color=randomize_lighting,
            ),
        )

        return cls(
            enabled=enable_dr,
            seed=seed,
            container_pose=ContainerPoseConfig(
                position_noise_x_min=container_x_min,
                position_noise_x_max=container_x_max,
                position_noise_y=position_noise,
                rotation_noise_yaw=container_rotation,
                randomize_position=randomize_container_position,
                position_noise_xy=position_noise,
                allow_90_degree_rotation=allow_90_degree_rotation,
            ),
            bowl_pose=ObjectPoseConfig(
                position_noise_xy=position_noise,
                rotation_noise_yaw=rotation_noise,
            ),
            collision=CollisionConfig(
                min_object_spacing=min_spacing,
            ),
            scene_variant=SceneVariantConfig(
                min_bowls=min_bowls,
                max_bowls=max_bowls,
            ),
            visual=visual,
        )


# Default nominal positions (from food_transfer_base.py)
# Robot base is at origin with ~12cm radius + 5cm bowl radius + 3cm margin = 20cm minimum distance
# So bowl X positions should be <= -0.20 to avoid robot collision
#
# Bowl spacing: With ±3cm noise and 12cm min_spacing, nominal distance should be >= 18cm
# Bottom bowls (1-4): X spacing is 14cm (marginal), Y spacing is 14cm (marginal)
# Top bowls (5-8): Mirror the bottom layout for consistency
NOMINAL_POSITIONS = {
    "container": np.array([-0.63, -0.15, 0.04]),
    # Bottom bowls: 2x2 grid in negative Y region
    "bowl_1": np.array([-0.22, -0.26, 0.04]),
    "bowl_2": np.array([-0.36, -0.26, 0.04]),
    "bowl_3": np.array([-0.36, -0.12, 0.04]),
    "bowl_4": np.array([-0.22, -0.12, 0.04]),
    # Top bowls: 2x2 grid in positive Y region (mirror of bottom)
    # Same X positions as bottom bowls for symmetric layout
    "bowl_5": np.array([-0.22, +0.12, 0.04]),  # Mirror of bowl_4
    "bowl_6": np.array([-0.36, +0.12, 0.04]),  # Mirror of bowl_3
    "bowl_7": np.array([-0.22, +0.26, 0.04]),  # Mirror of bowl_1
    "bowl_8": np.array([-0.36, +0.26, 0.04]),  # Mirror of bowl_2
}

# Possible container slot positions (where container can be placed when randomize_position=True)
# Container is ~28cm x 34cm (large!), so it covers multiple bowl positions
# Container half-dims: half_x=0.14, half_y=0.17 (after default 90-deg rotation)
#
# Bowl nominal positions: Y in {-0.26, -0.12, +0.12, +0.26}, X in {-0.22, -0.36}
# Innermost bowls (3, 6) are at X=-0.36
# For container to not block bowls: container_X + 0.14 < -0.36 - 0.05 - 0.02
#   => container_X < -0.57
# Using X=-0.58 for far-left positions (blocks 0 bowls, allows all 8)
#
# Strategy:
# - Far-left positions (X=-0.58): Block NO bowls, allow 5-8 bowl configurations
# - Bowl-area positions (X=-0.36): Block 4 bowls each, for 1-4 bowl variety
#
# Robot base is at origin with ~12cm radius. Container extends 14cm toward robot.
# So container center X must be <= -(0.12 + 0.14 + 0.03) = -0.29 minimum to avoid robot.
CONTAINER_SLOT_POSITIONS = [
    # Slots 0-2: Far-left positions - block NO bowls (for 5-8 bowl configs)
    np.array([-0.58, -0.12, 0.04]),  # Slot 0: Far-left, bottom side
    np.array([-0.58, 0.00, 0.04]),   # Slot 1: Far-left, center
    np.array([-0.58, +0.12, 0.04]),  # Slot 2: Far-left, top side
    # Slots 3-4: Bowl-area positions - block 4 bowls each (for variety with 1-4 bowls)
    np.array([-0.36, -0.22, 0.04]),  # Slot 3: Bottom bowl area -> use top bowls
    np.array([-0.36, +0.22, 0.04]),  # Slot 4: Top bowl area -> use bottom bowls
]

# Bowls that are blocked (too close) for each container slot position
# When container is at a slot, these bowls cannot be used
CONTAINER_SLOT_BLOCKED_BOWLS = [
    [],  # Slot 0: Far-left bottom - no bowls blocked
    [],  # Slot 1: Far-left center - no bowls blocked
    [],  # Slot 2: Far-left top - no bowls blocked
    ["bowl_1", "bowl_2", "bowl_3", "bowl_4"],  # Slot 3: Bottom area, blocks bottom bowls
    ["bowl_5", "bowl_6", "bowl_7", "bowl_8"],  # Slot 4: Top area, blocks top bowls
]

# Slots that allow all 8 bowls (for 5-8 bowl configurations)
CONTAINER_SLOTS_FULL_BOWLS = [0, 1, 2]
# Slots that block some bowls (for variety with fewer bowls)
CONTAINER_SLOTS_PARTIAL_BOWLS = [3, 4]

# Identity quaternion [w, x, y, z]
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
