"""
Configuration dataclasses for domain randomization.

This module defines all configuration parameters for domain randomization
including object pose noise, collision constraints, and scene variants.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
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


@dataclass
class SceneConfiguration:
    """
    Complete scene configuration for one episode.

    Contains poses for all objects and metadata about the configuration.
    """
    container_pose: ObjectPose
    bowl_poses: Dict[str, ObjectPose]  # bowl_name -> pose
    active_bowls: List[str]  # Which bowls are active in this episode
    target_bowl: str  # Which bowl is the target
    num_bowls: int  # Total number of active bowls
    scene_xml: str  # Which scene XML to use
    seed: int  # Random seed for reproducibility

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict for HDF5 storage."""
        # Ensure active_bowls are plain strings (not numpy strings)
        active_bowls = [str(b) for b in self.active_bowls]
        return {
            "container_pose": self.container_pose.to_dict(),
            "bowl_poses": {str(k): v.to_dict() for k, v in self.bowl_poses.items()},
            "active_bowls": active_bowls,
            "target_bowl": str(self.target_bowl),
            "num_bowls": int(self.num_bowls),  # Ensure native Python int
            "scene_xml": str(self.scene_xml),
            "seed": int(self.seed),  # Ensure native Python int
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SceneConfiguration":
        """Create from dict."""
        return cls(
            container_pose=ObjectPose.from_dict(d["container_pose"]),
            bowl_poses={k: ObjectPose.from_dict(v) for k, v in d["bowl_poses"].items()},
            active_bowls=d["active_bowls"],
            target_bowl=d["target_bowl"],
            num_bowls=d["num_bowls"],
            scene_xml=d["scene_xml"],
            seed=d["seed"],
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
    Special configuration for container pose randomization.

    Container has asymmetric noise because it's at workspace edge.
    Only allows movement TOWARD the robot (+X direction).

    Container nominal position is X=-0.63, IK workspace limit is X=-0.6.
    To ensure reachability, we need at least +0.03 X offset.
    """
    position_noise_x_min: float = 0.03  # Minimum +3cm toward robot (required for reachability)
    position_noise_x_max: float = 0.06  # Up to +6cm toward robot
    position_noise_y: float = 0.02  # ±2cm in Y
    position_noise_z: float = 0.005  # ±5mm in Z
    rotation_noise_yaw: float = 0.15  # ±0.15 rad (~9 degrees)


@dataclass
class CollisionConfig:
    """
    Configuration for collision-aware placement.

    Attributes:
        min_object_spacing: Minimum distance between object centers (meters)
        max_sample_attempts: Maximum rejection sampling attempts
        workspace_bounds: Dictionary of (min, max) bounds for x, y, z
    """
    min_object_spacing: float = 0.05  # 5cm between objects
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
    of domain randomization for the food transfer task.
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

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
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
        }

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
    ) -> "DomainRandomizationConfig":
        """Create config from CLI arguments."""
        # Container X noise: min=0.03 (required for reachability), max=0.03+position_noise
        container_x_min = 0.03  # Required minimum to reach IK workspace
        container_x_max = container_x_min + position_noise
        return cls(
            enabled=enable_dr,
            seed=seed,
            container_pose=ContainerPoseConfig(
                position_noise_x_min=container_x_min,
                position_noise_x_max=container_x_max,
                position_noise_y=position_noise,
                rotation_noise_yaw=container_rotation,
            ),
            bowl_pose=ObjectPoseConfig(
                position_noise_xy=position_noise,
                rotation_noise_yaw=rotation_noise,
            ),
            scene_variant=SceneVariantConfig(
                min_bowls=min_bowls,
                max_bowls=max_bowls,
            ),
        )


# Default nominal positions (from food_transfer_base.py)
NOMINAL_POSITIONS = {
    "container": np.array([-0.63, -0.15, 0.04]),
    "bowl_1": np.array([-0.22, -0.26, 0.04]),
    "bowl_2": np.array([-0.36, -0.26, 0.04]),
    "bowl_3": np.array([-0.36, -0.12, 0.04]),
    "bowl_4": np.array([-0.22, -0.12, 0.04]),
    # New bowl positions for 6 and 8 bowl scenes
    "bowl_5": np.array([-0.50, -0.26, 0.04]),
    "bowl_6": np.array([-0.50, -0.12, 0.04]),
    "bowl_7": np.array([-0.22, +0.02, 0.04]),
    "bowl_8": np.array([-0.36, +0.02, 0.04]),
}

# Identity quaternion [w, x, y, z]
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
