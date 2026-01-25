"""
Scene sampler for domain randomization.

This module provides collision-aware sampling of scene configurations,
including object poses and variable bowl counts.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import (
    DomainRandomizationConfig,
    ObjectPose,
    SceneConfiguration,
    NOMINAL_POSITIONS,
    CONTAINER_SLOT_POSITIONS,
    CONTAINER_SLOT_BLOCKED_BOWLS,
    CONTAINER_SLOTS_FULL_BOWLS,
    CONTAINER_SLOTS_PARTIAL_BOWLS,
    IDENTITY_QUAT,
)


class SceneSampler:
    """
    Samples valid scene configurations with collision checking.

    This class generates randomized scene configurations for domain
    randomization, ensuring all objects are collision-free and
    within the robot's workspace.
    """

    def __init__(
        self,
        config: DomainRandomizationConfig,
        nominal_positions: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Initialize the scene sampler.

        Args:
            config: Domain randomization configuration
            nominal_positions: Optional override for nominal object positions
        """
        self.config = config
        self.nominal = nominal_positions or NOMINAL_POSITIONS.copy()
        self.rng = np.random.default_rng(config.seed)
        self._current_container_slot = None  # Track which slot is selected

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def sample(self, target_bowl: Optional[str] = None) -> SceneConfiguration:
        """
        Sample a collision-free scene configuration.

        New logic (bowl-count-first):
        1. First pick the number of bowls
        2. Then choose a container slot that supports that bowl count
        3. This ensures all bowl counts 1-8 can appear with proper distribution

        Args:
            target_bowl: Specific target bowl to use. If None, randomly selected.

        Returns:
            SceneConfiguration with randomized poses

        Raises:
            RuntimeError: If unable to find valid configuration after max attempts
        """
        # STEP 1: Determine number of bowls FIRST
        num_bowls = self.rng.integers(
            self.config.scene_variant.min_bowls,
            self.config.scene_variant.max_bowls + 1
        )

        # STEP 2: Choose container slot based on bowl count
        self._current_container_slot = None
        blocked_bowls = []

        if (self.config.randomize_container and
            self.config.container_pose.randomize_position):

            if num_bowls > 4:
                # Need 5-8 bowls: must use a slot that doesn't block bowls
                # Choose from CONTAINER_SLOTS_FULL_BOWLS (slots 0, 1, 2)
                self._current_container_slot = self.rng.choice(CONTAINER_SLOTS_FULL_BOWLS)
            else:
                # 1-4 bowls: can use any slot (more variety)
                # Give equal weight to all slots for diversity
                self._current_container_slot = self.rng.integers(
                    0, len(CONTAINER_SLOT_POSITIONS)
                )

            blocked_bowls = CONTAINER_SLOT_BLOCKED_BOWLS[self._current_container_slot]

        # STEP 3: Determine scene and available bowls
        # Always use 8-bowl scene when container position is randomized
        # (we need access to all bowls, some will be hidden)
        if self.config.container_pose.randomize_position:
            scene_xml = "wxai/teleop_scene_8bowl.xml"
            available_bowls = ["bowl_1", "bowl_2", "bowl_3", "bowl_4",
                              "bowl_5", "bowl_6", "bowl_7", "bowl_8"]
        else:
            scene_xml = self.config.scene_variant.get_scene_xml(num_bowls)
            available_bowls = self.config.scene_variant.get_available_bowls(scene_xml)

        # Remove blocked bowls from available
        available_bowls = [b for b in available_bowls if b not in blocked_bowls]

        if not available_bowls:
            raise RuntimeError("No bowls available after removing blocked bowls")

        # STEP 4: Select target bowl
        if target_bowl is None:
            target_bowl = self.rng.choice(available_bowls)
        elif target_bowl not in available_bowls:
            # If specified target not available, pick one that is
            target_bowl = self.rng.choice(available_bowls)

        # Adjust num_bowls if we don't have enough available (shouldn't happen with new logic)
        num_bowls = min(num_bowls, len(available_bowls))

        # Select which bowls are active (target always included)
        active_bowls = self._select_active_bowls(
            available_bowls, target_bowl, num_bowls
        )

        # Now sample positions with collision checking
        for attempt in range(self.config.collision.max_sample_attempts):
            scene_config = self._sample_candidate(
                active_bowls, target_bowl, num_bowls, scene_xml
            )
            if self._validate_configuration(scene_config):
                return scene_config

        raise RuntimeError(
            f"Failed to sample valid scene configuration after "
            f"{self.config.collision.max_sample_attempts} attempts"
        )

    def _select_active_bowls(
        self,
        available_bowls: List[str],
        target_bowl: str,
        num_bowls: int,
    ) -> List[str]:
        """Select which bowls are active in this episode."""
        # Target bowl is always first
        active_bowls = [target_bowl]

        # Add additional bowls up to num_bowls
        other_bowls = [b for b in available_bowls if b != target_bowl]
        num_additional = min(num_bowls - 1, len(other_bowls))

        if num_additional > 0:
            additional = self.rng.choice(
                other_bowls, size=num_additional, replace=False
            )
            active_bowls.extend(additional.tolist())

        return active_bowls

    def _sample_candidate(
        self,
        active_bowls: List[str],
        target_bowl: str,
        num_bowls: int,
        scene_xml: str,
    ) -> SceneConfiguration:
        """Generate candidate configuration (may have collisions)."""
        # Store current bowl count for adaptive noise
        self._current_num_bowls = num_bowls

        # Sample container pose
        container_pose = self._sample_container_pose()

        # Sample bowl poses with adaptive noise based on bowl count
        bowl_poses = {}
        for bowl_name in active_bowls:
            bowl_poses[bowl_name] = self._sample_bowl_pose(bowl_name)

        # Generate unique seed for this configuration
        config_seed = int(self.rng.integers(0, 2**31))

        return SceneConfiguration(
            container_pose=container_pose,
            bowl_poses=bowl_poses,
            active_bowls=active_bowls,
            target_bowl=target_bowl,
            num_bowls=num_bowls,
            scene_xml=scene_xml,
            seed=config_seed,
        )

    def _sample_container_pose(self) -> ObjectPose:
        """
        Sample container pose.

        Position modes:
        1. randomize_position=False: Small noise around default edge position
        2. randomize_position=True: Use pre-selected slot position with noise

        Rotation modes:
        1. allow_90_degree_rotation=False: Small yaw noise around default (0 deg)
        2. allow_90_degree_rotation=True: Base rotation is 0 or 90 deg, plus small noise
        """
        cfg = self.config.container_pose

        if not self.config.randomize_container:
            # No randomization at all
            return ObjectPose(
                position=self.nominal["container"].copy(),
                quaternion=IDENTITY_QUAT.copy()
            )

        if cfg.randomize_position and self._current_container_slot is not None:
            # Use the pre-selected slot position (selected in sample())
            base_position = CONTAINER_SLOT_POSITIONS[self._current_container_slot].copy()

            # Add small noise around the slot position
            dx = self.rng.uniform(-cfg.position_noise_xy, cfg.position_noise_xy)
            dy = self.rng.uniform(-cfg.position_noise_xy, cfg.position_noise_xy)
            dz = self.rng.uniform(-cfg.position_noise_z, cfg.position_noise_z)

            position = base_position + np.array([dx, dy, dz])
        else:
            # Original mode: small noise around default edge position
            nominal = self.nominal["container"].copy()

            # Asymmetric X noise: only toward robot (positive direction)
            dx = self.rng.uniform(cfg.position_noise_x_min, cfg.position_noise_x_max)
            dy = self.rng.uniform(-cfg.position_noise_y, cfg.position_noise_y)
            dz = self.rng.uniform(-cfg.position_noise_z, cfg.position_noise_z)

            position = nominal + np.array([dx, dy, dz])

        # Determine base rotation
        if cfg.allow_90_degree_rotation:
            # Randomly choose between 0 and 90 degrees (pi/2) as base rotation
            base_yaw = self.rng.choice([0.0, np.pi / 2])
        else:
            base_yaw = 0.0

        # Add small yaw noise around the base rotation
        yaw_noise = self.rng.uniform(-cfg.rotation_noise_yaw, cfg.rotation_noise_yaw)
        total_yaw = base_yaw + yaw_noise

        quaternion = self._yaw_to_quaternion(total_yaw)

        return ObjectPose(position=position, quaternion=quaternion)

    def _sample_bowl_pose(self, bowl_name: str) -> ObjectPose:
        """Sample bowl pose with symmetric noise.

        Uses adaptive noise: reduced position noise when there are many bowls
        to avoid collision failures (bowl spacing is only 14cm nominal).
        """
        nominal = self.nominal[bowl_name].copy()
        cfg = self.config.bowl_pose

        if self.config.randomize_bowls:
            # Adaptive position noise based on bowl count
            # With 8 bowls and 14cm spacing, we need to reduce noise to avoid collisions
            # Scale: 1.0 for 1-4 bowls, linearly decrease to 0.3 for 8 bowls
            num_bowls = getattr(self, '_current_num_bowls', 4)
            if num_bowls <= 4:
                noise_scale = 1.0
            else:
                # Linear interpolation: 5 bowls -> 0.85, 6 -> 0.7, 7 -> 0.55, 8 -> 0.4
                noise_scale = 1.0 - 0.15 * (num_bowls - 4)
                noise_scale = max(0.4, noise_scale)

            position_noise = cfg.position_noise_xy * noise_scale

            # Symmetric XY noise
            dx = self.rng.uniform(-position_noise, position_noise)
            dy = self.rng.uniform(-position_noise, position_noise)
            dz = self.rng.uniform(-cfg.position_noise_z, cfg.position_noise_z)

            position = nominal + np.array([dx, dy, dz])

            # Yaw rotation (around Z axis)
            if cfg.keep_upright:
                yaw = self.rng.uniform(-cfg.rotation_noise_yaw, cfg.rotation_noise_yaw)
                quaternion = self._yaw_to_quaternion(yaw)
            else:
                quaternion = IDENTITY_QUAT.copy()
        else:
            position = nominal
            quaternion = IDENTITY_QUAT.copy()

        return ObjectPose(position=position, quaternion=quaternion)

    def _yaw_to_quaternion(self, yaw: float) -> np.ndarray:
        """Convert yaw angle to quaternion [w, x, y, z]."""
        return np.array([
            np.cos(yaw / 2),
            0.0,
            0.0,
            np.sin(yaw / 2),
        ])

    def _validate_configuration(self, config: SceneConfiguration) -> bool:
        """
        Validate scene configuration.

        Checks:
        1. No collisions between objects (minimum spacing)
        2. All objects within workspace bounds
        3. Container is reachable (critical for task success)
        """
        return (
            self._validate_no_collisions(config) and
            self._validate_workspace_bounds(config) and
            self._validate_reachability(config)
        )

    def _get_container_half_dims(self, quaternion: np.ndarray) -> Tuple[float, float]:
        """Get container half-dimensions in world frame based on rotation.

        Container local dimensions: ~0.28m x 0.34m (14cm x 17cm half-dims)
        The container in the XML has a default 90-deg Z rotation.

        When yaw is ~0 (default): half_x=0.14, half_y=0.17
        When yaw is ~90 deg: half_x=0.17, half_y=0.14 (swapped)
        """
        # Extract yaw from quaternion [w, x, y, z]
        # yaw = 2 * atan2(z, w)
        w, x, y, z = quaternion
        yaw = 2 * np.arctan2(z, w)

        # Normalize yaw to [0, pi) to determine if closer to 0 or 90 deg
        yaw_normalized = yaw % np.pi

        # If yaw is closer to 90 degrees (pi/2), swap dimensions
        if np.pi / 4 < yaw_normalized < 3 * np.pi / 4:
            # Rotated ~90 degrees - dimensions swap
            return 0.17, 0.14
        else:
            # Default orientation
            return 0.14, 0.17

    def _validate_no_collisions(self, config: SceneConfiguration) -> bool:
        """Check minimum spacing between all objects.

        Container is treated as an axis-aligned rectangle (based on rotation),
        bowls as circles (10cm diameter).
        Also checks container doesn't collide with robot base/mount.
        """
        min_spacing = self.config.collision.min_object_spacing
        bowl_radius = 0.05  # 10cm diameter bowls

        # Get container dimensions based on its rotation
        container_half_x, container_half_y = self._get_container_half_dims(
            config.container_pose.quaternion
        )

        container_pos = config.container_pose.position[:2]

        # Robot base is at origin (0, 0) with approximate radius of 0.12m
        # (includes base + mount structure)
        robot_base_pos = np.array([0.0, 0.0])
        robot_base_radius = 0.12

        # Check container-robot collision (rectangle-circle)
        # Find closest point on container rectangle to robot base center
        closest_x = np.clip(
            robot_base_pos[0],
            container_pos[0] - container_half_x,
            container_pos[0] + container_half_x
        )
        closest_y = np.clip(
            robot_base_pos[1],
            container_pos[1] - container_half_y,
            container_pos[1] + container_half_y
        )
        dist_to_robot = np.linalg.norm(robot_base_pos - np.array([closest_x, closest_y]))

        # Container must not overlap with robot base (with small margin)
        robot_margin = 0.03  # 3cm safety margin
        if dist_to_robot < robot_base_radius + robot_margin:
            return False

        # Check each bowl against the robot base (circle-circle collision)
        for bowl_name, pose in config.bowl_poses.items():
            bowl_pos = pose.position[:2]
            dist_to_robot = np.linalg.norm(bowl_pos - robot_base_pos)
            # Bowl must not overlap with robot base
            if dist_to_robot < robot_base_radius + bowl_radius + robot_margin:
                return False

        # Check each bowl against the container (rectangle-circle collision)
        for bowl_name, pose in config.bowl_poses.items():
            bowl_pos = pose.position[:2]

            # Find closest point on container rectangle to bowl center
            closest_x = np.clip(
                bowl_pos[0],
                container_pos[0] - container_half_x,
                container_pos[0] + container_half_x
            )
            closest_y = np.clip(
                bowl_pos[1],
                container_pos[1] - container_half_y,
                container_pos[1] + container_half_y
            )

            # Distance from bowl center to closest point on container
            dist_to_container = np.linalg.norm(bowl_pos - np.array([closest_x, closest_y]))

            # Bowl must be at least bowl_radius + small margin away from container
            # Using 2cm margin instead of full min_spacing (which is for bowl-bowl)
            container_margin = 0.02
            if dist_to_container < bowl_radius + container_margin:
                return False

        # Check bowl-to-bowl distances (circle-circle collision)
        bowl_positions = list(config.bowl_poses.values())
        for i, pose1 in enumerate(bowl_positions):
            for pose2 in bowl_positions[i + 1:]:
                dist_xy = np.linalg.norm(pose1.position[:2] - pose2.position[:2])
                if dist_xy < min_spacing:
                    return False

        return True

    def _validate_workspace_bounds(self, config: SceneConfiguration) -> bool:
        """Check all bowls are within workspace bounds.

        Note: Container is NOT checked against workspace bounds because
        it's at the workspace edge by design (-0.63). Container reachability
        is validated separately in _validate_reachability().
        """
        bounds = self.config.collision.workspace_bounds

        # Check all bowls (NOT container - it's at workspace edge by design)
        for pose in config.bowl_poses.values():
            if not self._in_bounds(pose.position, bounds):
                return False

        return True

    def _in_bounds(
        self,
        position: np.ndarray,
        bounds: Dict[str, Tuple[float, float]],
    ) -> bool:
        """Check if position is within bounds."""
        x, y, z = position
        return (
            bounds["x"][0] <= x <= bounds["x"][1] and
            bounds["y"][0] <= y <= bounds["y"][1] and
            bounds["z"][0] <= z <= bounds["z"][1]
        )

    def _validate_reachability(self, config: SceneConfiguration) -> bool:
        """
        Ensure all IK targets are within workspace.

        The container is most critical - if unreachable, the entire
        episode will fail at the first step.
        """
        # Approach height from TaskConfig
        approach_height = 0.12

        # Check container approach position
        container_approach = config.container_pose.position.copy()
        container_approach[2] += approach_height

        # Container must be reachable
        # IK workspace limit is X >= -0.6, so allow -0.60 (at the edge)
        # With container at -0.63 and max +0.03 noise, we get -0.60
        if container_approach[0] < -0.60:  # Actual IK limit
            return False

        # Check bowl approach positions
        for pose in config.bowl_poses.values():
            bowl_approach = pose.position.copy()
            bowl_approach[2] += approach_height

            # Check within workspace
            bounds = self.config.collision.workspace_bounds
            if not self._in_bounds(bowl_approach, bounds):
                return False

        return True

    def sample_multiple(
        self,
        count: int,
        target_bowl: Optional[str] = None,
    ) -> List[SceneConfiguration]:
        """
        Sample multiple scene configurations.

        Args:
            count: Number of configurations to sample
            target_bowl: Optional specific target bowl

        Returns:
            List of SceneConfigurations
        """
        configs = []
        for _ in range(count):
            configs.append(self.sample(target_bowl))
        return configs
