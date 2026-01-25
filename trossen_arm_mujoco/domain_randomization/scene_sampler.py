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

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def sample(self, target_bowl: Optional[str] = None) -> SceneConfiguration:
        """
        Sample a collision-free scene configuration.

        Args:
            target_bowl: Specific target bowl to use. If None, randomly selected.

        Returns:
            SceneConfiguration with randomized poses

        Raises:
            RuntimeError: If unable to find valid configuration after max attempts
        """
        # First, determine number of bowls and scene
        num_bowls = self.rng.integers(
            self.config.scene_variant.min_bowls,
            self.config.scene_variant.max_bowls + 1
        )
        scene_xml = self.config.scene_variant.get_scene_xml(num_bowls)
        available_bowls = self.config.scene_variant.get_available_bowls(scene_xml)

        # Select active bowls
        if target_bowl is None:
            target_bowl = self.rng.choice(available_bowls)
        elif target_bowl not in available_bowls:
            # If specified target not available, pick one that is
            target_bowl = self.rng.choice(available_bowls)

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
        # Sample container pose
        container_pose = self._sample_container_pose()

        # Sample bowl poses
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
        Sample container pose with asymmetric noise.

        Container is at workspace edge, so only allows movement
        TOWARD the robot (+X direction).
        """
        nominal = self.nominal["container"].copy()
        cfg = self.config.container_pose

        if self.config.randomize_container:
            # Asymmetric X noise: only toward robot (positive direction)
            dx = self.rng.uniform(cfg.position_noise_x_min, cfg.position_noise_x_max)
            dy = self.rng.uniform(-cfg.position_noise_y, cfg.position_noise_y)
            dz = self.rng.uniform(-cfg.position_noise_z, cfg.position_noise_z)

            position = nominal + np.array([dx, dy, dz])

            # Yaw rotation (around Z axis)
            yaw = self.rng.uniform(-cfg.rotation_noise_yaw, cfg.rotation_noise_yaw)
            quaternion = self._yaw_to_quaternion(yaw)
        else:
            position = nominal
            quaternion = IDENTITY_QUAT.copy()

        return ObjectPose(position=position, quaternion=quaternion)

    def _sample_bowl_pose(self, bowl_name: str) -> ObjectPose:
        """Sample bowl pose with symmetric noise."""
        nominal = self.nominal[bowl_name].copy()
        cfg = self.config.bowl_pose

        if self.config.randomize_bowls:
            # Symmetric XY noise
            dx = self.rng.uniform(-cfg.position_noise_xy, cfg.position_noise_xy)
            dy = self.rng.uniform(-cfg.position_noise_xy, cfg.position_noise_xy)
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

    def _validate_no_collisions(self, config: SceneConfiguration) -> bool:
        """Check minimum spacing between all objects."""
        min_spacing = self.config.collision.min_object_spacing

        # Collect all object positions
        all_positions = [config.container_pose.position]
        for pose in config.bowl_poses.values():
            all_positions.append(pose.position)

        # Check pairwise distances (XY only, Z doesn't matter for table objects)
        for i, p1 in enumerate(all_positions):
            for p2 in all_positions[i + 1:]:
                dist_xy = np.linalg.norm(p1[:2] - p2[:2])
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
