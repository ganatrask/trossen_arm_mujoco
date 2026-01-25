"""
Scene loader for domain randomization.

This module applies scene configurations to MuJoCo simulations,
modifying object positions and orientations at runtime.
"""

from typing import Dict, Optional
import mujoco
import numpy as np

from .config import ObjectPose, SceneConfiguration


class SceneLoader:
    """
    Loads scene configurations into MuJoCo simulation.

    This class modifies object positions in the MuJoCo model and
    resets the simulation to apply the changes.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Initialize the scene loader.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        self._body_ids: Dict[str, int] = {}
        self._cache_body_ids()

    def _cache_body_ids(self) -> None:
        """Cache body IDs for fast lookup."""
        # Container
        container_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "container"
        )
        if container_id >= 0:
            self._body_ids["container"] = container_id

        # Bowls (1-8)
        for i in range(1, 9):
            bowl_name = f"bowl_{i}"
            bowl_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, bowl_name
            )
            if bowl_id >= 0:
                self._body_ids[bowl_name] = bowl_id

    def apply(self, config: SceneConfiguration) -> None:
        """
        Apply scene configuration to simulation.

        For static bodies (attached to world), we modify model.body_pos
        and model.body_quat directly, then reset the simulation.

        Args:
            config: Scene configuration to apply
        """
        # Apply container pose
        if "container" in self._body_ids:
            self._set_body_pose("container", config.container_pose)

        # Apply bowl poses
        for bowl_name, pose in config.bowl_poses.items():
            if bowl_name in self._body_ids:
                self._set_body_pose(bowl_name, pose)

        # Hide inactive bowls (move far below table)
        self._hide_inactive_bowls(config.active_bowls)

        # Reset simulation to apply changes
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _set_body_pose(self, body_name: str, pose: ObjectPose) -> None:
        """
        Set pose for a static body by modifying the model.

        Args:
            body_name: Name of the body
            pose: Target pose
        """
        body_id = self._body_ids.get(body_name)
        if body_id is None:
            return

        # Modify model body position and quaternion
        self.model.body_pos[body_id] = pose.position
        self.model.body_quat[body_id] = pose.quaternion

    def _hide_inactive_bowls(self, active_bowls: list) -> None:
        """
        Hide bowls that are not active by moving them far below table.

        Args:
            active_bowls: List of bowl names that should remain visible
        """
        hidden_z = -10.0  # Far below table

        for bowl_name, body_id in self._body_ids.items():
            if bowl_name.startswith("bowl_") and bowl_name not in active_bowls:
                # Move far below table to hide
                current_pos = self.model.body_pos[body_id].copy()
                current_pos[2] = hidden_z
                self.model.body_pos[body_id] = current_pos

    def reset_to_nominal(self, nominal_positions: Dict[str, np.ndarray]) -> None:
        """
        Reset all objects to their nominal positions.

        Args:
            nominal_positions: Dictionary mapping object names to positions
        """
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

        for name, position in nominal_positions.items():
            if name in self._body_ids:
                body_id = self._body_ids[name]
                self.model.body_pos[body_id] = position
                self.model.body_quat[body_id] = identity_quat

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def get_body_pose(self, body_name: str) -> Optional[ObjectPose]:
        """
        Get current pose of a body from simulation.

        Args:
            body_name: Name of the body

        Returns:
            ObjectPose or None if body not found
        """
        body_id = self._body_ids.get(body_name)
        if body_id is None:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id < 0:
                return None

        # Get position from data.xpos (world position after forward)
        position = self.data.xpos[body_id].copy()

        # Get quaternion from data.xquat
        quaternion = self.data.xquat[body_id].copy()

        return ObjectPose(position=position, quaternion=quaternion)

    def get_all_object_poses(self) -> Dict[str, ObjectPose]:
        """
        Get poses of all tracked objects.

        Returns:
            Dictionary mapping object names to ObjectPose
        """
        poses = {}
        for name in self._body_ids.keys():
            pose = self.get_body_pose(name)
            if pose is not None:
                poses[name] = pose
        return poses

    @property
    def available_bodies(self) -> list:
        """Return list of body names that were found in the model."""
        return list(self._body_ids.keys())
