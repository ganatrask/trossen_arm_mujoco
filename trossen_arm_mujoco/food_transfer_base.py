#!/usr/bin/env python3
"""
Base classes for food transfer task.

This module provides shared components used by both the interactive
visualization script (food_transfer_ik.py) and the HDF5 recording
script (record_food_transfer_ik.py).

Components:
- TaskPhase: Enum defining the phases of a food transfer task
- TaskConfig: Dataclass with timing and height configuration
- FoodTransferBase: Base class with IK solving and object position logic
"""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import mujoco
import numpy as np

from trossen_arm_mujoco.mink_ik import TrossenMinkIK


# =============================================================================
# Collision Detection Constants
# =============================================================================

# Robot geoms that we want to check for collisions
# These are the collision geoms from teleop_follower_spoon.xml
ROBOT_COLLISION_GEOMS: Set[str] = {
    # Spoon collision geom
    "spoon_col",
    # Gripper collision geoms
    "gripper_left_col_1",
    "gripper_left_col_2",
    "gripper_left_tip",
    "gripper_right_col_1",
    "gripper_right_col_2",
    "gripper_right_tip",
    # Arm link collision geoms
    "link_6_col_1",
    "link_6_col_2",
    "camera_col",
    "link_5_col",
    "link_4_col",
    "link_3_col_1",
    "link_3_col_2",
    "link_2_col_1",
    "link_2_col_2",
    "base_link_col",
}

# Container collision geoms (from teleop_scene.xml)
CONTAINER_COLLISION_GEOMS: Set[str] = {
    "container_bottom",
    "container_front",
    "container_back",
    "container_left",
    "container_right",
}

# Bowl collision geoms (32 collision meshes per bowl, invisible)
# These are named "ramekin_collision_0" through "ramekin_collision_31"
# but MuJoCo may append suffixes for multiple instances
BOWL_COLLISION_PREFIXES: List[str] = [
    f"ramekin_collision_{i}" for i in range(32)
]

# All obstacle geoms (bowls + container)
# Note: Bowl geoms may have instance suffixes, so we use prefix matching
OBSTACLE_COLLISION_GEOMS: Set[str] = CONTAINER_COLLISION_GEOMS.copy()


class TaskPhase(Enum):
    """Task phases for food transfer."""
    HOME = auto()
    APPROACH_CONTAINER = auto()
    REACH_CONTAINER = auto()
    SCOOP = auto()
    LIFT = auto()
    APPROACH_BOWL = auto()
    LOWER_BOWL = auto()
    DUMP = auto()
    RETURN = auto()
    DONE = auto()


# Phase execution order
PHASE_ORDER = [
    TaskPhase.HOME,
    TaskPhase.APPROACH_CONTAINER,
    TaskPhase.REACH_CONTAINER,
    TaskPhase.SCOOP,
    TaskPhase.LIFT,
    TaskPhase.APPROACH_BOWL,
    TaskPhase.LOWER_BOWL,
    TaskPhase.DUMP,
    TaskPhase.RETURN,
    TaskPhase.DONE,
]


@dataclass
class TaskConfig:
    """Configuration for food transfer task."""
    # Heights above objects (meters)
    approach_height: float = 0.12      # Height for approach moves
    scoop_height: float = 0.06         # Height when scooping in container
    bowl_height: float = 0.08       # Height above bowl for dumping

    # Timing (seconds)
    move_duration: float = 2.0         # Time for each move
    scoop_duration: float = 1.5        # Time for scoop motion
    dump_duration: float = 1.0         # Time for dump motion
    hold_duration: float = 0.5         # Pause between phases

    # Wrist rotation for dump (radians)
    dump_rotation: float = -0.8        # Rotation for dumping food

    # Scoop rotation (radians)
    scoop_rotation: float = 0.8        # Rotation for scooping

    # Reward parameters
    reach_threshold: float = 0.06      # Distance threshold for "reached" (meters)
    dwell_time: float = 2.0            # Time spoon must stay near target (seconds)


# Default bowl positions (fallback if not found in simulation)
DEFAULT_BOWL_POSITIONS = {
    "bowl_1": np.array([-0.22, -0.26, 0.04]),
    "bowl_2": np.array([-0.36, -0.26, 0.04]),
    "bowl_3": np.array([-0.36, -0.12, 0.04]),
    "bowl_4": np.array([-0.22, -0.12, 0.04]),
}

# Default container position
DEFAULT_CONTAINER_POSITION = np.array([-0.63, -0.15, 0.04])

# All bowl names for cycling
ALL_BOWLS = ["bowl_1", "bowl_2", "bowl_3", "bowl_4"]

# Home joint position
HOME_JOINTS = np.array([0.085, 0.004, 0.006, 0.034, -0.029, -0.068])

# Simulation timestep
DT = 0.02  # 50 Hz


class FoodTransferBase:
    """
    Base class for food transfer task.

    Provides shared functionality for IK solving, object position lookup,
    and phase management. Subclasses add visualization or recording.
    """

    def __init__(self, target: str = "bowl_2", scene_xml: str = "wxai/teleop_scene.xml"):
        """
        Initialize food transfer task.

        Args:
            target: Which bowl to target ("bowl_1" to "bowl_4")
            scene_xml: Path to scene XML file, relative to assets/ directory
                       (e.g., "wxai/teleop_scene.xml" or "wxai/food_scene.xml")
        """
        self.target = target
        self.scene_xml = scene_xml
        self.config = TaskConfig()

        # Create IK solver
        self.ik = TrossenMinkIK(ee_name="spoon")

        # Create model/data for simulation
        assets_dir = Path(__file__).parent / "assets"
        model_path = assets_dir / scene_xml
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Object body IDs (will be looked up)
        self._container_id: Optional[int] = None
        self._bowl_ids: Dict[str, int] = {}

        # Current phase
        self.phase = TaskPhase.HOME

        # Home joint position
        self.home_joints = HOME_JOINTS.copy()

        # Reward tracking state
        self.max_reward = 2
        self._step_count = 0
        self._steps_per_second = int(1.0 / DT)  # 50 Hz
        self._container_enter_step: Optional[int] = None
        self._reached_container = False
        self._bowl_enter_step: Dict[str, Optional[int]] = {
            name: None for name in DEFAULT_BOWL_POSITIONS.keys()
        }
        self._reached_bowl = False

        self._lookup_object_ids()

    def _lookup_object_ids(self):
        """Look up body IDs for objects."""
        self._container_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "container"
        )

        for i in range(1, 5):
            name = f"bowl_{i}"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                self._bowl_ids[name] = body_id

    def get_container_position(self) -> np.ndarray:
        """Get current container position from simulation."""
        if self._container_id is None or self._container_id < 0:
            return DEFAULT_CONTAINER_POSITION.copy()
        return self.data.xpos[self._container_id].copy()

    def get_bowl_position(self, name: str) -> np.ndarray:
        """Get current bowl position from simulation."""
        if name not in self._bowl_ids:
            return DEFAULT_BOWL_POSITIONS.get(
                name, np.array([-0.3, -0.2, 0.04])
            ).copy()
        body_id = self._bowl_ids[name]
        return self.data.xpos[body_id].copy()

    def get_all_object_positions(self) -> Dict[str, np.ndarray]:
        """Get all object positions for debugging."""
        positions = {"container": self.get_container_position()}
        for name in self._bowl_ids:
            positions[name] = self.get_bowl_position(name)
        return positions

    def compute_target_for_phase(self, phase: TaskPhase) -> Optional[np.ndarray]:
        """
        Compute target XYZ position for given phase.

        Returns None if phase doesn't have a position target (uses joint target).
        """
        container_pos = self.get_container_position()
        bowl_pos = self.get_bowl_position(self.target)

        if phase == TaskPhase.HOME:
            return None

        elif phase == TaskPhase.APPROACH_CONTAINER:
            target = container_pos.copy()
            target[2] += self.config.approach_height
            return target

        elif phase == TaskPhase.REACH_CONTAINER:
            target = container_pos.copy()
            target[2] += self.config.scoop_height
            return target

        elif phase == TaskPhase.SCOOP:
            target = container_pos.copy()
            target[2] += self.config.scoop_height
            target[0] += 0.05  # Move forward slightly while scooping
            return target

        elif phase == TaskPhase.LIFT:
            target = container_pos.copy()
            target[2] += self.config.approach_height + 0.05
            return target

        elif phase == TaskPhase.APPROACH_BOWL:
            target = bowl_pos.copy()
            target[2] += self.config.approach_height
            return target

        elif phase == TaskPhase.LOWER_BOWL:
            target = bowl_pos.copy()
            target[2] += self.config.bowl_height
            return target

        elif phase == TaskPhase.DUMP:
            target = bowl_pos.copy()
            target[2] += self.config.bowl_height
            return target

        elif phase == TaskPhase.RETURN:
            return None

        return None

    def solve_for_phase(self, phase: TaskPhase, current_joints: np.ndarray) -> np.ndarray:
        """
        Solve IK for target phase.

        Args:
            phase: Target phase
            current_joints: Current joint configuration (for seeding IK)

        Returns:
            Target joint configuration
        """
        target_pos = self.compute_target_for_phase(phase)

        if target_pos is None:
            return self.home_joints.copy()

        # For SCOOP phase: rotate wrist roll to scoop
        if phase == TaskPhase.SCOOP:
            joints = current_joints.copy()
            joints[5] += self.config.scoop_rotation
            return joints

        # For DUMP phase: rotate wrist roll to pour out contents
        elif phase == TaskPhase.DUMP:
            joints = current_joints.copy()
            joints[5] = self.config.dump_rotation
            return joints

        # For other phases, use position-only IK
        joints, success, error = self.ik.solve(target_pos, initial_q=current_joints)

        if not success:
            print(f"  Warning: IK for {phase.name} did not converge (error={error:.4f}m)")

        return joints

    def get_phase_duration(self, phase: TaskPhase) -> float:
        """Get duration for a phase."""
        if phase == TaskPhase.SCOOP:
            return self.config.scoop_duration
        elif phase == TaskPhase.DUMP:
            return self.config.dump_duration
        elif phase in [TaskPhase.HOME, TaskPhase.DONE]:
            return self.config.hold_duration
        else:
            return self.config.move_duration

    def next_phase(self, phase: Optional[TaskPhase] = None) -> TaskPhase:
        """
        Get next phase in sequence.

        Args:
            phase: Phase to get next for. If None, uses self.phase.

        Returns:
            Next phase in sequence.
        """
        if phase is None:
            phase = self.phase

        try:
            idx = PHASE_ORDER.index(phase)
            if idx < len(PHASE_ORDER) - 1:
                return PHASE_ORDER[idx + 1]
        except ValueError:
            pass

        return TaskPhase.DONE

    def interpolate_joints(
        self,
        start_joints: np.ndarray,
        target_joints: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Smoothly interpolate between joint configurations.

        Args:
            start_joints: Starting joint configuration
            target_joints: Target joint configuration
            t: Interpolation factor (0.0 to 1.0)

        Returns:
            Interpolated joint configuration
        """
        # Smooth interpolation (ease in/out)
        t_smooth = 3 * t**2 - 2 * t**3
        return start_joints + t_smooth * (target_joints - start_joints)

    def reset(self):
        """Reset simulation and task state."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.phase = TaskPhase.HOME

        # Reset reward tracking state
        self._step_count = 0
        self._container_enter_step = None
        self._reached_container = False
        self._bowl_enter_step = {
            name: None for name in DEFAULT_BOWL_POSITIONS.keys()
        }
        self._reached_bowl = False

    def get_spoon_position(self) -> np.ndarray:
        """Get the current XYZ position of the spoon body."""
        spoon_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "spoon")
        return self.data.xpos[spoon_id].copy()

    def get_reward(self) -> int:
        """
        Computes the reward based on task progress.

        Reward stages:
            0: No reach (initial state)
            1: Reached container/bowl (spoon within threshold)
            2: Reached target bowl and stayed for dwell_time seconds

        Returns:
            The computed reward (0, 1, or 2).
        """
        self._step_count += 1
        spoon_pos = self.get_spoon_position()
        dwell_steps = int(self.config.dwell_time * self._steps_per_second)

        # Get actual container position from simulation
        container_pos = self.get_container_position()

        # Check container reach with dwell time (XY distance only)
        container_dist = np.linalg.norm(spoon_pos[:2] - container_pos[:2])
        if container_dist < self.config.reach_threshold:
            # Inside threshold zone
            if self._container_enter_step is None:
                self._container_enter_step = self._step_count
            # Check if we've dwelled long enough
            steps_in_zone = self._step_count - self._container_enter_step
            if steps_in_zone >= dwell_steps:
                self._reached_container = True
        else:
            # Left threshold zone, reset dwell timer
            self._container_enter_step = None

        # Check target bowl reach with dwell time (only the specified target)
        bowl_pos = self.get_bowl_position(self.target)
        bowl_dist = np.linalg.norm(spoon_pos[:2] - bowl_pos[:2])

        if bowl_dist < self.config.reach_threshold:
            # Inside threshold zone
            if self._bowl_enter_step[self.target] is None:
                self._bowl_enter_step[self.target] = self._step_count
            # Check if we've dwelled long enough
            steps_in_zone = self._step_count - self._bowl_enter_step[self.target]
            if steps_in_zone >= dwell_steps:
                self._reached_bowl = True
        else:
            # Left threshold zone, reset dwell timer
            self._bowl_enter_step[self.target] = None

        # Compute reward (monotonic - once reached, stays reached)
        if self._reached_bowl:
            return 2
        elif self._reached_container:
            return 1
        else:
            return 0

    def get_reward_info(self) -> Dict[str, any]:
        """
        Get detailed reward/evaluation info for logging.

        Returns:
            Dictionary with reward state details.
        """
        spoon_pos = self.get_spoon_position()
        container_pos = self.get_container_position()

        info = {
            "reward": self.get_reward(),
            "reached_container": self._reached_container,
            "reached_bowl": self._reached_bowl,
            "container_dist": float(np.linalg.norm(spoon_pos[:2] - container_pos[:2])),
            "bowl_dists": {},
        }

        for bowl_name in self._bowl_ids.keys():
            bowl_pos = self.get_bowl_position(bowl_name)
            info["bowl_dists"][bowl_name] = float(
                np.linalg.norm(spoon_pos[:2] - bowl_pos[:2])
            )

        return info

    # =========================================================================
    # Collision Detection Methods
    # =========================================================================

    def _is_robot_geom(self, geom_name: str) -> bool:
        """Check if a geom belongs to the robot."""
        if geom_name is None:
            return False
        return geom_name in ROBOT_COLLISION_GEOMS

    def _is_obstacle_geom(self, geom_name: str, geom_id: int) -> bool:
        """
        Check if a geom is an obstacle (bowl or container).

        Uses geom name for container and body ID for bowls.

        Args:
            geom_name: Name of the geom (may be None for unnamed geoms)
            geom_id: ID of the geom in the model
        """
        # Direct match for container geoms by name
        if geom_name is not None and geom_name in OBSTACLE_COLLISION_GEOMS:
            return True

        # For bowls, check if the geom's parent body is a bowl
        # This works even for unnamed geoms
        body_id = self.model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        if body_name is not None and body_name.startswith("bowl_"):
            return True

        return False

    def check_collisions(self) -> List[Tuple[str, str]]:
        """
        Check for collisions between robot and obstacles.

        Returns:
            List of (robot_geom, obstacle_geom/body) collision pairs.
            Empty list if no collisions.
        """
        collisions = []

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get geom names and IDs
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            geom1_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id
            )
            geom2_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id
            )

            # Check if this is a robot-obstacle collision
            is_robot1 = self._is_robot_geom(geom1_name)
            is_robot2 = self._is_robot_geom(geom2_name)
            is_obstacle1 = self._is_obstacle_geom(geom1_name, geom1_id)
            is_obstacle2 = self._is_obstacle_geom(geom2_name, geom2_id)

            if is_robot1 and is_obstacle2:
                # Get obstacle identifier (body name for bowls, geom name for container)
                obstacle_name = geom2_name
                if obstacle_name is None:
                    body_id = self.model.geom_bodyid[geom2_id]
                    obstacle_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                    )
                collisions.append((geom1_name, obstacle_name))
            elif is_robot2 and is_obstacle1:
                obstacle_name = geom1_name
                if obstacle_name is None:
                    body_id = self.model.geom_bodyid[geom1_id]
                    obstacle_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                    )
                collisions.append((geom2_name, obstacle_name))

        return collisions

    def has_collision(self) -> bool:
        """
        Check if there is any collision between robot and obstacles.

        Returns:
            True if collision detected, False otherwise.
        """
        return len(self.check_collisions()) > 0

    def get_collision_penalty(self, penalty: float = -1.0) -> float:
        """
        Get collision penalty for reward calculation.

        Args:
            penalty: Penalty value to return if collision detected.

        Returns:
            penalty if collision detected, 0.0 otherwise.
        """
        if self.has_collision():
            return penalty
        return 0.0

    def get_collision_info(self) -> Dict[str, any]:
        """
        Get detailed collision information for debugging.

        Returns:
            Dictionary with collision details.
        """
        collisions = self.check_collisions()

        info = {
            "has_collision": len(collisions) > 0,
            "num_collisions": len(collisions),
            "collision_pairs": collisions,
            "total_contacts": self.data.ncon,
        }

        # Categorize collisions by obstacle type
        bowl_collisions = []
        container_collisions = []

        for robot_geom, obstacle_name in collisions:
            if obstacle_name in CONTAINER_COLLISION_GEOMS:
                container_collisions.append((robot_geom, obstacle_name))
            elif obstacle_name is not None and obstacle_name.startswith("bowl_"):
                bowl_collisions.append((robot_geom, obstacle_name))
            else:
                bowl_collisions.append((robot_geom, obstacle_name))

        info["bowl_collisions"] = bowl_collisions
        info["container_collisions"] = container_collisions

        return info
