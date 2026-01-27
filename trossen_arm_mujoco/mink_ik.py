"""
Mink IK solver for Trossen WXAI arm with spoon follower.

This module provides inverse kinematics using the mink library for the
teleop follower spoon robot configuration.

Usage:
    from trossen_arm_mujoco.mink_ik import TrossenMinkIK

    # Create solver
    ik = TrossenMinkIK()

    # Solve for target position
    joints, success, error = ik.solve(target_xyz=[0.25, 0.0, 0.15])

    # Or use with existing dm_control physics
    ik = TrossenMinkIK.from_physics(physics)
    joints, success, error = ik.solve(target_xyz)
"""

import numpy as np
import mujoco
import mink
from pathlib import Path
from typing import Optional, Tuple, Union

# Path to assets
ASSETS_DIR = Path(__file__).parent / "assets"
DEFAULT_MODEL = ASSETS_DIR / "wxai" / "teleop_scene.xml"
FOLLOWER_SPOON_MODEL = ASSETS_DIR / "food_task" / "teleop_follower_spoon.xml"


class TrossenMinkIK:
    """
    Mink IK solver for Trossen WXAI arm with spoon follower.

    The robot has 6 arm joints (joint_0 to joint_5) plus 2 coupled gripper joints.

    End-effector options:
        - "spoon": The spoon body attached to link_6 (default)
        - "camera_color_frame": Camera site on link_6
        - "link_6": The wrist body

    Example:
        ik = TrossenMinkIK()
        joints, success, error = ik.solve([0.25, 0.0, 0.15])

        # With orientation control
        joints, success, error = ik.solve(
            target_xyz=[0.25, 0.0, 0.15],
            target_quat=[1, 0, 0, 0]  # [w, x, y, z]
        )
    """

    # Joint names for the 6-DOF arm
    ARM_JOINTS = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]

    # End-effector options
    EE_SPOON = "spoon"           # Spoon body (primary EE)
    EE_CAMERA = "camera_color_frame"  # Camera site
    EE_WRIST = "link_6"          # Wrist body

    def __init__(
        self,
        model_path: Optional[str] = None,
        ee_name: str = "spoon",
        position_cost: float = 1.0,
        orientation_cost: float = 0.0,
        dt: float = 0.01,
        max_iters: int = 150,
        pos_threshold: float = 1e-4,
        damping: float = 1e-3,
    ):
        """
        Initialize the mink IK solver.

        Args:
            model_path: Path to MJCF XML file. Defaults to teleop_scene.xml
            ee_name: End-effector frame name ("spoon", "camera_color_frame", or "link_6")
            position_cost: Weight for position error in IK objective
            orientation_cost: Weight for orientation error (0 = position-only IK)
            dt: Integration timestep for IK iterations
            max_iters: Maximum IK iterations
            pos_threshold: Position error threshold for convergence (meters)
            damping: Damping factor for regularization
        """
        # Load model
        if model_path is None:
            model_path = str(DEFAULT_MODEL)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Setup mink configuration
        self.configuration = mink.Configuration(self.model)

        # Determine frame type
        self.ee_name = ee_name
        self.frame_type = self._get_frame_type(ee_name)

        # print(f"Mink IK initialized with end-effector: {ee_name} (type: {self.frame_type})")

        # Create IK task
        self.ee_task = mink.FrameTask(
            frame_name=self.ee_name,
            frame_type=self.frame_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1.0,
        )

        # IK parameters
        self.dt = dt
        self.max_iters = max_iters
        self.pos_threshold = pos_threshold
        self.damping = damping

        # Joint indices for arm (first 6 joints)
        self.n_arm_joints = 6
        self.n_total_joints = self.model.nq  # Usually 8 (6 arm + 2 gripper)

        # Get joint limits
        self._joint_limits = self._get_joint_limits()

    def _get_frame_type(self, name: str) -> str:
        """Determine if the frame is a site or body."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return "site"

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return "body"

        raise ValueError(f"Frame '{name}' not found as site or body in model")

    def _get_joint_limits(self) -> np.ndarray:
        """Get joint position limits for the arm joints."""
        limits = np.zeros((self.n_arm_joints, 2))
        for i in range(self.n_arm_joints):
            limits[i, 0] = self.model.jnt_range[i, 0]
            limits[i, 1] = self.model.jnt_range[i, 1]
        return limits

    @classmethod
    def from_physics(
        cls,
        physics,
        ee_name: str = "spoon",
        **kwargs
    ) -> "TrossenMinkIK":
        """
        Create IK solver from an existing dm_control physics instance.

        This allows using mink IK alongside dm_control simulation.

        Args:
            physics: dm_control physics instance
            ee_name: End-effector frame name
            **kwargs: Additional arguments passed to __init__

        Returns:
            TrossenMinkIK instance
        """
        # Get the XML path from the physics model if available
        # For now, we create a new instance with default model
        instance = cls(ee_name=ee_name, **kwargs)
        return instance

    def get_ee_position(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get end-effector position for given joint configuration.

        Args:
            q: Joint configuration. If None, uses current configuration.

        Returns:
            [x, y, z] position
        """
        if q is not None:
            self.data.qpos[:len(q)] = q
            mujoco.mj_forward(self.model, self.data)

        if self.frame_type == "site":
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_name)
            return self.data.site_xpos[site_id].copy()
        else:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_name)
            return self.data.xpos[body_id].copy()

    def get_ee_pose(self, q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get end-effector position and orientation.

        Args:
            q: Joint configuration. If None, uses current configuration.

        Returns:
            (position [x,y,z], quaternion [w,x,y,z])
        """
        if q is not None:
            self.data.qpos[:len(q)] = q
            mujoco.mj_forward(self.model, self.data)

        if self.frame_type == "site":
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_name)
            pos = self.data.site_xpos[site_id].copy()
            # Get rotation matrix and convert to quaternion
            xmat = self.data.site_xmat[site_id].reshape(3, 3)
        else:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_name)
            pos = self.data.xpos[body_id].copy()
            xmat = self.data.xmat[body_id].reshape(3, 3)

        # Convert rotation matrix to quaternion [w, x, y, z]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, xmat.flatten())

        return pos, quat

    def _set_initial_config(self, initial_q: Optional[np.ndarray]) -> None:
        """Set initial joint configuration for IK, padding if needed."""
        if initial_q is not None:
            if len(initial_q) < self.n_total_joints:
                full_q = np.zeros(self.n_total_joints)
                full_q[:len(initial_q)] = initial_q
                initial_q = full_q
            self.configuration.update(initial_q)

    def _solve_ik_loop(self, task: mink.FrameTask) -> Tuple[np.ndarray, float, float]:
        """
        Run the iterative IK solve loop.

        Args:
            task: The mink FrameTask to solve for

        Returns:
            (q_solution, pos_error, ori_error)
        """
        pos_error = float('inf')
        ori_error = float('inf')

        for _ in range(self.max_iters):
            vel = mink.solve_ik(
                self.configuration,
                [task],
                self.dt,
                solver="daqp",
                damping=self.damping,
            )

            self.configuration.integrate_inplace(vel, self.dt)

            # Check convergence
            err = task.compute_error(self.configuration)
            pos_error = np.linalg.norm(err[:3])
            ori_error = np.linalg.norm(err[3:])

            if pos_error < self.pos_threshold:
                break

        return self.configuration.q.copy(), pos_error, ori_error

    def solve(
        self,
        target_xyz: Union[np.ndarray, list],
        target_quat: Optional[Union[np.ndarray, list]] = None,
        initial_q: Optional[np.ndarray] = None,
        return_full: bool = False,
    ) -> Tuple[np.ndarray, bool, float]:
        """
        Solve IK for target position (and optionally orientation).

        Args:
            target_xyz: Target [x, y, z] position in world frame
            target_quat: Target orientation as [w, x, y, z] quaternion (optional)
            initial_q: Initial joint configuration for IK seed (optional)
            return_full: If True, return full qpos (8 joints), else arm only (6 joints)

        Returns:
            (joint_positions, success, position_error)
            - joint_positions: Solution joint angles (6 or 8 values)
            - success: True if converged within threshold
            - position_error: Final position error in meters
        """
        target_xyz = np.array(target_xyz, dtype=np.float64)

        self._set_initial_config(initial_q)

        # Set target pose
        if target_quat is not None:
            target_quat = np.array(target_quat, dtype=np.float64)
            target_pose = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(target_quat),
                translation=target_xyz,
            )
        else:
            # Position-only target - use identity rotation
            target_pose = mink.SE3.from_translation(target_xyz)
        self.ee_task.set_target(target_pose)

        # Solve iteratively
        q_solution, pos_error, _ = self._solve_ik_loop(self.ee_task)
        success = pos_error < self.pos_threshold

        if return_full:
            return q_solution, success, pos_error
        else:
            return q_solution[:self.n_arm_joints], success, pos_error

    def solve_with_orientation(
        self,
        target_xyz: Union[np.ndarray, list],
        target_quat: Union[np.ndarray, list],
        initial_q: Optional[np.ndarray] = None,
        orientation_cost: float = 1.0,
    ) -> Tuple[np.ndarray, bool, float, float]:
        """
        Solve IK with both position and orientation constraints.

        Creates a temporary task with orientation cost for this solve.

        Args:
            target_xyz: Target [x, y, z] position
            target_quat: Target [w, x, y, z] quaternion
            initial_q: Initial joint configuration
            orientation_cost: Weight for orientation error

        Returns:
            (joint_positions, success, position_error, orientation_error)
        """
        target_xyz = np.array(target_xyz, dtype=np.float64)
        target_quat = np.array(target_quat, dtype=np.float64)

        # Create temporary task with orientation cost
        ori_task = mink.FrameTask(
            frame_name=self.ee_name,
            frame_type=self.frame_type,
            position_cost=1.0,
            orientation_cost=orientation_cost,
            lm_damping=1.0,
        )

        self._set_initial_config(initial_q)

        # Set target pose with orientation
        target_pose = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(target_quat),
            translation=target_xyz,
        )
        ori_task.set_target(target_pose)

        # Solve iteratively
        q_solution, pos_error, ori_error = self._solve_ik_loop(ori_task)
        success = pos_error < self.pos_threshold

        return q_solution[:self.n_arm_joints], success, pos_error, ori_error

    def solve_trajectory(
        self,
        waypoints: list,
        steps_per_segment: int = 50,
        smooth: bool = True,
    ) -> np.ndarray:
        """
        Generate joint trajectory through Cartesian waypoints.

        Args:
            waypoints: List of [x, y, z] positions
            steps_per_segment: Number of interpolation steps between waypoints
            smooth: Use smooth (cubic) interpolation vs linear

        Returns:
            Joint trajectory array [T, n_arm_joints]
        """
        trajectory = []

        for i in range(len(waypoints) - 1):
            start_pos = np.array(waypoints[i])
            end_pos = np.array(waypoints[i + 1])

            for step in range(steps_per_segment):
                t = step / steps_per_segment

                if smooth:
                    # Smooth cubic interpolation
                    t_smooth = 3 * t**2 - 2 * t**3
                else:
                    t_smooth = t

                target = start_pos + t_smooth * (end_pos - start_pos)

                joints, success, error = self.solve(target)
                if not success and error > 0.01:
                    print(f"Warning: IK failed at waypoint {i}, step {step}, error={error:.4f}")

                trajectory.append(joints)

        # Add final waypoint
        joints, _, _ = self.solve(waypoints[-1])
        trajectory.append(joints)

        return np.array(trajectory)


def solve_ik_mink(
    physics,
    target_pos: np.ndarray,
    ee_name: str = "spoon",
    max_iterations: int = 150,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """
    Drop-in replacement for the existing solve_ik function using mink.

    This function provides API compatibility with the existing Jacobian-based
    IK solver in scripted_policy_single_arm.py.

    Args:
        physics: dm_control physics instance
        target_pos: Target XYZ position [x, y, z]
        ee_name: End-effector frame name
        max_iterations: Maximum IK iterations
        tolerance: Position error tolerance

    Returns:
        Joint angles [j1, j2, j3, j4, j5, j6] that reach target
    """
    # Create solver (cached in practice for performance)
    ik = TrossenMinkIK(
        ee_name=ee_name,
        max_iters=max_iterations,
        pos_threshold=tolerance,
    )

    # Use current joint positions as seed
    initial_q = physics.data.qpos[:8].copy()

    # Solve
    joints, success, error = ik.solve(
        target_pos,
        initial_q=initial_q,
        return_full=False,
    )

    if not success:
        print(f"Warning: Mink IK did not converge, error={error:.6f}")

    return joints


# Convenience function for quick testing
def test_mink_ik():
    """Test the mink IK solver with the teleop scene."""
    print("=" * 60)
    print("Testing Mink IK for Trossen Teleop Spoon Follower")
    print("=" * 60)

    # Create solver
    ik = TrossenMinkIK(ee_name="spoon")

    # Get current EE position
    current_pos = ik.get_ee_position()
    print(f"\nCurrent EE position: {current_pos}")

    # Test targets (relative to scene)
    test_targets = [
        [0.0, 0.0, 0.2],    # Above robot
        [-0.3, 0.0, 0.15],  # Forward
        [-0.2, 0.1, 0.2],   # Forward-left
        [-0.4, -0.1, 0.1],  # Far forward-right (near bowls)
    ]

    print("\nTesting IK solutions:")
    print("-" * 60)

    for i, target in enumerate(test_targets):
        joints, success, error = ik.solve(target)
        status = "✓" if success else "✗"
        print(f"{status} Target {i+1}: {target}")
        print(f"   Solution: {np.round(joints, 4)}")
        print(f"   Error: {error:.6f} m")

        # Verify solution
        actual_pos = ik.get_ee_position(joints)
        print(f"   Actual:  {np.round(actual_pos, 4)}")

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    test_mink_ik()
