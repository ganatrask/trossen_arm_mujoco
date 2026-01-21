"""
Scripted policy for single-arm manipulation tasks.

This module provides waypoint-based trajectory control for a single WXAI arm
using joint-space control (qpos), not Cartesian control.

Supports both:
- Direct joint angles (qpos)
- XYZ positions (converted to joint angles via IK)

Action format: [joint1, joint2, joint3, joint4, joint5, joint6, gripper_left, gripper_right]
- Joints 0-5: Arm joint angles (radians)
- Joints 6-7: Gripper fingers (coupled, 0.044 = open, 0.012 = closed)
"""

import argparse
import time

import matplotlib.pyplot as plt
import mujoco
from dm_control.mujoco.engine import Physics
from dm_control.suite import base
from dm_env import TimeStep
from mujoco import viewer as mj_viewer
import numpy as np

from trossen_arm_mujoco.constants import START_ARM_POSE
from trossen_arm_mujoco.utils import (
    get_observation_base,
    make_sim_env,
    plot_observation_images,
    set_observation_images,
)
from trossen_arm_mujoco.assets.food_task.single_arm_env import FoodTransferTask


def solve_ik(
    physics: Physics,
    target_pos: np.ndarray,
    site_name: str = "camera_color_frame",
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """
    Solve inverse kinematics to find joint angles for a target XYZ position.

    Uses MuJoCo's Jacobian-based IK with damped least squares.

    :param physics: The MuJoCo physics instance.
    :param target_pos: Target XYZ position [x, y, z].
    :param site_name: Name of the site to move to target (end-effector).
    :param max_iterations: Maximum IK iterations.
    :param tolerance: Position error tolerance.
    :return: Joint angles [j1, j2, j3, j4, j5, j6] that reach target.
    """
    model = physics.model.ptr
    data = physics.data.ptr

    # Get site ID
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        # Fallback to link_6 body
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_6")
        use_body = True
    else:
        use_body = False

    # Save original qpos
    original_qpos = data.qpos.copy()

    # IK parameters
    damping = 0.1
    step_size = 0.5

    for iteration in range(max_iterations):
        mujoco.mj_forward(model, data)

        # Get current end-effector position
        if use_body:
            current_pos = data.xpos[body_id].copy()
        else:
            current_pos = data.site_xpos[site_id].copy()

        # Compute position error
        error = target_pos - current_pos
        error_norm = np.linalg.norm(error)

        if error_norm < tolerance:
            break

        # Compute Jacobian (3 x nv for position only)
        jacp = np.zeros((3, model.nv))
        if use_body:
            mujoco.mj_jacBody(model, data, jacp, None, body_id)
        else:
            mujoco.mj_jacSite(model, data, jacp, None, site_id)

        # Only use first 6 columns (arm joints)
        J = jacp[:, :6]

        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 * error
        JJT = J @ J.T + damping**2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error)

        # Update joint angles
        data.qpos[:6] += step_size * dq
        mujoco.mj_forward(model, data)

    # Get result
    result_qpos = data.qpos[:6].copy()

    # Restore original qpos
    data.qpos[:] = original_qpos
    mujoco.mj_forward(model, data)

    return result_qpos


class SingleArmTask(base.Task):
    """
    Single-arm task for scripted policy testing.
    Uses joint-space control (8 DOF: 6 arm + 2 gripper).
    """

    def __init__(
        self,
        random: int | None = None,
        onscreen_render: bool = False,
        cam_list: list[str] = [],
    ):
        super().__init__(random=random)
        self.cam_list = cam_list if cam_list else ["cam_high"]

    def before_step(self, action: np.ndarray, physics: Physics) -> None:
        """Process action before simulation step."""
        # Handle 8-value qpos action -> 7-value actuator control
        if action.shape[0] == 7:
            super().before_step(action, physics)
            return
        if action.shape[0] != 8:
            raise ValueError("Expected action length 7 (ctrl) or 8 (qpos).")

        arm_action = action[:6]
        gripper_action = action[6]
        env_action = np.concatenate([arm_action, [gripper_action]])
        super().before_step(env_action, physics)

    def initialize_episode(self, physics: Physics) -> None:
        """Reset robot to start pose."""
        with physics.reset_context():
            physics.named.data.qpos[:8] = START_ARM_POSE[:8]
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics: Physics) -> np.ndarray:
        return physics.data.qpos.copy()

    def get_observation(self, physics: Physics) -> dict:
        obs = get_observation_base(physics, self.cam_list)
        obs["qpos"] = physics.data.qpos.copy()[:8]
        obs["qvel"] = physics.data.qvel.copy()[:8]
        obs["env_state"] = self.get_env_state(physics)
        return obs

    def get_reward(self, physics: Physics) -> int:
        return 0


class BaseSingleArmPolicy:
    """
    Base class for single-arm trajectory-based policies using joint control.

    :param inject_noise: Whether to inject noise into actions for robustness testing.
    """

    def __init__(self, inject_noise: bool = False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory: list[dict] = []
        self.curr_waypoint: dict = None
        self.physics: Physics = None

    def set_physics(self, physics: Physics):
        """Set the physics instance for IK solving."""
        self.physics = physics

    def generate_trajectory(self, ts_first: TimeStep, physics: Physics = None):
        """
        Generate a trajectory based on the initial timestep.

        :param ts_first: The first observation of the episode.
        :param physics: Physics instance for IK solving (optional).
        :raises NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def interpolate(
        curr_waypoint: dict,
        next_waypoint: dict,
        t: int,
    ) -> np.ndarray:
        """
        Linearly interpolates joint positions between two waypoints.

        :param curr_waypoint: The current waypoint with 't' and 'qpos'.
        :param next_waypoint: The next waypoint with 't' and 'qpos'.
        :param t: The current timestep.
        :return: Interpolated joint positions (8 values).
        """
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_qpos = curr_waypoint["qpos"]
        next_qpos = next_waypoint["qpos"]
        qpos = curr_qpos + (next_qpos - curr_qpos) * t_frac
        return qpos

    def __call__(self, ts: TimeStep) -> np.ndarray:
        """
        Executes the policy for one timestep.

        :param ts: The current observation timestep.
        :return: The computed action (8 joint values) for the current timestep.
        """
        # Generate trajectory at first timestep
        if self.step_count == 0:
            self.generate_trajectory(ts, self.physics)

        # Get current and next waypoints
        if len(self.trajectory) > 0 and self.trajectory[0]["t"] == self.step_count:
            self.curr_waypoint = self.trajectory.pop(0)

        # If we're at the last waypoint, stay there
        if len(self.trajectory) == 0:
            action = self.curr_waypoint["qpos"].copy()
        else:
            next_waypoint = self.trajectory[0]
            action = self.interpolate(self.curr_waypoint, next_waypoint, self.step_count)

        # Inject noise if enabled
        if self.inject_noise:
            scale = 0.01
            action[:6] = action[:6] + np.random.uniform(-scale, scale, 6)

        self.step_count += 1
        return action


class BowlToPlatePolicy(BaseSingleArmPolicy):
    """
    Policy for scooping from bowl and transferring to plate.

    The scene has:
    - Bowl (bradshaw_bowl) at position (-0.05, 0.15, 0.04)
    - Plate at position (-0.05, -0.15, 0.04)
    - Food particles in the bowl
    """

    def generate_trajectory(self, ts_first: TimeStep, physics: Physics = None):
        """
        Generates a trajectory to scoop from bowl and move to plate.

        Uses IK to convert XYZ marker positions to joint angles.

        Trajectory:
        1. Move above the bowl (marker at -0.0496, 0.1520, 0.2497)
        2. Move above the plate (marker at -0.0447, -0.1417, 0.2019)
        3. Return home

        :param ts_first: The first observation of the episode.
        :param physics: Physics instance for IK solving (optional).
        """
        # Get starting joint positions
        start_qpos = ts_first.observation["qpos"].copy()
        print(f"Starting qpos: {start_qpos}")

        # Gripper values
        GRIPPER_OPEN = 0.044

        # Home/start position
        home_qpos = start_qpos.copy()

        # Target XYZ positions (from pose_helper marker positions)
        above_bowl_xyz = np.array([-0.0496, 0.1520, 0.2497])
        above_plate_xyz = np.array([-0.0447, -0.1417, 0.2019])

        # Solve IK if physics is available
        if physics is not None:
            print("Solving IK for target positions...")
            above_bowl_joints = solve_ik(physics, above_bowl_xyz)
            above_plate_joints = solve_ik(physics, above_plate_xyz)
            print(f"  Above bowl IK solution: {above_bowl_joints}")
            print(f"  Above plate IK solution: {above_plate_joints}")
        else:
            # Fallback to pre-computed values
            print("No physics provided, using pre-computed joint angles")
            above_bowl_joints = np.array([0.051, 0.577, 0.671, -0.098, 0.001, -0.001])
            above_plate_joints = np.array([-0.207, 0.495, 0.263, -0.107, 0.154, 0.006])

        # Build full qpos with gripper
        above_bowl_qpos = np.concatenate([above_bowl_joints, [GRIPPER_OPEN, GRIPPER_OPEN]])
        above_plate_qpos = np.concatenate([above_plate_joints, [GRIPPER_OPEN, GRIPPER_OPEN]])

        # Build trajectory with timing
        self.trajectory = [
            {"t": 0,   "qpos": home_qpos},            # Start at home
            {"t": 80,  "qpos": above_bowl_qpos},      # Point 1: Above bowl
            {"t": 140, "qpos": above_bowl_qpos},      # Pause above bowl
            {"t": 220, "qpos": above_plate_qpos},     # Point 2: Above plate
            {"t": 280, "qpos": above_plate_qpos},     # Pause above plate
            {"t": 360, "qpos": home_qpos},            # Return home
            {"t": 400, "qpos": home_qpos},            # Stay at home
        ]

        print("Generated bowl-to-plate trajectory:")
        print(f"  Point 1: Above bowl (xyz={above_bowl_xyz})")
        print(f"  Point 2: Above plate (xyz={above_plate_xyz})")
        for wp in self.trajectory:
            print(f"  t={wp['t']:3d}: joints={wp['qpos'][:6]}")


class TeleopPolicy(BaseSingleArmPolicy):
    """
    Policy based on teleoperated trajectory with key waypoints.
    Robot interpolates smoothly between waypoints.
    """

    def generate_trajectory(self, ts_first: TimeStep, physics: Physics = None):
        """
        Generates trajectory from key waypoints of recorded teleoperation.

        :param ts_first: The first observation of the episode.
        :param physics: Physics instance (unused, for API compatibility).
        """
        GRIPPER_OPEN = 0.044

        # Key waypoints extracted from teleoperation recording
        # Each is [j1, j2, j3, j4, j5, j6, gripper_l, gripper_r]

        # t=0s: Start/home position
        home = np.array([0.0849, 0.0036, 0.0059, 0.0334, -0.0292, -0.0681, GRIPPER_OPEN, GRIPPER_OPEN])

        # t=3s: Reach into bowl (arm extended)
        reach_bowl = np.array([0.5037, 1.8320, 1.8053, -0.7662, 0.2317, -0.8711, GRIPPER_OPEN, GRIPPER_OPEN])

        # t=8s: Scoop position (wrist rotated)
        scoop = np.array([0.0772, 1.7725, 1.7874, -0.6491, -0.7898, -0.6094, GRIPPER_OPEN, GRIPPER_OPEN])

        # t=12s: Move to bowl_2 area (front right bowl at -0.36, -0.26)
        above_plate = np.array([0.4080, 1.1198, 0.8974, -0.3222, -0.6323, -0.5343, GRIPPER_OPEN, GRIPPER_OPEN])

        # t=19s: Dump food (wrist rotation)
        dump = np.array([0.4736, 1.1870, 0.8703, -0.1677, -0.5152, -1.8313, GRIPPER_OPEN, GRIPPER_OPEN])

        # t=22s: Return position
        return_pos = np.array([0.2672, 1.1156, 1.1976, -0.3962, -0.3302, -0.1829, GRIPPER_OPEN, GRIPPER_OPEN])

        # Build trajectory (50 steps per second)
        self.trajectory = [
            {"t": 0,    "qpos": home},
            {"t": 150,  "qpos": reach_bowl},   # ~3s
            {"t": 400,  "qpos": scoop},        # ~8s
            {"t": 600,  "qpos": above_plate},  # ~12s
            {"t": 950,  "qpos": dump},         # ~19s
            {"t": 1100, "qpos": return_pos},   # ~22s
        ]

        print("Generated teleop trajectory (bowl_2) with key waypoints:")
        for wp in self.trajectory:
            print(f"  t={wp['t']:4d}: joints={np.round(wp['qpos'][:6], 3)}")


class TeleopPolicy2(BaseSingleArmPolicy):
    """
    Policy based on teleoperated trajectory targeting bowl_3 (back left).
    Robot interpolates smoothly between waypoints.
    """

    def generate_trajectory(self, ts_first: TimeStep, physics: Physics = None):
        """
        Generates trajectory from key waypoints targeting bowl_3.

        :param ts_first: The first observation of the episode.
        :param physics: Physics instance (unused, for API compatibility).
        """
        GRIPPER_OPEN = 0.044

        # Key waypoints extracted from teleoperation recording
        # Each is [j1, j2, j3, j4, j5, j6, gripper_l, gripper_r]

        # t=0s: Start/home position
        home = np.array([0.0849, 0.0036, 0.0059, 0.0334, -0.0292, -0.0681, GRIPPER_OPEN, GRIPPER_OPEN])

        # t=3s: Reach into bowl (arm extended)
        reach_bowl = np.array([0.5037, 1.8320, 1.8053, -0.7662, 0.2317, -0.8711, GRIPPER_OPEN, GRIPPER_OPEN])

        # t=8s: Scoop position (wrist rotated)
        scoop = np.array([0.0772, 1.7725, 1.7874, -0.6491, -0.7898, -0.6094, GRIPPER_OPEN, GRIPPER_OPEN])

        # t=10s: Lift position - raise arm to clear container before moving to bowl_3
        # Lower joint_1 and joint_2 to lift the arm higher
        lift = np.array([0.0772, 1.2, 1.2, -0.6491, -0.7898, -0.6094, GRIPPER_OPEN, GRIPPER_OPEN])

        # t=14s: Move to bowl_3 area (back left bowl at -0.36, -0.12)
        # Tuned using pose_tuner.py
        above_plate = np.array([0.1937, 1.1011, 0.9224, -0.9347, -0.5698, -0.5343, GRIPPER_OPEN, GRIPPER_OPEN])

        # t=21s: Dump food (wrist rotation) - adjusted joint_0 similarly
        dump = np.array([0.32, 1.1870, 0.8703, -0.1677, -0.5152, -1.8313, GRIPPER_OPEN, GRIPPER_OPEN])

        # t=20s: Return position
        return_pos = np.array([0.2672, 1.1156, 1.1976, -0.3962, -0.3302, -0.1829, GRIPPER_OPEN, GRIPPER_OPEN])

        # Build trajectory (50 steps per second, total 22s = 1100 steps)
        self.trajectory = [
            {"t": 0,    "qpos": home},
            {"t": 150,  "qpos": reach_bowl},   # ~3s
            {"t": 350,  "qpos": scoop},        # ~7s
            {"t": 450,  "qpos": lift},         # ~9s - lift to clear container
            {"t": 600,  "qpos": above_plate},  # ~12s
            {"t": 850,  "qpos": dump},         # ~17s
            {"t": 1000, "qpos": return_pos},   # ~20s
            {"t": 1100, "qpos": return_pos},   # ~22s - hold at end
        ]

        print("Generated teleop_2 trajectory (bowl_3) with key waypoints:")
        for wp in self.trajectory:
            print(f"  t={wp['t']:4d}: joints={np.round(wp['qpos'][:6], 3)}")


class SimplePickPolicy(BaseSingleArmPolicy):
    """
    Simple policy demonstrating pick motion - move to a position and close gripper.
    """

    def generate_trajectory(self, ts_first: TimeStep):
        """
        Generates a simple pick trajectory.

        :param ts_first: The first observation of the episode.
        """
        start_qpos = ts_first.observation["qpos"].copy()

        GRIPPER_OPEN = 0.044
        GRIPPER_CLOSED = 0.012

        # Target position (reach forward and down)
        target_qpos = np.array([
            0.0,   # j1: base rotation
            0.9,   # j2: shoulder forward
            0.9,   # j3: elbow bent
            0.0,   # j4: wrist rotation
            0.3,   # j5: wrist bend
            0.0,   # j6: gripper rotation
            GRIPPER_OPEN, GRIPPER_OPEN
        ])

        # Close gripper
        grasp_qpos = target_qpos.copy()
        grasp_qpos[6:8] = [GRIPPER_CLOSED, GRIPPER_CLOSED]

        # Lift up
        lift_qpos = np.array([
            0.0,
            0.5,
            0.5,
            0.0,
            0.3,
            0.0,
            GRIPPER_CLOSED, GRIPPER_CLOSED
        ])

        self.trajectory = [
            {"t": 0,   "qpos": start_qpos},
            {"t": 60,  "qpos": target_qpos},    # Move to target
            {"t": 80,  "qpos": target_qpos},    # Pause
            {"t": 100, "qpos": grasp_qpos},     # Close gripper
            {"t": 130, "qpos": grasp_qpos},     # Hold
            {"t": 180, "qpos": lift_qpos},      # Lift
            {"t": 250, "qpos": lift_qpos},      # Stay lifted
        ]


def test_policy(
    policy_name: str = "bowl_to_plate",
    episode_len: int = 450,
    onscreen_render: bool = True,
    inject_noise: bool = False,
    camera_view: bool = False,
    use_rewards: bool = False,
):
    """
    Tests the single-arm scripted policy in simulation.

    :param policy_name: Name of policy to use ('bowl_to_plate' or 'simple_pick').
    :param episode_len: Length of episode in timesteps.
    :param onscreen_render: Whether to show the MuJoCo viewer.
    :param inject_noise: Whether to add noise to actions.
    :param camera_view: Whether to show camera views in matplotlib window.
    :param use_rewards: Whether to use FoodTransferTask with rewards (for teleop/teleop_2).
    """
    # Setup environment based on policy
    # cam = wrist camera, cam_high = overhead
    cam_list = ["cam_high", "cam"]

    # Select policy and scene
    if policy_name == "bowl_to_plate":
        xml_file = "wxai/food_scene.xml"
        policy = BowlToPlatePolicy(inject_noise)
        task_cls = SingleArmTask
    elif policy_name == "simple_pick":
        xml_file = "wxai/food_scene.xml"
        policy = SimplePickPolicy(inject_noise)
        task_cls = SingleArmTask
    elif policy_name == "teleop":
        xml_file = "wxai/teleop_scene.xml"
        policy = TeleopPolicy(inject_noise)
        task_cls = FoodTransferTask if use_rewards else SingleArmTask
    elif policy_name == "teleop_2":
        xml_file = "wxai/teleop_scene.xml"
        policy = TeleopPolicy2(inject_noise)
        task_cls = FoodTransferTask if use_rewards else SingleArmTask
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    env = make_sim_env(
        task_cls,
        xml_file=xml_file,
        task_name="single arm",
        onscreen_render=onscreen_render,
        cam_list=cam_list,
    )

    ts = env.reset()

    # Pass physics to policy for IK solving
    policy.set_physics(env.physics)

    print(f"Running {policy_name} policy for {episode_len} steps...")
    print(f"Initial qpos: {ts.observation['qpos']}")
    if use_rewards:
        print(f"Rewards enabled: max_reward={env.task.max_reward}")
        print("  0 = No reach")
        print("  1 = Reached container (stayed 2s)")
        print("  2 = Reached bowl (stayed 2s)")

    # Reward tracking
    episode_rewards = []
    last_reward = 0

    if onscreen_render:
        # Setup camera view window if requested
        plt_imgs = None
        if camera_view:
            plt_imgs = plot_observation_images(ts.observation, cam_list)
            plt.pause(0.02)
            plt.show(block=False)

        # Use MuJoCo viewer
        # Real-time factor: 50 steps per second = 0.02s per step
        step_duration = 0.02
        with mj_viewer.launch_passive(env.physics.model.ptr, env.physics.data.ptr) as viewer:
            for step in range(episode_len):
                step_start = time.time()
                if not viewer.is_running():
                    break
                action = policy(ts)
                ts = env.step(action)

                # Track rewards
                if use_rewards:
                    reward = ts.reward if ts.reward is not None else 0
                    episode_rewards.append(reward)
                    if reward != last_reward:
                        print(f"[Step {step}] Reward changed: {last_reward} -> {reward}")
                        last_reward = reward

                # Update camera views if enabled
                if camera_view and plt_imgs is not None:
                    plt_imgs = set_observation_images(ts.observation, plt_imgs, cam_list)
                    plt.pause(0.001)
                viewer.sync()
                # Sleep to maintain real-time playback
                elapsed = time.time() - step_start
                if elapsed < step_duration:
                    time.sleep(step_duration - elapsed)

        # Print reward summary
        if use_rewards:
            print("\n" + "=" * 40)
            print("REWARD SUMMARY")
            print("=" * 40)
            max_reward = max(episode_rewards) if episode_rewards else 0
            print(f"Max reward achieved: {max_reward}")
            if max_reward == env.task.max_reward:
                print("SUCCESS! Task completed.")
            else:
                print("FAILED. Task not completed.")
            print("=" * 40)

        print("Simulation complete.")
    else:
        # Run without viewer
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)

            # Track rewards
            if use_rewards:
                reward = ts.reward if ts.reward is not None else 0
                episode_rewards.append(reward)
                if reward != last_reward:
                    print(f"[Step {step}] Reward changed: {last_reward} -> {reward}")
                    last_reward = reward

            if step % 100 == 0:
                print(f"Step {step}: qpos = {ts.observation['qpos'][:6]}")

        # Print reward summary
        if use_rewards:
            print("\n" + "=" * 40)
            print("REWARD SUMMARY")
            print("=" * 40)
            max_reward = max(episode_rewards) if episode_rewards else 0
            print(f"Max reward achieved: {max_reward}")
            if max_reward == env.task.max_reward:
                print("SUCCESS! Task completed.")
            else:
                print("FAILED. Task not completed.")
            print("=" * 40)

        print("Simulation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test single-arm scripted policies."
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="bowl_to_plate",
        choices=["bowl_to_plate", "simple_pick", "teleop", "teleop_2"],
        help="Policy to run.",
    )
    parser.add_argument(
        "--episode_len",
        type=int,
        default=450,
        help="Episode length in timesteps.",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Disable visualization.",
    )
    parser.add_argument(
        "--inject_noise",
        action="store_true",
        help="Inject noise into actions.",
    )
    parser.add_argument(
        "--camera_view",
        action="store_true",
        help="Show camera views (cam_high, cam_front, cam_wrist) in matplotlib window.",
    )
    parser.add_argument(
        "--use_rewards",
        action="store_true",
        help="Enable FoodTransferTask with rewards (for teleop/teleop_2 policies).",
    )

    args = parser.parse_args()

    test_policy(
        policy_name=args.policy,
        episode_len=args.episode_len,
        onscreen_render=not args.no_render,
        inject_noise=args.inject_noise,
        camera_view=args.camera_view,
        use_rewards=args.use_rewards,
    )
