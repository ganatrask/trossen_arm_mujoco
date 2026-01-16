
import collections

from mujoco import viewer as mj_viewer
from dm_control.mujoco.engine import Physics
from dm_control.suite import base
import matplotlib.pyplot as plt
import numpy as np

from trossen_arm_mujoco.constants import START_ARM_POSE
from trossen_arm_mujoco.utils import (
    get_observation_base,
    make_sim_env,
    plot_observation_images,
    set_observation_images,
)


class TrossenAISingleArmTask(base.Task):
    """
    A base task for single-arm manipulation with the WXAI follower arm.

    :param random: Random seed for environment variability, defaults to ``None``.
    :param onscreen_render: Whether to enable real-time rendering, defaults to ``False``.
    :param cam_list: List of cameras to capture observations, defaults to ``[]``.
    """

    def __init__(
        self,
        random: int | None = None,
        onscreen_render: bool = False,
        cam_list: list[str] = [],
    ):
        super().__init__(random=random)
        self.cam_list = cam_list
        if self.cam_list == []:
            self.cam_list = ["cam_high"]

    def before_step(self, action: np.ndarray, physics: Physics) -> None:
        """
        Processes the action before passing it to the simulation.

        Maps 8-value qpos action to 7-value actuator control.
        The gripper has 2 joints in qpos but only 1 actuator (coupled via equality constraint).

        :param action: The action array (7 or 8 values) matching actuator or qpos layout.
        :param physics: The MuJoCo physics simulation instance.
        """
        # Qpos action layout (8 values):
        #   [0:6] = arm joints (6)
        #   [6:8] = gripper joints (2, coupled)
        #
        # Actuator ctrl layout (7 values):
        #   [0:6] = arm actuators (6)
        #   [6]   = gripper actuator (1)
        if action.shape[0] == 7:
            super().before_step(action, physics)
            return
        if action.shape[0] != 8:
            raise ValueError("Expected action length 7 (ctrl) or 8 (qpos).")

        arm_action = action[:6]
        gripper_action = action[6]  # First gripper joint value
        env_action = np.concatenate([arm_action, [gripper_action]])
        super().before_step(env_action, physics)

    def initialize_episode(self, physics: Physics) -> None:
        """
        Sets the state of the environment at the start of each episode.

        :param physics: The MuJoCo physics simulation instance.
        """
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics: Physics) -> np.ndarray:
        """
        Retrieves the current state of the environment.

        :param physics: The MuJoCo physics simulation instance.
        :return: The environment state.
        """
        env_state = physics.data.qpos.copy()
        return env_state

    def get_position(self, physics: Physics) -> np.ndarray:
        """
        Retrieves the current joint positions.

        :param physics: The MuJoCo physics simulation instance.
        :return: The joint positions.
        """
        position = physics.data.qpos.copy()
        return position[:8]

    def get_velocity(self, physics: Physics) -> np.ndarray:
        """
        Retrieves the current joint velocities.

        :param physics: The MuJoCo physics simulation instance.
        :return: The joint velocities.
        """
        velocity = physics.data.qvel.copy()
        return velocity[:8]

    def get_observation(self, physics: Physics) -> collections.OrderedDict:
        """
        Collects the current observation from the environment.

        :param physics: The MuJoCo physics simulation instance.
        :return: An ordered dictionary containing joint positions, velocities, and environment state.
        """
        obs = get_observation_base(physics, self.cam_list)
        obs["qpos"] = self.get_position(physics)
        obs["qvel"] = self.get_velocity(physics)
        obs["env_state"] = self.get_env_state(physics)
        return obs

    def get_reward(self, physics: Physics) -> int:
        """
        Computes the reward for the current timestep.

        :param physics: The MuJoCo physics simulation instance.
        :raises NotImplementedError: This method must be implemented in subclasses.
        """
        return 0


class SingleArmTask(TrossenAISingleArmTask):
    """
    A simple single-arm task used for visualization and teleoperation testing.

    :param random: Random seed for environment variability, defaults to ``None``.
    :param onscreen_render: Whether to enable real-time rendering, defaults to ``False``.
    :param cam_list: List of cameras to capture observations, defaults to ``[]``.
    """

    def __init__(
        self,
        random: int | None = None,
        onscreen_render: bool = False,
        cam_list: list[str] = [],
    ):
        super().__init__(
            random=random,
            onscreen_render=onscreen_render,
            cam_list=cam_list,
        )
        self.max_reward = 0

    def initialize_episode(self, physics: Physics) -> None:
        """
        Initializes the episode, resetting the robot's pose.

        :param physics: The MuJoCo physics simulation instance.
        """
        with physics.reset_context():
            physics.named.data.qpos[:8] = START_ARM_POSE[:8]

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics: Physics) -> np.ndarray:
        """
        Retrieves the environment state.

        :param physics: The MuJoCo physics simulation instance.
        :return: The environment state.
        """
        env_state = physics.data.qpos.copy()
        return env_state

    def get_reward(self, physics: Physics) -> int:
        """
        Computes the reward for the current timestep.

        :param physics: The MuJoCo physics simulation instance.
        :return: The computed reward.
        """
        return 0


def test_sim_teleop():
    """
    Runs a simulation to test teleoperation with the WXAI base arm.
    """
    # setup the environment
    cam_list = ["cam_high", "cam"]
    env = make_sim_env(
        SingleArmTask,
        xml_file="wxai/food_scene.xml",
        task_name="single_arm",
        onscreen_render=True,
        cam_list=cam_list,
    )
    ts = env.reset()
    episode = [ts]
    # setup plotting
    print("Running single arm simulation with wxai_base.xml...")
    print(f"Action space: 8 DOF (6 arm joints + 2 gripper joints)")
    print(f"Control space: 7 actuators (6 arm + 1 gripper)")
    print(f"Press Ctrl+C to stop the simulation")

    plt_imgs = plot_observation_images(ts.observation, cam_list)
    plt.pause(0.02)
    plt.show(block=False)

    # Scripted policy: move to bowl, then move to plate.
    start_qpos = ts.observation["qpos"].copy()
    bowl_qpos = np.array(
        [0.3, 0.6, 1.1, -0.8, 0.0, 0.2, 0.044, 0.044]
    )
    plate_qpos = np.array(
        [0.0, 0.9, 0.9, -0.6, 0.0, 0.0, 0.044, 0.044]
    )
    move_steps = 60
    hold_steps = 10
    total_steps = move_steps + hold_steps + move_steps + hold_steps
    step_count = 0
    stuck_count = 0
    last_error = None
    stuck_error_threshold = 0.05
    stuck_patience = 50
    improvement_eps = 1e-3
    scripted_policy_enabled = False

    def compute_target(t: int) -> tuple[str, np.ndarray, np.ndarray]:
        if t < move_steps:
            if t == 0:
                print("Phase: moving to bowl")
            alpha = t / max(1, move_steps)
            qpos = (1.0 - alpha) * start_qpos + alpha * bowl_qpos
            return "moving_to_bowl", qpos, bowl_qpos
        if t < move_steps + hold_steps:
            if t == move_steps:
                print("Phase: holding at bowl")
            return "holding_bowl", bowl_qpos, bowl_qpos
        if t < 2 * move_steps + hold_steps:
            if t == move_steps + hold_steps:
                print("Phase: moving to plate")
            alpha = (t - move_steps - hold_steps) / max(1, move_steps)
            qpos = (1.0 - alpha) * bowl_qpos + alpha * plate_qpos
            return "moving_to_plate", qpos, plate_qpos
        if t == 2 * move_steps + hold_steps:
            print("Phase: holding at plate")
        return "holding_plate", plate_qpos, plate_qpos

    with mj_viewer.launch_passive(env.physics.model.ptr, env.physics.data.ptr) as viewer:
        while viewer.is_running():
            if scripted_policy_enabled:
                phase, action, target_qpos = compute_target(step_count)
                ts = env.step(action)
                error = np.linalg.norm(ts.observation["qpos"][:6] - target_qpos[:6])
                if error > stuck_error_threshold:
                    if last_error is not None and (last_error - error) < improvement_eps:
                        stuck_count += 1
                    else:
                        stuck_count = 0
                    if stuck_count >= stuck_patience:
                        print(f"Warning: possible stuck state in {phase} (error={error:.3f})")
                        stuck_count = 0
                else:
                    stuck_count = 0
                last_error = error
                step_count = (step_count + 1) % total_steps
            else:
                # Hold current pose so teleop can take over without scripted motion.
                action = ts.observation["qpos"].copy()
                ts = env.step(action)
            plt_imgs = set_observation_images(ts.observation, plt_imgs, cam_list)
            plt.pause(0.001)
            viewer.sync()


if __name__ == "__main__":
    test_sim_teleop()
