"""
Interactive pose tuner for adjusting robot waypoints.

Usage:
    python -m trossen_arm_mujoco.assets.food_task.pose_tuner

Controls (in terminal):
    - 0-5: Select joint to adjust
    - +/=: Increase selected joint by step
    - -: Decrease selected joint by step
    - [/]: Decrease/increase step size
    - p: Print current pose
    - r: Reset to initial pose
    - q: Quit and print final pose
"""

import argparse
import sys
import termios
import tty
import numpy as np
from mujoco import viewer as mj_viewer
import mujoco
import threading
import time

from trossen_arm_mujoco.utils import make_sim_env
from trossen_arm_mujoco.assets.food_task.scripted_policy_single_arm import SingleArmTask


# Starting poses to tune
POSES = {
    "above_plate_teleop2": np.array([0.25, 1.1198, 0.8974, -0.3222, -0.6323, -0.5343, 0.044, 0.044]),
    "dump_teleop2": np.array([0.32, 1.1870, 0.8703, -0.1677, -0.5152, -1.8313, 0.044, 0.044]),
    "above_plate_teleop": np.array([0.4080, 1.1198, 0.8974, -0.3222, -0.6323, -0.5343, 0.044, 0.044]),
    "dump_teleop": np.array([0.4736, 1.1870, 0.8703, -0.1677, -0.5152, -1.8313, 0.044, 0.044]),
}

JOINT_NAMES = ["joint_0 (base)", "joint_1 (shoulder)", "joint_2 (elbow)",
               "joint_3 (wrist1)", "joint_4 (wrist2)", "joint_5 (wrist3)"]


def print_pose(qpos: np.ndarray, name: str = "current"):
    """Print pose in a format ready to copy into code."""
    joints = qpos[:8]
    print(f"\n{'='*60}")
    print(f"Pose: {name}")
    print(f"{'='*60}")
    print(f"Joint values: {np.round(joints, 4).tolist()}")
    print(f"\nCopy-paste format:")
    print(f"np.array([{joints[0]:.4f}, {joints[1]:.4f}, {joints[2]:.4f}, {joints[3]:.4f}, {joints[4]:.4f}, {joints[5]:.4f}, GRIPPER_OPEN, GRIPPER_OPEN])")
    print(f"{'='*60}\n")


def print_status(qpos: np.ndarray, selected_joint: int, step_size: float):
    """Print current status."""
    print(f"\r[Joint {selected_joint}: {JOINT_NAMES[selected_joint]}] "
          f"Value: {qpos[selected_joint]:+.4f} | Step: {step_size:.3f} | "
          f"Keys: 0-5=select, +/-=adjust, [/]=step, p=print, r=reset, q=quit", end="", flush=True)


def getch():
    """Get a single character from stdin."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def run_pose_tuner(pose_name: str = "above_plate_teleop2"):
    """
    Run the interactive pose tuner.

    :param pose_name: Name of the starting pose from POSES dict.
    """
    # Get initial pose
    if pose_name not in POSES:
        print(f"Unknown pose: {pose_name}")
        print(f"Available poses: {list(POSES.keys())}")
        return

    initial_pose = POSES[pose_name].copy()
    current_pose = initial_pose.copy()

    # Setup environment with teleop scene
    cam_list = ["cam_high", "cam"]
    env = make_sim_env(
        SingleArmTask,
        xml_file="wxai/teleop_scene.xml",
        task_name="pose_tuner",
        onscreen_render=True,
        cam_list=cam_list,
    )

    ts = env.reset()

    # Set initial pose
    env.physics.data.qpos[:8] = current_pose
    mujoco.mj_forward(env.physics.model.ptr, env.physics.data.ptr)

    print(f"\n{'='*60}")
    print("POSE TUNER - Keyboard Joint Control")
    print(f"{'='*60}")
    print(f"Starting pose: {pose_name}")
    print(f"\nControls (press keys in THIS terminal window):")
    print("  0-5: Select joint to adjust")
    print("  + or =: Increase joint value")
    print("  - or _: Decrease joint value")
    print("  [ : Decrease step size")
    print("  ] : Increase step size")
    print("  p : Print current pose")
    print("  r : Reset to initial pose")
    print("  q : Quit and print final pose")
    print(f"{'='*60}\n")

    print_pose(initial_pose, f"Initial ({pose_name})")

    selected_joint = 0
    step_size = 0.05
    running = True

    def viewer_thread():
        """Run the viewer in a separate thread."""
        nonlocal running
        with mj_viewer.launch_passive(
            env.physics.model.ptr,
            env.physics.data.ptr,
        ) as viewer:
            while viewer.is_running() and running:
                mujoco.mj_forward(env.physics.model.ptr, env.physics.data.ptr)
                viewer.sync()
                time.sleep(0.02)
        running = False

    # Start viewer in background thread
    viewer_t = threading.Thread(target=viewer_thread, daemon=True)
    viewer_t.start()

    print_status(current_pose, selected_joint, step_size)

    try:
        while running:
            ch = getch()

            if ch == 'q':
                running = False
                break
            elif ch in '012345':
                selected_joint = int(ch)
            elif ch in '+=':
                current_pose[selected_joint] += step_size
                env.physics.data.qpos[:8] = current_pose
            elif ch in '-_':
                current_pose[selected_joint] -= step_size
                env.physics.data.qpos[:8] = current_pose
            elif ch == '[':
                step_size = max(0.001, step_size / 2)
            elif ch == ']':
                step_size = min(0.5, step_size * 2)
            elif ch == 'p':
                print()  # New line
                print_pose(current_pose, "Current")
            elif ch == 'r':
                current_pose = initial_pose.copy()
                env.physics.data.qpos[:8] = current_pose
                print("\n>>> Reset to initial pose")

            print_status(current_pose, selected_joint, step_size)

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        print("\n")

    # Print final pose on exit
    print("\n" + "="*60)
    print("FINAL POSE (copy this to your policy):")
    print("="*60)
    print_pose(current_pose, "Final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive pose tuner")
    parser.add_argument(
        "--pose",
        type=str,
        default="above_plate_teleop2",
        choices=list(POSES.keys()),
        help="Starting pose to tune.",
    )
    args = parser.parse_args()

    run_pose_tuner(args.pose)
