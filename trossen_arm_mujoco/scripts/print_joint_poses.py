#!/usr/bin/env python3

"""
Interactive joint pose viewer for MuJoCo scenes.

Opens a MuJoCo viewer where you can manipulate robot joints and prints
the current joint positions to the terminal. Use this to find joint
configurations for your robot.

Usage:
    python -m trossen_arm_mujoco.scripts.print_joint_poses
    python -m trossen_arm_mujoco.scripts.print_joint_poses --xml wxai/scene.xml

Controls:
    - Press 'F' to freeze/lock the arm at current position and print qpos
    - Press 'U' to unlock the arm and allow free movement
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from trossen_arm_mujoco.constants import ASSETS_DIR

# Global state for key callback
arm_locked = False
locked_qpos = None


def key_callback(keycode):
    """Handle key presses in the viewer."""
    global arm_locked, locked_qpos
    # 'F' key (ASCII 70) or 'f' (ASCII 102)
    if keycode == 70 or keycode == 102:
        arm_locked = True
    # 'U' key (ASCII 85) or 'u' (ASCII 117)
    elif keycode == 85 or keycode == 117:
        arm_locked = False
        locked_qpos = None
        print("\n*** ARM UNLOCKED - Free movement enabled ***\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a MuJoCo XML and interactively print joint poses."
    )
    parser.add_argument(
        "--xml",
        default="wxai/scene.xml",
        help="Path to XML file relative to assets directory.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Print rate in Hz (default: 10).",
    )
    args = parser.parse_args()

    xml_path = f"{ASSETS_DIR}/{args.xml}"
    print(f"Loading model from: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Robot has 8 joints: 6 arm joints + 2 gripper joints
    num_robot_joints = 8

    # Set initial robot pose and hold it
    start_qpos = [0.31211, 1.04882, 0.94984, 0.09885, 0.00084, -0.00089, 0.00019, 0.00019]
    data.qpos[:num_robot_joints] = start_qpos
    # Set actuator controls to hold position (6 arm joints + 1 gripper actuator = 7 actuators)
    # Actuator order: joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, left_gripper
    data.ctrl[:6] = start_qpos[:6]  # Arm joints
    data.ctrl[6] = start_qpos[6]     # Gripper
    mujoco.mj_forward(model, data)

    # Print joint names for reference
    print("\nJoint names and indices:")
    print("-" * 40)
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        qpos_addr = model.jnt_qposadr[i]
        joint_type = model.jnt_type[i]
        type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
        type_str = type_names.get(joint_type, "unknown")
        print(f"  [{i}] {joint_name} (type: {type_str}, qpos_addr: {qpos_addr})")
    print("-" * 40)
    print(f"\nTotal qpos size: {model.nq}")
    print(f"Total joints: {model.njnt}")
    print("\nInstructions:")
    print("  - Double-click on a body to select it")
    print("  - Use Ctrl+Right-click drag to apply forces")
    print("  - Press 'Space' to pause/unpause simulation")
    print("  - Press 'Backspace' to reset")
    print("  - Press 'F' to FREEZE arm at current position and print qpos")
    print("  - Press 'U' to UNLOCK arm and allow free movement")
    print("-" * 40 + "\n")

    global arm_locked, locked_qpos

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            robot_qpos = data.qpos[:num_robot_joints].copy()

            # Check if 'F' was pressed to lock arm
            if arm_locked and locked_qpos is None:
                locked_qpos = robot_qpos.copy()
                qpos_str = np.array2string(
                    locked_qpos,
                    precision=5,
                    suppress_small=True,
                    separator=", ",
                    max_line_width=200,
                )
                print("\n" + "=" * 50)
                print("*** ARM LOCKED ***")
                print(f"robot_qpos = {qpos_str}")
                print("=" * 50 + "\n")

            # If locked, hold arm at locked position
            if locked_qpos is not None:
                data.ctrl[:6] = locked_qpos[:6]  # Arm joints
                data.ctrl[6] = locked_qpos[6]     # Gripper
            else:
                # Not locked - no control (arm moves freely)
                data.ctrl[:] = 0

            # Step simulation and sync viewer
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)  # Small sleep to prevent CPU spinning


if __name__ == "__main__":
    main()
