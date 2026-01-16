"""
Joint Tuner - Interactive tool to find joint positions for waypoints.

This tool lets you:
1. Manually adjust each joint using keyboard
2. See the end-effector position in real-time
3. Copy the joint values for your trajectory

Usage:
    python examples/joint_tuner.py

Controls (in terminal):
    1-6: Select joint to adjust
    +/= : Increase joint angle by 0.1 rad
    -   : Decrease joint angle by 0.1 rad
    g   : Toggle gripper open/closed
    p   : Print current state (copy-paste ready)
    r   : Reset to home position
    q   : Quit
"""

import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
import sys
import termios
import tty

from trossen_arm_mujoco.constants import START_ARM_POSE, ASSETS_DIR


SCENE_XML = """
<mujoco model="joint tuner scene">
  <include file="wxai_follower_spoon.xml"/>

  <compiler meshdir="../meshes" texturedir="../"/>
  <statistic center="0 0 0.3" extent="0.8"/>

  <visual>
    <headlight diffuse="0.6 0.65 0.75" ambient="0.5 0.5 0.6" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-25"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.9725 0.9608 0.8706" rgb2="0.5 0.5 0.5" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="table_mat" rgba="0.8 0.7 0.6 1"/>
  </asset>

  <worldbody>
    <light name="top_light" pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <geom name="tabletop" type="box" size="0.4 0.4 0.02" pos="0 0 0.02" material="table_mat"/>
    <camera name="cam_high" pos="0.6 0.6 0.6" xyaxes="-1 1 0 -0.4 -0.4 1.4" mode="fixed" fovy="50"/>

    <!-- Bowl position marker -->
    <geom name="bowl_marker" type="cylinder" size="0.08 0.005" pos="-0.05 0.15 0.05" rgba="0.3 0.6 0.3 0.5"/>
    <!-- Plate position marker -->
    <geom name="plate_marker" type="cylinder" size="0.08 0.005" pos="-0.05 -0.15 0.05" rgba="0.3 0.3 0.6 0.5"/>
  </worldbody>
</mujoco>
"""


JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
GRIPPER_OPEN = 0.044
GRIPPER_CLOSED = 0.012


def get_ee_position(data, model):
    """Get the end-effector position."""
    ee_body_id = model.body('link_6').id
    return data.xpos[ee_body_id].copy()


def print_state(model, data, selected_joint):
    """Print current state."""
    qpos = data.qpos[:8].copy()
    ee_pos = get_ee_position(data, model)

    print("\033[H\033[J", end="")  # Clear screen
    print("="*65)
    print("JOINT TUNER - Use keyboard to adjust joints")
    print("="*65)
    print(f"\nEnd-Effector Position: x={ee_pos[0]:.4f}, y={ee_pos[1]:.4f}, z={ee_pos[2]:.4f}")
    print(f"\nJoints (selected: {selected_joint + 1} - {JOINT_NAMES[selected_joint]}):")
    print("-"*65)

    for i in range(6):
        marker = " >>>" if i == selected_joint else "    "
        print(f"{marker} [{i+1}] j{i+1} ({JOINT_NAMES[i]:8s}): {qpos[i]:7.3f} rad ({np.degrees(qpos[i]):7.1f} deg)")

    gripper_state = "OPEN" if qpos[6] > 0.03 else "CLOSED"
    print(f"\n     [g] Gripper: {qpos[6]:.3f} ({gripper_state})")

    print("\n" + "-"*65)
    print("Controls: 1-6=select joint, +/-=adjust, g=gripper, p=print, r=reset, q=quit")
    print("-"*65)
    print("\nCopy for trajectory:")
    print(f"np.array([{qpos[0]:.3f}, {qpos[1]:.3f}, {qpos[2]:.3f}, {qpos[3]:.3f}, {qpos[4]:.3f}, {qpos[5]:.3f}, {qpos[6]:.3f}, {qpos[7]:.3f}])")


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


def main():
    import os
    xml_path = os.path.join(ASSETS_DIR, "wxai")
    temp_xml = os.path.join(xml_path, "_joint_tuner_temp.xml")

    with open(temp_xml, 'w') as f:
        f.write(SCENE_XML)

    try:
        model = mujoco.MjModel.from_xml_path(temp_xml)
        data = mujoco.MjData(model)

        # Set initial pose
        data.qpos[:8] = START_ARM_POSE[:8]
        mujoco.mj_forward(model, data)

        selected_joint = 0
        running = [True]
        step_size = 0.1  # radians

        def input_loop():
            nonlocal selected_joint, step_size
            while running[0]:
                print_state(model, data, selected_joint)
                ch = getch()

                if ch == 'q':
                    running[0] = False
                    break
                elif ch in '123456':
                    selected_joint = int(ch) - 1
                elif ch in '+=':
                    data.qpos[selected_joint] += step_size
                elif ch == '-':
                    data.qpos[selected_joint] -= step_size
                elif ch == 'g':
                    if data.qpos[6] > 0.03:
                        data.qpos[6] = data.qpos[7] = GRIPPER_CLOSED
                    else:
                        data.qpos[6] = data.qpos[7] = GRIPPER_OPEN
                elif ch == 'r':
                    data.qpos[:8] = START_ARM_POSE[:8]
                elif ch == 'p':
                    qpos = data.qpos[:8]
                    print("\n\n" + "="*65)
                    print("SAVED POSITION:")
                    print(f"np.array([{qpos[0]:.3f}, {qpos[1]:.3f}, {qpos[2]:.3f}, {qpos[3]:.3f}, {qpos[4]:.3f}, {qpos[5]:.3f}, {qpos[6]:.3f}, {qpos[7]:.3f}])")
                    print("="*65)
                    input("\nPress Enter to continue...")

                mujoco.mj_forward(model, data)

        # Start input thread
        input_t = threading.Thread(target=input_loop)
        input_t.daemon = True
        input_t.start()

        # Launch viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and running[0]:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.02)

        running[0] = False
        print("\n\nGoodbye!")

    finally:
        if os.path.exists(temp_xml):
            os.remove(temp_xml)


if __name__ == "__main__":
    main()
