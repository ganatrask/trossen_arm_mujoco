"""
Pose Helper Tool for Single-Arm Robot.

This tool lets you:
1. Drag a red marker (sphere) with long axes in the MuJoCo viewer to define target positions
2. Move the robot arm by dragging it
3. Press SPACE in the MuJoCo viewer to save/print the current pose

The scene includes the actual bowl and plate from scene.xml.

Usage:
    python examples/pose_helper.py

In the MuJoCo viewer:
- Double-click on the red marker or robot to select it
- Ctrl + Right-click drag to move
- Press SPACE to save current pose (prints to terminal)
- Press 'r' to reset robot to home position
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os

from trossen_arm_mujoco.constants import START_ARM_POSE, ASSETS_DIR


# Global storage for saved poses
saved_poses = []  # List of (qpos, marker_xyz, ee_xyz) tuples


def get_ee_position(data, model):
    """Get the end-effector (gripper) position."""
    ee_body_id = model.body('link_6').id
    return data.xpos[ee_body_id].copy()


def print_and_save_pose(model, data):
    """Print current marker position and robot state, and save it."""
    global saved_poses

    marker_pos = data.mocap_pos[0].copy()
    ee_pos = get_ee_position(data, model)
    qpos = data.qpos[:8].copy()

    pose_num = len(saved_poses) + 1
    saved_poses.append((qpos.copy(), marker_pos.copy(), ee_pos.copy()))

    print("\n" + "="*70)
    print(f"POSE #{pose_num} SAVED!")
    print("="*70)
    print(f"Marker (red sphere) XYZ: x={marker_pos[0]:.4f}, y={marker_pos[1]:.4f}, z={marker_pos[2]:.4f}")
    print(f"End-Effector XYZ:        x={ee_pos[0]:.4f}, y={ee_pos[1]:.4f}, z={ee_pos[2]:.4f}")
    print(f"\nJoint Angles:")
    print(f"  j1 (base):     {qpos[0]:7.3f} rad ({np.degrees(qpos[0]):7.1f} deg)")
    print(f"  j2 (shoulder): {qpos[1]:7.3f} rad ({np.degrees(qpos[1]):7.1f} deg)")
    print(f"  j3 (elbow):    {qpos[2]:7.3f} rad ({np.degrees(qpos[2]):7.1f} deg)")
    print(f"  j4 (wrist1):   {qpos[3]:7.3f} rad ({np.degrees(qpos[3]):7.1f} deg)")
    print(f"  j5 (wrist2):   {qpos[4]:7.3f} rad ({np.degrees(qpos[4]):7.1f} deg)")
    print(f"  j6 (wrist3):   {qpos[5]:7.3f} rad ({np.degrees(qpos[5]):7.1f} deg)")
    print(f"  gripper:       {qpos[6]:7.3f}")
    print("\n>>> Copy for trajectory (joints):")
    print(f"np.array([{qpos[0]:.3f}, {qpos[1]:.3f}, {qpos[2]:.3f}, {qpos[3]:.3f}, {qpos[4]:.3f}, {qpos[5]:.3f}, {qpos[6]:.3f}, {qpos[7]:.3f}])")
    print(f"\n>>> Marker position (for reference):")
    print(f"marker_xyz = np.array([{marker_pos[0]:.4f}, {marker_pos[1]:.4f}, {marker_pos[2]:.4f}])")
    print("="*70)


def print_all_saved_poses():
    """Print all saved poses at the end."""
    global saved_poses

    if not saved_poses:
        print("\nNo poses saved.")
        return

    print("\n" + "="*70)
    print(f"ALL SAVED POSES ({len(saved_poses)} total)")
    print("="*70)

    # Print summary table
    print("\n# Summary of saved positions:")
    print("-"*70)
    print(f"{'#':<4} {'Marker XYZ':<30} {'End-Effector XYZ':<30}")
    print("-"*70)
    for i, (qpos, marker_xyz, ee_xyz) in enumerate(saved_poses):
        marker_str = f"({marker_xyz[0]:.3f}, {marker_xyz[1]:.3f}, {marker_xyz[2]:.3f})"
        ee_str = f"({ee_xyz[0]:.3f}, {ee_xyz[1]:.3f}, {ee_xyz[2]:.3f})"
        print(f"{i+1:<4} {marker_str:<30} {ee_str:<30}")
    print("-"*70)

    print("\n# Copy this into your trajectory:\n")
    print("GRIPPER_OPEN = 0.044")
    print("")

    for i, (qpos, marker_xyz, ee_xyz) in enumerate(saved_poses):
        print(f"# Point {i+1} - Marker at ({marker_xyz[0]:.3f}, {marker_xyz[1]:.3f}, {marker_xyz[2]:.3f})")
        print(f"point_{i+1}_qpos = np.array([{qpos[0]:.3f}, {qpos[1]:.3f}, {qpos[2]:.3f}, {qpos[3]:.3f}, {qpos[4]:.3f}, {qpos[5]:.3f}, {qpos[6]:.3f}, {qpos[7]:.3f}])")
        print("")

    print("\n# Trajectory:")
    print("self.trajectory = [")
    print('    {"t": 0, "qpos": home_qpos},')
    for i in range(len(saved_poses)):
        t = (i + 1) * 60
        print(f'    {{"t": {t}, "qpos": point_{i+1}_qpos}},')
    print("]")
    print("="*70)


def key_callback(keycode):
    """Handle key presses in the viewer."""
    global model, data

    # Space bar = save pose
    if keycode == 32:  # Space
        print_and_save_pose(model, data)
    # 'r' = reset
    elif keycode == 82 or keycode == 114:  # 'R' or 'r'
        data.qpos[:8] = START_ARM_POSE[:8]
        mujoco.mj_forward(model, data)
        print("\n[Robot reset to home position]")


def main():
    global model, data

    # Read the original scene.xml
    scene_path = os.path.join(ASSETS_DIR, "wxai", "scene.xml")
    with open(scene_path, 'r') as f:
        scene_xml = f.read()

    # Add a mocap marker body before </worldbody>
    marker_xml = """
    <!-- DRAGGABLE TARGET MARKER (mocap body) - position above bowl -->
    <body name="target_marker" mocap="true" pos="-0.05 0.15 0.12">
      <!-- Center sphere -->
      <geom type="sphere" size="0.015" rgba="1 0 0 0.9" contype="0" conaffinity="0"/>

      <!-- LONG X-axis (RED) - 10cm each direction -->
      <geom type="cylinder" size="0.004 0.10" pos="0.10 0 0" euler="0 90 0" rgba="1 0.2 0.2 0.8" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.004 0.10" pos="-0.10 0 0" euler="0 90 0" rgba="1 0.2 0.2 0.8" contype="0" conaffinity="0"/>
      <geom type="sphere" size="0.008" pos="0.20 0 0" rgba="1 0 0 1" contype="0" conaffinity="0"/>

      <!-- LONG Y-axis (GREEN) - 10cm each direction -->
      <geom type="cylinder" size="0.004 0.10" pos="0 0.10 0" euler="90 0 0" rgba="0.2 1 0.2 0.8" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.004 0.10" pos="0 -0.10 0" euler="90 0 0" rgba="0.2 1 0.2 0.8" contype="0" conaffinity="0"/>
      <geom type="sphere" size="0.008" pos="0 0.20 0" rgba="0 1 0 1" contype="0" conaffinity="0"/>

      <!-- LONG Z-axis (BLUE) - 10cm each direction -->
      <geom type="cylinder" size="0.004 0.10" pos="0 0 0.10" rgba="0.2 0.2 1 0.8" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.004 0.10" pos="0 0 -0.10" rgba="0.2 0.2 1 0.8" contype="0" conaffinity="0"/>
      <geom type="sphere" size="0.008" pos="0 0 0.20" rgba="0 0 1 1" contype="0" conaffinity="0"/>
    </body>

  </worldbody>
"""

    modified_xml = scene_xml.replace("</worldbody>", marker_xml)

    # Write temp XML file
    xml_dir = os.path.join(ASSETS_DIR, "wxai")
    temp_xml = os.path.join(xml_dir, "_pose_helper_temp.xml")

    with open(temp_xml, 'w') as f:
        f.write(modified_xml)

    try:
        model = mujoco.MjModel.from_xml_path(temp_xml)
        data = mujoco.MjData(model)

        # Set initial robot pose
        data.qpos[:8] = START_ARM_POSE[:8]
        mujoco.mj_forward(model, data)

        print("\n" + "="*70)
        print("POSE HELPER - Interactive Position Tool")
        print("="*70)
        print("\nControls (in MuJoCo viewer window):")
        print("  SPACE  - Save current pose")
        print("  R      - Reset robot to home position")
        print("  Double-click + Ctrl+Right-drag - Move objects")
        print("\nReference positions:")
        print("  Bowl center:  x=-0.05, y=0.15,  z=0.04")
        print("  Plate center: x=-0.05, y=-0.15, z=0.04")
        print("="*70)
        print("\nMove the robot, then press SPACE to save poses...")

        # Launch viewer with key callback
        with mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=key_callback
        ) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)

        # Print all saved poses when viewer closes
        print_all_saved_poses()
        print("\nViewer closed. Goodbye!")

    finally:
        # Clean up temp file
        if os.path.exists(temp_xml):
            os.remove(temp_xml)


if __name__ == "__main__":
    main()
