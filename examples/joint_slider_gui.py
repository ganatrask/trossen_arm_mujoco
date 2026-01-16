"""
Joint Slider GUI - Control robot joints with sliders in real-time.

This tool provides a GUI with sliders for each joint angle.
Move the sliders and see the robot move in the MuJoCo viewer.

Usage:
    python examples/joint_slider_gui.py

Controls:
- Sliders: Adjust each joint angle
- Save Pose: Save current joint configuration
- Reset: Return to home position
- Close the viewer window to exit and see all saved poses
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import tkinter as tk
from tkinter import ttk

from trossen_arm_mujoco.constants import START_ARM_POSE, ASSETS_DIR


# Joint limits (approximate, in radians)
JOINT_LIMITS = [
    (-3.14, 3.14),   # j1: base rotation
    (-1.5, 2.0),     # j2: shoulder
    (-1.5, 2.0),     # j3: elbow
    (-3.14, 3.14),   # j4: wrist1
    (-2.0, 2.0),     # j5: wrist2
    (-3.14, 3.14),   # j6: wrist3
]

JOINT_NAMES = [
    "J1 - Base",
    "J2 - Shoulder",
    "J3 - Elbow",
    "J4 - Wrist 1",
    "J5 - Wrist 2",
    "J6 - Wrist 3",
]

GRIPPER_OPEN = 0.044
GRIPPER_CLOSED = 0.012


class JointSliderApp:
    """Single-threaded joint slider application."""

    def __init__(self):
        self.model = None
        self.data = None
        self.viewer = None
        self.saved_poses = []

        # Load MuJoCo model
        scene_path = os.path.join(ASSETS_DIR, "wxai", "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        # Set initial pose
        self.data.qpos[:8] = START_ARM_POSE[:8]
        mujoco.mj_forward(self.model, self.data)

        # Create tkinter GUI
        self.root = tk.Tk()
        self.root.title("Joint Slider Control")
        self.root.geometry("420x520")

        self.joint_vars = []
        self.gripper_var = tk.DoubleVar(value=GRIPPER_OPEN)

        self._create_gui()

        print("\n" + "="*50)
        print("JOINT SLIDER GUI")
        print("="*50)
        print("Use sliders to control joint angles")
        print("Press 'S' to save pose, 'R' to reset")
        print("Close the MuJoCo viewer to exit")
        print("="*50)

    def _create_gui(self):
        """Create the GUI elements."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="Joint Angle Control", font=('Helvetica', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # Create sliders for each joint
        for i in range(6):
            label = ttk.Label(main_frame, text=JOINT_NAMES[i], width=15)
            label.grid(row=i+1, column=0, sticky=tk.W, pady=5)

            var = tk.DoubleVar(value=START_ARM_POSE[i])
            self.joint_vars.append(var)

            min_val, max_val = JOINT_LIMITS[i]
            slider = ttk.Scale(
                main_frame,
                from_=min_val,
                to=max_val,
                variable=var,
                orient=tk.HORIZONTAL,
                length=200,
                command=lambda v, idx=i: self.on_slider_change(idx)
            )
            slider.grid(row=i+1, column=1, padx=10, pady=5)

            # Value display
            value_label = ttk.Label(main_frame, width=10)
            value_label.grid(row=i+1, column=2, pady=5)
            # Store reference for updating
            setattr(self, f'value_label_{i}', value_label)

        # Gripper slider
        gripper_label = ttk.Label(main_frame, text="Gripper", width=15)
        gripper_label.grid(row=7, column=0, sticky=tk.W, pady=5)

        gripper_slider = ttk.Scale(
            main_frame,
            from_=GRIPPER_CLOSED,
            to=GRIPPER_OPEN,
            variable=self.gripper_var,
            orient=tk.HORIZONTAL,
            length=200,
            command=lambda v: self.on_gripper_change()
        )
        gripper_slider.grid(row=7, column=1, padx=10, pady=5)

        self.gripper_value_label = ttk.Label(main_frame, width=10)
        self.gripper_value_label.grid(row=7, column=2, pady=5)

        # End-effector position display
        self.ee_label = ttk.Label(main_frame, text="EE Position: (-, -, -)", font=('Helvetica', 10))
        self.ee_label.grid(row=8, column=0, columnspan=3, pady=15)

        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=9, column=0, columnspan=3, pady=10)

        save_btn = ttk.Button(button_frame, text="Save Pose (S)", command=self.save_pose)
        save_btn.grid(row=0, column=0, padx=5)

        reset_btn = ttk.Button(button_frame, text="Reset (R)", command=self.reset_pose)
        reset_btn.grid(row=0, column=1, padx=5)

        # Keyboard bindings
        self.root.bind('s', lambda e: self.save_pose())
        self.root.bind('S', lambda e: self.save_pose())
        self.root.bind('r', lambda e: self.reset_pose())
        self.root.bind('R', lambda e: self.reset_pose())

    def get_ee_position(self):
        """Get the end-effector position."""
        ee_body_id = self.model.body('link_6').id
        return self.data.xpos[ee_body_id].copy()

    def on_slider_change(self, joint_idx):
        """Called when a slider is moved."""
        value = self.joint_vars[joint_idx].get()
        self.data.qpos[joint_idx] = value
        mujoco.mj_forward(self.model, self.data)

    def on_gripper_change(self):
        """Called when gripper slider is moved."""
        value = self.gripper_var.get()
        self.data.qpos[6] = value
        self.data.qpos[7] = value
        mujoco.mj_forward(self.model, self.data)

    def reset_pose(self):
        """Reset to home position."""
        for i in range(6):
            self.joint_vars[i].set(START_ARM_POSE[i])
            self.data.qpos[i] = START_ARM_POSE[i]
        self.gripper_var.set(GRIPPER_OPEN)
        self.data.qpos[6] = GRIPPER_OPEN
        self.data.qpos[7] = GRIPPER_OPEN
        mujoco.mj_forward(self.model, self.data)
        print("\n[Reset to home position]")

    def save_pose(self):
        """Save current pose."""
        qpos = self.data.qpos[:8].copy()
        ee_pos = self.get_ee_position()
        self.saved_poses.append((qpos.copy(), ee_pos.copy()))
        print(f"\n>>> POSE #{len(self.saved_poses)} SAVED!")
        print(f"    EE Position: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
        print(f"    np.array([{qpos[0]:.3f}, {qpos[1]:.3f}, {qpos[2]:.3f}, {qpos[3]:.3f}, {qpos[4]:.3f}, {qpos[5]:.3f}, {qpos[6]:.3f}, {qpos[7]:.3f}])")

    def update_display(self):
        """Update GUI displays."""
        ee_pos = self.get_ee_position()
        self.ee_label.config(text=f"EE Position: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")

        # Update value labels
        for i in range(6):
            label = getattr(self, f'value_label_{i}')
            label.config(text=f"{self.data.qpos[i]:.3f}")
        self.gripper_value_label.config(text=f"{self.data.qpos[6]:.3f}")

    def print_all_poses(self):
        """Print all saved poses."""
        if not self.saved_poses:
            print("\nNo poses saved.")
            return

        print("\n" + "="*70)
        print(f"ALL SAVED POSES ({len(self.saved_poses)} total)")
        print("="*70)
        print("\nGRIPPER_OPEN = 0.044\n")

        for i, (qpos, ee_pos) in enumerate(self.saved_poses):
            print(f"# Point {i+1} - EE at ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
            print(f"point_{i+1}_qpos = np.array([{qpos[0]:.3f}, {qpos[1]:.3f}, {qpos[2]:.3f}, {qpos[3]:.3f}, {qpos[4]:.3f}, {qpos[5]:.3f}, {qpos[6]:.3f}, {qpos[7]:.3f}])")
            print()

        print("\n# Trajectory:")
        print("self.trajectory = [")
        print('    {"t": 0, "qpos": home_qpos},')
        for i in range(len(self.saved_poses)):
            t = (i + 1) * 60
            print(f'    {{"t": {t}, "qpos": point_{i+1}_qpos}},')
        print("]")
        print("="*70)

    def run(self):
        """Main loop - runs MuJoCo viewer with tkinter updates."""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # Process tkinter events (non-blocking)
                try:
                    self.root.update()
                    self.update_display()
                except tk.TclError:
                    # Window was closed
                    break

                time.sleep(0.01)

        # Clean up
        try:
            self.root.destroy()
        except tk.TclError:
            pass

        # Print all saved poses
        self.print_all_poses()
        print("\nGoodbye!")


def main():
    app = JointSliderApp()
    app.run()


if __name__ == "__main__":
    main()
