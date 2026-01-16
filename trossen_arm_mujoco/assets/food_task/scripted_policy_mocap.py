"""
Scripted policy for single-arm manipulation using MOCAP control.

This module provides Cartesian (XYZ) waypoint control for the spoon arm.
Just specify XYZ positions - MuJoCo handles the IK automatically via
mocap body + weld constraint.

Action format: [x, y, z, gripper]
- xyz: End-effector target position
- gripper: Gripper opening (0.044 = open, 0.012 = closed)
"""

import argparse
import os
import time

import mujoco
from mujoco import viewer as mj_viewer
import numpy as np

from trossen_arm_mujoco.constants import ASSETS_DIR


def run_mocap_trajectory(
    waypoints: list[dict],
    episode_len: int = 500,
):
    """
    Run a trajectory using mocap control.

    :param waypoints: List of waypoints with 't', 'xyz', and optional 'gripper'.
    :param episode_len: Total episode length in timesteps.
    """
    # Load mocap scene
    scene_path = os.path.join(ASSETS_DIR, "wxai", "food_scene_mocap.xml")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # Get mocap body ID
    mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap_target")
    if mocap_body_id == -1:
        raise ValueError("mocap_target body not found in scene")

    # Get gripper actuator
    gripper_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")

    # Initialize
    mujoco.mj_forward(model, data)

    # Set initial mocap position to first waypoint
    if waypoints:
        data.mocap_pos[0] = waypoints[0]["xyz"]

    print("=" * 60)
    print("MOCAP TRAJECTORY CONTROL")
    print("=" * 60)
    print(f"Waypoints: {len(waypoints)}")
    for i, wp in enumerate(waypoints):
        print(f"  t={wp['t']:3d}: xyz={wp['xyz']}")
    print("=" * 60)

    def interpolate(wp1, wp2, t):
        """Interpolate between two waypoints."""
        t_frac = (t - wp1["t"]) / (wp2["t"] - wp1["t"])
        xyz = wp1["xyz"] + (wp2["xyz"] - wp1["xyz"]) * t_frac
        gripper = wp1.get("gripper", 0.044)
        if "gripper" in wp2:
            gripper = wp1.get("gripper", 0.044) + (wp2["gripper"] - wp1.get("gripper", 0.044)) * t_frac
        return xyz, gripper

    curr_wp_idx = 0
    curr_wp = waypoints[0] if waypoints else {"t": 0, "xyz": data.mocap_pos[0].copy()}

    with mj_viewer.launch_passive(model, data) as viewer:
        for step in range(episode_len):
            if not viewer.is_running():
                break

            # Update current waypoint
            if curr_wp_idx < len(waypoints) - 1:
                if step >= waypoints[curr_wp_idx + 1]["t"]:
                    curr_wp_idx += 1
                    curr_wp = waypoints[curr_wp_idx]

            # Interpolate to next waypoint
            if curr_wp_idx < len(waypoints) - 1:
                next_wp = waypoints[curr_wp_idx + 1]
                xyz, gripper = interpolate(curr_wp, next_wp, step)
            else:
                xyz = curr_wp["xyz"]
                gripper = curr_wp.get("gripper", 0.044)

            # Set mocap position (this is the control!)
            data.mocap_pos[0] = xyz

            # Set gripper
            if gripper_actuator_id != -1:
                data.ctrl[gripper_actuator_id] = gripper

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            # Slow down to real-time (roughly 50Hz)
            time.sleep(0.02)

    print("\nTrajectory complete!")


def bowl_to_plate_trajectory():
    """
    Create a realistic scooping trajectory from bowl to plate.

    Trajectory phases:
    1. Start at home position
    2. Move above the bowl (approach)
    3. Lower spoon into bowl (descend)
    4. Scoop motion - drag through bowl (scoop)
    5. Lift spoon out of bowl (lift)
    6. Move above plate (transfer)
    7. Lower over plate (descend to plate)
    8. Tilt/release motion (release)
    9. Lift away from plate
    10. Return home
    """
    GRIPPER_OPEN = 0.044

    # Key positions
    # Bowl center: (-0.05, 0.15, 0.04) - from scene.xml
    # Plate center: (-0.05, -0.15, 0.04)

    bowl_x, bowl_y, bowl_z = -0.05, 0.15, 0.04
    plate_x, plate_y, plate_z = -0.05, -0.15, 0.04

    # Home position
    home_xyz = np.array([-0.15, 0.0, 0.30])

    # Bowl positions
    above_bowl = np.array([bowl_x, bowl_y, bowl_z + 0.20])       # 20cm above bowl
    bowl_entry = np.array([bowl_x + 0.03, bowl_y, bowl_z + 0.08])  # Entry point (side of bowl)
    bowl_bottom = np.array([bowl_x, bowl_y, bowl_z + 0.05])      # Near bottom of bowl
    bowl_scoop_end = np.array([bowl_x - 0.03, bowl_y, bowl_z + 0.07])  # End of scoop motion
    bowl_lift = np.array([bowl_x - 0.02, bowl_y, bowl_z + 0.18])  # Lift out of bowl

    # Transfer position (high arc to avoid spilling)
    transfer_high = np.array([-0.05, 0.0, 0.30])  # High point in middle

    # Plate positions
    above_plate = np.array([plate_x, plate_y, plate_z + 0.15])   # Above plate
    plate_release = np.array([plate_x, plate_y, plate_z + 0.08]) # Lower to release
    plate_tilt = np.array([plate_x + 0.02, plate_y, plate_z + 0.10])  # Slight tilt to release food

    waypoints = [
        # Phase 1: Start
        {"t": 0,   "xyz": home_xyz, "gripper": GRIPPER_OPEN},

        # Phase 2: Approach bowl
        {"t": 60,  "xyz": above_bowl, "gripper": GRIPPER_OPEN},

        # Phase 3: Descend into bowl
        {"t": 100, "xyz": bowl_entry, "gripper": GRIPPER_OPEN},

        # Phase 4: Scoop motion (drag through bowl)
        {"t": 140, "xyz": bowl_bottom, "gripper": GRIPPER_OPEN},
        {"t": 180, "xyz": bowl_scoop_end, "gripper": GRIPPER_OPEN},

        # Phase 5: Lift out of bowl (careful, carrying food!)
        {"t": 220, "xyz": bowl_lift, "gripper": GRIPPER_OPEN},

        # Phase 6: Transfer to plate (high arc)
        {"t": 280, "xyz": transfer_high, "gripper": GRIPPER_OPEN},
        {"t": 340, "xyz": above_plate, "gripper": GRIPPER_OPEN},

        # Phase 7: Lower to plate
        {"t": 380, "xyz": plate_release, "gripper": GRIPPER_OPEN},

        # Phase 8: Tilt to release food
        {"t": 420, "xyz": plate_tilt, "gripper": GRIPPER_OPEN},
        {"t": 460, "xyz": plate_tilt, "gripper": GRIPPER_OPEN},  # Hold

        # Phase 9: Lift away
        {"t": 500, "xyz": above_plate, "gripper": GRIPPER_OPEN},

        # Phase 10: Return home
        {"t": 580, "xyz": home_xyz, "gripper": GRIPPER_OPEN},
        {"t": 620, "xyz": home_xyz, "gripper": GRIPPER_OPEN},
    ]

    return waypoints


def test_mocap():
    """Quick test - move the arm around using mocap."""
    waypoints = bowl_to_plate_trajectory()
    run_mocap_trajectory(waypoints, episode_len=650)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test mocap-based control")
    parser.add_argument(
        "--episode_len",
        type=int,
        default=650,
        help="Episode length in timesteps.",
    )
    args = parser.parse_args()

    test_mocap()
