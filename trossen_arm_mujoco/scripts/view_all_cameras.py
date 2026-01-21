#!/usr/bin/env python3
"""View all camera feeds from the teleop scene in a grid layout."""

import argparse
import mujoco
import mujoco.viewer
import numpy as np
import cv2
from pathlib import Path


def view_all_cameras():
    """Display all cameras in a grid layout."""
    script_dir = Path(__file__).parent
    scene_path = script_dir.parent / "assets" / "wxai" / "teleop_scene.xml"

    print(f"Loading scene from: {scene_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    camera_names = ["cam_high", "cam_front", "main_view"]

    camera_ids = {}
    for name in camera_names:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id >= 0:
            camera_ids[name] = cam_id
            print(f"  Found camera: {name} (id={cam_id})")
        else:
            print(f"  Warning: Camera '{name}' not found")

    width, height = 640, 480
    renderer = mujoco.Renderer(model, height, width)

    print("\nControls:")
    print("  - Press 'q' or ESC to quit")
    print("  - Press SPACE to pause/resume simulation")
    print("  - Press 'r' to reset simulation")

    paused = False
    cv2.namedWindow("Teleop Scene - All Cameras", cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            mujoco.mj_step(model, data)

        images = []
        labels = []
        for name, cam_id in camera_ids.items():
            renderer.update_scene(data, camera=cam_id)
            img = renderer.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append(img_bgr)
            labels.append(name)

        while len(images) < 6:
            images.append(np.zeros((height, width, 3), dtype=np.uint8))
            labels.append("")

        for i, (img, label) in enumerate(zip(images, labels)):
            if label:
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2, cv2.LINE_AA)

        row1 = np.hstack(images[0:3])
        row2 = np.hstack(images[3:6])
        grid = np.vstack([row1, row2])

        cv2.imshow("Teleop Scene - All Cameras", grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):
            mujoco.mj_resetData(model, data)
            print("Reset simulation")

    cv2.destroyAllWindows()
    renderer.close()
    print("Viewer closed.")


def interactive_camera_tuner():
    """Launch interactive viewer to position camera and print XML attributes."""
    script_dir = Path(__file__).parent
    scene_path = script_dir.parent / "assets" / "wxai" / "teleop_scene.xml"

    print(f"Loading scene from: {scene_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    print("\n" + "="*60)
    print("INTERACTIVE CAMERA TUNER")
    print("="*60)
    print("\nMouse Controls:")
    print("  - Left drag:   Rotate view")
    print("  - Right drag:  Pan view")
    print("  - Scroll:      Zoom in/out")
    print("\nKeyboard:")
    print("  - Press 'p' to print current camera pose (XML format)")
    print("  - Press ESC to close")
    print("\nTip: Position the view where you want, then press 'p' to get")
    print("     the camera attributes for your XML file.")
    print("="*60 + "\n")

    # Track if we should print camera info
    print_camera = [False]

    def key_callback(keycode):
        if keycode == ord('p') or keycode == ord('P'):
            print_camera[0] = True

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

            if print_camera[0]:
                print_camera[0] = False
                cam = viewer.cam

                # Get camera position (lookat + distance along viewing direction)
                azimuth = np.radians(cam.azimuth)
                elevation = np.radians(cam.elevation)
                distance = cam.distance

                # Calculate camera position from lookat point
                dx = distance * np.cos(elevation) * np.cos(azimuth)
                dy = distance * np.cos(elevation) * np.sin(azimuth)
                dz = distance * np.sin(elevation)

                pos = np.array([
                    cam.lookat[0] + dx,
                    cam.lookat[1] + dy,
                    cam.lookat[2] + dz
                ])

                # Calculate xyaxes from azimuth/elevation
                # X-axis points right in camera frame
                # Y-axis points up in camera frame
                ca, sa = np.cos(azimuth), np.sin(azimuth)
                ce, se = np.cos(elevation), np.sin(elevation)

                # Camera forward direction (pointing at lookat)
                fwd = np.array([-ce*ca, -ce*sa, -se])
                # Camera right direction
                right = np.array([sa, -ca, 0])
                # Camera up direction
                up = np.cross(fwd, right)
                up = up / np.linalg.norm(up)

                print("\n" + "-"*60)
                print("CURRENT CAMERA POSE")
                print("-"*60)
                print(f"Lookat:    [{cam.lookat[0]:.3f}, {cam.lookat[1]:.3f}, {cam.lookat[2]:.3f}]")
                print(f"Distance:  {distance:.3f}")
                print(f"Azimuth:   {cam.azimuth:.1f}°")
                print(f"Elevation: {cam.elevation:.1f}°")
                print(f"\nCamera position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                print("\n--- Copy this for your XML file: ---\n")
                print(f'<camera name="my_camera" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" '
                      f'xyaxes="{right[0]:.3f} {right[1]:.3f} {right[2]:.3f} '
                      f'{up[0]:.3f} {up[1]:.3f} {up[2]:.3f}" mode="fixed" fovy="60"/>')
                print("-"*60 + "\n")

    print("Viewer closed.")


def main():
    parser = argparse.ArgumentParser(description="View teleop scene cameras")
    parser.add_argument(
        "--tune", "-t",
        action="store_true",
        help="Launch interactive camera tuner to position camera and get XML attributes"
    )
    args = parser.parse_args()

    if args.tune:
        interactive_camera_tuner()
    else:
        view_all_cameras()


if __name__ == "__main__":
    main()
