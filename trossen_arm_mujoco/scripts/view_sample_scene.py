#!/usr/bin/env python3
"""Simple viewer for the tray and bowl sample scene."""

import mujoco
import mujoco.viewer
from pathlib import Path

def main():
    # Get the path to the scene XML
    script_dir = Path(__file__).parent
    scene_path = script_dir.parent / "assets" / "sample_scenes" / "tray_ramekin_preview.xml"

    print(f"Loading scene from: {scene_path}")

    # Load the model
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    print("\nScene loaded successfully!")
    print(f"  - Tray: Threshold_Tray_Rectangle_Porcelain (~6.7\" x 12.4\" x 1.4\")")
    print(f"  - Ramekin: Threshold_Ramekin_White_Porcelain (~4.4\" x 4.4\" x 2.1\")")
    print("\nControls:")
    print("  - Left mouse: Rotate view")
    print("  - Right mouse: Pan view")
    print("  - Scroll: Zoom")
    print("  - ESC: Close viewer")

    # Launch the viewer
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()
