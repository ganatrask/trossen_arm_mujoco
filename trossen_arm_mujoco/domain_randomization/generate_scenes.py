#!/usr/bin/env python3
"""
Generate scene XML variants with different bowl counts.

This script generates teleop_scene_6bowl.xml and teleop_scene_8bowl.xml
by reading the base teleop_scene.xml and adding additional bowl bodies.
"""

import re
from pathlib import Path


# Bowl positions for 8-bowl configuration
BOWL_POSITIONS = {
    "bowl_1": "-0.22 -0.26 0.04",  # existing
    "bowl_2": "-0.36 -0.26 0.04",  # existing
    "bowl_3": "-0.36 -0.12 0.04",  # existing
    "bowl_4": "-0.22 -0.12 0.04",  # existing
    "bowl_5": "-0.50 -0.26 0.04",  # new - left of bowl_2
    "bowl_6": "-0.50 -0.12 0.04",  # new - left of bowl_3
    "bowl_7": "-0.22 +0.02 0.04",  # new - behind bowl_4
    "bowl_8": "-0.36 +0.02 0.04",  # new - behind bowl_3
}


def generate_bowl_body(bowl_name: str, position: str) -> str:
    """Generate XML for a single bowl body with all collision geoms."""
    collision_geoms = "\n".join([
        f'      <geom type="mesh" mesh="ramekin_collision_{i}" contype="1" conaffinity="1" friction="1 0.005 0.0001" rgba="0 0 0 0"/>'
        for i in range(32)
    ])

    return f'''    <!-- {bowl_name} -->
    <body name="{bowl_name}" pos="{position}">
      <geom type="mesh" mesh="threshold_ramekin_mesh" material="threshold_ramekin_mat" contype="0" conaffinity="0" group="2"/>
{collision_geoms}
    </body>
'''


def generate_scene_xml(num_bowls: int, base_xml_path: Path, output_path: Path):
    """Generate a scene XML with specified number of bowls."""
    # Read base XML
    with open(base_xml_path, 'r') as f:
        content = f.read()

    # Update model name
    content = re.sub(
        r'model="wxai base scene with table"',
        f'model="wxai base scene with table - {num_bowls} bowls"',
        content
    )

    # Find where to insert additional bowls (after bowl_4, before closing </worldbody>)
    # We'll add new bowls after the last existing bowl

    # Generate additional bowl XML
    additional_bowls = []
    for i in range(5, num_bowls + 1):
        bowl_name = f"bowl_{i}"
        position = BOWL_POSITIONS[bowl_name]
        additional_bowls.append(generate_bowl_body(bowl_name, position))

    if additional_bowls:
        # Find the position after bowl_4's closing </body> tag
        # Look for the comment about white beans or end of worldbody
        insert_marker = "<!-- White beans"
        if insert_marker not in content:
            insert_marker = "</worldbody>"

        additional_xml = "\n" + "".join(additional_bowls)
        content = content.replace(insert_marker, additional_xml + "\n    " + insert_marker)

    # Write output
    with open(output_path, 'w') as f:
        f.write(content)

    print(f"Generated {output_path} with {num_bowls} bowls")


def main():
    """Generate all scene variants."""
    assets_dir = Path(__file__).parent.parent / "assets" / "wxai"
    base_xml = assets_dir / "teleop_scene.xml"

    if not base_xml.exists():
        print(f"Error: Base XML not found at {base_xml}")
        return

    # Generate 6-bowl and 8-bowl scenes
    generate_scene_xml(6, base_xml, assets_dir / "teleop_scene_6bowl.xml")
    generate_scene_xml(8, base_xml, assets_dir / "teleop_scene_8bowl.xml")

    print("Done generating scene variants!")


if __name__ == "__main__":
    main()
