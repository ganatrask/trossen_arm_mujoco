"""
Dataset card (README.md) generator for HuggingFace datasets.
"""

from datetime import datetime
from typing import Any, Dict, Optional


def generate_dataset_card(
    repo_id: str,
    num_episodes: int,
    dr_config: Optional[Dict[str, Any]] = None,
    visual_dr_config: Optional[Dict[str, Any]] = None,
    additional_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a dataset card (README.md) for a HuggingFace dataset.

    Args:
        repo_id: HuggingFace repository ID
        num_episodes: Total number of episodes in the dataset
        dr_config: Domain randomization configuration dict
        visual_dr_config: Visual domain randomization configuration dict
        additional_info: Additional metadata to include

    Returns:
        Markdown string with YAML frontmatter
    """
    # YAML frontmatter
    yaml_frontmatter = f"""---
license: mit
task_categories:
  - robotics
tags:
  - robotics
  - manipulation
  - imitation-learning
  - mujoco
  - simulation
size_categories:
  - {_get_size_category(num_episodes)}
---
"""

    # Main content
    content = f"""# {repo_id.split('/')[-1]}

Simulated robot manipulation dataset for imitation learning.

## Dataset Description

This dataset contains **{num_episodes:,} episodes** of a Trossen WXAI robotic arm performing food transfer tasks in MuJoCo simulation.

### Task Description

The robot performs a scooping motion to transfer food from a source container to target bowls:

1. **HOME** - Start at home position
2. **APPROACH_CONTAINER** - Move above the source container
3. **REACH_CONTAINER** - Lower into container
4. **SCOOP** - Perform scooping motion (wrist rotation)
5. **LIFT** - Lift from container
6. **APPROACH_BOWL** - Move above target bowl
7. **LOWER_BOWL** - Lower to bowl
8. **DUMP** - Rotate wrist to dump food
9. **RETURN** - Return to home position

## Dataset Structure

```
{repo_id}/
├── README.md
├── manifest.json
└── data/
    ├── batch_000/
    │   ├── episode_0.hdf5
    │   ├── episode_1.hdf5
    │   ├── ...
    │   └── videos/
    │       ├── episode_0.mp4
    │       ├── episode_1.mp4
    │       └── ...
    ├── batch_001/
    └── ...
```

### HDF5 Episode Format

Each `episode_X.hdf5` file contains:

```
episode_X.hdf5
├── observations/
│   ├── images/
│   │   ├── main_view  (T, 480, 640, 3) uint8 - overhead camera
│   │   └── cam        (T, 480, 640, 3) uint8 - wrist camera
│   ├── qpos           (T, 8) float64 - joint positions
│   └── qvel           (T, 8) float64 - joint velocities
├── action             (T, 8) float64 - joint commands
├── success            bool - episode success flag
├── env_state/
│   ├── source_container  (7,) float64 - [x,y,z,qw,qx,qy,qz]
│   ├── target_container  (7,) float64 - target bowl pose
│   └── bowl_*            (7,) float64 - all bowl poses
└── attrs:
    ├── sim: True
    ├── source: "food_transfer_ik"
    ├── target: bowl name
    ├── dr_enabled: bool
    └── dr_config: JSON (if DR enabled)
```

### MP4 Videos

Each episode has a corresponding MP4 video showing both camera views side-by-side at 50 FPS.

## Robot Configuration

- **Robot**: Trossen WXAI 6-DOF robotic arm
- **End-effector**: Spoon attached to wrist
- **Joints**: 6 arm joints + 2 gripper joints
- **Action space**: 8-dimensional joint position commands
- **Cameras**: Overhead (main_view) + Wrist-mounted (cam)
- **Image resolution**: 640x480 RGB

"""

    # Domain randomization section
    if dr_config:
        content += """## Domain Randomization

This dataset was generated with geometric domain randomization:

| Parameter | Value |
|-----------|-------|
"""
        dr_params = [
            ("Position noise", f"±{dr_config.get('position_noise', 0.03)*100:.1f} cm"),
            ("Rotation noise", f"±{dr_config.get('rotation_noise', 0.1):.2f} rad"),
            ("Container rotation", f"±{dr_config.get('container_rotation', 0.15):.2f} rad"),
            ("Bowl count range", f"{dr_config.get('min_bowls', 1)} - {dr_config.get('max_bowls', 8)}"),
            ("Min object spacing", f"{dr_config.get('min_spacing', 0.12)*100:.1f} cm"),
            ("Container randomization", "Yes" if dr_config.get('randomize_container_position', False) else "No"),
            ("90-degree rotation", "Yes" if dr_config.get('allow_90_degree_rotation', False) else "No"),
        ]
        for name, value in dr_params:
            content += f"| {name} | {value} |\n"
        content += "\n"

    # Visual domain randomization section
    if visual_dr_config:
        content += """## Visual Domain Randomization

This dataset includes visual domain randomization:

| Parameter | Value |
|-----------|-------|
"""
        visual_params = [
            ("Table textures", f"{visual_dr_config.get('num_table_textures', 100)} variations" if visual_dr_config.get('randomize_table_texture', True) else "Disabled"),
            ("Floor textures", f"{visual_dr_config.get('num_floor_textures', 100)} variations" if visual_dr_config.get('randomize_floor_texture', True) else "Disabled"),
            ("Container color", "Randomized" if visual_dr_config.get('randomize_container_color', True) else "Fixed"),
            ("Bowl color", "Randomized" if visual_dr_config.get('randomize_bowl_color', False) else "Fixed"),
            ("Lighting", "Randomized" if visual_dr_config.get('randomize_lighting', True) else "Fixed"),
            ("Light position noise", f"±{visual_dr_config.get('light_position_noise', 0.3):.1f} m"),
            ("Light intensity range", f"{visual_dr_config.get('light_intensity_min', 0.5):.1f} - {visual_dr_config.get('light_intensity_max', 1.2):.1f}x"),
        ]
        for name, value in visual_params:
            content += f"| {name} | {value} |\n"
        content += "\n"

    # Usage section
    content += """## Usage

### Loading with Python

```python
import h5py
from huggingface_hub import hf_hub_download

# Download a single episode
path = hf_hub_download(
    repo_id="{repo_id}",
    filename="data/batch_000/episode_0.hdf5",
    repo_type="dataset"
)

# Load the episode
with h5py.File(path, "r") as f:
    images = f["/observations/images/main_view"][:]
    actions = f["/action"][:]
    qpos = f["/observations/qpos"][:]
    success = f["success"][()]

print(f"Episode length: {{len(actions)}} timesteps")
print(f"Success: {{success}}")
```

### Loading with datasets library

```python
from datasets import load_dataset

# Load manifest to get episode list
dataset = load_dataset("{repo_id}", split="train")
```

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{{repo_id.split('/')[-1].replace('-', '_')}},
    title={{{repo_id.split('/')[-1]}}},
    author={{Trossen Robotics}},
    year={{{datetime.now().year}}},
    publisher={{HuggingFace}},
    url={{https://huggingface.co/datasets/{repo_id}}}
}}
```

## License

This dataset is released under the MIT License.

""".replace("{repo_id}", repo_id)

    # Additional info
    if additional_info:
        content += "## Additional Information\n\n"
        for key, value in additional_info.items():
            content += f"- **{key}**: {value}\n"
        content += "\n"

    # Generation info
    content += f"""---

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using trossen_arm_mujoco*
"""

    return yaml_frontmatter + content


def _get_size_category(num_episodes: int) -> str:
    """Get HuggingFace size category based on episode count."""
    if num_episodes < 100:
        return "n<1K"
    elif num_episodes < 1000:
        return "1K<n<10K"
    elif num_episodes < 10000:
        return "10K<n<100K"
    elif num_episodes < 100000:
        return "100K<n<1M"
    else:
        return "1M<n<10M"
