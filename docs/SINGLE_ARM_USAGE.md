# Single Arm Simulation with wxai_base.xml

This guide explains how to use the single arm simulation environment with the `wxai_base.xml` model.

## Overview

The single arm environment provides a MuJoCo simulation of a single Trossen WXAI robotic arm mounted on a table with:
- **6 arm joints** (joint_0 through joint_5)
- **1 gripper** with 2 coupled joints (controlled by 1 actuator)
- **Total DOF**: 8 (6 arm + 2 gripper)
- **Control space**: 7 actuators (6 arm + 1 gripper)
- **Scene**: Robot mounted on a table at z=0.04m (table surface at z=0.02-0.04m)
- **Cameras**: 3 camera views (cam_high, cam_front, cam_side)

## Files

- **Model**: `trossen_arm_mujoco/assets/wxai/wxai_base.xml` - Single arm definition
- **Scene**: `trossen_arm_mujoco/assets/wxai/scene.xml` - Complete scene with floor, lighting, and camera
- **Environment**: `trossen_arm_mujoco/single_arm_env.py` - Task and environment definitions
- **Example**: `examples/single_arm_example.py` - Demo scripts

## Quick Start

### Run the basic test:

```bash
python -m trossen_arm_mujoco.single_arm_env
```

This will run a simulation with random actions for 1000 steps.

### Run the example demos:

```bash
# Random action demo with default camera (cam_high)
python examples/single_arm_example.py

# Random action demo with front camera view
python examples/single_arm_example.py cam_front

# Random action demo with side camera view
python examples/single_arm_example.py cam_side

# Controlled movement demo
python examples/single_arm_example.py controlled
```

## Using in Your Code

### Basic Usage

```python
from trossen_arm_mujoco.single_arm_env import SimpleSingleArmTask
from trossen_arm_mujoco.utils import make_sim_env
import numpy as np

# Create environment with default camera (cam_high)
env = make_sim_env(
    SimpleSingleArmTask,
    xml_file="wxai/scene.xml",
    task_name="single_arm",
    onscreen_render=True,
    cam_list=["cam_high"],  # or ["cam_front"] or ["cam_side"]
)

# Reset
ts = env.reset()

# Run simulation loop
for t in range(1000):
    # Create action: 8 values [6 arm joints, 2 gripper joints]
    arm_action = np.random.uniform(-1.0, 1.0, 6)
    gripper_action = np.random.uniform(0.0, 0.044, 2)
    action = np.concatenate([arm_action, gripper_action])

    # Step
    ts = env.step(action)

    # Get observations
    qpos = ts.observation["qpos"]  # Joint positions
    qvel = ts.observation["qvel"]  # Joint velocities
    images = ts.observation["images"]  # Camera images
```

### Custom Task

Create your own task by subclassing `TrossenAISingleArmTask`:

```python
from trossen_arm_mujoco.single_arm_env import TrossenAISingleArmTask
from dm_control.mujoco.engine import Physics

class MyCustomTask(TrossenAISingleArmTask):
    def initialize_episode(self, physics: Physics) -> None:
        """Set initial state"""
        with physics.reset_context():
            # Set custom initial joint positions
            physics.named.data.qpos[:8] = [0, 1.0, 1.0, 0, 0, 0, 0.044, 0.044]
        super().initialize_episode(physics)

    def get_reward(self, physics: Physics) -> float:
        """Compute custom reward"""
        # Example: reward for reaching a target position
        end_effector_pos = physics.named.data.xpos["link_6"]
        target_pos = np.array([0.3, 0.0, 0.2])
        distance = np.linalg.norm(end_effector_pos - target_pos)
        return -distance
```

## Joint Information

### Arm Joints (from wxai_base.xml)

| Joint | Name | Type | Range (rad) | Description |
|-------|------|------|-------------|-------------|
| 0 | joint_0 | Revolute | -3.05 to 3.05 | Base rotation |
| 1 | joint_1 | Revolute | 0 to 3.14 | Shoulder |
| 2 | joint_2 | Revolute | 0 to 2.36 | Elbow |
| 3 | joint_3 | Revolute | -1.57 to 1.57 | Wrist roll |
| 4 | joint_4 | Revolute | -1.57 to 1.57 | Wrist pitch |
| 5 | joint_5 | Revolute | -3.14 to 3.14 | Wrist yaw |

### Gripper Joints

| Joint | Name | Type | Range (m) | Description |
|-------|------|------|-----------|-------------|
| 6 | right_carriage_joint | Slide | 0 to 0.044 | Right gripper finger |
| 7 | left_carriage_joint | Slide | 0 to 0.044 | Left gripper finger |

**Note**: The gripper joints are coupled via an equality constraint, so they move together. Only one actuator controls both.

## Action Space

The action space can be provided in two formats:

1. **8-DOF format** (qpos): `[joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, gripper_left, gripper_right]`
2. **7-DOF format** (ctrl): `[joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, gripper]`

The environment automatically converts 8-DOF actions to 7-DOF control signals.

## Observation Space

```python
observation = {
    "qpos": np.ndarray,      # Joint positions (8,)
    "qvel": np.ndarray,      # Joint velocities (8,)
    "env_state": np.ndarray, # Environment state (8,)
    "images": {
        "cam_high": np.ndarray  # RGB image (480, 640, 3)
    }
}
```

## Camera Configuration

The scene includes three camera views to observe the robot from different angles:

### cam_high (Default)
- Position: (0.6, 0.6, 0.6)
- View: Angled top view showing the entire robot and table
- Best for: Overall workspace visualization

### cam_front
- Position: (0.7, 0, 0.4)
- View: Front view of the robot
- Best for: Observing forward/backward arm movements

### cam_side
- Position: (0, 0.7, 0.4)
- View: Side view of the robot
- Best for: Observing left/right arm movements

You can switch between cameras by specifying the camera name in the `cam_list` parameter or add more cameras by editing `trossen_arm_mujoco/assets/wxai/scene.xml`.

## Comparison with Bimanual Setup

| Feature | Single Arm (wxai_base) | Bimanual (stationary_ai) |
|---------|------------------------|--------------------------|
| Arms | 1 | 2 |
| Total DOF | 8 | 16 |
| Actuators | 7 | 14 |
| Model file | wxai/scene.xml | stationary_ai/scene_joint.xml |
| Task class | TrossenAISingleArmTask | TrossenAIStationaryTask |

## Scene Structure

The simulation environment consists of:

1. **Floor**: Ground plane at z=0 with checker pattern
2. **Table**:
   - Tabletop: 0.8m x 0.8m at z=0.02-0.04m
   - Four legs at corners
   - Wood-like texture
3. **Robot**:
   - Base mounted at z=0.04m (on table surface)
   - Total height: ~0.5m from floor when extended
4. **Lighting**:
   - Top light for overall illumination
   - Side light for better depth perception

## Troubleshooting

### No visualization window appears
- Make sure you set `onscreen_render=True` when creating the environment
- Check that matplotlib backend supports interactive plotting

### "Camera not found" error
- Ensure you're using one of the available cameras: cam_high, cam_front, or cam_side
- Verify the camera name in your cam_list matches the XML

### Robot appears to be floating or clipping
- The robot base should be at z=0.04m (on top of table)
- Check that wxai_base.xml has `pos="0 0 0.04"` for base_link

### Actions have no effect
- Check that action values are within joint limits
- Verify action array has correct shape (8 for qpos, 7 for ctrl)

## Advanced Usage

### Accessing Physics Directly

```python
# Get physics instance from environment
physics = env.physics

# Get end-effector position
ee_pos = physics.named.data.xpos["link_6"]

# Get joint torques
torques = physics.data.qfrc_actuator[:7]

# Set joint positions directly (useful for testing)
with physics.reset_context():
    physics.named.data.qpos[:8] = [0, 1, 1, 0, 0, 0, 0.02, 0.02]
```

### Recording Videos

```python
import imageio

frames = []
for t in range(500):
    ts = env.step(action)
    frame = ts.observation["images"]["cam_high"]
    frames.append(frame)

imageio.mimsave("simulation.mp4", frames, fps=30)
```

## Next Steps

- Implement your own reward function for reinforcement learning
- Add objects to the scene for manipulation tasks
- Create multi-camera setups for better observation
- Integrate with real robot teleoperation

## References

- [dm_control documentation](https://github.com/deepmind/dm_control)
- [MuJoCo documentation](https://mujoco.readthedocs.io/)
- Main bimanual simulation: `trossen_arm_mujoco/sim_env.py`
