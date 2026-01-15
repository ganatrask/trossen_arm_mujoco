# Quick Reference Guide

## 🚀 Quick Start

```bash
# Install
cd trossen_arm_mujoco
pip install .

# Run demo (bimanual)
python -m trossen_arm_mujoco.sim_env

# Run demo (single arm)
python -m trossen_arm_mujoco.single_arm_env
```

---

## 📦 Project Components

| Component | File | Purpose |
|-----------|------|---------|
| **EE Control** | `ee_sim_env.py` | Mocap-based Cartesian control |
| **Joint Control** | `sim_env.py` | Joint-space position control |
| **Single Arm** | `single_arm_env.py` | Simplified single arm setup |
| **Policies** | `scripted_policy.py` | Pre-programmed behaviors |
| **Config** | `constants.py` | Task configs & parameters |
| **Utils** | `utils.py` | Helper functions |

---

## 🤖 Robot Configurations

### Stationary AI (Bimanual)
- **DOF**: 16 (2 arms × 8)
- **Actuators**: 14
- **Scene**: `stationary_ai/scene_joint.xml`
- **With Mocap**: `stationary_ai/scene_mocap.xml`

### WXAI Base (Single Arm)
- **DOF**: 8 (6 joints + 2 gripper)
- **Actuators**: 7
- **Scene**: `wxai/scene.xml`

---

## 🎮 Control Modes

### End-Effector Control (Mocap)
```python
# Action: [x, y, z, qw, qx, qy, qz, gripper]
action_left = [0.2, 0.3, 0.1, 1, 0, 0, 0, 0.044]
action_right = [0.2, -0.3, 0.1, 1, 0, 0, 0, 0.044]
action = np.concatenate([action_left, action_right])
```

### Joint Control
```python
# Action: [j0, j1, j2, j3, j4, j5, grip_l, grip_r, ...]
# Bimanual: 16 values
action = np.array([0, 0.3, 0.3, 0, 0, 0, 0.044, 0.044,  # Left
                   0, 0.3, 0.3, 0, 0, 0, 0.044, 0.044]) # Right

# Single arm: 8 values
action = np.array([0, 0.3, 0.3, 0, 0, 0, 0.044, 0.044])
```

---

## 📸 Camera Names

### Bimanual (Stationary AI)
- `cam_high` - Top-down workspace view
- `cam_low` - Table-level view
- `cam_left_wrist` - Left arm wrist camera
- `cam_right_wrist` - Right arm wrist camera

### Single Arm (WXAI)
- `cam_high` - Angled top view
- `cam_front` - Front view
- `cam_side` - Side view
- `cam` - Wrist camera (wxai_follower only)

---

## 🛠️ Common Commands

### Record Demonstrations
```bash
python trossen_arm_mujoco/scripts/record_sim_episodes.py \
    --task_name sim_transfer_cube \
    --data_dir my_demos \
    --num_episodes 10 \
    --onscreen_render
```

### Visualize Episodes
```bash
python trossen_arm_mujoco/scripts/visualize_eps.py \
    --data_dir my_demos \
    --output_dir videos \
    --fps 50
```

### Replay on Real Robot
```bash
python trossen_arm_mujoco/scripts/replay_episode_real.py \
    --data_dir my_demos \
    --episode_idx 0 \
    --left_ip 192.168.1.5 \
    --right_ip 192.168.1.4
```

### Single Arm Examples
```bash
# Default camera
python examples/single_arm_example.py

# Different cameras
python examples/single_arm_example.py cam_front
python examples/single_arm_example.py cam_side

# Controlled movement
python examples/single_arm_example.py controlled
```

---

## 📁 Important Paths

| Path | Description |
|------|-------------|
| `~/.trossen/mujoco/data/` | Default data directory |
| `trossen_arm_mujoco/assets/` | XML models & meshes |
| `trossen_arm_mujoco/scripts/` | CLI tools |
| `examples/` | Example scripts |

---

## 🔧 Creating Custom Tasks

### 1. Define Task Class

```python
from trossen_arm_mujoco.sim_env import TrossenAIStationaryTask
from dm_control.mujoco.engine import Physics

class MyTask(TrossenAIStationaryTask):
    def initialize_episode(self, physics: Physics):
        """Set up initial state"""
        super().initialize_episode(physics)
        # Your initialization
        physics.named.data.qpos[:16] = START_ARM_POSE

    def get_reward(self, physics: Physics) -> float:
        """Compute reward"""
        # Your reward logic
        return 0.0
```

### 2. Create Environment

```python
from trossen_arm_mujoco.utils import make_sim_env

env = make_sim_env(
    MyTask,
    xml_file="stationary_ai/scene_joint.xml",
    task_name="my_task"
)
```

### 3. Run Simulation

```python
ts = env.reset()
for t in range(1000):
    action = ... # Your policy
    ts = env.step(action)
```

---

## 🎯 Scripted Policy Template

```python
from trossen_arm_mujoco.scripted_policy import BasePolicy

class MyPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        """Define movement waypoints"""
        self.left_trajectory = [
            {
                "t": 0,
                "xyz": [x, y, z],
                "quat": [w, x, y, z],
                "gripper": 0.044  # Open
            },
            {
                "t": 100,
                "xyz": [x2, y2, z2],
                "quat": [w, x, y, z],
                "gripper": 0.0  # Closed
            },
        ]
        self.right_trajectory = [...]  # Similar for right arm

    def __call__(self, ts):
        """Execute policy at timestep"""
        action_dict = self.interpolate_trajectory(ts)
        return action_dict
```

---

## 📊 Data Format (HDF5)

### Structure
```
episode_0.hdf5
├── /observations
│   ├── qpos       # Joint positions [T, 16]
│   ├── qvel       # Joint velocities [T, 16]
│   ├── images     # Camera images
│   │   ├── cam_high       [T, 480, 640, 3]
│   │   ├── cam_low        [T, 480, 640, 3]
│   │   ├── cam_left_wrist [T, 480, 640, 3]
│   │   └── cam_right_wrist[T, 480, 640, 3]
│   └── env_state  # Environment state [T, 7]
├── /action        # Actions taken [T, 16]
└── /reward        # Rewards [T]
```

### Read Episode
```python
import h5py

with h5py.File('episode_0.hdf5', 'r') as f:
    qpos = f['/observations/qpos'][:]
    images = f['/observations/images/cam_high'][:]
    actions = f['/action'][:]
    rewards = f['/reward'][:]
```

---

## 🔍 Debugging Tips

### Check Scene Loading
```python
from dm_control import mujoco
from trossen_arm_mujoco.constants import ASSETS_DIR
import os

xml_path = os.path.join(ASSETS_DIR, 'wxai/scene.xml')
physics = mujoco.Physics.from_xml_path(xml_path)
print(f"Bodies: {physics.model.nbody}")
print(f"Joints: {physics.model.njnt}")
print(f"Cameras: {physics.model.ncam}")
```

### Inspect Robot State
```python
# After env.reset() or env.step()
qpos = ts.observation['qpos']
print(f"Joint positions: {qpos}")

# End-effector position
ee_pos = physics.named.data.xpos['follower_left_link_6']
print(f"Left EE: {ee_pos}")
```

### Render Camera View
```python
image = physics.render(height=480, width=640, camera_id='cam_high')
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()
```

---

## ⚡ Performance Tips

1. **Disable rendering** for fast data collection:
   ```python
   env = make_sim_env(MyTask, onscreen_render=False)
   ```

2. **Reduce image resolution** if not needed:
   ```python
   image = physics.render(height=240, width=320, camera_id='cam_high')
   ```

3. **Parallel collection**: Run multiple processes
   ```bash
   for i in {0..9}; do
       python record_sim_episodes.py --num_episodes 10 &
   done
   wait
   ```

---

## 🐛 Common Errors

### "gladLoadGL error"
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

### "Camera not found"
Check camera name matches XML definition:
```bash
grep -r "camera name=" trossen_arm_mujoco/assets/
```

### "Invalid action shape"
- Bimanual: Needs 16 values (2 × 8 DOF)
- Single arm: Needs 8 values (1 × 8 DOF)

### Import errors
```bash
pip install -e .  # Install in editable mode
```

---

## 📚 Key Concepts

| Concept | Description |
|---------|-------------|
| **Mocap Bodies** | Virtual objects for Cartesian control |
| **Weld Constraint** | Links mocap to end-effector |
| **Equality Constraint** | Couples gripper joints |
| **TimeStep** | Data structure with observations, reward, discount |
| **Physics** | MuJoCo simulation instance |
| **Task** | Defines initialization, observations, rewards |

---

## 🎓 Learning Resources

### In This Repo
1. [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Comprehensive guide
2. [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) - Visual diagrams
3. [SINGLE_ARM_USAGE.md](SINGLE_ARM_USAGE.md) - Single arm guide
4. [README.md](README.md) - Installation & basic usage

### External
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [dm_control Documentation](https://github.com/deepmind/dm_control)
- [Trossen Robotics](https://www.trossenrobotics.com/)

---

## 🔄 Typical Workflow

```
1. Design Task
   └─> Create task class (sim_env.py or ee_sim_env.py)

2. Create Policy
   └─> Define waypoints (scripted_policy.py)

3. Test in Simulation
   └─> python -m trossen_arm_mujoco.ee_sim_env

4. Record Demonstrations
   └─> python scripts/record_sim_episodes.py

5. Visualize Results
   └─> python scripts/visualize_eps.py

6. (Optional) Train RL Policy
   └─> Use HDF5 data with your favorite RL library

7. Deploy to Real Robot
   └─> python scripts/replay_episode_real.py
```

---

## 🆘 Getting Help

- Check error messages carefully
- Read relevant .md files in repo
- Inspect XML files to understand scene structure
- Use Python debugger: `import pdb; pdb.set_trace()`
- Check MuJoCo/dm_control docs for physics questions

---

## 💡 Pro Tips

1. **Start simple**: Test with single arm before bimanual
2. **Visualize first**: Always use `--onscreen_render` when debugging
3. **Check joint limits**: See XML files for valid ranges
4. **Use small timesteps**: DT=0.02 is good balance
5. **Save often**: Record episodes incrementally
6. **Version control**: Git commit after successful changes
7. **Document changes**: Update this file when adding features!

---

Last Updated: January 2026
