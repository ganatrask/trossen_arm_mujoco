# Trossen Arm MuJoCo - Complete Project Overview

## 📋 Table of Contents
1. [Project Purpose](#project-purpose)
2. [Project Structure](#project-structure)
3. [Robot Hardware](#robot-hardware)
4. [Simulation Environments](#simulation-environments)
5. [Key Concepts](#key-concepts)
6. [Workflow](#workflow)
7. [File-by-File Guide](#file-by-file-guide)
8. [Recent Additions](#recent-additions)

---

## 🎯 Project Purpose

This project provides **MuJoCo simulation environments** for Trossen Robotics robotic arms, specifically designed for:

1. **Robot Learning**: Training reinforcement learning policies for manipulation tasks
2. **Data Collection**: Recording demonstrations for imitation learning
3. **Sim-to-Real Transfer**: Testing policies in simulation before deploying to real robots
4. **Manipulation Tasks**: Bimanual tasks like transferring objects, pick-and-place, etc.

### Key Use Cases
- Generate demonstration data in simulation
- Train robot policies without hardware
- Test policies safely before real-world deployment
- Visualize and debug robot behavior
- Collect multi-camera observations for vision-based learning

---

## 📁 Project Structure

```
trossen_arm_mujoco/
├── trossen_arm_mujoco/          # Main Python package
│   ├── assets/                  # MuJoCo XML models & meshes
│   │   ├── stationary_ai/       # Bimanual robot configs
│   │   ├── wxai/                # Single arm configs
│   │   ├── mobile_ai/           # Mobile robot configs
│   │   └── meshes/              # STL files for robot parts
│   ├── scripts/                 # Command-line tools
│   ├── *.py                     # Core Python modules
│   └── bowl_food.xml            # Bowl & food objects
├── examples/                    # Example scripts
├── mujoco_scanned_objects/      # 3D scanned objects library
├── README.md                    # Installation & usage
├── SINGLE_ARM_USAGE.md          # Single arm guide
└── setup.py                     # Package configuration
```

---

## 🤖 Robot Hardware

The project supports three Trossen robotics kits:

### 1. **Stationary AI** (Bimanual - Dual Arms)
- **DOF**: 16 (2 arms × 6 joints + 2 grippers × 2 joints)
- **Actuators**: 14 (6 arm joints + 1 gripper per arm)
- **Use Case**: Bimanual manipulation tasks
- **XML Files**:
  - `stationary_ai.xml` - Base model
  - `scene_mocap.xml` - With motion capture control
  - `scene_joint.xml` - With joint control

### 2. **WXAI Base** (Single Arm)
- **DOF**: 8 (6 arm joints + 2 gripper joints)
- **Actuators**: 7 (6 arm + 1 gripper)
- **Use Case**: Single arm tasks, testing, learning
- **XML Files**:
  - `wxai_base.xml` - Base model
  - `wxai_follower.xml` - With camera mount
  - `scene.xml` - Complete scene with table

### 3. **Mobile AI** (Mobile Base + Arms)
- Mobile robot platform with manipulation capabilities
- **XML Files**: `mobile_ai/` directory

---

## 🎮 Simulation Environments

The project provides **two control paradigms**:

### A. End-Effector (EE) Control - `ee_sim_env.py`

**How it works:**
- Uses **motion capture (mocap) bodies** to control end-effector positions
- You specify XYZ positions and orientations for the gripper
- MuJoCo's inverse kinematics solves for joint angles
- Best for task definition and scripted policies

**XML**: `stationary_ai/scene_mocap.xml`

**Use Case**: Generating demonstrations by defining Cartesian waypoints

```python
# Example: Move gripper to position (x, y, z)
action = [x, y, z, qw, qx, qy, qz, gripper_pos]
```

**Classes**:
- `TrossenAIStationaryEETask` - Base class for EE control
- `TransferCubeEETask` - Cube transfer task with EE control

---

### B. Joint Control - `sim_env.py`

**How it works:**
- Directly controls **joint positions** (like real hardware)
- More realistic simulation of actual robot behavior
- Used for replaying recorded trajectories
- Better for training policies that transfer to real robots

**XML**: `stationary_ai/scene_joint.xml`

**Use Case**: Clean replay and data collection

```python
# Example: Set joint positions directly
action = [j0, j1, j2, j3, j4, j5, grip_left, grip_left,  # Left arm
          j0, j1, j2, j3, j4, j5, grip_right, grip_right] # Right arm
```

**Classes**:
- `TrossenAIStationaryTask` - Base class for joint control
- `TransferCubeTask` - Cube transfer task with joint control

---

### C. Single Arm Environment - `single_arm_env.py` ⭐ NEW

**How it works:**
- Single WXAI arm mounted on a table
- 3 camera views (cam_high, cam_front, cam_side)
- Simplified environment for single-arm learning

**XML**: `wxai/scene.xml`

**Use Case**: Single arm manipulation, simpler learning tasks

**Classes**:
- `TrossenAISingleArmTask` - Base class
- `SimpleSingleArmTask` - Basic implementation

---

## 🧠 Key Concepts

### 1. **Motion Capture (Mocap) Bodies**

Mocap bodies are virtual objects in MuJoCo that:
- Are positioned directly by your code
- Pull the robot's end-effector to follow them (via weld constraints)
- Enable Cartesian space control
- Used in `ee_sim_env.py`

**Why use them?**
- Easier to define tasks (think in XYZ, not joint angles)
- Natural for scripting demonstrations
- Automatically handles inverse kinematics

### 2. **Two-Phase Data Collection**

The project uses a clever two-phase approach:

**Phase 1: Generate with EE Control (`ee_sim_env.py`)**
```
Scripted Policy → Mocap Control → Record Joint Positions
```

**Phase 2: Replay with Joint Control (`sim_env.py`)**
```
Recorded Joints → Joint Control → Clean Visuals + Camera Data
```

**Why?**
- Phase 1: Easy to define behaviors in Cartesian space
- Phase 2: Clean recordings without mocap bodies visible
- Result: Realistic data that transfers to real robots

### 3. **Gripper Coupling**

Each gripper has 2 joints (left finger, right finger) but only 1 actuator:
- Both fingers move symmetrically (coupled via equality constraint)
- When you command gripper position, both fingers respond
- This mirrors real hardware behavior

### 4. **Camera System**

Multiple cameras provide different viewpoints:
- `cam_high`: Top-down view of workspace
- `cam_low`: Table-level view
- `cam_left_wrist`: Left arm wrist camera
- `cam_right_wrist`: Right arm wrist camera

Images captured at 480×640 resolution, RGB format

---

## 🔄 Workflow

### Typical Data Collection Workflow

```
1. Define Task
   └─> Create task class in ee_sim_env.py or sim_env.py
       ├─> initialize_episode() - Set up scene
       ├─> get_reward() - Define success criteria
       └─> get_observation() - Specify what data to record

2. Create Policy
   └─> Implement scripted_policy.py
       └─> Define trajectory waypoints (xyz, quat, gripper)

3. Record Episodes
   └─> Run: record_sim_episodes.py
       ├─> Phase 1: Execute in ee_sim_env (mocap control)
       ├─> Record joint trajectories
       └─> Phase 2: Replay in sim_env (joint control)
           └─> Save: observations, actions, rewards, images

4. Visualize
   └─> Run: visualize_eps.py
       └─> Convert HDF5 files to videos

5. Deploy to Real Robot
   └─> Run: replay_episode_real.py
       └─> Send recorded trajectories to physical robots
```

---

## 📄 File-by-File Guide

### Core Modules (`trossen_arm_mujoco/`)

#### **constants.py**
- Configuration constants
- Task definitions (SIM_TASK_CONFIGS)
- Default poses (START_ARM_POSE, BOX_POSE)
- File paths (ASSETS_DIR, ROOT_DIR)

```python
START_ARM_POSE = [0, π/12, π/12, 0, 0, 0, 0.044, 0.044, ...]  # 16 joints
DT = 0.02  # Simulation timestep
```

#### **ee_sim_env.py**
End-effector control environment using mocap bodies

**Key Classes**:
- `TrossenAIStationaryEETask` - Base task with mocap control
- `TransferCubeEETask` - Cube transfer demonstration

**Key Methods**:
- `before_step()` - Maps actions to mocap positions
- `initialize_robots()` - Resets arm poses
- `get_observation()` - Collects camera images + joint states

#### **sim_env.py**
Joint-space control environment (realistic simulation)

**Key Classes**:
- `TrossenAIStationaryTask` - Base task with joint control
- `TransferCubeTask` - Cube transfer with joint commands

**Key Methods**:
- `before_step()` - Maps 16-DOF actions to 14 actuators
- `get_position()` - Returns joint positions
- `get_reward()` - Calculates task success

#### **single_arm_env.py** ⭐
Single arm environment with table and multiple cameras

**Key Features**:
- Robot mounted on table (z=0.04m)
- 3 camera views for different perspectives
- Simplified 8-DOF control
- Interactive MuJoCo viewer integration
- Scripted bowl-to-plate movement demo

**Key Classes**:
- `TrossenAISingleArmTask` - Base class
- `SingleArmTask` - With initialization
- `test_sim_teleop()` - Demo with viewer

#### **scripted_policy.py**
Pre-programmed movement policies for demonstrations

**Key Classes**:
- `BasePolicy` - Abstract base
- `PickAndTransferPolicy` - Pick cube and transfer

**How Trajectories Work**:
```python
trajectory = [
    {"t": 0, "xyz": [x, y, z], "quat": [w, x, y, z], "gripper": 0.044},
    {"t": 100, "xyz": [x2, y2, z2], "quat": [w, x, y, z], "gripper": 0.0},
    ...
]
```

#### **utils.py**
Utility functions for simulation

**Key Functions**:
- `make_sim_env()` - Creates environment from task class
- `get_observation_base()` - Captures camera images
- `plot_observation_images()` - Visualizes observations
- `sample_box_pose()` - Randomizes object positions

---

### Scripts (`trossen_arm_mujoco/scripts/`)

#### **record_sim_episodes.py**
Generates and saves demonstration episodes

**What it does**:
1. Runs scripted policy in `ee_sim_env`
2. Records joint trajectories
3. Replays in `sim_env` with cameras
4. Saves to HDF5 files

**Usage**:
```bash
python trossen_arm_mujoco/scripts/record_sim_episodes.py \
    --task_name sim_transfer_cube \
    --data_dir my_demos \
    --num_episodes 10 \
    --onscreen_render
```

#### **visualize_eps.py**
Converts HDF5 files to MP4 videos

**Usage**:
```bash
python trossen_arm_mujoco/scripts/visualize_eps.py \
    --data_dir my_demos \
    --output_dir videos \
    --fps 50
```

#### **replay_episode_real.py**
Deploys recorded trajectories to physical robots

**Usage**:
```bash
python trossen_arm_mujoco/scripts/replay_episode_real.py \
    --data_dir my_demos \
    --episode_idx 0 \
    --left_ip 192.168.1.5 \
    --right_ip 192.168.1.4
```

---

### Assets (`trossen_arm_mujoco/assets/`)

#### **Stationary AI Files**

**stationary_ai.xml** - Complete bimanual robot definition
- Two arms with wrist cameras
- Tabletop with frame
- High and low workspace cameras
- All joints, actuators, and constraints

**stationary_ai_mocap.xml** - Mocap-enabled version
- Adds mocap bodies at end-effectors
- Weld constraints link mocap to link_6
- Used for Cartesian control

**scene_mocap.xml** - Complete scene with mocap control
- Includes robot, table, lights
- Free-floating red cube
- 4 cameras (high, low, wrist×2)

**scene_joint.xml** - Complete scene with joint control
- Same as scene_mocap but without mocap bodies
- Used for clean replay

#### **WXAI Files**

**wxai_base.xml** - Single arm without camera
- 6 arm joints + 2-finger gripper
- Base positioned at z=0.04m (on table)
- 7 actuators (6 arm + 1 coupled gripper)

**wxai_follower.xml** - Single arm with D405 camera
- Same as base but with wrist camera mount
- Camera positioned on link_6

**scene.xml** ⭐ - Complete single arm scene
- Table with wooden texture (0.8m × 0.8m)
- 3 camera views (high, front, side)
- Includes wxai_follower robot
- Bowl and plate objects
- Food particles for manipulation

#### **Other Files**

**bowl_food.xml** - Bowl and food objects
- Multiple bowl sizes (small, medium, large)
- Plate with texture
- Food spheres with physics
- Uses MuJoCo SDF plugin for bowls

**meshes/** - STL/OBJ files
- Robot part geometries
- Visual and collision meshes

---

## 🆕 Recent Additions

### Single Arm Environment (Jan 2026)

You recently worked on creating a complete single-arm setup:

1. **Scene with Table**
   - Added table geometry to `wxai/scene.xml`
   - Positioned robot on table (z=0.04m)
   - Table surface at z=0.02-0.04m

2. **Multiple Camera Views**
   - `cam_high`: Angled top view (0.6, 0.6, 0.6)
   - `cam_front`: Front view (0.7, 0, 0.4)
   - `cam_side`: Side view (0, 0.7, 0.4)

3. **Bowl and Plate Integration**
   - Medium bowl at (-0.05, 0.15) in front of robot
   - Plate at (-0.05, -0.15)
   - Food particles on bowl and plate
   - Realistic manipulation scenario

4. **Interactive Demo**
   - Modified `single_arm_env.py` with MuJoCo viewer
   - Scripted policy: bowl → plate movement
   - Real-time visualization with matplotlib

5. **Documentation**
   - [SINGLE_ARM_USAGE.md](SINGLE_ARM_USAGE.md) - Complete guide
   - [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) - What was changed
   - [examples/single_arm_example.py](examples/single_arm_example.py) - Demos

---

## 🎓 Learning Path

### If you want to...

**1. Run a demo simulation**
```bash
# Bimanual
python -m trossen_arm_mujoco.sim_env

# Single arm
python -m trossen_arm_mujoco.single_arm_env
```

**2. Collect demonstration data**
```bash
python trossen_arm_mujoco/scripts/record_sim_episodes.py \
    --task_name sim_transfer_cube \
    --data_dir my_dataset \
    --num_episodes 50
```

**3. Create a custom task**
- Copy `TransferCubeTask` in `sim_env.py`
- Modify `initialize_episode()` for your scene
- Implement `get_reward()` for your success criteria

**4. Test on single arm**
```bash
python examples/single_arm_example.py cam_high
```

**5. Deploy to real robots**
```bash
python trossen_arm_mujoco/scripts/replay_episode_real.py \
    --data_dir my_dataset \
    --episode_idx 0 \
    --left_ip <IP> \
    --right_ip <IP>
```

---

## 🔑 Key Takeaways

1. **Two Control Modes**: EE (mocap) for task definition, Joint for realistic simulation
2. **Two-Phase Collection**: Generate with EE, replay with Joint for clean data
3. **Modular Design**: Easy to create new tasks, policies, and environments
4. **Sim-to-Real**: Direct deployment of simulated trajectories to hardware
5. **Single + Dual Arm**: Flexibility for different complexity levels
6. **Rich Observations**: Multi-camera visual + proprioceptive data

---

## 📚 Quick Reference

### File Extensions
- `.xml` - MuJoCo model definitions
- `.stl` - 3D mesh files (visual/collision geometry)
- `.py` - Python code
- `.hdf5` - Recorded episode data

### Common Commands
```bash
# Install
pip install .

# Record demos
python trossen_arm_mujoco/scripts/record_sim_episodes.py --help

# Visualize
python trossen_arm_mujoco/scripts/visualize_eps.py --help

# Single arm
python -m trossen_arm_mujoco.single_arm_env
```

### Important Directories
- Data: `~/.trossen/mujoco/data/`
- Assets: `trossen_arm_mujoco/assets/`
- Scripts: `trossen_arm_mujoco/scripts/`

---

## 🎯 Next Steps

Based on your current work, you might want to:

1. **Create manipulation tasks** with bowl/plate in single arm env
2. **Train RL policies** using the collected demonstrations
3. **Add more objects** from `mujoco_scanned_objects/`
4. **Test sim-to-real transfer** on physical hardware
5. **Implement custom reward functions** for your tasks

---

For specific guides:
- Single Arm: See [SINGLE_ARM_USAGE.md](SINGLE_ARM_USAGE.md)
- Installation: See [README.md](README.md)
- Changes: See [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)
