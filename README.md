# Trossen Arm MuJoCo

## Overview

This package provides the necessary scripts and assets for simulating and training robotic policies using the Trossen AI kits in MuJoCo.
It includes URDFs, mesh models, and MuJoCo XML files for robot configuration, as well as Python scripts for policy execution, reward-based evaluation, data collection, and visualization.

This package supports two types of simulation environments:

1. End-Effector (EE) Controlled Simulation ([`ee_sim_env.py`](./trossen_arm_mujoco/ee_sim_env.py)): Uses motion capture bodies to move the arms
2. Joint-Controlled Simulation ([`sim_env.py`](./trossen_arm_mujoco/sim_env.py)): Uses joint position controllers

## Installation

First, clone this repository:

```bash
git clone https://github.com/TrossenRobotics/trossen_arm_mujoco.git
```

It is recommended to create a virtual environment before installing dependencies.
Create a Conda environment with Python 3.10 or above.

```bash
conda create --name trossen_mujoco_env python=3.10
```

After creation, activate the environment with:

```bash
conda activate trossen_mujoco_env
```

Install the package and required dependencies using:

```bash
cd trossen_arm_mujoco
pip install .
```

To verify the installation, run:

```bash
python trossen_arm_mujoco/ee_sim_env.py
```

If the simulation window appears, the setup was successful.

## 1. Assets ([`assets/`](./trossen_arm_mujoco/assets/))

This folder contains all required MuJoCo XML configuration files, URDF files, and mesh models for the simulation.

### Key Files:

- `stationary_ai/` → Contains the Stationary AI robot configuration files:
  - `scene_mocap.xml` → Uses mocap bodies to control the simulated arms.
  - `scene_joint.xml` → Uses joint controllers similar to real hardware to control the simulated arms.
  - `stationary_ai.xml` → Base model definition of the Stationary AI robot.
  - `stationary_ai_mocap.xml` → Mocap-enabled model with weld constraints.
- `wxai/` → Contains the WXAI robot configuration files.
- `meshes/` → Contains STL and OBJ files for the robot components, including arms, cameras, and environmental objects.

### Motion Capture vs Joint-Controlled Environments:

- Motion Capture (`stationary_ai/scene_mocap.xml`): Uses predefined mocap bodies that move the robot arms based on scripted end effector movements.
- Joint Control (`stationary_ai/scene_joint.xml`): Uses position controllers for each joint, similar to a real-world robot setup.

## 2. Modules ([`trossen_arm_mujoco`](./trossen_arm_mujoco/))

This folder contains all Python modules necessary for running simulations, executing policies, recording episodes, and visualizing results.

### 2.1 Simulations

- `ee_sim_env.py`
  - Loads `stationary_ai/scene_mocap.xml` (motion capture-based control).
  - The arms move by following the positions commanded to the mocap bodies.
  - Used for generating scripted policies that control the robot's arms in predefined ways.

- `sim_env.py`
  - Loads `stationary_ai/scene_joint.xml` (position-controlled joints).
  - Uses joint controllers instead of mocap bodies.
  - Replays joint trajectories from `ee_sim_env.py`, enabling clean simulation visuals without mocap bodies visible in the rendered output.

### 2.2 Scripted Policy Execution

- `scripted_policy.py`
  - Defines pre-scripted movements for the robot arms to perform tasks like picking up objects.
  - Uses the motion capture bodies to generate smooth movement trajectories.
  - In the current setup, a policy is designed to pick up a red block, with randomized block positions in the environment.

## 3. How the Data Collection Works

The data collection process involves two simulation phases:

1. Running a scripted policy in `ee_sim_env.py` to record observations (joint positions).
2. Replaying the recorded joint positions in `sim_env.py` to capture full episode data.

### Step-by-Step Process

1. Run `record_sim_episodes.py`

    - Starts `ee_sim_env.py` and executes a scripted policy.
    - Captures observations in the form of joint positions.
    - Saves these joint positions for later replay.
    - Immediately replays the episode in sim_env.py using the recorded joint positions.
    - During replay, captures:
      - Camera feeds from 4 different viewpoints
      - Joint states (actual positions during execution)
      - Actions (input joint positions)
      - Reward values indicating success or failure

2. Save the Data

    - All observations and actions are stored in HDF5 format, with one file per episode.
    - Each episode is saved as `episode_X.hdf5` inside the `~/.trossen/mujoco/data/` folder.

3. Visualizing the Data

    - The stored HDF5 files can be converted into videos using `visualize_eps.py`.

4. Sim-to-real

    - Run `replay_episode_real.py`
    - This script:
      - Loads the joint position trajectory from a selected HDF5 file.
      - Sends commands to both arms using IP addresses (--left_ip, --right_ip).
      - Plays back the motions based on the saved trajectory.
      - Monitors position error between commanded and actual joint states.
      - Returns arms to home and sleep positions after execution.


## 4. Script Arguments Explanation

### a. record_sim_episodes.py

This script generates and saves demonstration episodes using a scripted policy in simulation. It supports both end-effector control (for task definition) and joint-space replay (for clean data collection), storing all observations in `.hdf5` format.

To generate and save simulation episodes, use:

```bash
python trossen_arm_mujoco/scripts/record_sim_episodes.py \
    --task_name sim_transfer_cube \
    --data_dir sim_transfer_cube \
    --num_episodes 5 \
    --onscreen_render
```
Arguments:

- `--task_name`: Name of the task (default: sim_transfer_cube).
- `--num_episodes`: Number of episodes to generate.
- `--data_dir`: Directory where episodes will be saved (required).
- `--root_dir`: Directory where the root is (optional). Default: `~/.trossen/mujoco/data/`
- `--episode_len`: Number of simulation steps of each episode.
- `--onscreen_render` : Enables on-screen rendering. Default: False (only true if explicitly set)
- `--inject_noise`: Injects noise into actions. Default: False (only true if explicitly set)
- `--cam_names`: Comma-separated list of camera names for image collection

**Note:**

- When you pass `--task_name`, the script will automatically load the corresponding configuration from constants.py.

- You can extend `SIM_TASK_CONFIGS` in `constants.py` to support new task configurations.

- All parameters loaded from `constants.py` can be individually overridden via command-line arguments.

### b. visualize_eps.py

To convert saved episodes to videos, run:

```bash
python trossen_arm_mujoco/scripts/visualize_eps.py \
    --data_dir sim_transfer_cube \
    --output_dir videos \
    --fps 50
```
Arguments:

- `--data_dir`: Directory containing .hdf5 files (required), relative to --root_dir if provided.
- `--root_dir`: Root path prefix for locating data_dir. Default: ~/.trossen/mujoco/data/
- `--output_dir`: Subdirectory inside data_dir where generated .mp4 videos will be saved. Default: videos
- `--fps`: Frames per second for the generated videos (default: 50)
- `--root_dir`: Directory where the root is (optional). Default: `~/.trossen/mujoco/`

**Note:** If you do not specify `--root_dir`, videos will be saved to `~/.trossen/mujoco/data/<data_dir>/<output_dir>`.
You can customize the output path by changing `--root_dir`, `--data_dir`, or `--output_dir` as needed.

### c. replay_episode_real.py

This script replays recorded joint-space episodes on real Trossen robotic arms using data saved in .hdf5 files.
It configures each arm, plays back the actions with a user-defined frame rate, and returns both arms to a safe rest pose after execution.

To perform sim to real, run:

```bash
python trossen_arm_mujoco/scripts/replay_episode_real.py \
    --data_dir sim_transfer_cube \
    --episode_idx 0 \
    --fps 10 \
    --left_ip 192.168.1.5 \
    --right_ip 192.168.1.4
```

Arguments:

- `--data_dir`: Directory containing `.hdf5` files (required).
- `--root_dir`: Directory where the root is (optional). Default: `~/.trossen/mujoco/data/`
- `--episode_idx`: Index of the episode to replay. Default: 0
- `--fps`: Playback frame rate (Hz). Controls the action replay speed. Default: 10
- `--left_ip` : IP address of the left Trossen arm. Default: 192.168.1.5
- `--right_ip`: 	IP address of the right Trossen arm. Default: 192.168.1.4

## 5. Single-Arm Food Task

The package includes a single-arm manipulation task for food scooping and transfer operations using the WXAI robot arm with a spoon attachment.

### 5.1 Scene Files

- `assets/wxai/teleop_scene.xml` - Teleop scene with container, bowl, and 4 ramekins arranged in a 2x2 grid
- `assets/food_task/teleop_follower_spoon.xml` - Single arm robot model with spoon end-effector

### 5.2 Running Single-Arm Policies

The single-arm scripted policies demonstrate food scooping from a bowl and transferring to ramekins.

**Available policies:**
- `teleop` - Targets ramekin_2 (front right)
- `teleop_2` - Targets ramekin_3 (back left)
- `bowl_to_plate` - Basic bowl-to-plate transfer using IK
- `simple_pick` - Simple pick demonstration

Run a policy with real-time visualization (22 seconds):

```bash
python -m trossen_arm_mujoco.assets.food_task.scripted_policy_single_arm \
    --policy teleop_2 \
    --episode_len 1100
```

Arguments:
- `--policy`: Policy to run (`bowl_to_plate`, `simple_pick`, `teleop`, `teleop_2`)
- `--episode_len`: Episode length in timesteps (50 Hz, so 1100 = 22 seconds)
- `--no_render`: Disable visualization
- `--inject_noise`: Add noise to actions for robustness testing
- `--camera_view`: Show camera views in matplotlib window

#### 5.2.1 Replay Recorded Teleop Data

You can replay recorded teleoperation data in simulation:

```bash
# Replay a single episode
python -m trossen_arm_mujoco.scripts.replay_episode_teleop \
    --data_dir /home/shyam/projects/cc/dataset/data_from_raven/dual_arm_recording_20260113_131443  \
    --arm right \
    --role follower

# Replay multiple episodes from a root directory
python -m trossen_arm_mujoco.scripts.replay_episode_teleop \
    --data_root /home/shyam/projects/cc/data_from_raven \
    --num_episodes 4 \
    --arm right \
    --role follower
```

Arguments:
- `--data_dir`: Single episode directory containing `arm_data` folder with CSV files
- `--data_root`: Root directory containing multiple `dual_arm_recording_*` episode folders
- `--num_episodes`: Number of episodes to replay (default: all)
- `--arm`: Which arm to replay (`left` or `right`, default: `right`)
- `--role`: Role of the arm (`leader` or `follower`, default: `follower`)
- `--speed`: Playback speed multiplier (default: 1.0)

#### 5.2.2 Convert Teleop CSV to HDF5 Dataset

Convert real teleoperation CSV recordings to HDF5 datasets by replaying them in simulation. This captures simulated camera images and joint states for training.

```bash
python -m trossen_arm_mujoco.scripts.convert_teleop_to_hdf5 \
    --data_root /home/shyam/projects/cc/dataset/data_from_raven \
    --output_dir teleop_hdf5_dataset
```

Arguments:
- `--data_root`: Root directory containing multiple `dual_arm_recording_*` episode folders
- `--data_dir`: Single episode directory (alternative to `--data_root`)
- `--output_dir`: Directory to save HDF5 files
- `--num_episodes`: Number of episodes to convert (default: all)
- `--arm`: Which arm to use (`left` or `right`, default: `right`)
- `--role`: Which role to use (`leader` or `follower`, default: `follower`)
- `--realtime`: Replay at original timing (slower, useful for verification)

HDF5 output structure:
```
episode_X.hdf5
├── observations/
│   ├── images/
│   │   ├── cam_high  (timesteps, 480, 640, 3) uint8
│   │   └── cam       (timesteps, 480, 640, 3) uint8
│   ├── qpos          (timesteps, 8) float64
│   └── qvel          (timesteps, 8) float64
├── action            (timesteps, 8) float64
└── attrs: sim=True, source="teleop_replay", original_episode=<name>
```

#### 5.2.3 Visualize HDF5 Episodes as Videos

Convert HDF5 episode files to MP4 videos for verification:

```bash
python -m trossen_arm_mujoco.scripts.visualize_eps \
    --data_dir teleop_hdf5_dataset \
    --root_dir . \
    --output_dir videos \
    --fps 200
```

Arguments:
- `--data_dir`: Directory containing HDF5 episode files
- `--root_dir`: Root directory (use `.` for current directory)
- `--output_dir`: Subdirectory for output videos (default: `videos`)
- `--fps`: Frames per second for video (use ~200 to match original teleop recording rate)

Videos are saved to `<data_dir>/<output_dir>/episode_X.mp4`.

### 5.3 Pose Tuner Tool

An interactive tool for adjusting robot waypoints in real-time using keyboard controls.

```bash
python -m trossen_arm_mujoco.assets.food_task.pose_tuner --pose above_plate_teleop2
```

**Keyboard Controls (in terminal):**
- `0-5`: Select joint to adjust (0=base, 1=shoulder, 2=elbow, 3-5=wrist)
- `+` or `=`: Increase selected joint value
- `-` or `_`: Decrease selected joint value
- `[` / `]`: Decrease/increase step size
- `p`: Print current pose in copy-paste format
- `r`: Reset to initial pose
- `q`: Quit and print final pose

**Available poses to tune:**
- `above_plate_teleop2` - Position above ramekin_3
- `dump_teleop2` - Dump position for ramekin_3
- `above_plate_teleop` - Position above ramekin_2
- `dump_teleop` - Dump position for ramekin_2

### 5.4 Trajectory Waypoints

The teleop policies use waypoint-based trajectories with linear interpolation between poses. Each waypoint consists of:
- 6 arm joint angles (radians)
- 2 gripper values (0.044 = open, 0.012 = closed)

Example trajectory sequence for `teleop_2`:
1. `home` (0s) - Starting position
2. `reach_bowl` (3s) - Extend arm into bowl
3. `scoop` (7s) - Scoop position with wrist rotation
4. `lift` (9s) - Lift arm to clear container
5. `above_plate` (12s) - Position above target ramekin
6. `dump` (17s) - Dump food with wrist rotation
7. `return_pos` (20s) - Return to neutral position

### 5.5 Food Transfer with IK

Run the food transfer task with interactive visualization using the Mink IK solver:

```bash
python -m trossen_arm_mujoco.scripts.food_transfer_ik --target ramekin_2 --scene wxai/food_scene.xml
```

Arguments:
- `--target`: Target bowl (`bowl_1`, `bowl_2`, `bowl_3`, `bowl_4`, or `all` to cycle)
- `--scene`: Scene XML file (relative to assets/ directory)

### 5.6 Record Food Transfer with IK

Record food transfer demonstrations using the Mink IK solver:

```bash
python -m trossen_arm_mujoco.scripts.record_food_transfer_ik \
    --output_dir food_transfer_ik_dataset_s2_b3 \
    --num_episodes 5 \
    --scene wxai/teleop_scene_1.xml \
    --target bowl_3
```

Arguments:
- `--output_dir`: Directory to save HDF5 episode files
- `--num_episodes`: Number of episodes to record
- `--scene`: Scene XML file (relative to assets/ directory)
- `--target`: Target bowl (`bowl_1`, `bowl_2`, `bowl_3`, `bowl_4`, or `all` to cycle)

### 5.7 Domain Randomization

Record food transfer demonstrations with domain randomization for improved policy generalization. Domain randomization varies object positions, orientations, and the number of bowls in the scene.

```bash
# Record 50 episodes with full domain randomization (1-8 bowls)
python -m trossen_arm_mujoco.scripts.record_food_transfer_ik \
    --output_dir dr_dataset \
    --num_episodes 50 \
    --enable_dr \
    --dr_position_noise 0.03 \
    --dr_rotation_noise 0.1 \
    --dr_container_rotation 0.15 \
    --dr_min_bowls 1 \
    --dr_max_bowls 8 \
    --dr_seed 42
```

Arguments:
- `--enable_dr`: Enable domain randomization
- `--dr_position_noise`: Position noise standard deviation in meters (default: 0.03)
- `--dr_rotation_noise`: Bowl rotation noise standard deviation in radians (default: 0.1)
- `--dr_container_rotation`: Container rotation noise standard deviation in radians (default: 0.15)
- `--dr_min_bowls`: Minimum number of bowls in scene (default: 1)
- `--dr_max_bowls`: Maximum number of bowls in scene (default: 8)
- `--dr_seed`: Random seed for reproducibility (optional)

## Customization

### 1. Modifying Tasks

To create a custom task, modify `ee_sim_env.py` or `sim_env.py` and define a new subclass of `TrossenAIStationary(EE)Task`.
Implement:

- `initialize_episode(self, physics)`: Set up the initial environment state, including robot and object positions.
- `get_observation(self, physics)`: Define what data should be recorded as observations.
- `get_reward(self, physics)`: Implement the reward function to determine task success criteria.

### 2. Changing Policy Behavior

Modify `scripted_policy.py` to define new behavior for the robotic arms.
Update the trajectory generation logic in `PickAndTransferPolicy.generate_trajectory()` to create different movement patterns.

Each movement step in the trajectory is defined by:

- `t`: The time step at which the movement shall occur.
- `xyz`: The target position of the end effector in 3D space.
- `quat`: The target orientation of the end effector, represented as a quaternion.
- `gripper`: The target gripper finger position 0~0.044 where 0 is closed and 0.044 is fully open.

Example:

```python
def generate_trajectory(self, ts_first: TimeStep):
    self.left_trajectory = [
        {"t": 0, "xyz": [0, 0, 0.4], "quat": [1, 0, 0, 0], "gripper": 0},
        {"t": 100, "xyz": [0.1, 0, 0.3], "quat": [1, 0, 0, 0], "gripper": 0.044}
    ]
```

### 3. Adding New Environment Setups

The simulation uses XML files stored in the `assets/` directory. To introduce a new environment setup:

1. Create a new XML configuration file in `assets/` with desired object placements and constraints.
2. Modify `sim_env.py` to load the new environment by specifying the new XML file.
3. Update the scripted policies in `scripted_policy.py` to accommodate new task goals and constraints.

## TroubleshootingConfiguration
Tasks
Limits
Inverse kinematics
Utilities
Lie

If you encounter into Mesa Loader or `mujoco.FatalError: gladLoadGL error` errors:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```
