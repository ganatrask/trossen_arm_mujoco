# MuJoCo ROS 2 Bridge

Connects the MuJoCo simulation to a full ROS 2 stack — replacing Isaac Sim as the physics backend while keeping the rest of the cuMotion/nvblox pipeline unchanged.

## Overview

The bridge (`mujoco_ros2_bridge.py`) is a single ROS 2 node that:

| Direction | Topic | Purpose |
|-----------|-------|---------|
| Publish | `/joint_states` | Feeds current joint positions/velocities to `ros2_control` |
| Subscribe | `/joint_commands` | Receives commanded positions from `ros2_control` and applies to MuJoCo |
| Publish | `/depth` (32FC1) | Depth image for nvblox scene reconstruction |
| Publish | `/rgb` (rgb8) | Color image for nvblox |
| Publish | `/camera_info` | Camera intrinsics (pinhole, computed from MuJoCo fovy) |
| Publish | `/tf_static` | `base_link → <depth_camera>` transform for nvblox TF lookup |

The `isaac_sim` hardware type in `trossen_arm_bringup` is `topic_based_ros2_control` — it reads `/joint_states` and writes `/joint_commands`. The bridge implements the same interface, so **T2–T7 run completely unchanged**.

## Threading Architecture

```
Main thread (rclpy.spin)
  ├── _sim_step     @ 200 Hz  — mj_step + viewer sync
  ├── _pub_joints   @ 50 Hz   — publishes /joint_states
  └── _cmd_callback @ 100 Hz  — applies /joint_commands to ctrl

Render thread (Python threading.Thread)
  └── _render_loop  @ ~6 Hz   — creates its own MjData + Renderer
                                 snapshots qpos → mj_forward → render
                                 publishes /depth, /rgb, /camera_info
```

**Why a dedicated render thread?**
MuJoCo `Renderer` uses EGL for offscreen rendering. EGL contexts are thread-affine — they can only be made current on the thread that created them. Creating the renderer inside the render thread ensures it is always called from the correct thread, avoiding `EGL_BAD_ACCESS` errors.

**Why a separate `MjData` for rendering?**
`mj_forward` and `mj_step` both use MuJoCo's internal stack in `mjData`. Calling them concurrently on the same `mjData` causes stack overflow. The render thread creates its own `render_data = mujoco.MjData(model)`, snapshots `qpos/qvel` (fast copy), then calls `mj_forward(model, render_data)` — completely isolated from the sim.

## Full Pipeline

```
T1  python3 -m trossen_arm_mujoco.scripts.mujoco_ros2_bridge
T2  ros2 launch trossen_arm_bringup trossen_arm.launch.py arm_variant:=follower ros2_control_hardware_type:=isaac_sim use_rviz:=false
T3  python3 scripts/trajectory_relay.py
T4  ros2 launch trossen_curobo moveit_remote.launch.py arm_variant:=follower use_cumotion:=true
T5  (docker) ros2 launch isaac_ros_cumotion robot_segmentation.launch.py ...
T6  (docker) ros2 launch nvblox_trossen.launch.py
T7  (docker) ros2 launch isaac_ros_cumotion isaac_ros_cumotion.launch.py ...
```

Only T1 changes from the Isaac Sim setup. Set these env vars before all terminals:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=134
```

## Usage

```bash
# Default — opens MuJoCo viewer, uses main_view camera for depth
python3 -m trossen_arm_mujoco.scripts.mujoco_ros2_bridge

# Custom scene or camera
python3 -m trossen_arm_mujoco.scripts.mujoco_ros2_bridge \
    --scene wxai/teleop_scene.xml \
    --depth_camera main_view

# Headless (no viewer window — for servers or dataset recording)
python3 -m trossen_arm_mujoco.scripts.mujoco_ros2_bridge --no_viewer

# Lower depth rate to reduce CPU load
python3 -m trossen_arm_mujoco.scripts.mujoco_ros2_bridge --depth_hz 5
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scene` | `wxai/teleop_scene.xml` | MuJoCo scene XML (relative to assets dir or absolute path) |
| `--depth_camera` | `main_view` | Camera name in the XML to use for `/depth` and `/rgb` |
| `--control_hz` | `200` | Physics simulation step rate (Hz) |
| `--publish_hz` | `50` | `/joint_states` publish rate (Hz) |
| `--depth_hz` | `15` | Target depth/rgb publish rate (Hz) — actual rate is render-limited |
| `--no_viewer` | off | Disable the MuJoCo viewer window |

## Available Cameras

| Camera | Description | Notes |
|--------|-------------|-------|
| `main_view` | Overhead fixed camera | **Recommended** for nvblox |
| `cam` | Wrist camera (body-attached) | Moves with arm; static TF will be wrong |
| `cam_high` | DO NOT USE | Points at sky |
| `cam_front` | DO NOT USE | Shows only floor |

For body-attached cameras (e.g. `cam`), the static TF published by the bridge will be incorrect since it reads the initial pose only. A dynamic TF publisher would be needed.

## Topic Verification

After starting the bridge, verify topics with:

```bash
# Joint control loop working (should be ~100 Hz)
ros2 topic hz /joint_commands

# Joint states published (should be ~50 Hz)
ros2 topic hz /joint_states

# Depth publishing (render-limited, typically ~6 Hz at 320×424)
ros2 topic hz /depth

# Check message content
ros2 topic echo /joint_states --once
ros2 topic echo /camera_info --once
ros2 topic echo /tf_static --once
```

## Render Rate

Depth/RGB rendering is GPU-limited. Expected rates:

| Resolution | Approx. rate |
|------------|-------------|
| 640 × 480 | ~4–6 Hz |
| 320 × 424 | ~8–12 Hz |

To change resolution, edit `IMG_H, IMG_W` at the top of `mujoco_ros2_bridge.py`. nvblox works fine at 6 Hz.

## Joint Mapping

| ros2_control joint | MuJoCo actuator | MuJoCo joint |
|-------------------|-----------------|--------------|
| `joint_0` | `joint_0` | `joint_0` |
| `joint_1` | `joint_1` | `joint_1` |
| `joint_2` | `joint_2` | `joint_2` |
| `joint_3` | `joint_3` | `joint_3` |
| `joint_4` | `joint_4` | `joint_4` |
| `joint_5` | `joint_5` | `joint_5` |
| `left_carriage_joint` | `left_gripper` | `left_carriage_joint` |

## Troubleshooting

**`EGL_BAD_ACCESS` on startup**
EGL context created on wrong thread. Ensure `MUJOCO_GL=egl` is set before any renderer is created — the script sets this automatically via `os.environ.setdefault`.

**`mj_stackAlloc: out of memory`**
`mj_forward` called on `self.data` from the render thread concurrently with `mj_step`. Fixed by using a separate `render_data` in the render thread.

**`GLX: Failed to make context current`**
Viewer (GLX) and renderer (EGL) context conflict. Fixed by `MUJOCO_GL=egl` which directs the renderer to use EGL instead of GLX.

**`/joint_states` not found**
Check spelling — the topic is `/joint_states` (with `s`). Confirm with `ros2 topic list | grep joint`.

**Viewer opens but arm doesn't move**
Confirm T2 is running with `ros2_control_hardware_type:=isaac_sim`. Check `/joint_commands` is publishing: `ros2 topic hz /joint_commands`.
