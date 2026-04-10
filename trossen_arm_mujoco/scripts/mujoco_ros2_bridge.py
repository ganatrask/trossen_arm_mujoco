#!/usr/bin/env python3
"""
MuJoCo ↔ ROS 2 bridge for Trossen WXAI arm.

Replaces Isaac Sim (T1) in the cuMotion pipeline while keeping T2–T7 unchanged.

What this node does:
  - Publishes  /joint_states   → topic_based_ros2_control (T2) reads it
  - Subscribes /joint_commands ← topic_based_ros2_control (T2) writes it
  - Publishes  /depth + /camera_info + /rgb → nvblox (T6) / robot_segmentation (T5)
  - Publishes  /tf_static: base_link → <depth_camera> (nvblox global_frame=base_link)

Threading model:
  - Main thread : rclpy.spin() — sim_step (200Hz), joint_states (50Hz), cmd_callback
  - Render thread: dedicated Python thread — creates its own Renderer so the EGL
                   context stays on the thread that uses it (EGL is thread-affine).

T2 (unchanged):
  ros2 launch trossen_arm_bringup trossen_arm.launch.py \\
      arm_variant:=follower ros2_control_hardware_type:=isaac_sim use_rviz:=false
"""

import argparse
import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import StaticTransformBroadcaster

from trossen_arm_mujoco.constants import ASSETS_DIR

# Joints that ros2_control expects (must match _wxai.ros2_control.xacro)
CONTROLLED_JOINTS = [
    'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5',
    'left_carriage_joint',
]

# MuJoCo actuator name for each ros2_control joint
JOINT_TO_ACTUATOR = {
    'joint_0': 'joint_0',
    'joint_1': 'joint_1',
    'joint_2': 'joint_2',
    'joint_3': 'joint_3',
    'joint_4': 'joint_4',
    'joint_5': 'joint_5',
    'left_carriage_joint': 'left_gripper',
}

# Depth image size — reduce for faster rendering (320x240 ≈ 4x faster than 640x480)
IMG_H, IMG_W = 320, 424


class MuJoCoROS2Bridge(Node):
    def __init__(
        self,
        scene_path: str,
        depth_camera: str,
        control_hz: float,
        publish_hz: float,
        depth_hz: float,
        show_viewer: bool = True,
    ):
        super().__init__('mujoco_ros2_bridge')

        self.get_logger().info(f'Loading MuJoCo model: {scene_path}')
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self._depth_camera = depth_camera
        self._depth_hz = depth_hz

        # --- Joint index maps ---
        self._qpos_idx = {}
        self._qvel_idx = {}
        self._ctrl_idx = {}

        for jname in CONTROLLED_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                self.get_logger().warn(f'Joint {jname!r} not found in model')
                continue
            self._qpos_idx[jname] = self.model.jnt_qposadr[jid]
            self._qvel_idx[jname] = self.model.jnt_dofadr[jid]
            aname = JOINT_TO_ACTUATOR.get(jname)
            if aname:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
                if aid >= 0:
                    self._ctrl_idx[jname] = aid

        self._active_joints = [j for j in CONTROLLED_JOINTS if j in self._qpos_idx]

        # --- Depth camera metadata (no renderer here — created in render thread) ---
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, depth_camera)
        if cam_id < 0:
            raise ValueError(f'Camera {depth_camera!r} not found in model')
        self._cam_id = cam_id
        self._cam_fovy = float(self.model.cam_fovy[cam_id])

        # --- ROS 2 publishers / subscribers (all on main thread) ---
        self._js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self._depth_pub = self.create_publisher(Image, '/depth', 10)
        self._rgb_pub = self.create_publisher(Image, '/rgb', 10)
        self._info_pub = self.create_publisher(CameraInfo, '/camera_info', 10)

        self._cmd_sub = self.create_subscription(
            JointState, '/joint_commands', self._cmd_callback, 10
        )

        # Static TF
        self._tf_static = StaticTransformBroadcaster(self)
        self._publish_camera_tf()

        # --- Sim timers (main thread, single-threaded executor) ---
        self._sim_timer = self.create_timer(1.0 / control_hz, self._sim_step)
        self._js_timer = self.create_timer(1.0 / publish_hz, self._publish_joint_states)

        # --- Viewer ---
        self._viewer = None
        if show_viewer:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.get_logger().info('MuJoCo viewer opened')

        # --- Render thread ---
        # Renderer is created INSIDE the thread so its EGL context is bound
        # to that thread. Never create/use a Renderer on a different thread.
        self._stop_render = threading.Event()
        self._render_thread = threading.Thread(
            target=self._render_loop,
            daemon=True,
            name='mujoco-render',
        )
        self._render_thread.start()

        self.get_logger().info(
            f'Bridge ready | depth_camera={depth_camera} | '
            f'sim={control_hz:.0f}Hz js={publish_hz:.0f}Hz depth={depth_hz:.0f}Hz'
        )

    def shutdown(self):
        self._stop_render.set()
        self._render_thread.join(timeout=3.0)
        if self._viewer is not None:
            self._viewer.close()

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _stamp(self):
        t = self.get_clock().now().nanoseconds
        from builtin_interfaces.msg import Time
        ts = Time()
        ts.sec = int(t // 1_000_000_000)
        ts.nanosec = int(t % 1_000_000_000)
        return ts

    def _publish_camera_tf(self):
        cam_pos = self.model.cam_pos[self._cam_id]
        cam_mat = self.model.cam_mat0[self._cam_id]
        R = cam_mat.reshape(3, 3)
        qw = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
        qx = (R[2, 1] - R[1, 2]) / (4.0 * qw + 1e-10)
        qy = (R[0, 2] - R[2, 0]) / (4.0 * qw + 1e-10)
        qz = (R[1, 0] - R[0, 1]) / (4.0 * qw + 1e-10)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self._stamp()
        tf_msg.header.frame_id = 'base_link'
        tf_msg.child_frame_id = self._depth_camera
        tf_msg.transform.translation.x = float(cam_pos[0])
        tf_msg.transform.translation.y = float(cam_pos[1])
        tf_msg.transform.translation.z = float(cam_pos[2])
        tf_msg.transform.rotation.w = float(qw)
        tf_msg.transform.rotation.x = float(qx)
        tf_msg.transform.rotation.y = float(qy)
        tf_msg.transform.rotation.z = float(qz)
        self._tf_static.sendTransform(tf_msg)

    def _make_camera_info(self, stamp) -> CameraInfo:
        fovy_rad = self._cam_fovy * np.pi / 180.0
        fy = (IMG_H / 2.0) / np.tan(fovy_rad / 2.0)
        fx = fy
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self._depth_camera
        msg.width = IMG_W
        msg.height = IMG_H
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [fx, 0.0, IMG_W / 2.0, 0.0, fy, IMG_H / 2.0, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, IMG_W / 2.0, 0.0, 0.0, fy, IMG_H / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    # ─────────────────────────────────────────────
    # Main-thread callbacks
    # ─────────────────────────────────────────────

    def _cmd_callback(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if name in self._ctrl_idx and i < len(msg.position):
                self.data.ctrl[self._ctrl_idx[name]] = msg.position[i]

    def _sim_step(self):
        mujoco.mj_step(self.model, self.data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def _publish_joint_states(self):
        stamp = self._stamp()
        positions = [float(self.data.qpos[self._qpos_idx[j]]) for j in self._active_joints]
        velocities = [float(self.data.qvel[self._qvel_idx[j]]) for j in self._active_joints]
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = list(self._active_joints)
        msg.position = positions
        msg.velocity = velocities
        msg.effort = [0.0] * len(positions)
        self._js_pub.publish(msg)

    # ─────────────────────────────────────────────
    # Render thread — EGL context lives here
    # ─────────────────────────────────────────────

    def _render_loop(self):
        """Dedicated render thread.

        - Renderer created here → EGL context bound to this thread.
        - render_data is a private MjData copy → mj_forward here never
          touches self.data, so there is no stack conflict with mj_step
          running on the main thread.
        """
        renderer = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
        render_data = mujoco.MjData(self.model)   # private copy for rendering
        period = 1.0 / self._depth_hz

        while not self._stop_render.is_set():
            t0 = time.monotonic()
            try:
                self._render_and_publish(renderer, render_data)
            except Exception as e:
                self.get_logger().error(f'Render error: {e}')

            elapsed = time.monotonic() - t0
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        renderer.close()

    def _render_and_publish(self, renderer: mujoco.Renderer, render_data: mujoco.MjData):
        # Snapshot qpos/qvel from the sim (fast copy, benign race for display).
        # mj_forward runs on render_data — completely separate from self.data,
        # so it never conflicts with mj_step on the main thread.
        render_data.qpos[:] = self.data.qpos
        render_data.qvel[:] = self.data.qvel
        mujoco.mj_forward(self.model, render_data)
        renderer.update_scene(render_data, camera=self._depth_camera)
        stamp = self._stamp()

        # Depth (32FC1)
        renderer.enable_depth_rendering()
        depth_m = renderer.render().astype(np.float32)
        depth_msg = Image()
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self._depth_camera
        depth_msg.height = IMG_H
        depth_msg.width = IMG_W
        depth_msg.encoding = '32FC1'
        depth_msg.is_bigendian = False
        depth_msg.step = IMG_W * 4
        depth_msg.data = depth_m.tobytes()
        self._depth_pub.publish(depth_msg)

        # RGB (rgb8)
        renderer.disable_depth_rendering()
        rgb = renderer.render()
        rgb_msg = Image()
        rgb_msg.header.stamp = stamp
        rgb_msg.header.frame_id = self._depth_camera
        rgb_msg.height = IMG_H
        rgb_msg.width = IMG_W
        rgb_msg.encoding = 'rgb8'
        rgb_msg.is_bigendian = False
        rgb_msg.step = IMG_W * 3
        rgb_msg.data = rgb.tobytes()
        self._rgb_pub.publish(rgb_msg)

        # Camera info
        self._info_pub.publish(self._make_camera_info(stamp))


def main():
    parser = argparse.ArgumentParser(description='MuJoCo ROS 2 bridge for Trossen WXAI')
    parser.add_argument('--scene', default='wxai/teleop_scene.xml',
                        help='Scene XML path (relative to assets dir or absolute)')
    parser.add_argument('--depth_camera', default='main_view',
                        help='MuJoCo camera name for /depth publishing')
    parser.add_argument('--control_hz', type=float, default=200.0,
                        help='MuJoCo physics step rate (Hz)')
    parser.add_argument('--publish_hz', type=float, default=50.0,
                        help='/joint_states publish rate (Hz)')
    parser.add_argument('--depth_hz', type=float, default=15.0,
                        help='/depth publish rate (Hz)')
    parser.add_argument('--no_viewer', action='store_true',
                        help='Disable the MuJoCo viewer window (headless mode)')
    args, ros_args = parser.parse_known_args()

    # Must be set before any mujoco.Renderer is created.
    # EGL = offscreen GPU rendering, independent of the viewer's GLFW/GLX context.
    os.environ.setdefault('MUJOCO_GL', 'egl')

    scene_path = args.scene
    if not scene_path.startswith('/'):
        scene_path = os.path.join(ASSETS_DIR, args.scene)

    rclpy.init(args=ros_args)
    node = MuJoCoROS2Bridge(
        scene_path=scene_path,
        depth_camera=args.depth_camera,
        control_hz=args.control_hz,
        publish_hz=args.publish_hz,
        depth_hz=args.depth_hz,
        show_viewer=not args.no_viewer,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
