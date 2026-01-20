#!/usr/bin/env python3
"""
Dynamic Food Transfer Task using Mink IK.

This script generates trajectories dynamically based on actual object positions
in the simulation. If you move the container or ramekins, the IK planner will
find new joint configurations to reach them.

Task phases:
1. HOME -> Move to home/ready position
2. APPROACH_CONTAINER -> Move above container
3. REACH_CONTAINER -> Lower into container (scoop position)
4. SCOOP -> Perform scooping motion
5. LIFT -> Lift spoon from container
6. APPROACH_RAMEKIN -> Move above target ramekin
7. LOWER_RAMEKIN -> Lower to ramekin
8. DUMP -> Rotate wrist to dump food
9. RETURN -> Return to home position

Usage:
    python -m trossen_arm_mujoco.scripts.food_transfer_ik
    python -m trossen_arm_mujoco.scripts.food_transfer_ik --target ramekin_3
    python -m trossen_arm_mujoco.scripts.food_transfer_ik --loop 3
"""

import argparse
import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import mujoco
from mujoco import viewer as mj_viewer

from trossen_arm_mujoco.mink_ik import TrossenMinkIK


class TaskPhase(Enum):
    """Task phases for food transfer."""
    HOME = auto()
    APPROACH_CONTAINER = auto()
    REACH_CONTAINER = auto()
    SCOOP = auto()
    LIFT = auto()
    APPROACH_RAMEKIN = auto()
    LOWER_RAMEKIN = auto()
    DUMP = auto()
    RETURN = auto()
    DONE = auto()


@dataclass
class TaskConfig:
    """Configuration for food transfer task."""
    # Heights above objects (meters)
    approach_height: float = 0.12      # Height for approach moves
    scoop_height: float = 0.06         # Height when scooping in container
    ramekin_height: float = 0.08       # Height above ramekin for dumping

    # Timing (seconds)
    move_duration: float = 2.0         # Time for each move
    scoop_duration: float = 1.5        # Time for scoop motion
    dump_duration: float = 1.0         # Time for dump motion
    hold_duration: float = 0.5         # Pause between phases
    debug_hold: float = 3.0            # Debug hold time after each phase (seconds)

    # Wrist rotation for dump (radians)
    dump_rotation: float = -0.8        # Rotation for dumping food


class FoodTransferIK:
    """
    Dynamic food transfer task using mink IK.

    Reads object positions from simulation and generates IK solutions
    to reach them. Adapts to object position changes.

    NOTE: Uses separate model/data for visualization vs IK solving.
    The IK solver computes solutions internally without affecting the
    visualization model until we explicitly apply the solution.
    """

    def __init__(self, target_ramekin: str = "ramekin_2"):
        """
        Initialize food transfer task.

        Args:
            target_ramekin: Which ramekin to target ("ramekin_1" to "ramekin_4")
        """
        self.target_ramekin = target_ramekin
        self.config = TaskConfig()

        # Create IK solver (has its own internal model for solving)
        self.ik = TrossenMinkIK(ee_name="spoon")

        # Create SEPARATE model/data for visualization
        # This prevents IK iterations from being visible during solving
        from pathlib import Path
        assets_dir = Path(__file__).parent.parent / "assets"
        model_path = assets_dir / "wxai" / "teleop_scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Object body IDs (will be looked up)
        self._container_id = None
        self._ramekin_ids = {}

        # Current phase
        self.phase = TaskPhase.HOME
        self.phase_start_time = 0.0

        # Home joint position
        self.home_joints = np.array([0.085, 0.004, 0.006, 0.034, -0.029, -0.068])

        self._lookup_object_ids()

    def _lookup_object_ids(self):
        """Look up body IDs for objects."""
        self._container_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "container"
        )

        for i in range(1, 5):
            name = f"ramekin_{i}"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                self._ramekin_ids[name] = body_id

        print(f"Container ID: {self._container_id}")
        print(f"Ramekin IDs: {self._ramekin_ids}")

    def get_container_position(self) -> np.ndarray:
        """Get current container position from simulation."""
        if self._container_id is None or self._container_id < 0:
            # Fallback to default position
            return np.array([-0.63, -0.15, 0.04])
        return self.data.xpos[self._container_id].copy()

    def get_ramekin_position(self, name: str) -> np.ndarray:
        """Get current ramekin position from simulation."""
        if name not in self._ramekin_ids:
            # Fallback positions
            defaults = {
                "ramekin_1": np.array([-0.22, -0.26, 0.04]),
                "ramekin_2": np.array([-0.36, -0.26, 0.04]),
                "ramekin_3": np.array([-0.36, -0.12, 0.04]),
                "ramekin_4": np.array([-0.22, -0.12, 0.04]),
            }
            return defaults.get(name, np.array([-0.3, -0.2, 0.04]))

        body_id = self._ramekin_ids[name]
        return self.data.xpos[body_id].copy()

    def get_all_object_positions(self) -> Dict[str, np.ndarray]:
        """Get all object positions for debugging."""
        positions = {"container": self.get_container_position()}
        for name in self._ramekin_ids:
            positions[name] = self.get_ramekin_position(name)
        return positions

    def compute_target_for_phase(self, phase: TaskPhase) -> Optional[np.ndarray]:
        """
        Compute target XYZ position for given phase.

        Returns None if phase doesn't have a position target.
        """
        container_pos = self.get_container_position()
        ramekin_pos = self.get_ramekin_position(self.target_ramekin)

        if phase == TaskPhase.HOME:
            # Use FK to get home position (or just return None to use joint target)
            return None

        elif phase == TaskPhase.APPROACH_CONTAINER:
            # Above container center
            target = container_pos.copy()
            target[2] += self.config.approach_height
            return target

        elif phase == TaskPhase.REACH_CONTAINER:
            # Inside container (lower)
            target = container_pos.copy()
            target[2] += self.config.scoop_height
            return target

        elif phase == TaskPhase.SCOOP:
            # Scoop motion - slight movement within container
            target = container_pos.copy()
            target[2] += self.config.scoop_height
            target[0] += 0.05  # Move forward slightly while scooping
            return target

        elif phase == TaskPhase.LIFT:
            # Lift above container
            target = container_pos.copy()
            target[2] += self.config.approach_height + 0.05
            return target

        elif phase == TaskPhase.APPROACH_RAMEKIN:
            # Above target ramekin
            target = ramekin_pos.copy()
            target[2] += self.config.approach_height
            return target

        elif phase == TaskPhase.LOWER_RAMEKIN:
            # Lower to ramekin
            target = ramekin_pos.copy()
            target[2] += self.config.ramekin_height
            return target

        elif phase == TaskPhase.DUMP:
            # Same position, different orientation (handled separately)
            target = ramekin_pos.copy()
            target[2] += self.config.ramekin_height
            return target

        elif phase == TaskPhase.RETURN:
            return None  # Use joint target for home

        return None

    def solve_for_phase(self, phase: TaskPhase, current_joints: np.ndarray) -> np.ndarray:
        """
        Solve IK for target phase.

        Args:
            phase: Target phase
            current_joints: Current joint configuration (for seeding IK)

        Returns:
            Target joint configuration
        """
        target_pos = self.compute_target_for_phase(phase)

        if target_pos is None:
            # Return home joints
            return self.home_joints.copy()

        # For SCOOP phase: just rotate wrist roll to scoop (like inverse of dump)
        # Only rotate joint 5 (wrist roll) - same as dump but opposite direction
        if phase == TaskPhase.SCOOP:
            joints = current_joints.copy()
            joints[5] += 0.8  # Rotate wrist roll to scoop (positive = tilt spoon to scoop)
            return joints

        # For DUMP phase: rotate wrist roll to pour out contents
        elif phase == TaskPhase.DUMP:
            joints = current_joints.copy()
            joints[5] = self.config.dump_rotation  # Absolute roll for dumping (-1.5)
            return joints

        # For other phases, use position-only IK
        joints, success, error = self.ik.solve(target_pos, initial_q=current_joints)

        if not success:
            print(f"  Warning: IK for {phase.name} did not converge (error={error:.4f}m)")

        return joints

    def get_phase_duration(self, phase: TaskPhase) -> float:
        """Get duration for a phase."""
        if phase == TaskPhase.SCOOP:
            return self.config.scoop_duration
        elif phase == TaskPhase.DUMP:
            return self.config.dump_duration
        elif phase in [TaskPhase.HOME, TaskPhase.DONE]:
            return self.config.hold_duration
        else:
            return self.config.move_duration

    def next_phase(self) -> TaskPhase:
        """Get next phase in sequence."""
        phase_order = [
            TaskPhase.HOME,
            TaskPhase.APPROACH_CONTAINER,
            TaskPhase.REACH_CONTAINER,
            TaskPhase.SCOOP,
            TaskPhase.LIFT,
            TaskPhase.APPROACH_RAMEKIN,
            TaskPhase.LOWER_RAMEKIN,
            TaskPhase.DUMP,
            TaskPhase.RETURN,
            TaskPhase.DONE,
        ]

        try:
            idx = phase_order.index(self.phase)
            if idx < len(phase_order) - 1:
                return phase_order[idx + 1]
        except ValueError:
            pass

        return TaskPhase.DONE


def run_food_transfer(
    target_ramekin: str = "all",
    num_loops: int = -1,
    speed: float = 1.0,
    debug_hold: float = 3.0,
):
    """
    Run the food transfer task with dynamic IK.

    Args:
        target_ramekin: Target ramekin name, or "all" to cycle through all 4
        num_loops: Number of transfer cycles (-1 = infinite until viewer closed)
        speed: Speed multiplier (1.0 = normal)
        debug_hold: Hold time after each phase for debugging (seconds)
    """
    # List of all ramekins for cycling
    all_ramekins = ["ramekin_1", "ramekin_2", "ramekin_3", "ramekin_4"]
    cycle_all = (target_ramekin == "all")

    print("=" * 60)
    print("Dynamic Food Transfer Task with Mink IK")
    print("=" * 60)
    if cycle_all:
        print(f"Target: ALL bowls (cycling through 1-4)")
    else:
        print(f"Target: {target_ramekin}")
    print(f"Loops: {'infinite' if num_loops < 0 else num_loops}")
    print(f"Speed: {speed}x")
    print(f"Debug hold: {debug_hold}s after each phase")
    print()

    # Initialize task with first ramekin (will be updated in loop if cycling)
    initial_target = all_ramekins[0] if cycle_all else target_ramekin
    task = FoodTransferIK(target_ramekin=initial_target)
    task.config.debug_hold = debug_hold

    # Get spoon body ID for debug output
    spoon_id = mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_BODY, "spoon")

    # Print object positions
    print("Object positions (from simulation):")
    for name, pos in task.get_all_object_positions().items():
        print(f"  {name}: {np.round(pos, 4)}")
    print()

    # Simulation parameters
    dt = 0.02  # 50 Hz
    time_scale = 1.0 / speed

    print("Starting simulation...")
    print("Close viewer to exit (Ctrl+C or close window).")
    print("-" * 60)

    with mj_viewer.launch_passive(task.model, task.data) as viewer:
        loop = 0
        ramekin_idx = 0  # Index for cycling through ramekins

        while viewer.is_running():
            # Check if we've done enough loops (if not infinite)
            if num_loops >= 0 and loop >= num_loops:
                break

            # Determine current target ramekin
            if cycle_all:
                current_target = all_ramekins[ramekin_idx]
                task.target_ramekin = current_target
                print(f"\n=== Loop {loop + 1} | Bowl: {current_target} (bowl_{ramekin_idx + 1}) ===")
            else:
                print(f"\n=== Loop {loop + 1}{'/' + str(num_loops) if num_loops >= 0 else ''} ===")

            # Reset to home
            task.phase = TaskPhase.HOME
            current_joints = task.data.qpos[:6].copy()

            while task.phase != TaskPhase.DONE:
                if not viewer.is_running():
                    break

                # Get target for current phase
                target_joints = task.solve_for_phase(task.phase, current_joints)
                duration = task.get_phase_duration(task.phase) * time_scale

                # Print phase info with detailed joint tracking
                target_pos = task.compute_target_for_phase(task.phase)
                print(f"\n{'='*50}")
                print(f"[{task.phase.name}] STARTING MOVEMENT")
                print(f"{'='*50}")
                if target_pos is not None:
                    print(f"  Cartesian target: {np.round(target_pos, 4)}")
                else:
                    print(f"  Using joint target (no Cartesian target)")
                print(f"  START joints: {np.round(current_joints, 4)}")
                print(f"  END joints:   {np.round(target_joints, 4)}")
                print(f"  Joint DELTA:  {np.round(target_joints - current_joints, 4)}")
                print(f"  Duration: {duration:.2f}s, Steps: {int(duration / dt)}")

                # Animate movement
                start_joints = current_joints.copy()
                steps = int(duration / dt)

                # Track actual qpos at start
                print(f"  Actual qpos before move: {np.round(task.data.qpos[:6], 4)}")

                for step in range(max(steps, 1)):
                    if not viewer.is_running():
                        break

                    # Smooth interpolation
                    t = step / max(steps, 1)
                    t_smooth = 3 * t**2 - 2 * t**3

                    # Interpolate joints
                    q = start_joints + t_smooth * (target_joints - start_joints)

                    # Apply to simulation
                    task.data.qpos[:6] = q
                    task.data.ctrl[:6] = q

                    mujoco.mj_step(task.model, task.data)
                    viewer.sync()
                    time.sleep(dt)

                # Update current joints
                current_joints = target_joints.copy()

                # Get actual EE position after movement
                actual_pos = task.data.xpos[spoon_id].copy()
                target_pos = task.compute_target_for_phase(task.phase)

                # Print debug info
                print(f"  Actual qpos after move:  {np.round(task.data.qpos[:6], 4)}")
                print(f"  Actual ctrl after move:  {np.round(task.data.ctrl[:6], 4)}")
                print(f"\n    -> Phase [{task.phase.name}] COMPLETE")
                print(f"       Actual EE pos: {np.round(actual_pos, 4)}")
                if target_pos is not None:
                    pos_error = np.linalg.norm(actual_pos - target_pos)
                    print(f"       Target pos:    {np.round(target_pos, 4)}")
                    print(f"       Position error: {pos_error*1000:.2f} mm")
                print(f"       Joint angles: {np.round(current_joints, 4)}")
                print(f"       Holding for {debug_hold}s...")

                # Debug hold with countdown - ACTIVELY HOLD POSITION
                debug_steps = int(debug_hold / dt)
                hold_start_qpos = task.data.qpos[:6].copy()
                for step in range(debug_steps):
                    if not viewer.is_running():
                        break

                    # Actively hold position by setting qpos and ctrl each step
                    task.data.qpos[:6] = target_joints
                    task.data.ctrl[:6] = target_joints

                    mujoco.mj_step(task.model, task.data)
                    viewer.sync()
                    time.sleep(dt)

                    # Print countdown every second
                    elapsed = step * dt
                    if step > 0 and step % 50 == 0:
                        remaining = debug_hold - elapsed
                        # Check if joints drifted during hold
                        current_qpos = task.data.qpos[:6]
                        drift = np.linalg.norm(current_qpos - hold_start_qpos)
                        print(f"       {remaining:.0f}s remaining... (joint drift: {drift:.6f})")

                # Check final position after hold
                hold_end_qpos = task.data.qpos[:6].copy()
                hold_drift = np.linalg.norm(hold_end_qpos - hold_start_qpos)
                print(f"       Hold complete. Total joint drift: {hold_drift:.6f}")
                print(f"       qpos at hold end: {np.round(hold_end_qpos, 4)}")
                print(f"    -> Proceeding to next phase\n")

                # Next phase
                task.phase = task.next_phase()

            if task.phase == TaskPhase.DONE:
                print("Transfer complete!")

                # Increment loop counter
                loop += 1

                # Cycle to next ramekin if in "all" mode
                if cycle_all:
                    ramekin_idx = (ramekin_idx + 1) % len(all_ramekins)

        # Hold at end (only if finite loops completed)
        if viewer.is_running() and num_loops >= 0:
            print("\nAll loops complete. Press close to exit.")
            while viewer.is_running():
                mujoco.mj_step(task.model, task.data)
                viewer.sync()
                time.sleep(dt)

    print("\n" + "=" * 60)
    print("Task finished!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic food transfer with mink IK"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all", "ramekin_1", "ramekin_2", "ramekin_3", "ramekin_4",
                 "bowl_1", "bowl_2", "bowl_3", "bowl_4"],
        help="Target bowl/ramekin. 'all' cycles through all 4 bowls (default: all).",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=-1,
        help="Number of transfer loops (-1 = infinite until closed, default: -1)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--debug_hold",
        type=float,
        default=0.0,
        help="Hold time after each phase for debugging (seconds, default: 0.0)",
    )

    args = parser.parse_args()

    # Convert bowl_N to ramekin_N if needed (keep "all" as-is)
    target = args.target
    if target.startswith("bowl_"):
        target = target.replace("bowl_", "ramekin_")

    run_food_transfer(
        target_ramekin=target,
        num_loops=args.loop,
        speed=args.speed,
        debug_hold=args.debug_hold,
    )


if __name__ == "__main__":
    main()
