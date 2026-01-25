#!/usr/bin/env python3
"""
Dynamic Food Transfer Task using Mink IK with interactive visualization.

This script generates trajectories dynamically based on actual object positions
in the simulation. If you move the container or bowls, the IK planner will
find new joint configurations to reach them.

Task phases:
1. HOME -> Move to home/ready position
2. APPROACH_CONTAINER -> Move above container
3. REACH_CONTAINER -> Lower into container (scoop position)
4. SCOOP -> Perform scooping motion
5. LIFT -> Lift spoon from container
6. APPROACH_BOWL -> Move above target bowl
7. LOWER_BOWL -> Lower to bowl
8. DUMP -> Rotate wrist to dump food
9. RETURN -> Return to home position

Usage:
    python -m trossen_arm_mujoco.scripts.food_transfer_ik
    python -m trossen_arm_mujoco.scripts.food_transfer_ik --target bowl_3
    python -m trossen_arm_mujoco.scripts.food_transfer_ik --loop 3
"""

import argparse
import time

import mujoco
import numpy as np
from mujoco import viewer as mj_viewer

from trossen_arm_mujoco.food_transfer_base import (
    ALL_BOWLS,
    DT,
    FoodTransferBase,
    TaskPhase,
)


class FoodTransferIK(FoodTransferBase):
    """
    Interactive food transfer task with visualization.

    Extends FoodTransferBase with debug output and viewer support.
    """

    def __init__(
        self,
        target: str = "bowl_2",
        scene_xml: str = "wxai/teleop_scene.xml",
        debug_hold: float = 0.0,
    ):
        """
        Initialize food transfer task.

        Args:
            target: Which bowl to target ("bowl_1" to "bowl_4")
            scene_xml: Path to scene XML file, relative to assets/ directory
            debug_hold: Hold time after each phase for debugging (seconds)
        """
        super().__init__(target, scene_xml)
        self.debug_hold = debug_hold

    def print_object_positions(self):
        """Print all object positions for debugging."""
        print("Object positions (from simulation):")
        for name, pos in self.get_all_object_positions().items():
            print(f"  {name}: {np.round(pos, 4)}")
        print()


def run_food_transfer(
    target: str = "all",
    num_loops: int = -1,
    speed: float = 1.0,
    debug_hold: float = 0.0,
    scene_xml: str = "wxai/teleop_scene.xml",
):
    """
    Run the food transfer task with dynamic IK.

    Args:
        target: Target bowl name, or "all" to cycle through all 4
        num_loops: Number of transfer cycles (-1 = infinite until viewer closed)
        speed: Speed multiplier (1.0 = normal)
        debug_hold: Hold time after each phase for debugging (seconds)
        scene_xml: Path to scene XML file, relative to assets/ directory
    """
    cycle_all = (target == "all")

    print("=" * 60)
    print("Dynamic Food Transfer Task with Mink IK")
    print("=" * 60)
    print(f"Scene: {scene_xml}")
    if cycle_all:
        print("Target: ALL bowls (cycling through 1-4)")
    else:
        print(f"Target: {target}")
    print(f"Loops: {'infinite' if num_loops < 0 else num_loops}")
    print(f"Speed: {speed}x")
    print(f"Debug hold: {debug_hold}s after each phase")
    print()

    # Initialize task with first bowl (will be updated in loop if cycling)
    initial_target = ALL_BOWLS[0] if cycle_all else target
    task = FoodTransferIK(
        target=initial_target,
        scene_xml=scene_xml,
        debug_hold=debug_hold,
    )

    # Get spoon body ID for debug output
    spoon_id = mujoco.mj_name2id(task.model, mujoco.mjtObj.mjOBJ_BODY, "spoon")

    # Print object positions
    task.print_object_positions()

    # Time scale
    time_scale = 1.0 / speed

    print("Starting simulation...")
    print("Close viewer to exit (Ctrl+C or close window).")
    print("-" * 60)

    with mj_viewer.launch_passive(task.model, task.data) as viewer:
        loop = 0
        bowl_idx = 0

        while viewer.is_running():
            # Check if we've done enough loops (if not infinite)
            if num_loops >= 0 and loop >= num_loops:
                break

            # Determine current target bowl
            if cycle_all:
                current_target = ALL_BOWLS[bowl_idx]
                task.target = current_target
                print(f"\n=== Loop {loop + 1} | Bowl: {current_target} (bowl_{bowl_idx + 1}) ===")
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

                # Print phase info
                target_pos = task.compute_target_for_phase(task.phase)
                print(f"\n{'='*50}")
                print(f"[{task.phase.name}] STARTING MOVEMENT")
                print(f"{'='*50}")
                if target_pos is not None:
                    print(f"  Cartesian target: {np.round(target_pos, 4)}")
                else:
                    print("  Using joint target (no Cartesian target)")
                print(f"  START joints: {np.round(current_joints, 4)}")
                print(f"  END joints:   {np.round(target_joints, 4)}")
                print(f"  Joint DELTA:  {np.round(target_joints - current_joints, 4)}")
                print(f"  Duration: {duration:.2f}s, Steps: {int(duration / DT)}")

                # Animate movement
                start_joints = current_joints.copy()
                steps = int(duration / DT)

                print(f"  Actual qpos before move: {np.round(task.data.qpos[:6], 4)}")

                prev_reward = task.get_reward()
                collision_count = 0
                collision_reported = False

                for step in range(max(steps, 1)):
                    if not viewer.is_running():
                        break

                    t = step / max(steps, 1)
                    q = task.interpolate_joints(start_joints, target_joints, t)

                    # Apply to simulation
                    task.data.qpos[:6] = q
                    task.data.ctrl[:6] = q

                    mujoco.mj_step(task.model, task.data)
                    viewer.sync()
                    time.sleep(DT)

                    # Check for collisions
                    collisions = task.check_collisions()
                    if collisions:
                        collision_count += 1
                        if not collision_reported:
                            # Print first collision details
                            print(f"  [COLLISION] Robot-obstacle collision detected!")
                            for robot_geom, obstacle_geom in collisions[:3]:  # Show max 3
                                print(f"    - {robot_geom} <-> {obstacle_geom}")
                            if len(collisions) > 3:
                                print(f"    ... and {len(collisions) - 3} more")
                            collision_reported = True

                    # Check reward and print if changed
                    current_reward = task.get_reward()
                    if current_reward != prev_reward:
                        reward_labels = {0: "No reach", 1: "Container reached", 2: "Bowl reached"}
                        print(f"  [REWARD] {prev_reward} -> {current_reward} ({reward_labels.get(current_reward, '?')})")
                        prev_reward = current_reward

                # Summary of collisions for this phase
                if collision_count > 0:
                    print(f"  [COLLISION SUMMARY] {collision_count} steps with collisions in this phase")

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

                # Debug hold if enabled
                if debug_hold > 0:
                    print(f"       Holding for {debug_hold}s...")
                    debug_steps = int(debug_hold / DT)
                    hold_start_qpos = task.data.qpos[:6].copy()

                    for step in range(debug_steps):
                        if not viewer.is_running():
                            break

                        # Actively hold position
                        task.data.qpos[:6] = target_joints
                        task.data.ctrl[:6] = target_joints

                        mujoco.mj_step(task.model, task.data)
                        viewer.sync()
                        time.sleep(DT)

                        # Print countdown every second
                        if step > 0 and step % 50 == 0:
                            elapsed = step * DT
                            remaining = debug_hold - elapsed
                            current_qpos = task.data.qpos[:6]
                            drift = np.linalg.norm(current_qpos - hold_start_qpos)
                            print(f"       {remaining:.0f}s remaining... (joint drift: {drift:.6f})")

                    # Check final position after hold
                    hold_end_qpos = task.data.qpos[:6].copy()
                    hold_drift = np.linalg.norm(hold_end_qpos - hold_start_qpos)
                    print(f"       Hold complete. Total joint drift: {hold_drift:.6f}")
                    print(f"       qpos at hold end: {np.round(hold_end_qpos, 4)}")

                print("    -> Proceeding to next phase\n")

                # Next phase
                task.phase = task.next_phase()

            if task.phase == TaskPhase.DONE:
                final_reward = task.get_reward()
                reward_labels = {0: "No reach", 1: "Container reached", 2: "Bowl reached"}
                success = "SUCCESS" if final_reward == task.max_reward else "INCOMPLETE"
                print(f"\nTransfer complete! Final reward: {final_reward}/{task.max_reward} ({reward_labels.get(final_reward, '?')}) [{success}]")
                loop += 1

                # Reset reward tracking for next loop
                task._step_count = 0
                task._container_enter_step = None
                task._reached_container = False
                task._bowl_enter_step = {name: None for name in task._bowl_ids.keys()}
                task._reached_bowl = False

                # Cycle to next bowl if in "all" mode
                if cycle_all:
                    bowl_idx = (bowl_idx + 1) % len(ALL_BOWLS)

        # Hold at end (only if finite loops completed)
        if viewer.is_running() and num_loops >= 0:
            print("\nAll loops complete. Press close to exit.")
            while viewer.is_running():
                mujoco.mj_step(task.model, task.data)
                viewer.sync()
                time.sleep(DT)

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
        choices=["all", "bowl_1", "bowl_2", "bowl_3", "bowl_4"],
        help="Target bowl. 'all' cycles through all 4 bowls (default: all).",
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
    parser.add_argument(
        "--scene",
        type=str,
        default="wxai/teleop_scene.xml",
        help="Scene XML file path relative to assets/ (default: wxai/teleop_scene.xml)",
    )

    args = parser.parse_args()

    run_food_transfer(
        target=args.target,
        num_loops=args.loop,
        speed=args.speed,
        debug_hold=args.debug_hold,
        scene_xml=args.scene,
    )


if __name__ == "__main__":
    main()
