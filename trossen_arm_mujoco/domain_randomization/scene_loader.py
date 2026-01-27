"""
Scene loader for domain randomization.

This module applies scene configurations to MuJoCo simulations,
modifying object positions, orientations, textures, colors, and lighting at runtime.
"""

from typing import Dict, Optional, Tuple
import mujoco
import numpy as np

from .config import ObjectPose, SceneConfiguration, VisualConfiguration


class SceneLoader:
    """
    Loads scene configurations into MuJoCo simulation.

    This class modifies object positions in the MuJoCo model and
    resets the simulation to apply the changes.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Initialize the scene loader.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        self._body_ids: Dict[str, int] = {}
        self._material_ids: Dict[str, int] = {}
        self._texture_ids: Dict[str, int] = {}
        self._light_ids: Dict[str, int] = {}

        # Store default values for reset
        self._default_mat_rgba: Dict[int, np.ndarray] = {}
        self._default_mat_texid: Dict[int, int] = {}
        self._default_light_pos: Dict[int, np.ndarray] = {}
        self._default_light_diffuse: Dict[int, np.ndarray] = {}
        self._default_headlight_diffuse: Optional[np.ndarray] = None
        self._default_headlight_ambient: Optional[np.ndarray] = None

        self._cache_body_ids()
        self._cache_material_ids()
        self._cache_texture_ids()
        self._cache_light_ids()
        self._store_defaults()

    def _cache_body_ids(self) -> None:
        """Cache body IDs for fast lookup."""
        # Container
        container_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "container"
        )
        if container_id >= 0:
            self._body_ids["container"] = container_id

        # Bowls (1-8)
        for i in range(1, 9):
            bowl_name = f"bowl_{i}"
            bowl_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, bowl_name
            )
            if bowl_id >= 0:
                self._body_ids[bowl_name] = bowl_id

    def _cache_material_ids(self) -> None:
        """Cache material IDs for visual randomization."""
        material_names = [
            "table_mat",
            "groundplane",
            "container_mat",
            "threshold_ramekin_mat",
        ]
        for name in material_names:
            mat_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_MATERIAL, name
            )
            if mat_id >= 0:
                self._material_ids[name] = mat_id

    def _cache_texture_ids(self) -> None:
        """Cache texture IDs for table and floor texture variants."""
        # Original textures
        for tex_name in ["table_tex", "groundplane"]:
            tex_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_TEXTURE, tex_name
            )
            if tex_id >= 0:
                self._texture_ids[tex_name] = tex_id

        # Counter/table textures (counter_tex_0 through counter_tex_99)
        for i in range(100):
            tex_name = f"counter_tex_{i}"
            tex_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_TEXTURE, tex_name
            )
            if tex_id >= 0:
                self._texture_ids[tex_name] = tex_id

        # Floor textures (floor_tex_0 through floor_tex_99)
        for i in range(100):
            tex_name = f"floor_tex_{i}"
            tex_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_TEXTURE, tex_name
            )
            if tex_id >= 0:
                self._texture_ids[tex_name] = tex_id

    def _cache_light_ids(self) -> None:
        """Cache light IDs for lighting randomization."""
        light_names = ["top_light", "side_light"]
        for name in light_names:
            light_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_LIGHT, name
            )
            if light_id >= 0:
                self._light_ids[name] = light_id

    def _store_defaults(self) -> None:
        """Store default values for materials and lights for reset."""
        # Material defaults
        # mat_texid has shape (nmat, 10) - we store the whole row
        for name, mat_id in self._material_ids.items():
            self._default_mat_rgba[mat_id] = self.model.mat_rgba[mat_id].copy()
            self._default_mat_texid[mat_id] = self.model.mat_texid[mat_id].copy()

        # Light defaults
        for name, light_id in self._light_ids.items():
            self._default_light_pos[light_id] = self.model.light_pos[light_id].copy()
            self._default_light_diffuse[light_id] = self.model.light_diffuse[light_id].copy()

        # Headlight defaults (in visual section)
        self._default_headlight_diffuse = self.model.vis.headlight.diffuse.copy()
        self._default_headlight_ambient = self.model.vis.headlight.ambient.copy()

    def apply(self, config: SceneConfiguration) -> None:
        """
        Apply scene configuration to simulation.

        For static bodies (attached to world), we modify model.body_pos
        and model.body_quat directly, then reset the simulation.
        Also applies visual configuration if present.

        Args:
            config: Scene configuration to apply
        """
        # Apply container pose
        if "container" in self._body_ids:
            self._set_body_pose("container", config.container_pose)

        # Apply bowl poses
        for bowl_name, pose in config.bowl_poses.items():
            if bowl_name in self._body_ids:
                self._set_body_pose(bowl_name, pose)

        # Hide inactive bowls (move far below table)
        self._hide_inactive_bowls(config.active_bowls)

        # Apply visual configuration if present
        if config.visual_config is not None:
            self._apply_visual_config(config.visual_config)

        # Reset simulation to apply changes
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _apply_visual_config(self, visual: VisualConfiguration) -> None:
        """
        Apply visual configuration to model.

        Modifies textures, colors, and lighting parameters.

        Args:
            visual: Visual configuration to apply
        """
        self._apply_table_texture(visual.table_texture_index)
        self._apply_floor_texture(visual.floor_texture_index)
        self._apply_container_color(visual.container_color)
        self._apply_bowl_tints(visual.bowl_tints)
        self._apply_lighting(visual)

    def _apply_table_texture(self, texture_index: int) -> None:
        """
        Swap table material's texture to the specified counter texture.

        Args:
            texture_index: Index of counter texture (0-99)
        """
        mat_id = self._material_ids.get("table_mat")
        if mat_id is None:
            return

        tex_name = f"counter_tex_{texture_index}"
        tex_id = self._texture_ids.get(tex_name)
        if tex_id is not None:
            # mat_texid[mat_id] is an array of 10 texture slots
            # Index 1 is typically used for cube/2d textures on materials
            # Find the non-negative slot in the default and update that one
            default_texid = self._default_mat_texid.get(mat_id)
            if default_texid is not None:
                for i in range(len(default_texid)):
                    if default_texid[i] >= 0:
                        self.model.mat_texid[mat_id, i] = tex_id
                        break

    def _apply_floor_texture(self, texture_index: int) -> None:
        """
        Swap floor material's texture to the specified floor texture.

        Args:
            texture_index: Index of floor texture (0-99)
        """
        mat_id = self._material_ids.get("groundplane")
        if mat_id is None:
            return

        tex_name = f"floor_tex_{texture_index}"
        tex_id = self._texture_ids.get(tex_name)
        if tex_id is not None:
            # mat_texid[mat_id] is an array of 10 texture slots
            # Find the non-negative slot in the default and update that one
            default_texid = self._default_mat_texid.get(mat_id)
            if default_texid is not None:
                for i in range(len(default_texid)):
                    if default_texid[i] >= 0:
                        self.model.mat_texid[mat_id, i] = tex_id
                        break

    def _apply_container_color(self, rgba: Tuple[float, float, float, float]) -> None:
        """
        Modify container material color.

        Args:
            rgba: RGBA color tuple
        """
        mat_id = self._material_ids.get("container_mat")
        if mat_id is None:
            return
        self.model.mat_rgba[mat_id] = np.array(rgba)

    def _apply_bowl_tints(
        self, tints: Dict[str, Tuple[float, float, float, float]]
    ) -> None:
        """
        Apply tint to bowl materials.

        Note: All bowls share the same material (threshold_ramekin_mat).
        To support per-bowl tints, we average the tints for now.
        For true per-bowl colors, per-bowl materials would be needed.

        Args:
            tints: Dictionary mapping bowl names to RGBA tint values
        """
        mat_id = self._material_ids.get("threshold_ramekin_mat")
        if mat_id is None or not tints:
            return

        # Use average of all bowl tints
        tint_values = list(tints.values())
        avg_tint = np.mean([np.array(t) for t in tint_values], axis=0)
        self.model.mat_rgba[mat_id] = avg_tint

    def _apply_lighting(self, visual: VisualConfiguration) -> None:
        """
        Apply lighting configuration to model.

        Args:
            visual: Visual configuration containing light parameters
        """
        # Top light
        top_light_id = self._light_ids.get("top_light")
        if top_light_id is not None:
            self.model.light_pos[top_light_id] = np.array(visual.top_light_pos)
            self.model.light_diffuse[top_light_id] = np.array(visual.top_light_diffuse)

        # Side light
        side_light_id = self._light_ids.get("side_light")
        if side_light_id is not None:
            self.model.light_pos[side_light_id] = np.array(visual.side_light_pos)
            self.model.light_diffuse[side_light_id] = np.array(visual.side_light_diffuse)

        # Headlight (in visual section)
        self.model.vis.headlight.diffuse[:] = np.array(visual.headlight_diffuse)
        self.model.vis.headlight.ambient[:] = np.array(visual.headlight_ambient)

    def reset_visual_to_defaults(self) -> None:
        """Reset all visual parameters to default values."""
        # Reset materials
        for mat_id, rgba in self._default_mat_rgba.items():
            self.model.mat_rgba[mat_id] = rgba
        for mat_id, texid in self._default_mat_texid.items():
            self.model.mat_texid[mat_id] = texid

        # Reset lights
        for light_id, pos in self._default_light_pos.items():
            self.model.light_pos[light_id] = pos
        for light_id, diffuse in self._default_light_diffuse.items():
            self.model.light_diffuse[light_id] = diffuse

        # Reset headlight
        if self._default_headlight_diffuse is not None:
            self.model.vis.headlight.diffuse[:] = self._default_headlight_diffuse
        if self._default_headlight_ambient is not None:
            self.model.vis.headlight.ambient[:] = self._default_headlight_ambient

    @property
    def available_textures(self) -> Dict[str, int]:
        """Return dictionary of texture names to IDs that were found."""
        return dict(self._texture_ids)

    @property
    def available_materials(self) -> Dict[str, int]:
        """Return dictionary of material names to IDs that were found."""
        return dict(self._material_ids)

    @property
    def available_lights(self) -> Dict[str, int]:
        """Return dictionary of light names to IDs that were found."""
        return dict(self._light_ids)

    def _set_body_pose(self, body_name: str, pose: ObjectPose) -> None:
        """
        Set pose for a static body by modifying the model.

        Args:
            body_name: Name of the body
            pose: Target pose
        """
        body_id = self._body_ids.get(body_name)
        if body_id is None:
            return

        # Modify model body position and quaternion
        self.model.body_pos[body_id] = pose.position
        self.model.body_quat[body_id] = pose.quaternion

    def _hide_inactive_bowls(self, active_bowls: list) -> None:
        """
        Hide bowls that are not active by moving them far below table.

        Args:
            active_bowls: List of bowl names that should remain visible
        """
        hidden_z = -10.0  # Far below table

        for bowl_name, body_id in self._body_ids.items():
            if bowl_name.startswith("bowl_") and bowl_name not in active_bowls:
                # Move far below table to hide
                current_pos = self.model.body_pos[body_id].copy()
                current_pos[2] = hidden_z
                self.model.body_pos[body_id] = current_pos

    def reset_to_nominal(self, nominal_positions: Dict[str, np.ndarray]) -> None:
        """
        Reset all objects to their nominal positions.

        Args:
            nominal_positions: Dictionary mapping object names to positions
        """
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

        for name, position in nominal_positions.items():
            if name in self._body_ids:
                body_id = self._body_ids[name]
                self.model.body_pos[body_id] = position
                self.model.body_quat[body_id] = identity_quat

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def get_body_pose(self, body_name: str) -> Optional[ObjectPose]:
        """
        Get current pose of a body from simulation.

        Args:
            body_name: Name of the body

        Returns:
            ObjectPose or None if body not found
        """
        body_id = self._body_ids.get(body_name)
        if body_id is None:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id < 0:
                return None

        # Get position from data.xpos (world position after forward)
        position = self.data.xpos[body_id].copy()

        # Get quaternion from data.xquat
        quaternion = self.data.xquat[body_id].copy()

        return ObjectPose(position=position, quaternion=quaternion)

    def get_all_object_poses(self) -> Dict[str, ObjectPose]:
        """
        Get poses of all tracked objects.

        Returns:
            Dictionary mapping object names to ObjectPose
        """
        poses = {}
        for name in self._body_ids.keys():
            pose = self.get_body_pose(name)
            if pose is not None:
                poses[name] = pose
        return poses

    @property
    def available_bodies(self) -> list:
        """Return list of body names that were found in the model."""
        return list(self._body_ids.keys())

    def validate_visual_dr_assets(
        self,
        num_table_textures: int = 100,
        num_floor_textures: int = 100,
    ) -> None:
        """
        Validate that all required visual DR assets are loaded.

        Raises an error if textures are missing, preventing silent failures
        where visual DR appears to work but actually does nothing.

        Args:
            num_table_textures: Expected number of table/counter textures
            num_floor_textures: Expected number of floor textures

        Raises:
            RuntimeError: If required textures are missing
        """
        errors = []

        # Check counter/table textures
        counter_textures_found = sum(
            1 for i in range(num_table_textures)
            if f"counter_tex_{i}" in self._texture_ids
        )
        if counter_textures_found == 0:
            errors.append(
                f"No counter/table textures found (expected {num_table_textures}). "
                f"Check that texture files exist at the path specified in the scene XML. "
                f"The XML references '../domain_randomization/assets/counter/' - "
                f"ensure this path resolves correctly from the scene file location."
            )
        elif counter_textures_found < num_table_textures:
            errors.append(
                f"Only {counter_textures_found}/{num_table_textures} counter textures found. "
                f"Some texture files may be missing."
            )

        # Check floor textures
        floor_textures_found = sum(
            1 for i in range(num_floor_textures)
            if f"floor_tex_{i}" in self._texture_ids
        )
        if floor_textures_found == 0:
            errors.append(
                f"No floor textures found (expected {num_floor_textures}). "
                f"Check that texture files exist at the path specified in the scene XML. "
                f"The XML references '../domain_randomization/assets/floor/' - "
                f"ensure this path resolves correctly from the scene file location."
            )
        elif floor_textures_found < num_floor_textures:
            errors.append(
                f"Only {floor_textures_found}/{num_floor_textures} floor textures found. "
                f"Some texture files may be missing."
            )

        # Additional check: verify texture data is non-trivial
        # MuJoCo may create texture IDs even if files don't exist, but the
        # textures will have minimal size. Check a sample texture's dimensions.
        if counter_textures_found > 0:
            sample_tex_name = "counter_tex_0"
            tex_id = self._texture_ids.get(sample_tex_name)
            if tex_id is not None:
                # Check texture dimensions - empty/failed textures have tiny sizes
                tex_width = self.model.tex_width[tex_id]
                tex_height = self.model.tex_height[tex_id]
                if tex_width < 64 or tex_height < 64:
                    errors.append(
                        f"Counter texture '{sample_tex_name}' has invalid dimensions "
                        f"({tex_width}x{tex_height}). The texture file may not have "
                        f"loaded correctly. Check that the texture files exist at the "
                        f"correct path."
                    )

        if floor_textures_found > 0:
            sample_tex_name = "floor_tex_0"
            tex_id = self._texture_ids.get(sample_tex_name)
            if tex_id is not None:
                tex_width = self.model.tex_width[tex_id]
                tex_height = self.model.tex_height[tex_id]
                if tex_width < 64 or tex_height < 64:
                    errors.append(
                        f"Floor texture '{sample_tex_name}' has invalid dimensions "
                        f"({tex_width}x{tex_height}). The texture file may not have "
                        f"loaded correctly. Check that the texture files exist at the "
                        f"correct path."
                    )

        # Check required materials
        required_materials = ["table_mat", "groundplane", "container_mat"]
        missing_materials = [
            m for m in required_materials if m not in self._material_ids
        ]
        if missing_materials:
            errors.append(
                f"Missing required materials: {missing_materials}. "
                f"Check that the scene XML defines these materials."
            )

        # Check lights
        required_lights = ["top_light", "side_light"]
        missing_lights = [l for l in required_lights if l not in self._light_ids]
        if missing_lights:
            errors.append(
                f"Missing required lights: {missing_lights}. "
                f"Lighting randomization will not work correctly."
            )

        if errors:
            error_msg = (
                "\n" + "=" * 70 + "\n"
                "VISUAL DOMAIN RANDOMIZATION VALIDATION FAILED\n"
                "=" * 70 + "\n\n"
                "The following issues were detected:\n\n"
            )
            for i, err in enumerate(errors, 1):
                error_msg += f"{i}. {err}\n\n"
            error_msg += (
                "This validation prevents generating datasets with missing visual DR.\n"
                "To fix texture path issues, you may need to create a symlink:\n\n"
                "  mkdir -p <project>/trossen_arm_mujoco/assets/domain_randomization\n"
                "  ln -s ../../domain_randomization/assets \\\n"
                "        <project>/trossen_arm_mujoco/assets/domain_randomization/assets\n\n"
                "Or update the texture paths in the scene XML files.\n"
                "=" * 70
            )
            raise RuntimeError(error_msg)

    def get_visual_dr_status(self) -> dict:
        """
        Get status of visual DR assets for diagnostics.

        Returns:
            Dictionary with counts and status of visual DR components
        """
        counter_count = sum(
            1 for k in self._texture_ids if k.startswith("counter_tex_")
        )
        floor_count = sum(
            1 for k in self._texture_ids if k.startswith("floor_tex_")
        )

        return {
            "counter_textures": counter_count,
            "floor_textures": floor_count,
            "materials": list(self._material_ids.keys()),
            "lights": list(self._light_ids.keys()),
            "table_mat_found": "table_mat" in self._material_ids,
            "groundplane_found": "groundplane" in self._material_ids,
            "container_mat_found": "container_mat" in self._material_ids,
        }
