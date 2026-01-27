"""
Visual parameter sampler for domain randomization.

This module provides sampling of texture, color, and lighting parameters
for visual domain randomization in the food transfer task.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import (
    VisualRandomizationConfig,
    VisualConfiguration,
)


# Default light positions (from teleop_scene.xml)
DEFAULT_TOP_LIGHT_POS = np.array([0.0, 0.0, 1.5])
DEFAULT_SIDE_LIGHT_POS = np.array([0.5, 0.5, 1.0])

# Default light colors (from teleop_scene.xml)
DEFAULT_TOP_LIGHT_DIFFUSE = np.array([0.8, 0.8, 0.8])
DEFAULT_SIDE_LIGHT_DIFFUSE = np.array([0.6, 0.6, 0.6])
DEFAULT_HEADLIGHT_DIFFUSE = np.array([0.6, 0.65, 0.75])
DEFAULT_HEADLIGHT_AMBIENT = np.array([0.5, 0.5, 0.6])

# Default container color (from teleop_scene.xml: rgba="0.3 0.3 0.35 1")
DEFAULT_CONTAINER_COLOR = np.array([0.3, 0.3, 0.35, 1.0])


class VisualSampler:
    """
    Samples visual parameters for domain randomization.

    This class generates randomized visual configurations including
    texture assignments, material colors, and lighting parameters.
    """

    def __init__(
        self,
        config: VisualRandomizationConfig,
        active_bowls: Optional[List[str]] = None,
    ):
        """
        Initialize visual sampler.

        Args:
            config: Visual randomization configuration
            active_bowls: List of bowl names for tint sampling
        """
        self.config = config
        self.active_bowls = active_bowls or []
        self.rng = np.random.default_rng(config.seed)

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def set_active_bowls(self, active_bowls: List[str]) -> None:
        """Update the list of active bowls for tint sampling."""
        self.active_bowls = active_bowls

    def sample(self) -> VisualConfiguration:
        """
        Sample a complete visual configuration.

        Returns:
            VisualConfiguration with randomized parameters
        """
        config_seed = int(self.rng.integers(0, 2**31))

        return VisualConfiguration(
            table_texture_index=self._sample_table_texture(),
            floor_texture_index=self._sample_floor_texture(),
            container_color=self._sample_container_color(),
            bowl_tints=self._sample_bowl_tints(),
            top_light_pos=self._sample_light_position(DEFAULT_TOP_LIGHT_POS),
            top_light_diffuse=self._sample_light_diffuse(DEFAULT_TOP_LIGHT_DIFFUSE),
            side_light_pos=self._sample_light_position(DEFAULT_SIDE_LIGHT_POS),
            side_light_diffuse=self._sample_light_diffuse(DEFAULT_SIDE_LIGHT_DIFFUSE),
            headlight_diffuse=self._sample_headlight_diffuse(),
            headlight_ambient=self._sample_headlight_ambient(),
            seed=config_seed,
        )

    def _sample_table_texture(self) -> int:
        """Sample table texture index (0-based)."""
        if not self.config.enabled or not self.config.texture.randomize_table:
            return 0  # Default to first texture

        num_textures = self.config.texture.num_table_textures
        return int(self.rng.integers(0, num_textures))

    def _sample_floor_texture(self) -> int:
        """Sample floor texture index (0-based)."""
        if not self.config.enabled or not self.config.texture.randomize_floor:
            return 0  # Default to first texture

        num_textures = self.config.texture.num_floor_textures
        return int(self.rng.integers(0, num_textures))

    def _sample_container_color(self) -> Tuple[float, float, float, float]:
        """Sample container color with RGB noise."""
        if not self.config.enabled or not self.config.color.randomize_container:
            return tuple(DEFAULT_CONTAINER_COLOR)

        noise = self.config.color.container_rgb_noise
        color = DEFAULT_CONTAINER_COLOR.copy()

        # Add noise to RGB channels
        for i in range(3):
            color[i] += self.rng.uniform(-noise, noise)

        # Clamp to valid range
        color = np.clip(color, 0.0, 1.0)
        color[3] = 1.0  # Keep alpha at 1.0

        return tuple(float(c) for c in color)

    def _sample_bowl_tints(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Sample bowl tint colors."""
        tints = {}

        if not self.config.enabled or not self.config.color.randomize_bowls:
            # Default white tint (no change to texture)
            for bowl in self.active_bowls:
                tints[bowl] = (1.0, 1.0, 1.0, 1.0)
            return tints

        min_tint, max_tint = self.config.color.bowl_tint_range
        for bowl in self.active_bowls:
            r = self.rng.uniform(min_tint, max_tint)
            g = self.rng.uniform(min_tint, max_tint)
            b = self.rng.uniform(min_tint, max_tint)
            tints[bowl] = (float(r), float(g), float(b), 1.0)

        return tints

    def _sample_light_position(
        self, default_pos: np.ndarray
    ) -> Tuple[float, float, float]:
        """Sample light position with noise."""
        if not self.config.enabled or not self.config.lighting.randomize_position:
            return tuple(float(x) for x in default_pos)

        noise = self.config.lighting.position_noise
        pos = default_pos.copy()

        # Add uniform noise to XYZ
        pos += self.rng.uniform(-noise, noise, size=3)

        # Clamp Z to stay above table (minimum 0.5m)
        pos[2] = max(0.5, pos[2])

        return tuple(float(x) for x in pos)

    def _sample_light_diffuse(
        self, default_diffuse: np.ndarray
    ) -> Tuple[float, float, float]:
        """Sample light diffuse color with intensity and color variation."""
        if not self.config.enabled:
            return tuple(float(x) for x in default_diffuse)

        diffuse = default_diffuse.copy()

        # Apply intensity scaling
        if self.config.lighting.randomize_intensity:
            lo, hi = self.config.lighting.intensity_range
            intensity = self.rng.uniform(lo, hi)
            diffuse = diffuse * intensity

        # Apply color temperature shift
        if self.config.lighting.randomize_color:
            lo, hi = self.config.lighting.color_temp_range
            temp_shift = self.rng.uniform(lo, hi)

            # Warm: boost R, reduce B. Cool: boost B, reduce R
            if temp_shift < 1.0:
                # Warm tone
                warm_factor = 1.0 - temp_shift
                diffuse[0] *= 1.0 + warm_factor * 0.2  # Boost red
                diffuse[2] *= 1.0 - warm_factor * 0.15  # Reduce blue
            else:
                # Cool tone
                cool_factor = temp_shift - 1.0
                diffuse[2] *= 1.0 + cool_factor * 0.2  # Boost blue
                diffuse[0] *= 1.0 - cool_factor * 0.15  # Reduce red

        # Clamp to valid range
        diffuse = np.clip(diffuse, 0.0, 1.0)

        return tuple(float(x) for x in diffuse)

    def _sample_headlight_diffuse(self) -> Tuple[float, float, float]:
        """Sample headlight diffuse color."""
        return self._sample_light_diffuse(DEFAULT_HEADLIGHT_DIFFUSE)

    def _sample_headlight_ambient(self) -> Tuple[float, float, float]:
        """Sample headlight ambient color."""
        if not self.config.enabled:
            return tuple(float(x) for x in DEFAULT_HEADLIGHT_AMBIENT)

        ambient = DEFAULT_HEADLIGHT_AMBIENT.copy()

        # Apply similar but gentler variation to ambient
        if self.config.lighting.randomize_intensity:
            lo, hi = self.config.lighting.intensity_range
            # Use narrower range for ambient to avoid too dark/bright scenes
            intensity = self.rng.uniform(
                max(0.7, lo),
                min(1.1, hi)
            )
            ambient = ambient * intensity

        # Clamp to valid range
        ambient = np.clip(ambient, 0.0, 1.0)

        return tuple(float(x) for x in ambient)

    def get_nominal_config(self) -> VisualConfiguration:
        """
        Get a nominal (non-randomized) visual configuration.

        Useful for testing or when visual DR is disabled.
        """
        return VisualConfiguration(
            table_texture_index=0,
            floor_texture_index=0,
            container_color=tuple(DEFAULT_CONTAINER_COLOR),
            bowl_tints={bowl: (1.0, 1.0, 1.0, 1.0) for bowl in self.active_bowls},
            top_light_pos=tuple(DEFAULT_TOP_LIGHT_POS),
            top_light_diffuse=tuple(DEFAULT_TOP_LIGHT_DIFFUSE),
            side_light_pos=tuple(DEFAULT_SIDE_LIGHT_POS),
            side_light_diffuse=tuple(DEFAULT_SIDE_LIGHT_DIFFUSE),
            headlight_diffuse=tuple(DEFAULT_HEADLIGHT_DIFFUSE),
            headlight_ambient=tuple(DEFAULT_HEADLIGHT_AMBIENT),
            seed=0,
        )
