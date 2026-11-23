"""Render simulation mosaics as zooming MP4 videos."""

from __future__ import annotations

from collections.abc import Iterator
from math import ceil, floor, log2
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter, map_coordinates

from spin_system import FileManager

from .zoom_mosaic import MosaicLayout


class ZoomMosaicVideo:
    """Create a 3:2 video that zooms from a mosaic to six simulations."""

    def __init__(
        self,
        config_name: str,
        first_sim: str | None = None,
        last_sim: str | None = None,
        source: str = "grids",
        cmap: str = "berlin",
        file_manager: FileManager | None = None,
    ) -> None:
        self.config_name = config_name
        self.first_sim = first_sim
        self.last_sim = last_sim
        self.source = source
        self.cmap = cmap
        self.file_manager = file_manager or FileManager()
        self.layout: MosaicLayout | None = None
        self._mosaic: np.ndarray | None = None
        self._mipmaps: tuple[np.ndarray, ...] = ()

    def create(
        self,
        output_path: str | Path,
        duration: float = 8,
        fps: int = 30,
        dpi: int = 300,
    ) -> Path:
        """Render frames and encode them as an H.264 MP4 file."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            import imageio.v2 as imageio
        except ImportError as error:
            raise RuntimeError(
                "MP4 export requires the 'imageio[ffmpeg]' dependency"
            ) from error

        with imageio.get_writer(
            output,
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            pixelformat="yuv420p",
            output_params=["-crf", "18", "-preset", "medium"],
        ) as writer:
            for frame in self.frames(duration=duration, fps=fps, dpi=dpi):
                writer.append_data(frame)
        return output

    def frames(
        self,
        duration: float = 8,
        fps: int = 30,
        dpi: int = 300,
    ) -> Iterator[np.ndarray]:
        """Yield RGB frames with a fixed 3:2 output format."""
        if duration <= 0:
            raise ValueError("Duration must be greater than zero")
        if fps <= 0:
            raise ValueError("Frames per second must be greater than zero")
        if dpi <= 0:
            raise ValueError("DPI must be greater than zero")

        self._prepare_mosaic()
        frame_count = max(2, round(duration * fps))
        frame_width = 6 * dpi
        frame_height = 4 * dpi
        max_scale = max(
            self.layout.camera_window(0)[1] / frame_height,
            self.layout.camera_window(0)[3] / frame_width,
        )
        self._mipmaps = self._create_mipmaps(max_scale)

        for frame_index in range(frame_count):
            progress = frame_index / (frame_count - 1)
            yield self._render_camera(progress, frame_height, frame_width)

    def _prepare_mosaic(self) -> None:
        if self._mosaic is not None:
            return

        names = self._simulation_names()
        values = [self._display_values(name) for name in names]
        tile_height = max(value.shape[0] for value in values)
        tile_width = max(value.shape[1] for value in values)
        self.layout = MosaicLayout.create(names, (tile_height, tile_width))
        tiles = self._colour_tiles(values, tile_height, tile_width)

        mosaic_height, mosaic_width = self.layout.mosaic_shape
        mosaic = np.full((mosaic_height, mosaic_width, 3), 255, dtype=np.uint8)
        for index, tile in enumerate(tiles):
            row, column = divmod(index, self.layout.columns)
            top = row * tile_height
            left = column * tile_width
            mosaic[top : top + tile_height, left : left + tile_width] = tile
        self._mosaic = mosaic

    def _simulation_names(self) -> list[str]:
        names = self.file_manager.list_simulations(self.config_name, self.source)
        if self.first_sim is not None:
            names = [name for name in names if name >= self.first_sim]
        if self.last_sim is not None:
            names = [name for name in names if name <= self.last_sim]
        if not names:
            raise FileNotFoundError(
                f"No simulation arrays found for '{self.config_name}' in '{self.source}'"
            )
        return names

    def _display_values(self, simulation_name: str) -> np.ndarray:
        array = self.file_manager.load_array(
            self.config_name, simulation_name, source=self.source
        )
        if array.ndim == 2:
            values = array
        elif array.ndim == 3:
            values = array[..., -1]
        else:
            raise ValueError(
                f"Simulation '{simulation_name}' must contain a 2D or 3D array, got {array.shape}"
            )
        if np.iscomplexobj(values):
            raise ValueError(f"Simulation '{simulation_name}' contains complex values")
        return np.asarray(values, dtype=float)

    def _colour_tiles(
        self, values: list[np.ndarray], tile_height: int, tile_width: int
    ) -> list[np.ndarray]:
        finite_values = np.concatenate(
            [value[np.isfinite(value)] for value in values if np.isfinite(value).any()]
        )
        if not finite_values.size:
            raise ValueError("Simulation arrays do not contain finite values")

        value_min = float(finite_values.min())
        value_max = float(finite_values.max())
        if value_min == value_max:
            value_min -= 0.5
            value_max += 0.5
        normalizer = Normalize(vmin=value_min, vmax=value_max, clip=True)
        colour_map = colormaps.get_cmap(self.cmap).copy()
        colour_map.set_bad("white")

        tiles = []
        for value in values:
            padded = np.full((tile_height, tile_width), np.nan)
            height, width = value.shape
            padded[:height, :width] = value
            rgba = colour_map(np.ma.masked_invalid(normalizer(padded)), bytes=True)
            tiles.append(rgba[..., :3])
        return tiles

    def _render_camera(
        self, progress: float, frame_height: int, frame_width: int
    ) -> np.ndarray:
        if self.layout is None or self._mosaic is None or not self._mipmaps:
            raise RuntimeError("Mosaic must be prepared before rendering frames")

        top, bottom, left, right = self.layout.camera_window(progress)
        pixel_scale = max(
            (bottom - top) / frame_height,
            (right - left) / frame_width,
        )
        level_position = min(max(log2(max(pixel_scale, 1)), 0), len(self._mipmaps) - 1)
        lower_level = floor(level_position)
        upper_level = min(lower_level + 1, len(self._mipmaps) - 1)
        upper_weight = level_position - lower_level

        lower = self._sample_mipmap(
            self._mipmaps[lower_level],
            lower_level,
            top,
            bottom,
            left,
            right,
            frame_height,
            frame_width,
        )
        if lower_level == upper_level:
            return lower
        upper = self._sample_mipmap(
            self._mipmaps[upper_level],
            upper_level,
            top,
            bottom,
            left,
            right,
            frame_height,
            frame_width,
        )
        return np.clip(
            np.rint((1 - upper_weight) * lower + upper_weight * upper), 0, 255
        ).astype(np.uint8)

    def _create_mipmaps(self, max_scale: float) -> tuple[np.ndarray, ...]:
        if self._mosaic is None:
            raise RuntimeError("Mosaic must be prepared before creating mipmaps")

        levels = [self._mosaic]
        for _ in range(ceil(log2(max(max_scale, 1)))):
            image = gaussian_filter(
                levels[-1].astype(np.float32),
                sigma=(0.6, 0.6, 0),
                mode="nearest",
            )
            height, width = image.shape[:2]
            if height % 2:
                image = np.pad(image, ((0, 1), (0, 0), (0, 0)), mode="edge")
            if width % 2:
                image = np.pad(image, ((0, 0), (0, 1), (0, 0)), mode="edge")
            levels.append(
                image.reshape(image.shape[0] // 2, 2, image.shape[1] // 2, 2, 3)
                .mean(axis=(1, 3))
                .astype(np.uint8)
            )
        return tuple(levels)

    @staticmethod
    def _sample_mipmap(
        image: np.ndarray,
        level: int,
        top: float,
        bottom: float,
        left: float,
        right: float,
        frame_height: int,
        frame_width: int,
    ) -> np.ndarray:
        rows = top + (np.arange(frame_height) + 0.5) * (bottom - top) / frame_height - 0.5
        columns = (
            left + (np.arange(frame_width) + 0.5) * (right - left) / frame_width - 0.5
        )
        scale = 2**level
        rows = (rows + 0.5) / scale - 0.5
        columns = (columns + 0.5) / scale - 0.5
        source_rows, source_columns = np.meshgrid(rows, columns, indexing="ij")
        coordinates = np.stack((source_rows, source_columns))
        channels = [
            map_coordinates(
                image[..., channel],
                coordinates,
                output=np.float32,
                order=1,
                mode="constant",
                cval=255,
                prefilter=False,
            )
            for channel in range(3)
        ]
        return np.clip(np.rint(np.stack(channels, axis=-1)), 0, 255).astype(np.uint8)