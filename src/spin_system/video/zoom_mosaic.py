"""Layout calculations for zooming simulation mosaics."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class MosaicLayout:
    """Tile layout and crop positions for a zooming mosaic."""

    simulation_names: tuple[str, ...]
    tile_height: int
    tile_width: int
    rows: int
    columns: int
    final_row: int
    final_column: int

    @classmethod
    def create(
        cls,
        simulation_names: list[str],
        tile_shape: tuple[int, int],
        frame_ratio: float = 3 / 2,
    ) -> MosaicLayout:
        """Arrange simulations and select a central complete 3-by-2 block."""
        if len(simulation_names) < 6:
            raise ValueError("At least six simulations are required for a zoom video")

        tile_height, tile_width = tile_shape
        if tile_height <= 0 or tile_width <= 0:
            raise ValueError("Tile dimensions must be positive")

        columns, rows = cls._grid_shape(
            len(simulation_names), tile_height, tile_width, frame_ratio
        )
        final_row, final_column = cls._final_block(
            len(simulation_names), rows, columns
        )
        return cls(
            simulation_names=tuple(simulation_names),
            tile_height=tile_height,
            tile_width=tile_width,
            rows=rows,
            columns=columns,
            final_row=final_row,
            final_column=final_column,
        )

    @staticmethod
    def _grid_shape(
        simulation_count: int,
        tile_height: int,
        tile_width: int,
        frame_ratio: float,
    ) -> tuple[int, int]:
        candidates = []
        for columns in range(3, simulation_count + 1):
            rows = ceil(simulation_count / columns)
            if rows < 2:
                continue
            ratio = columns * tile_width / (rows * tile_height)
            candidates.append((abs(ratio - frame_ratio), columns, rows))

        _, columns, rows = min(candidates)
        return columns, rows

    @staticmethod
    def _final_block(
        simulation_count: int, rows: int, columns: int
    ) -> tuple[int, int]:
        candidates = []
        mosaic_center = (rows / 2, columns / 2)
        for row in range(rows - 1):
            for column in range(columns - 2):
                indices = (
                    row * columns + column,
                    row * columns + column + 2,
                    (row + 1) * columns + column,
                    (row + 1) * columns + column + 2,
                )
                if max(indices) >= simulation_count:
                    continue
                block_center = (row + 1, column + 1.5)
                distance = sum(
                    (candidate - center) ** 2
                    for candidate, center in zip(block_center, mosaic_center)
                )
                candidates.append((distance, row, column))

        if not candidates:
            raise ValueError("Could not arrange six complete simulations in a 3-by-2 block")
        _, row, column = min(candidates)
        return row, column

    @property
    def mosaic_shape(self) -> tuple[int, int]:
        """Return the height and width of the complete mosaic."""
        return self.rows * self.tile_height, self.columns * self.tile_width

    @property
    def final_bounds(self) -> tuple[int, int, int, int]:
        """Return the final crop as top, bottom, left, right pixel bounds."""
        top = self.final_row * self.tile_height
        left = self.final_column * self.tile_width
        return top, top + 2 * self.tile_height, left, left + 3 * self.tile_width

    def camera_window(
        self, progress: float, frame_ratio: float = 3 / 2
    ) -> tuple[float, float, float, float]:
        """Return a smooth, fixed-ratio camera window for one video frame."""
        if not 0 <= progress <= 1:
            raise ValueError("Zoom progress must be between zero and one")
        if frame_ratio <= 0:
            raise ValueError("Frame ratio must be greater than zero")

        start = self._ratio_window(
            (0, self.mosaic_shape[0], 0, self.mosaic_shape[1]), frame_ratio
        )
        end = self._ratio_window(self.final_bounds, frame_ratio)
        eased_progress = progress * progress * (3 - 2 * progress)
        start_center = ((start[0] + start[1]) / 2, (start[2] + start[3]) / 2)
        end_center = ((end[0] + end[1]) / 2, (end[2] + end[3]) / 2)
        center_row = start_center[0] + eased_progress * (end_center[0] - start_center[0])
        center_column = start_center[1] + eased_progress * (
            end_center[1] - start_center[1]
        )
        height = start[1] - start[0] + eased_progress * (
            end[1] - end[0] - (start[1] - start[0])
        )
        width = height * frame_ratio
        return (
            center_row - height / 2,
            center_row + height / 2,
            center_column - width / 2,
            center_column + width / 2,
        )

    @staticmethod
    def _ratio_window(
        bounds: tuple[int, int, int, int], frame_ratio: float
    ) -> tuple[float, float, float, float]:
        top, bottom, left, right = bounds
        height = bottom - top
        width = right - left
        center_row = (top + bottom) / 2
        center_column = (left + right) / 2
        if width / height < frame_ratio:
            width = height * frame_ratio
        else:
            height = width / frame_ratio
        return (
            center_row - height / 2,
            center_row + height / 2,
            center_column - width / 2,
            center_column + width / 2,
        )