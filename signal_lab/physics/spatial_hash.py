"""Uniform-grid spatial hashing for toroidal neighbor lookup."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


class SpatialHash:
    """A deterministic uniform grid whose cell size is the interaction radius."""

    def __init__(self, width: float, height: float, cell_size: float) -> None:
        if width <= 0 or height <= 0 or cell_size <= 0:
            raise ValueError("world dimensions and cell_size must be positive")
        self.width = float(width)
        self.height = float(height)
        self.cell_size = float(cell_size)
        self.columns = max(1, int(np.ceil(self.width / self.cell_size)))
        self.rows = max(1, int(np.ceil(self.height / self.cell_size)))
        self._cells: dict[tuple[int, int], list[int]] = {}

    def _cell_for(self, position: np.ndarray) -> tuple[int, int]:
        x = float(position[0]) % self.width
        y = float(position[1]) % self.height
        return (int(x / self.cell_size) % self.columns, int(y / self.cell_size) % self.rows)

    def rebuild(self, positions: np.ndarray) -> None:
        """Index all positions. Particle index order is preserved in each cell."""
        cells: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
        for index, position in enumerate(positions):
            cells[self._cell_for(position)].append(index)
        self._cells = dict(cells)

    def neighbors(self, position: np.ndarray) -> list[int]:
        """Return candidate indices from the current and adjacent wrapped cells."""
        cell_x, cell_y = self._cell_for(position)
        found: list[int] = []
        seen: set[tuple[int, int]] = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = ((cell_x + dx) % self.columns, (cell_y + dy) % self.rows)
                if key not in seen:
                    found.extend(self._cells.get(key, ()))
                    seen.add(key)
        return found
