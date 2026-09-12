"""Particle state container and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ParticleState:
    """The complete per-particle state: id, species, position, velocity."""

    ids: np.ndarray
    species: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray

    def __post_init__(self) -> None:
        self.ids = np.asarray(self.ids, dtype=np.int64)
        self.species = np.asarray(self.species, dtype=np.int8)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.velocities = np.asarray(self.velocities, dtype=np.float64)
        self.validate()

    def validate(self) -> None:
        n = len(self.ids)
        if self.ids.ndim != 1 or self.species.shape != (n,):
            raise ValueError("ids and species must be one-dimensional arrays of equal length")
        if self.positions.shape != (n, 2) or self.velocities.shape != (n, 2):
            raise ValueError("positions and velocities must have shape (N, 2)")
        if not all(np.all(np.isfinite(array)) for array in (self.positions, self.velocities)):
            raise ValueError("particle positions and velocities must be finite")
        if np.any(self.species < 0):
            raise ValueError("species IDs must be non-negative")

    @property
    def count(self) -> int:
        return len(self.ids)

    def copy(self) -> "ParticleState":
        return ParticleState(self.ids.copy(), self.species.copy(), self.positions.copy(), self.velocities.copy())

    def save(self, path: str | Path) -> None:
        """Save arrays in a lossless compressed NumPy archive."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            ids=self.ids,
            species=self.species,
            positions=self.positions,
            velocities=self.velocities,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ParticleState":
        with np.load(path, allow_pickle=False) as data:
            return cls(data["ids"], data["species"], data["positions"], data["velocities"])
