"""Interaction genome and serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Genome:
    """Parameters that define one universe's local interaction physics.

    The matrix is directional: row ``i`` is the responding species and column
    ``j`` is the neighboring species. Symmetric matrices are accepted for
    controls and unit tests, although random genomes are deliberately made
    asymmetric.
    """

    species_count: int = 3
    interaction_matrix: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0.2, -0.7, 0.8], [0.6, 0.1, -0.5], [-0.2, 0.9, 0.3]],
            dtype=np.float64,
        )
    )
    interaction_radius: float = 60.0
    core_radius_ratio: float = 0.18
    core_repulsion: float = 1.0
    force_gain: float = 1.0
    damping: float = 0.94
    dt: float = 0.1

    def __post_init__(self) -> None:
        self.interaction_matrix = np.asarray(self.interaction_matrix, dtype=np.float64)
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` for physically invalid parameters."""
        if not 1 <= self.species_count <= 8:
            raise ValueError("species_count must be between 1 and 8")
        expected = (self.species_count, self.species_count)
        if self.interaction_matrix.shape != expected:
            raise ValueError(f"interaction_matrix must have shape {expected}")
        if not np.all(np.isfinite(self.interaction_matrix)):
            raise ValueError("interaction_matrix must contain finite values")
        if np.any(self.interaction_matrix < -1.0) or np.any(self.interaction_matrix > 1.0):
            raise ValueError("interaction values must be in [-1, 1]")
        if self.interaction_radius <= 0:
            raise ValueError("interaction_radius must be positive")
        if not 0 < self.core_radius_ratio < 1:
            raise ValueError("core_radius_ratio must be in (0, 1)")
        if self.core_repulsion < 0 or self.force_gain < 0:
            raise ValueError("repulsion and force_gain must be non-negative")
        if not 0 <= self.damping <= 1:
            raise ValueError("damping must be in [0, 1]")
        if self.dt <= 0:
            raise ValueError("dt must be positive")

    @classmethod
    def random(cls, species_count: int = 3, seed: int | None = None) -> "Genome":
        """Create a deterministic random, generally asymmetric genome."""
        rng = np.random.default_rng(seed)
        matrix = rng.uniform(-1.0, 1.0, (species_count, species_count))
        # Exact equality is vanishingly unlikely, but enforce asymmetry for
        # reproducible semantics if a custom RNG or seed ever creates it.
        if species_count > 1 and np.array_equal(matrix, matrix.T):
            matrix[0, 1] = np.nextafter(matrix[0, 1], 1.0)
        return cls(species_count=species_count, interaction_matrix=matrix)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "species_count": self.species_count,
            "interaction_matrix": self.interaction_matrix.tolist(),
            "interaction_radius": self.interaction_radius,
            "core_radius_ratio": self.core_radius_ratio,
            "core_repulsion": self.core_repulsion,
            "force_gain": self.force_gain,
            "damping": self.damping,
            "dt": self.dt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Genome":
        return cls(**data)

    def save(self, path: str | Path) -> None:
        """Save this genome as readable, stable JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Genome":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def genome_hash(self) -> str:
        """Return a content hash useful for experiment bookkeeping."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
