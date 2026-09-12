"""Deterministic simulation engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .genome import Genome
from .particle import ParticleState
from .spatial_hash import SpatialHash


@dataclass
class SimulationConfig:
    width: float = 1000.0
    height: float = 1000.0
    particle_count: int = 1000
    seed: int = 0


class SimulationEngine:
    """Advance particles under a fixed genome and deterministic initial seed."""

    def __init__(self, genome: Genome | None = None, config: SimulationConfig | None = None) -> None:
        self.genome = genome or Genome()
        self.config = config or SimulationConfig()
        if self.config.particle_count < 1:
            raise ValueError("particle_count must be positive")
        if self.config.width <= 0 or self.config.height <= 0:
            raise ValueError("world dimensions must be positive")
        self.state = self._make_initial_state()
        self.spatial_hash = SpatialHash(
            self.config.width, self.config.height, self.genome.interaction_radius
        )
        self.step_count = 0

    def _make_initial_state(self) -> ParticleState:
        rng = np.random.default_rng(self.config.seed)
        n = self.config.particle_count
        positions = rng.uniform(
            [0.0, 0.0], [self.config.width, self.config.height], size=(n, 2)
        )
        # Small deterministic velocities make early structure formation visible
        # without encoding any per-particle behavior.
        velocities = rng.normal(0.0, 0.05, size=(n, 2))
        species = rng.integers(0, self.genome.species_count, size=n, dtype=np.int8)
        return ParticleState(np.arange(n, dtype=np.int64), species, positions, velocities)

    def reset(self, seed: int | None = None) -> None:
        """Reset to the exact seeded initial state and timestep zero."""
        if seed is not None:
            self.config.seed = int(seed)
        self.state = self._make_initial_state()
        self.step_count = 0

    def replace_state(self, state: ParticleState, step_count: int = 0) -> None:
        """Replace the state for replay/tests without changing the genome."""
        if np.any(state.species >= self.genome.species_count):
            raise ValueError("state contains species not present in the genome")
        self.state = state.copy()
        self.step_count = int(step_count)

    def step(self, steps: int = 1) -> ParticleState:
        """Advance the world by ``steps`` and return the live state."""
        if steps < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(steps):
            self._step_once()
        return self.state

    def _step_once(self) -> None:
        positions = self.state.positions
        velocities = self.state.velocities
        species = self.state.species
        self.spatial_hash.rebuild(positions)
        accelerations = np.zeros_like(positions)
        # Keep neighbor discovery spatially hashed, but batch the force law.
        # This is algebraically the same directed pair force as pair_force();
        # batching removes thousands of Python/NumPy call boundaries per step.
        pair_i: list[int] = []
        pair_j: list[int] = []
        for i in range(self.state.count):
            for j in self.spatial_hash.neighbors(positions[i]):
                if i != j:
                    pair_i.append(i)
                    pair_j.append(j)
        if pair_i:
            i_indices = np.asarray(pair_i, dtype=np.int64)
            j_indices = np.asarray(pair_j, dtype=np.int64)
            size = np.array([self.config.width, self.config.height], dtype=np.float64)
            delta = positions[j_indices] - positions[i_indices]
            delta = (delta + size / 2.0) % size - size / 2.0
            distances = np.linalg.norm(delta, axis=1)
            valid = (distances > 0.0) & (distances < self.genome.interaction_radius)
            if np.any(valid):
                i_indices = i_indices[valid]
                j_indices = j_indices[valid]
                delta = delta[valid]
                distances = distances[valid]
                q = distances / self.genome.interaction_radius
                phi = np.empty_like(q)
                core = q < self.genome.core_radius_ratio
                phi[core] = -self.genome.core_repulsion * (1.0 - q[core] / self.genome.core_radius_ratio)
                beta = self.genome.core_radius_ratio
                interaction = ~core
                phi[interaction] = self.genome.interaction_matrix[species[i_indices[interaction]], species[j_indices[interaction]]] * np.sin(np.pi * (q[interaction] - beta) / (1.0 - beta))
                forces = self.genome.force_gain * phi[:, None] * delta / distances[:, None]
                np.add.at(accelerations, i_indices, forces)
        velocities *= self.genome.damping
        velocities += accelerations * self.genome.dt
        positions += velocities * self.genome.dt
        positions[:, 0] %= self.config.width
        positions[:, 1] %= self.config.height
        self.step_count += 1

    def clone(self) -> "SimulationEngine":
        """Return a bitwise state clone suitable for later branch experiments."""
        clone = SimulationEngine(self.genome, self.config)
        clone.replace_state(self.state, self.step_count)
        return clone
