"""Toroidal distance and directional interaction force calculations."""

from __future__ import annotations

import numpy as np

from .genome import Genome


def toroidal_delta(position_j: np.ndarray, position_i: np.ndarray, width: float, height: float) -> np.ndarray:
    """Return minimum-image displacement from ``i`` to ``j``."""
    delta = np.asarray(position_j, dtype=np.float64) - np.asarray(position_i, dtype=np.float64)
    size = np.array([width, height], dtype=np.float64)
    return (delta + size / 2.0) % size - size / 2.0


def toroidal_distance(position_a: np.ndarray, position_b: np.ndarray, width: float, height: float) -> float:
    """Return minimum-image distance between two points on a torus."""
    return float(np.linalg.norm(toroidal_delta(position_b, position_a, width, height)))


def pair_force(
    position_i: np.ndarray,
    position_j: np.ndarray,
    species_i: int,
    species_j: int,
    genome: Genome,
    width: float,
    height: float,
) -> np.ndarray:
    """Calculate force on particle ``i`` due to particle ``j``."""
    delta = toroidal_delta(position_j, position_i, width, height)
    distance = float(np.linalg.norm(delta))
    if distance == 0.0 or distance >= genome.interaction_radius:
        return np.zeros(2, dtype=np.float64)
    q = distance / genome.interaction_radius
    if q < genome.core_radius_ratio:
        phi = -genome.core_repulsion * (1.0 - q / genome.core_radius_ratio)
    else:
        beta = genome.core_radius_ratio
        phi = float(genome.interaction_matrix[species_i, species_j]) * np.sin(
            np.pi * (q - beta) / (1.0 - beta)
        )
    return genome.force_gain * phi * (delta / distance)
