"""Deterministic synthetic validation suite for Phase 2B structure scoring."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Callable

import numpy as np

from signal_lab.experiment.clustering import ClusterDetector, ClusterObservation, ClusterTracker
from signal_lab.physics.particle import ParticleState


WIDTH = 200.0
HEIGHT = 200.0
LINK_RADIUS = 12.0


@dataclass
class ValidationResult:
    name: str
    metrics: dict[str, float]
    passed: bool
    detail: str


def _state(positions: np.ndarray, velocities: np.ndarray | None = None) -> ParticleState:
    positions = np.asarray(positions, dtype=np.float64)
    velocity_array = np.zeros_like(positions) if velocities is None else np.asarray(velocities, dtype=np.float64)
    return ParticleState(np.arange(len(positions)), np.zeros(len(positions), dtype=np.int8), positions, velocity_array)


def _circle(center: tuple[float, float], count: int, radius: float) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))


def _best(state: ParticleState, minimum_size: int = 5) -> ClusterObservation | None:
    tracker = ClusterTracker(WIDTH, HEIGHT, LINK_RADIUS, min_cluster_size=minimum_size)
    clusters = tracker.update(state, 0).clusters
    return max(clusters, key=lambda item: item.cohesion_score, default=None)


def _track(states: list[ParticleState]) -> tuple[ClusterTracker, list[ClusterObservation]]:
    tracker = ClusterTracker(WIDTH, HEIGHT, LINK_RADIUS, min_cluster_size=5)
    observations: list[ClusterObservation] = []
    for step, state in enumerate(states):
        result = tracker.update(state, step * 100)
        if result.clusters:
            observations.append(max(result.clusters, key=lambda item: item.particle_count))
    return tracker, observations


def run_validation() -> list[ValidationResult]:
    """Run eight deterministic cases and return their human-readable results."""
    results: list[ValidationResult] = []

    compact_positions = _circle((100, 100), 40, 8)
    compact_velocity = np.tile([0.35, 0.0], (40, 1))
    compact = _best(_state(compact_positions, compact_velocity))
    results.append(ValidationResult("Compact moving blob", {"Cohesion": compact.cohesion_score if compact else 0.0, "Structure": compact.structure_score if compact else 0.0}, bool(compact and compact.cohesion_score >= 0.65), "compact local graph"))

    bridged = np.vstack((_circle((70, 100), 18, 8), _circle((130, 100), 18, 8), np.column_stack((np.arange(78, 123, 9), np.full(5, 100.0)))))
    bridged_cluster = _best(_state(bridged, np.tile([0.2, 0.0], (len(bridged), 1))))
    bridge_score = bridged_cluster.structure_score if bridged_cluster else 0.0
    results.append(ValidationResult("Bridged blobs", {"Structure": bridge_score, "Bridge Fraction": bridged_cluster.bridge_fraction if bridged_cluster else 0.0, "Compactness": bridged_cluster.compactness if bridged_cluster else 0.0}, bool(bridged_cluster and bridge_score < 0.65), "sparse bridge is penalized"))

    rng = np.random.default_rng(2026)
    random_cluster = _best(_state(rng.uniform(10, 190, (100, 2))))
    random_score = random_cluster.structure_score if random_cluster else 0.0
    results.append(ValidationResult("Uniform random cloud", {"Structure": random_score}, random_score < 0.65, "no persistent cohesive component"))

    expanding_states = []
    for step, radius in enumerate((5.0, 8.0, 11.0, 14.0)):
        positions = _circle((100 + step * 2, 100), 40, radius)
        expanding_states.append(_state(positions, np.tile([0.2, 0.0], (40, 1))))
    _, expanding_observations = _track(expanding_states)
    expanding = expanding_observations[-1] if expanding_observations else None
    results.append(ValidationResult("Expanding cloud", {"Structure": expanding.structure_score if expanding else 0.0, "Rg CV": expanding.rg_cv if expanding else 0.0, "Shape": expanding.shape_score if expanding else 0.0}, bool(expanding and (expanding.shape_score < 0.7 or expanding.structure_score < 0.65)), "growing Rg lowers temporal shape score"))

    crystal_positions = np.asarray([(70 + x * 8, 70 + y * 8) for x in range(6) for y in range(6)], dtype=np.float64)
    crystal = _best(_state(crystal_positions))
    results.append(ValidationResult("Frozen crystal", {"Cohesion": crystal.cohesion_score if crystal else 0.0, "Dynamics": crystal.dynamic_score if crystal else 0.0}, bool(crystal and crystal.cohesion_score >= 0.65 and crystal.dynamic_score <= 0.1), "cohesive but frozen"))

    circulation_positions = _circle((100, 100), 40, 8)
    centered = circulation_positions - np.array([100, 100])
    circulation_velocity = np.column_stack((-centered[:, 1], centered[:, 0])) / 20.0
    _, dynamic_observations = _track([_state(circulation_positions, circulation_velocity)] * 4)
    dynamic = dynamic_observations[-1] if dynamic_observations else None
    results.append(ValidationResult("Dynamic circulation", {"Structure": dynamic.structure_score if dynamic else 0.0, "Dynamics": dynamic.dynamic_score if dynamic else 0.0}, bool(dynamic and dynamic.structure_score >= 0.55 and dynamic.dynamic_score >= 0.5), "nonzero internal circulation"))

    first = np.vstack((_circle((55, 70), 15, 7), _circle((145, 130), 15, 7)))
    second = np.vstack((_circle((75, 70), 15, 7), _circle((125, 130), 15, 7)))
    third = np.vstack((_circle((95, 70), 15, 7), _circle((105, 130), 15, 7)))
    tracker, crossing_observations = _track([_state(first, np.ones_like(first)), _state(second, np.ones_like(second)), _state(third, np.ones_like(third))])
    identity = crossing_observations[-1].identity_score if crossing_observations else 0.0
    results.append(ValidationResult("Two structures crossing", {"Identity": identity, "Tracked": float(len(tracker.active))}, identity >= 0.40, "particle IDs preserved during a pass"))

    toroidal_states = [_state(np.vstack((_circle((198, 100), 20, 5),)), np.tile([1.0, 0.0], (20, 1))), _state(np.vstack((_circle((1, 100), 20, 5),)), np.tile([1.0, 0.0], (20, 1)))]
    toroidal_tracker, toroidal_observations = _track(toroidal_states)
    toroidal_identity = toroidal_observations[-1].identity_score if toroidal_observations else 0.0
    toroidal_centroid = toroidal_observations[-1].centroid[0] if toroidal_observations else 0.0
    results.append(ValidationResult("Toroidal boundary crossing", {"Identity": toroidal_identity, "Centroid X": toroidal_centroid, "Rg": toroidal_observations[-1].radius_of_gyration if toroidal_observations else 0.0}, toroidal_identity >= 0.40 and (toroidal_centroid < 5 or toroidal_centroid > 195), "minimum-image centroid and identity"))

    return results


def format_validation_report(results: list[ValidationResult] | None = None) -> str:
    """Render the validation results in a concise console-friendly report."""
    results = results if results is not None else run_validation()
    output = StringIO()
    output.write("STRUCTURE DETECTOR VALIDATION\n\n")
    for result in results:
        output.write(f"{result.name}:\n")
        for key, value in result.metrics.items():
            output.write(f"  {key:<16} {value:.2f}")
            if key in {"Structure", "Cohesion", "Identity", "Dynamics"}:
                output.write(f" {'PASS' if result.passed else 'FAIL'}")
            output.write("\n")
        output.write(f"  Detail           {result.detail}\n\n")
    passed = sum(result.passed for result in results)
    output.write(f"{passed}/{len(results)} VALIDATION TESTS PASSED\n")
    return output.getvalue()


if __name__ == "__main__":
    print(format_validation_report())
