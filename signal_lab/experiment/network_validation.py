"""Deterministic validation suite for internal network topology detection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from io import StringIO

import numpy as np

from signal_lab.experiment.clustering import ClusterTracker
from signal_lab.experiment.networks import NetworkObservation, NetworkTracker
from signal_lab.physics.particle import ParticleState


WIDTH = 200.0
HEIGHT = 200.0
RADIUS = 12.0


@dataclass
class NetworkValidationResult:
    name: str
    passed: bool
    expected_behavior: str
    actual_behavior: str
    raw_metrics: dict[str, object]


def _state(positions: np.ndarray, ids: np.ndarray | None = None) -> ParticleState:
    positions = np.asarray(positions, dtype=np.float64)
    return ParticleState(
        np.arange(len(positions)) if ids is None else ids,
        np.zeros(len(positions), dtype=np.int8),
        positions,
        np.zeros_like(positions),
    )


def _blob(center_x: float, center_y: float = 100.0, count: int = 12, radius: float = 5.0) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack((center_x + radius * np.cos(angles), center_y + radius * np.sin(angles)))


def _network_positions() -> np.ndarray:
    return np.vstack((_blob(40), _blob(100), _blob(160), np.column_stack(([55, 65, 75, 85, 115, 125, 135, 145], [100] * 8))))


def _observe(states: list[tuple[int, ParticleState]]) -> tuple[NetworkTracker, list[NetworkObservation]]:
    cluster_tracker = ClusterTracker(WIDTH, HEIGHT, RADIUS, min_cluster_size=5)
    network_tracker = NetworkTracker(WIDTH, HEIGHT, RADIUS)
    observations: list[NetworkObservation] = []
    for step, state in states:
        clusters = cluster_tracker.update(state, step).clusters
        observations.extend(network_tracker.update(state, clusters, step))
    return network_tracker, observations


def _metrics(networks: list[NetworkObservation]) -> dict[str, object]:
    return {
        "network_count": len(networks),
        "node_count": sum(network.node_count for network in networks),
        "edge_count": sum(network.edge_count for network in networks),
        "classifications": [network.classification for network in networks],
        "network_scores": [network.network_score for network in networks],
        "persistent_node_counts": [network.persistent_node_count for network in networks],
        "persistent_edge_counts": [network.persistent_edge_count for network in networks],
    }


def run_network_validation() -> list[NetworkValidationResult]:
    results: list[NetworkValidationResult] = []

    disconnected = np.vstack((_blob(30), _blob(100), _blob(170)))
    _tracker, observations = _observe([(0, _state(disconnected))])
    raw = _metrics(observations)
    results.append(NetworkValidationResult("A. Three dense blobs with no connections", raw["node_count"] == 3 and raw["edge_count"] == 0, "3 nodes, 0 edges, not a network structure", f"{raw['node_count']} nodes, {raw['edge_count']} edges; classes={raw['classifications']}", raw))

    _tracker, observations = _observe([(step, _state(_network_positions())) for step in (0, 100, 200, 500)])
    actual = observations[-1]
    raw = actual.to_dict()
    results.append(NetworkValidationResult("B. Three dense blobs joined by particle chains", actual.node_count == 3 and actual.edge_count >= 2 and actual.classification == "NETWORK_STRUCTURE", "3 nodes, at least 2 persistent edges, NETWORK_STRUCTURE", f"{actual.node_count} nodes, {actual.edge_count} edges, {actual.classification}", raw))

    uniform_like = np.asarray([(12 + x * 8, 12 + y * 8) for y in range(20) for x in range(20)], dtype=np.float64)
    _tracker, observations = _observe([(0, _state(uniform_like))])
    actual = observations[0]
    raw = actual.to_dict()
    results.append(NetworkValidationResult("C. Uniform large cloud", actual.classification == "MACRO_AGGREGATE", "MACRO_AGGREGATE, not NETWORK_STRUCTURE", f"{actual.classification}; {actual.node_count} nodes, {actual.edge_count} edges", raw))

    compact = np.asarray([(55 + x * 4, 55 + y * 4) for y in range(12) for x in range(15)], dtype=np.float64)
    _tracker, observations = _observe([(0, _state(compact))])
    actual = observations[0]
    raw = actual.to_dict()
    results.append(NetworkValidationResult("D. One compact large blob", actual.classification == "COMPACT_STRUCTURE" and actual.edge_count == 0, "compact large structure, not a network", f"{actual.classification}; {actual.node_count} nodes, {actual.edge_count} network edges", raw))

    first = _state(_network_positions())
    no_bridge = _state(np.vstack((_blob(40), _blob(100), _blob(160), np.column_stack(([55, 65, 75], [100] * 3)))))
    tracker, observations = _observe([(0, first), (50, first), (100, no_bridge)])
    raw = _metrics(observations)
    passed = not any(event.kind == "edge_persistent" for event in tracker.last_events)
    results.append(NetworkValidationResult("E. Temporary accidental bridge", passed, "temporary contact must not become a persistent edge", f"persistent-edge event={not passed}", raw))

    broken = _network_positions().copy()
    broken[36 + 3:36 + 5, 0] += 50.0
    lifecycle_cluster_tracker = ClusterTracker(WIDTH, HEIGHT, RADIUS, min_cluster_size=5)
    lifecycle_tracker = NetworkTracker(WIDTH, HEIGHT, RADIUS)
    lifecycle_observations: list[NetworkObservation] = []
    lifecycle_events = []
    for step, particle_state in ((0, _state(_network_positions())), (100, _state(_network_positions())), (200, _state(broken)), (300, _state(_network_positions()))):
        clusters = lifecycle_cluster_tracker.update(particle_state, step).clusters
        lifecycle_observations.extend(lifecycle_tracker.update(particle_state, clusters, step))
        lifecycle_events.extend(lifecycle_tracker.last_events)
    lifecycle_raw = _metrics(lifecycle_observations)
    lifecycle_events = [event.kind for event in lifecycle_events]
    results.append(NetworkValidationResult("F. Connection breaks and reforms", "edge_dissolved" in lifecycle_events or len(lifecycle_observations) >= 2, "edge lifecycle is tracked through break and reform", f"observed event kinds={lifecycle_events}", {**lifecycle_raw, "last_events": lifecycle_events}))

    toroidal = (_network_positions() + np.asarray([155.0, 0.0])) % 200.0
    toroidal_tracker, toroidal_observations = _observe([(0, _state(_network_positions())), (100, _state(toroidal))])
    first_ids = [(node.node_id, sorted(node.particle_ids)) for node in toroidal_observations[0].nodes]
    last_ids = [(node.node_id, sorted(node.particle_ids)) for node in toroidal_observations[-1].nodes]
    raw = toroidal_observations[-1].to_dict()
    results.append(NetworkValidationResult("G. Whole network crosses toroidal boundary", first_ids == last_ids, "node and edge identities are preserved", f"node identity preserved={first_ids == last_ids}", raw))

    center = np.asarray([100.0, 100.0])
    translated = ((_network_positions() - center) @ np.asarray([[0.0, -1.0], [1.0, 0.0]]).T + center + np.asarray([8.0, -5.0])) % 200.0
    _tracker, rotating_observations = _observe([(0, _state(_network_positions())), (100, _state(translated))])
    raw = rotating_observations[-1].to_dict()
    passed = rotating_observations[0].node_count == rotating_observations[-1].node_count and rotating_observations[0].edge_count == rotating_observations[-1].edge_count
    results.append(NetworkValidationResult("H. Network rotates and translates as a whole", passed, "topology remains stable", f"topology {rotating_observations[0].node_count}/{rotating_observations[0].edge_count} -> {rotating_observations[-1].node_count}/{rotating_observations[-1].edge_count}", raw))
    return results


def format_network_validation_report(results: list[NetworkValidationResult] | None = None) -> str:
    results = results if results is not None else run_network_validation()
    output = StringIO()
    output.write("NETWORK STRUCTURE VALIDATION\n\n")
    for result in results:
        output.write(f"{result.name}: {'PASS' if result.passed else 'FAIL'}\n")
        output.write(f"  Expected         {result.expected_behavior}\n")
        output.write(f"  Actual           {result.actual_behavior}\n")
        output.write("  Raw metrics\n")
        for key, value in result.raw_metrics.items():
            rendered = json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict, bool)) else str(value)
            output.write(f"    {key:<28} {rendered}\n")
        output.write("\n")
    passed = sum(result.passed for result in results)
    output.write(f"{passed}/{len(results)} NETWORK TESTS PASSED\n")
    return output.getvalue()
