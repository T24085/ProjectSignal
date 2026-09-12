import unittest
from pathlib import Path
import tempfile

import numpy as np

from signal_lab.experiment.clustering import ClusterTracker
from signal_lab.experiment.networks import NetworkTracker
from signal_lab.physics.particle import ParticleState
from signal_lab.physics.engine import SimulationConfig, SimulationEngine
from signal_lab.physics.genome import Genome
from signal_lab.storage.replay import load_network_snapshot, save_network_snapshot


def state(positions: np.ndarray, velocities: np.ndarray | None = None) -> ParticleState:
    positions = np.asarray(positions, dtype=np.float64)
    return ParticleState(
        np.arange(len(positions)),
        np.zeros(len(positions), dtype=np.int8),
        positions,
        np.zeros_like(positions) if velocities is None else np.asarray(velocities, dtype=np.float64),
    )


def blob(center_x: float, center_y: float = 100.0, count: int = 12, radius: float = 5.0) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack((center_x + radius * np.cos(angles), center_y + radius * np.sin(angles)))


def three_blob_network(with_bridges: bool = True) -> np.ndarray:
    positions = [blob(40.0), blob(100.0), blob(160.0)]
    if with_bridges:
        positions.append(np.column_stack(([55, 65, 75, 85, 115, 125, 135, 145], [100] * 8)))
    return np.vstack(positions)


class NetworkDetectionTests(unittest.TestCase):
    def detect(self, positions: np.ndarray, minimum_cluster_size: int = 5):
        particle_state = state(positions)
        clusters = ClusterTracker(200, 200, 12, min_cluster_size=minimum_cluster_size).update(particle_state, 0).clusters
        networks = NetworkTracker(200, 200, 12).update(particle_state, clusters, 0)
        return particle_state, clusters, networks

    def test_three_unconnected_blobs_are_nodes_without_edges(self):
        _state, clusters, networks = self.detect(np.vstack((blob(30), blob(100), blob(170))))
        self.assertEqual(len(clusters), 3)
        self.assertEqual(sum(network.node_count for network in networks), 3)
        self.assertEqual(sum(network.edge_count for network in networks), 0)
        self.assertNotIn("NETWORK_STRUCTURE", {network.classification for network in networks})

    def test_three_blobs_and_bridges_become_network_after_persistence(self):
        particle_state, _clusters, networks = self.detect(three_blob_network())
        tracker = ClusterTracker(200, 200, 12, min_cluster_size=5)
        network_tracker = NetworkTracker(200, 200, 12)
        observed = None
        for step in (0, 100, 200, 500):
            clusters = tracker.update(particle_state, step).clusters
            observed = network_tracker.update(particle_state, clusters, step)[0]
        assert observed is not None
        self.assertEqual(observed.node_count, 3)
        self.assertGreaterEqual(observed.edge_count, 2)
        self.assertEqual(observed.classification, "NETWORK_STRUCTURE")
        self.assertGreaterEqual(observed.persistent_node_count, 3)
        self.assertGreaterEqual(observed.persistent_edge_count, 2)
        self.assertTrue(observed.is_candidate)

    def test_large_uniform_like_cloud_is_macro_aggregate(self):
        grid = np.asarray([(12 + x * 8, 12 + y * 8) for y in range(20) for x in range(20)], dtype=np.float64)
        _state, clusters, networks = self.detect(grid)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].size_class, "MACRO_CLUSTER")
        self.assertEqual(networks[0].classification, "MACRO_AGGREGATE")

    def test_large_compact_blob_is_not_a_network(self):
        grid = np.asarray([(55 + x * 4, 55 + y * 4) for y in range(12) for x in range(15)], dtype=np.float64)
        _state, clusters, networks = self.detect(grid)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].size_class, "LARGE_STRUCTURE")
        self.assertEqual(networks[0].classification, "COMPACT_STRUCTURE")
        self.assertEqual(networks[0].edge_count, 0)

    def test_temporary_bridge_does_not_become_persistent_edge(self):
        first = state(three_blob_network())
        second = state(three_blob_network(False))
        cluster_tracker = ClusterTracker(200, 200, 12, min_cluster_size=5)
        network_tracker = NetworkTracker(200, 200, 12)
        events = []
        for step, particle_state in ((0, first), (50, first), (100, second)):
            clusters = cluster_tracker.update(particle_state, step).clusters
            network_tracker.update(particle_state, clusters, step)
            events.extend(network_tracker.last_events)
        self.assertFalse(any(event.kind == "edge_persistent" for event in events))

    def test_edge_lifecycle_and_identity_survive_translation(self):
        first = state(three_blob_network())
        translated = state((three_blob_network() + np.asarray([7.0, 4.0])) % 200.0)
        cluster_tracker = ClusterTracker(200, 200, 12, min_cluster_size=5)
        network_tracker = NetworkTracker(200, 200, 12)
        ids = []
        for step, particle_state in ((0, first), (100, translated)):
            clusters = cluster_tracker.update(particle_state, step).clusters
            observation = network_tracker.update(particle_state, clusters, step)[0]
            ids.append(([node.node_id for node in observation.nodes], [(edge.node_a, edge.node_b) for edge in observation.edges]))
        self.assertEqual(ids[0][0], ids[1][0])
        self.assertEqual(ids[0][1], ids[1][1])
        self.assertEqual(network_tracker.active[1].classification, "NETWORK_STRUCTURE")

    def test_qualifying_network_snapshot_round_trips(self):
        particle_state = state(three_blob_network())
        cluster_tracker = ClusterTracker(200, 200, 12, min_cluster_size=5)
        network_tracker = NetworkTracker(200, 200, 12)
        observation = None
        for step in (0, 100, 200, 500):
            clusters = cluster_tracker.update(particle_state, step).clusters
            observation = network_tracker.update(particle_state, clusters, step)[0]
        assert observation is not None
        engine = SimulationEngine(Genome.random(seed=42), SimulationConfig(width=200, height=200, particle_count=len(particle_state.ids), seed=7))
        engine.replace_state(particle_state, 500)
        with tempfile.TemporaryDirectory() as directory:
            path = save_network_snapshot(engine, observation, Path(directory))
            replay, metadata = load_network_snapshot(path)
        np.testing.assert_array_equal(replay.state.positions, particle_state.positions)
        self.assertEqual(metadata["network"]["network_id"], observation.network_id)
        self.assertEqual(metadata["network"]["edge_count"], observation.edge_count)


if __name__ == "__main__":
    unittest.main()
