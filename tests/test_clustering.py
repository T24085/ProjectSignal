import tempfile
import unittest
from pathlib import Path

import numpy as np

from signal_lab.experiment.clustering import (
    ClusterDetector,
    ClusterTracker,
    lifetime_class,
    membership_overlap,
    toroidal_centroid,
)
from signal_lab.physics.engine import SimulationConfig, SimulationEngine
from signal_lab.physics.genome import Genome
from signal_lab.physics.particle import ParticleState
from signal_lab.storage.replay import load_structure_snapshot, save_structure_snapshot


def make_state(positions: list[tuple[float, float]], velocities: list[tuple[float, float]] | None = None) -> ParticleState:
    count = len(positions)
    return ParticleState(
        np.arange(count),
        np.zeros(count, dtype=np.int8),
        np.asarray(positions, dtype=np.float64),
        np.asarray(velocities if velocities is not None else [(0.1, 0.0)] * count, dtype=np.float64),
    )


class ClusteringTests(unittest.TestCase):
    def test_cluster_detection(self):
        state = make_state([(10, 10), (11, 10), (12, 10), (13, 10), (80, 80)])
        clusters = ClusterDetector(100, 100, 4, min_cluster_size=3).detect(state)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].particle_count, 4)

    def test_membership_overlap(self):
        self.assertAlmostEqual(membership_overlap({1, 2, 3}, {2, 3, 4}), 0.5)

    def test_cluster_identity_through_movement(self):
        tracker = ClusterTracker(100, 100, 5, min_cluster_size=3)
        first = tracker.update(make_state([(10, 10), (11, 10), (12, 10)]), 0)
        second = tracker.update(make_state([(14, 10), (15, 10), (16, 10)]), 1)
        self.assertEqual(first.clusters[0].cluster_id, second.clusters[0].cluster_id)

    def test_toroidal_boundary_crossing(self):
        centroid = toroidal_centroid(np.asarray([(99, 50), (0, 50), (1, 50)]), 100, 100)
        self.assertTrue(centroid[0] < 2 or centroid[0] > 98, centroid)
        tracker = ClusterTracker(100, 100, 5, min_cluster_size=3)
        first = tracker.update(make_state([(98, 50), (99, 50), (0, 50)]), 0)
        second = tracker.update(make_state([(99, 50), (0, 50), (1, 50)]), 1)
        self.assertEqual(first.clusters[0].cluster_id, second.clusters[0].cluster_id)

    def test_cluster_split_event(self):
        tracker = ClusterTracker(100, 100, 5, min_cluster_size=3)
        tracker.update(make_state([(0, 10), (1, 10), (2, 10), (3, 10), (4, 10), (5, 10)]), 0)
        result = tracker.update(make_state([(0, 10), (1, 10), (2, 10), (10, 10), (11, 10), (12, 10)]), 1)
        self.assertTrue(any(event.kind == "split" for event in result.events))

    def test_cluster_merge_event(self):
        tracker = ClusterTracker(100, 100, 5, min_cluster_size=3)
        tracker.update(make_state([(0, 10), (1, 10), (2, 10), (10, 10), (11, 10), (12, 10)]), 0)
        result = tracker.update(make_state([(4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10)]), 1)
        self.assertTrue(any(event.kind == "merged" for event in result.events))

    def test_cluster_lifetime(self):
        self.assertEqual(lifetime_class(99), "TRANSIENT")
        self.assertEqual(lifetime_class(100), "SHORT_LIVED")
        self.assertEqual(lifetime_class(500), "PERSISTENT")
        self.assertEqual(lifetime_class(2000), "LONG_LIVED")

    def test_snapshot_replay(self):
        genome = Genome.random(seed=5)
        engine = SimulationEngine(genome, SimulationConfig(particle_count=12, seed=8))
        tracker = ClusterTracker(1000, 1000, genome.interaction_radius, min_cluster_size=1)
        cluster = tracker.update(engine.state, 0).clusters[0]
        with tempfile.TemporaryDirectory() as directory:
            path = save_structure_snapshot(engine, cluster, Path(directory))
            replay, metadata = load_structure_snapshot(path)
        np.testing.assert_array_equal(engine.state.positions, replay.state.positions)
        np.testing.assert_array_equal(engine.state.velocities, replay.state.velocities)
        self.assertEqual(metadata["cluster"]["cluster_id"], cluster.cluster_id)


if __name__ == "__main__":
    unittest.main()
