import tempfile
import unittest
from pathlib import Path

import numpy as np

from signal_lab.physics.engine import SimulationConfig, SimulationEngine
from signal_lab.physics.forces import pair_force, toroidal_distance
from signal_lab.physics.genome import Genome
from signal_lab.physics.particle import ParticleState
from signal_lab.physics.spatial_hash import SpatialHash


class PhysicsTests(unittest.TestCase):
    def test_toroidal_distance(self):
        self.assertAlmostEqual(toroidal_distance([1, 50], [999, 50], 1000, 1000), 2.0)

    def test_force_symmetry_when_matrix_symmetric(self):
        matrix = np.array([[0.2, -0.7], [-0.7, 0.4]])
        genome = Genome(2, matrix)
        force_ij = pair_force([100, 100], [130, 100], 0, 1, genome, 1000, 1000)
        force_ji = pair_force([130, 100], [100, 100], 1, 0, genome, 1000, 1000)
        np.testing.assert_allclose(force_ij, -force_ji)

    def test_asymmetric_force_behavior(self):
        genome = Genome(2, np.array([[1.0, 1.0], [-1.0, -1.0]]))
        force_a_on_b = pair_force([100, 100], [130, 100], 0, 1, genome, 1000, 1000)
        force_b_on_a = pair_force([130, 100], [100, 100], 1, 0, genome, 1000, 1000)
        self.assertGreater(force_a_on_b[0], 0)
        self.assertGreater(force_b_on_a[0], 0)

    def test_zero_interaction_matrix(self):
        genome = Genome(2, np.zeros((2, 2)))
        force = pair_force([100, 100], [130, 100], 0, 1, genome, 1000, 1000)
        np.testing.assert_allclose(force, [0, 0])

    def test_core_repulsion(self):
        genome = Genome(2, np.zeros((2, 2)), core_radius_ratio=0.5, core_repulsion=2.0)
        force = pair_force([100, 100], [110, 100], 0, 1, genome, 1000, 1000)
        self.assertLess(force[0], 0)

    def test_deterministic_seed(self):
        genome = Genome.random(seed=4)
        a = SimulationEngine(genome, SimulationConfig(particle_count=80, seed=12))
        b = SimulationEngine(Genome.from_dict(genome.to_dict()), SimulationConfig(particle_count=80, seed=12))
        a.step(20)
        b.step(20)
        np.testing.assert_array_equal(a.state.positions, b.state.positions)
        np.testing.assert_array_equal(a.state.velocities, b.state.velocities)

    def test_save_load_genome(self):
        genome = Genome.random(seed=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "genome.json"
            genome.save(path)
            loaded = Genome.load(path)
        self.assertEqual(genome.genome_hash, loaded.genome_hash)

    def test_save_load_state(self):
        engine = SimulationEngine(config=SimulationConfig(particle_count=30, seed=5))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            engine.state.save(path)
            loaded = ParticleState.load(path)
        np.testing.assert_array_equal(engine.state.ids, loaded.ids)
        np.testing.assert_array_equal(engine.state.positions, loaded.positions)
        np.testing.assert_array_equal(engine.state.velocities, loaded.velocities)

    def test_replay_matches_original(self):
        genome = Genome.random(seed=3)
        original = SimulationEngine(genome, SimulationConfig(particle_count=60, seed=9))
        original.step(15)
        replay = SimulationEngine(genome, SimulationConfig(particle_count=60, seed=9))
        replay.replace_state(original.state, original.step_count)
        original.step(25)
        replay.step(25)
        np.testing.assert_array_equal(original.state.positions, replay.state.positions)

    def test_spatial_hash_matches_bruteforce_small_sample(self):
        rng = np.random.default_rng(1)
        positions = rng.uniform(0, 100, (50, 2))
        spatial = SpatialHash(100, 100, 20)
        spatial.rebuild(positions)
        for i, position in enumerate(positions):
            candidates = set(spatial.neighbors(position))
            brute = {j for j, other in enumerate(positions) if np.linalg.norm(((other - position + 50) % 100) - 50) < 20}
            self.assertTrue(brute.issubset(candidates), (i, brute - candidates))

    def test_batched_engine_step_matches_pair_force_reference(self):
        genome = Genome.random(seed=22)
        engine = SimulationEngine(genome, SimulationConfig(particle_count=40, seed=23))
        reference = engine.state.copy()
        spatial = SpatialHash(1000, 1000, genome.interaction_radius)
        spatial.rebuild(reference.positions)
        accelerations = np.zeros_like(reference.positions)
        for i in range(reference.count):
            for j in spatial.neighbors(reference.positions[i]):
                if i != j:
                    accelerations[i] += pair_force(reference.positions[i], reference.positions[j], int(reference.species[i]), int(reference.species[j]), genome, 1000, 1000)
        reference.velocities *= genome.damping
        reference.velocities += accelerations * genome.dt
        reference.positions += reference.velocities * genome.dt
        reference.positions %= 1000
        engine.step()
        np.testing.assert_allclose(engine.state.positions, reference.positions, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(engine.state.velocities, reference.velocities, rtol=1e-12, atol=1e-12)

    def test_reset_is_deterministic(self):
        engine = SimulationEngine(config=SimulationConfig(particle_count=50, seed=21))
        initial = engine.state.copy()
        engine.step(10)
        engine.reset()
        np.testing.assert_array_equal(initial.positions, engine.state.positions)
        np.testing.assert_array_equal(initial.velocities, engine.state.velocities)


if __name__ == "__main__":
    unittest.main()
