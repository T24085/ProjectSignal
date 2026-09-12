import tempfile
import unittest
from pathlib import Path

from signal_lab.experiment.protocol import ExperimentSpec, capture_reference, load_experiment, run_experiment, save_experiment
from signal_lab.physics.engine import SimulationConfig, SimulationEngine


class ExperimentProtocolTests(unittest.TestCase):
    def test_three_branches_run_from_one_reference_and_persist(self):
        engine = SimulationEngine(config=SimulationConfig(particle_count=24, seed=19))
        reference = capture_reference(engine)
        spec = ExperimentSpec(experiment_id="EXP_PROTOCOL_TEST", total_steps=6, measurement_interval=2, target_type="CUSTOM REGION", custom_x=500.0, custom_y=500.0, target_radius=20.0)

        result = run_experiment(spec, reference)

        self.assertEqual(set(result.branches), {"CONTROL", "BIT-0", "BIT-1"})
        self.assertTrue(all(branch.final_step == 6 for branch in result.branches.values()))
        self.assertTrue(all(branch.measurements for branch in result.branches.values()))
        self.assertEqual(reference.step, 0)
        with tempfile.TemporaryDirectory() as directory:
            destination = save_experiment(result, directory)
            loaded_spec, loaded_reference, _payload = load_experiment(destination)
            self.assertEqual(loaded_spec.experiment_id, "EXP_PROTOCOL_TEST")
            self.assertEqual(loaded_reference.state.count, 24)
            self.assertTrue((Path(destination) / "control_measurements.csv").exists())
            self.assertTrue((Path(destination) / "preview.png").exists())

    def test_equal_magnitude_alternative_angles_are_required(self):
        spec = ExperimentSpec(bit0_angle=10.0, bit1_angle=20.0)
        with self.assertRaises(ValueError):
            spec.validate()


if __name__ == "__main__":
    unittest.main()
