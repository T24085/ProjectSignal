import csv
import json
from pathlib import Path
import tempfile
import unittest
from threading import Event

from signal_lab.experiment.clustering import cluster_classification
from signal_lab.experiment.config import load_structure_config
from signal_lab.search.runner import SearchConfig, load_baseline_rows, run_search


class BaselineTests(unittest.TestCase):
    def test_detector_thresholds_and_macro_classification_are_configured(self):
        config = load_structure_config()
        self.assertEqual(config.primary_min_particle_count, 20)
        self.assertEqual(config.primary_max_particle_count, 300)
        self.assertEqual(config.primary_min_age_steps, 500)
        self.assertAlmostEqual(config.primary_min_structure_score, 0.70)
        self.assertAlmostEqual(config.primary_min_cohesion, 0.60)
        self.assertAlmostEqual(config.primary_min_identity, 0.70)
        self.assertAlmostEqual(config.primary_min_shape, 0.50)
        self.assertAlmostEqual(config.primary_min_dynamics, 0.20)
        self.assertAlmostEqual(config.primary_max_bridge_fraction, 0.20)
        self.assertEqual(config.size_class(2), "MICRO_CLUSTER")
        self.assertEqual(config.size_class(9), "MICRO_CLUSTER")
        self.assertEqual(config.size_class(10), "SMALL_STRUCTURE")
        self.assertEqual(config.size_class(49), "SMALL_STRUCTURE")
        self.assertEqual(config.size_class(50), "MEDIUM_STRUCTURE")
        self.assertEqual(config.size_class(149), "MEDIUM_STRUCTURE")
        self.assertEqual(config.size_class(150), "LARGE_STRUCTURE")
        self.assertEqual(config.size_class(300), "LARGE_STRUCTURE")
        self.assertEqual(config.size_class(301), "MACRO_CLUSTER")
        self.assertEqual(cluster_classification(500, 300), "PERSISTENT")
        self.assertEqual(cluster_classification(500, 301), "MACRO_CLUSTER")

    def test_baseline_writes_all_requested_outputs_and_correlation_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SearchConfig(
                runs=2,
                steps=20,
                particle_count=30,
                result_root=Path(directory),
                minimum_cluster_size=3,
                observation_interval=10,
            )
            run_search(config, progress=lambda _message: None)
            baseline = config.baseline_dir
            self.assertTrue((baseline / "experiment_001_baseline.sqlite3").exists())
            self.assertTrue((baseline / "experiment_001_baseline.csv").exists())
            self.assertTrue((baseline / "experiment_001_baseline.json").exists())
            self.assertTrue((baseline / "experiment_001_summary.md").exists())
            with (baseline / "experiment_001_baseline.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            payload = json.loads((baseline / "experiment_001_baseline.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["genomes"]), 2)
            self.assertEqual(len(payload["correlations"]), 9)
            self.assertIn("network_structures", payload["genomes"][0])
            self.assertIn("network_candidates", payload["genomes"][0])
            self.assertEqual(len(load_baseline_rows(Path(directory))), 2)

    def test_stop_commits_completed_genomes_without_partial_row(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Event()

            def progress(message: str) -> None:
                if message.startswith("GENOME 1"):
                    stop.set()

            config = SearchConfig(
                runs=3,
                steps=20,
                particle_count=30,
                result_root=Path(directory),
                minimum_cluster_size=3,
                observation_interval=10,
                stop_event=stop,
            )
            run_search(config, progress=progress)
            rows = load_baseline_rows(Path(directory))
            self.assertEqual([int(row["run"]) for row in rows], [1])
            self.assertTrue(all(int(row["simulation_steps"]) == 20 for row in rows))


if __name__ == "__main__":
    unittest.main()
