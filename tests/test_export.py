from __future__ import annotations

from collections import deque
import importlib.util
from pathlib import Path
import tempfile
import unittest

from signal_lab.experiment.clustering import ClusterTracker
from signal_lab.physics.engine import SimulationConfig, SimulationEngine
from signal_lab.storage.export import collect_export_data, export_all


class ExportTests(unittest.TestCase):
    def _data(self):
        engine = SimulationEngine(config=SimulationConfig(particle_count=24, seed=4))
        tracker = ClusterTracker(engine.config.width, engine.config.height, engine.genome.interaction_radius, min_cluster_size=3)
        tracking = tracker.update(engine.state, 0)
        history = {"Average Speed": deque([0.1, 0.2]), "Cluster Count": deque([len(tracking.clusters), len(tracking.clusters)])}
        return engine, collect_export_data(engine, tracking.clusters, history, ["[000000] Smoke export"])

    def test_export_bundle_contains_portable_formats(self) -> None:
        engine, data = self._data()
        self.assertEqual(data["summary"]["particle_count"], engine.state.count)
        with tempfile.TemporaryDirectory() as directory:
            exported = export_all(data, Path(directory))
            self.assertTrue(Path(exported["pdf"]).exists())
            self.assertTrue(Path(exported["json"]).exists())
            self.assertEqual(len(exported["csv"]), 9)
            if importlib.util.find_spec("openpyxl"):
                self.assertTrue(Path(exported["xlsx"]).exists())
            else:
                self.assertIn("xlsx_error", exported)

