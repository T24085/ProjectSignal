"""Headless persistent-structure search for Phase 2A."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np

from signal_lab.experiment.clustering import ClusterObservation, ClusterTracker, lifetime_class
from signal_lab.physics.engine import SimulationConfig, SimulationEngine
from signal_lab.physics.genome import Genome
from signal_lab.storage.replay import save_structure_snapshot


@dataclass
class SearchConfig:
    runs: int = 1000
    steps: int = 5000
    particle_count: int = 1000
    min_cluster_size: int = 20
    persistence_threshold: float = 0.55
    observation_interval: int = 10
    seed: int = 0
    result_root: Path = Path("results")
    stop_event: Event | None = None


@dataclass
class SearchRunResult:
    run: int
    genome_hash: str
    seed: int
    clusters_detected: int
    persistent_structures: int
    best_cluster: ClusterObservation | None
    candidate_path: Path | None


def run_search(config: SearchConfig, progress: Callable[[str], None] = print) -> list[SearchRunResult]:
    """Run deterministic random genomes without importing or rendering the GUI."""
    if config.runs < 1 or config.steps < 1 or config.particle_count < 1:
        raise ValueError("runs, steps, and particle_count must be positive")
    if config.observation_interval < 1:
        raise ValueError("observation_interval must be positive")
    config.result_root.mkdir(parents=True, exist_ok=True)
    output_csv = config.result_root / "search_results.csv"
    write_header = not output_csv.exists()
    rng = np.random.default_rng(config.seed)
    results: list[SearchRunResult] = []
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("run", "genome", "seed", "clusters_detected", "persistent_structures", "best_cluster", "particles", "lifetime", "persistence", "candidate_path"))
        if write_header:
            writer.writeheader()
        for run_number in range(1, config.runs + 1):
            if config.stop_event is not None and config.stop_event.is_set():
                break
            genome_seed = int(rng.integers(0, 2**63 - 1))
            universe_seed = int(rng.integers(0, 2**63 - 1))
            genome = Genome.random(3, seed=genome_seed)
            engine = SimulationEngine(genome, SimulationConfig(particle_count=config.particle_count, seed=universe_seed))
            tracker = ClusterTracker(engine.config.width, engine.config.height, genome.interaction_radius, config.min_cluster_size)
            all_clusters: dict[int, ClusterObservation] = {}
            candidate_path = None
            for step in range(0, config.steps + 1, config.observation_interval):
                tracker_result = tracker.update(engine.state, step)
                for cluster in tracker_result.clusters:
                    all_clusters[cluster.cluster_id] = cluster
                if candidate_path is None:
                    persistent_transition = next((event for event in tracker_result.events if event.kind == "reached" and lifetime_class(event.particle_count) in {"PERSISTENT", "LONG_LIVED"}), None)
                    if persistent_transition is not None:
                        candidate = next((item for item in tracker_result.clusters if item.cluster_id == persistent_transition.cluster_id), None)
                        if candidate is not None:
                            candidate_path = save_structure_snapshot(engine, candidate, config.result_root / "structures")
                if step < config.steps:
                    engine.step(min(config.observation_interval, config.steps - step))
            for cluster in tracker.active.values():
                all_clusters[cluster.cluster_id] = cluster
            best = max(all_clusters.values(), key=lambda item: (item.persistence_score, item.age_steps, item.particle_count), default=None)
            persistent = [cluster for cluster in all_clusters.values() if cluster.classification in {"PERSISTENT", "LONG_LIVED"} and cluster.persistence_score >= config.persistence_threshold]
            if persistent and candidate_path is None:
                selected = max(persistent, key=lambda item: (item.persistence_score, item.age_steps, item.particle_count))
                candidate_path = save_structure_snapshot(engine, selected, config.result_root / "structures")
            progress(f"RUN {run_number}/{config.runs}")
            progress(f"Genome: {genome.genome_hash[:8]}   Seed: {universe_seed}")
            progress(f"Clusters detected: {len(all_clusters)}   Persistent structures: {len(persistent)}")
            if best is not None:
                progress(f"BEST STRUCTURE\nC{best.cluster_id}\nParticles: {best.particle_count}\nLifetime: {best.age_steps} steps\nPersistence: {best.persistence_score:.2f}")
            if candidate_path:
                progress(f"CANDIDATE SAVED\n{candidate_path}")
            writer.writerow({"run": run_number, "genome": genome.genome_hash, "seed": universe_seed, "clusters_detected": len(all_clusters), "persistent_structures": len(persistent), "best_cluster": f"C{best.cluster_id}" if best else "", "particles": best.particle_count if best else "", "lifetime": best.age_steps if best else "", "persistence": f"{best.persistence_score:.6f}" if best else "", "candidate_path": str(candidate_path) if candidate_path else ""})
            handle.flush()
            results.append(SearchRunResult(run_number, genome.genome_hash, universe_seed, len(all_clusters), len(persistent), best, candidate_path))
    return results
