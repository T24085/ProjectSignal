"""Append-safe, non-optimizing baseline search for Project SIGNAL Experiment 001."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
import sqlite3
from threading import Event
import time
from typing import Any, Callable

import numpy as np

from signal_lab.experiment.clustering import ClusterObservation, ClusterTracker
from signal_lab.experiment.config import StructureDetectionConfig, load_structure_config
from signal_lab.experiment.networks import NetworkObservation, NetworkTracker
from signal_lab.physics.engine import SimulationConfig, SimulationEngine
from signal_lab.physics.genome import Genome
from signal_lab.storage.replay import save_network_snapshot, save_structure_snapshot


ProgressCallback = Callable[[str], None]
ProgressStateCallback = Callable[[dict[str, object]], None]


@dataclass
class SearchConfig:
    runs: int = 1000
    steps: int = 5000
    particle_count: int = 1000
    species_count: int = 3
    minimum_cluster_size: int | None = None
    observation_interval: int | None = None
    tracker_persistence_threshold: float | None = None
    seed: int = 0
    result_root: Path = Path("results")
    experiment_id: str = "experiment_001"
    stop_event: Event | None = None
    structure_config: StructureDetectionConfig = field(default_factory=load_structure_config)

    def __post_init__(self) -> None:
        self.result_root = Path(self.result_root)
        if self.minimum_cluster_size is None:
            self.minimum_cluster_size = self.structure_config.minimum_cluster_size
        if self.observation_interval is None:
            self.observation_interval = self.structure_config.observation_interval
        if self.tracker_persistence_threshold is None:
            self.tracker_persistence_threshold = self.structure_config.tracker_persistence_threshold
        if self.runs < 1 or self.steps < 1 or self.particle_count < 1 or self.species_count < 1:
            raise ValueError("runs, steps, particle_count, and species_count must be positive")
        if self.minimum_cluster_size < 1 or self.observation_interval < 1:
            raise ValueError("minimum_cluster_size and observation_interval must be positive")

    @property
    def baseline_dir(self) -> Path:
        return self.result_root / f"{self.experiment_id}_baseline"


@dataclass
class SearchRunResult:
    run: int
    genome_hash: str
    genome_seed: int
    seed: int
    clusters_detected: int
    persistent_structures: int
    macro_clusters: int
    best_structure: ClusterObservation | None
    best_candidate: ClusterObservation | None
    candidate_path: Path | None
    status: str = "COMPLETED"
    simulation_steps: int = 0
    error: str = ""
    network_structures: int = 0
    network_candidates: int = 0
    best_network: NetworkObservation | None = None
    network_path: Path | None = None


BASELINE_FIELDS = (
    "experiment_id", "run", "status", "genome_hash", "genome_seed", "seed", "species_count",
    "interaction_matrix", "interaction_radius", "core_radius_ratio", "core_repulsion", "force_gain",
    "damping", "dt", "simulation_steps", "clusters_detected", "persistent_clusters", "macro_clusters",
    "best_structure_score", "best_structure_lifetime", "best_candidate_particle_count", "best_candidate_lifetime",
    "best_cohesion", "best_identity", "best_shape", "best_dynamics", "best_bridge_fraction", "candidate",
    "candidate_path", "network_structures", "network_candidates", "best_network_score", "best_network_node_count",
    "best_network_edge_count", "best_network_lifetime", "network_path", "error",
)


def _database_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS genomes (
            run INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL, status TEXT NOT NULL,
            genome_hash TEXT NOT NULL, genome_seed INTEGER NOT NULL, seed INTEGER NOT NULL,
            species_count INTEGER NOT NULL, interaction_matrix TEXT NOT NULL,
            interaction_radius REAL NOT NULL, core_radius_ratio REAL NOT NULL, core_repulsion REAL NOT NULL,
            force_gain REAL NOT NULL, damping REAL NOT NULL, dt REAL NOT NULL, simulation_steps INTEGER NOT NULL,
            clusters_detected INTEGER NOT NULL, persistent_clusters INTEGER NOT NULL, macro_clusters INTEGER NOT NULL,
            best_structure_score REAL, best_structure_lifetime INTEGER, best_candidate_particle_count INTEGER,
            best_candidate_lifetime INTEGER, best_cohesion REAL, best_identity REAL, best_shape REAL,
            best_dynamics REAL, best_bridge_fraction REAL, candidate INTEGER NOT NULL, candidate_path TEXT, error TEXT
        )
        """
    )
    connection.execute("CREATE TABLE IF NOT EXISTS baseline_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    existing_columns = {row[1] for row in connection.execute("PRAGMA table_info(genomes)")}
    additions = {
        "network_structures": "INTEGER",
        "network_candidates": "INTEGER",
        "best_network_score": "REAL",
        "best_network_node_count": "INTEGER",
        "best_network_edge_count": "INTEGER",
        "best_network_lifetime": "INTEGER",
        "network_path": "TEXT",
    }
    for name, column_type in additions.items():
        if name not in existing_columns:
            connection.execute(f"ALTER TABLE genomes ADD COLUMN {name} {column_type}")


def _config_signature(config: SearchConfig) -> dict[str, object]:
    return {
        "experiment_id": config.experiment_id, "runs": config.runs, "steps": config.steps,
        "particle_count": config.particle_count, "species_count": config.species_count,
        "minimum_cluster_size": config.minimum_cluster_size, "observation_interval": config.observation_interval,
        "tracker_persistence_threshold": config.tracker_persistence_threshold,
        "seed": config.seed, "structure_detection": config.structure_config.to_dict(),
    }


def _open_baseline(config: SearchConfig) -> sqlite3.Connection:
    config.baseline_dir.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(config.baseline_dir / "experiment_001_baseline.sqlite3")
    _database_schema(database)
    signature = json.dumps(_config_signature(config), sort_keys=True)
    row = database.execute("SELECT value FROM baseline_metadata WHERE key = 'config'").fetchone()
    if row is None:
        database.execute("INSERT INTO baseline_metadata(key, value) VALUES('config', ?)", (signature,))
        database.commit()
    elif row[0] != signature:
        database.close()
        raise FileExistsError("Experiment 001 already exists with a different configuration; it will not be overwritten.")
    return database


def _row_from_result(config: SearchConfig, genome: Genome, result: SearchRunResult) -> dict[str, object]:
    best = result.best_structure
    candidate = result.best_candidate
    return {
        "experiment_id": config.experiment_id, "run": result.run, "status": result.status,
        "genome_hash": result.genome_hash, "genome_seed": result.genome_seed, "seed": result.seed,
        "species_count": genome.species_count,
        "interaction_matrix": json.dumps(genome.interaction_matrix.tolist(), separators=(",", ":")),
        "interaction_radius": genome.interaction_radius, "core_radius_ratio": genome.core_radius_ratio,
        "core_repulsion": genome.core_repulsion, "force_gain": genome.force_gain, "damping": genome.damping,
        "dt": genome.dt, "simulation_steps": result.simulation_steps, "clusters_detected": result.clusters_detected,
        "persistent_clusters": result.persistent_structures, "macro_clusters": result.macro_clusters,
        "best_structure_score": best.structure_score if best else None,
        "best_structure_lifetime": best.age_steps if best else None,
        "best_candidate_particle_count": candidate.particle_count if candidate else None,
        "best_candidate_lifetime": candidate.age_steps if candidate else None,
        "best_cohesion": candidate.cohesion_score if candidate else None,
        "best_identity": candidate.identity_score if candidate else None,
        "best_shape": candidate.shape_score if candidate else None,
        "best_dynamics": candidate.dynamic_score if candidate else None,
        "best_bridge_fraction": candidate.bridge_fraction if candidate else None,
        "candidate": int(candidate is not None), "candidate_path": str(result.candidate_path) if result.candidate_path else "",
        "network_structures": result.network_structures,
        "network_candidates": result.network_candidates,
        "best_network_score": result.best_network.network_score if result.best_network else None,
        "best_network_node_count": result.best_network.node_count if result.best_network else None,
        "best_network_edge_count": result.best_network.edge_count if result.best_network else None,
        "best_network_lifetime": result.best_network.age_steps if result.best_network else None,
        "network_path": str(result.network_path) if result.network_path else "",
        "error": result.error,
    }


def _insert_row(connection: sqlite3.Connection, row: dict[str, object]) -> None:
    columns = ", ".join(BASELINE_FIELDS)
    placeholders = ", ".join("?" for _ in BASELINE_FIELDS)
    connection.execute(f"INSERT OR IGNORE INTO genomes ({columns}) VALUES ({placeholders})", tuple(row.get(field) for field in BASELINE_FIELDS))
    connection.commit()


def _all_rows(connection: sqlite3.Connection) -> list[dict[str, object]]:
    connection.row_factory = sqlite3.Row
    return [dict(row) for row in connection.execute("SELECT * FROM genomes ORDER BY run")]


def load_baseline_rows(result_root: Path | str = Path("results"), experiment_id: str = "experiment_001") -> list[dict[str, object]]:
    """Read the preserved baseline database for the Results tab."""
    database_path = Path(result_root) / f"{experiment_id}_baseline" / "experiment_001_baseline.sqlite3"
    if not database_path.exists():
        return []
    connection = sqlite3.connect(database_path)
    try:
        _database_schema(connection)
        return _all_rows(connection)
    finally:
        connection.close()


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _correlations(rows: list[dict[str, object]], species_count: int) -> list[dict[str, object]]:
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    if not completed:
        return []
    matrices = np.asarray([json.loads(str(row["interaction_matrix"])) for row in completed], dtype=np.float64)
    structure = np.asarray([float(row["best_structure_score"] or 0.0) for row in completed])
    candidates = np.asarray([float(row["candidate"]) for row in completed])
    cluster_count = np.asarray([float(row["clusters_detected"]) for row in completed])
    output: list[dict[str, object]] = []
    for row_index in range(species_count):
        for column_index in range(species_count):
            values = matrices[:, row_index, column_index]
            output.append({
                "matrix_element": f"M[{row_index},{column_index}]", "from_species": row_index, "to_species": column_index,
                "structure_score_correlation": _safe_correlation(values, structure),
                "candidate_probability_correlation": _safe_correlation(values, candidates),
                "cluster_count_correlation": _safe_correlation(values, cluster_count),
            })
    return output


def _summary(rows: list[dict[str, object]], correlations: list[dict[str, object]], config: SearchConfig) -> dict[str, object]:
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    failed = [row for row in rows if row["status"] == "FAILED"]
    with_clusters = [row for row in completed if int(row["clusters_detected"]) > 0]
    with_persistent = [row for row in completed if int(row["persistent_clusters"]) > 0]
    with_candidates = [row for row in completed if int(row["candidate"]) == 1]
    with_macro = [row for row in completed if int(row["macro_clusters"]) > 0]
    scores = np.asarray([float(row["best_structure_score"]) for row in completed if row["best_structure_score"] is not None], dtype=np.float64)
    sizes = np.asarray([int(row["best_candidate_particle_count"]) for row in with_candidates if row["best_candidate_particle_count"] is not None], dtype=np.float64)
    scored = [row for row in completed if row["best_structure_score"] is not None]
    lived = [row for row in completed if row["best_structure_lifetime"] is not None]
    highest = max(scored, key=lambda row: float(row["best_structure_score"]), default=None)
    longest = max(lived, key=lambda row: int(row["best_structure_lifetime"]), default=None)
    network_structures = sum(int(row.get("network_structures") or 0) for row in completed)
    network_candidates = sum(int(row.get("network_candidates") or 0) for row in completed)
    return {
        "experiment_id": config.experiment_id, "status": "COMPLETE" if len(completed) >= config.runs else "PARTIAL",
        "total_genomes_tested": len(rows), "completed_genomes": len(completed), "failed_genomes": len(failed),
        "total_simulation_steps": int(sum(int(row["simulation_steps"]) for row in completed)),
        "genomes_producing_clusters": len(with_clusters), "genomes_producing_persistent_clusters": len(with_persistent),
        "genomes_producing_primary_candidates": len(with_candidates), "candidate_rate": len(with_candidates) / max(len(rows), 1),
        "macro_cluster_rate": len(with_macro) / max(len(rows), 1),
        "network_structures": network_structures, "network_candidates": network_candidates,
        "mean_structure_score": float(np.mean(scores)) if len(scores) else None,
        "median_structure_score": float(np.median(scores)) if len(scores) else None,
        "max_structure_score": float(np.max(scores)) if len(scores) else None,
        "mean_candidate_size": float(np.mean(sizes)) if len(sizes) else None,
        "longest_lived_structure": int(longest["best_structure_lifetime"]) if longest and longest["best_structure_lifetime"] is not None else None,
        "longest_lived_run": int(longest["run"]) if longest else None,
        "highest_scoring_structure": float(highest["best_structure_score"]) if highest and highest["best_structure_score"] is not None else None,
        "highest_scoring_run": int(highest["run"]) if highest else None,
        "completed_runs": [int(row["run"]) for row in completed], "failed_runs": [int(row["run"]) for row in failed],
        "correlation_count": len(correlations),
    }


def _write_baseline_outputs(config: SearchConfig, rows: list[dict[str, object]]) -> None:
    baseline = config.baseline_dir
    correlations = _correlations(rows, config.species_count)
    summary = _summary(rows, correlations, config)
    csv_path = baseline / "experiment_001_baseline.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASELINE_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in BASELINE_FIELDS} for row in rows)
    payload = {"experiment": _config_signature(config), "detector_thresholds": config.structure_config.to_dict(), "summary": summary, "correlations": correlations, "genomes": rows}
    (baseline / "experiment_001_baseline.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [
        "# Project SIGNAL Experiment 001 Baseline Summary", "", f"Status: {summary['status']}", "",
        "## Detector calibration", "", "Thresholds are loaded from configs/default.toml. Correlations are descriptive associations only and must not be interpreted as causation.", "",
        "## Baseline results", "",
    ]
    for key in ("total_genomes_tested", "completed_genomes", "failed_genomes", "total_simulation_steps", "genomes_producing_clusters", "genomes_producing_persistent_clusters", "genomes_producing_primary_candidates", "candidate_rate", "macro_cluster_rate", "network_structures", "network_candidates", "mean_structure_score", "median_structure_score", "max_structure_score", "mean_candidate_size", "longest_lived_structure", "highest_scoring_structure"):
        value = summary[key]
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"- {key.replace('_', ' ').title()}: {rendered}")
    lines.extend(["", "## Highest-scoring and longest-lived runs", "", f"- Highest-scoring run: {summary['highest_scoring_run']}", f"- Longest-lived run: {summary['longest_lived_run']}", "", "## Matrix correlations", "", "Correlation does not imply causation. Values are Pearson correlations across completed genomes.", "", "| Matrix element | Structure score | Candidate probability | Cluster count |", "|---|---:|---:|---:|"])
    for item in correlations:
        def value_text(value: object) -> str:
            return "n/a" if value is None else f"{float(value):.4f}"
        lines.append(f"| {item['matrix_element']} | {value_text(item['structure_score_correlation'])} | {value_text(item['candidate_probability_correlation'])} | {value_text(item['cluster_count_correlation'])} |")
    (baseline / "experiment_001_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _progress_state(rows: list[dict[str, object]], config: SearchConfig, elapsed: float, total_steps: int) -> dict[str, object]:
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    candidates = sum(int(row["candidate"]) for row in completed)
    scores = [float(row["best_structure_score"]) for row in completed if row["best_structure_score"] is not None]
    completed_runs = len(completed)
    return {
        "completed_runs": completed_runs, "total_runs": config.runs, "percent": 100.0 * completed_runs / max(config.runs, 1),
        "simulation_steps_completed": total_steps, "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": (elapsed / completed_runs) * max(config.runs - completed_runs, 0) if completed_runs else 0.0,
        "candidates_discovered": candidates, "best_structure_score": max(scores, default=0.0),
    }


def run_search(config: SearchConfig, progress: ProgressCallback = print, progress_state: ProgressStateCallback | None = None) -> list[SearchRunResult]:
    """Run or resume Experiment 001 with independently seeded random genomes."""
    database = _open_baseline(config)
    started = time.monotonic()
    existing_rows = _all_rows(database)
    completed_runs = {int(row["run"]) for row in existing_rows}
    total_steps = sum(int(row["simulation_steps"]) for row in existing_rows if row["status"] == "COMPLETED")
    if progress_state:
        progress_state(_progress_state(existing_rows, config, 0.0, total_steps))
    results: list[SearchRunResult] = []
    structures_root = config.baseline_dir / "structures"
    for run_number in range(1, config.runs + 1):
        if run_number in completed_runs:
            continue
        if config.stop_event is not None and config.stop_event.is_set():
            break
        seed_sequence = np.random.SeedSequence([config.seed, run_number])
        children = seed_sequence.spawn(2)
        genome_seed = int(np.random.default_rng(children[0]).integers(0, 2**63 - 1))
        universe_seed = int(np.random.default_rng(children[1]).integers(0, 2**63 - 1))
        genome = Genome.random(config.species_count, seed=genome_seed)
        try:
            engine = SimulationEngine(genome, SimulationConfig(particle_count=config.particle_count, seed=universe_seed))
            tracker = ClusterTracker(engine.config.width, engine.config.height, genome.interaction_radius, config.minimum_cluster_size, config.structure_config.minimum_membership_overlap)
            network_tracker = NetworkTracker(engine.config.width, engine.config.height, genome.interaction_radius, config.structure_config)
            all_clusters: dict[int, ClusterObservation] = {}
            all_networks: dict[int, NetworkObservation] = {}
            best_candidate: ClusterObservation | None = None
            best_candidate_state = None
            best_candidate_step = 0
            best_network: NetworkObservation | None = None
            best_network_state = None
            best_network_step = 0
            stopped = False
            for step in range(0, config.steps + 1, config.observation_interval):
                if config.stop_event is not None and config.stop_event.is_set():
                    stopped = True
                    break
                tracking = tracker.update(engine.state, step)
                networks = network_tracker.update(engine.state, tracking.clusters, step)
                for cluster in tracking.clusters:
                    all_clusters[cluster.cluster_id] = cluster
                    if config.structure_config.is_primary_candidate(cluster) and (best_candidate is None or cluster.structure_score > best_candidate.structure_score):
                        best_candidate = cluster
                        best_candidate_state = engine.state.copy()
                        best_candidate_step = engine.step_count
                for network in networks:
                    all_networks[network.network_id] = network
                    if network.is_candidate and (best_network is None or network.network_score > best_network.network_score):
                        best_network = network
                        best_network_state = engine.state.copy()
                        best_network_step = engine.step_count
                if step < config.steps:
                    engine.step(min(config.observation_interval, config.steps - step))
            if stopped:
                break
            all_clusters.update(tracker.completed)
            all_clusters.update(tracker.active)
            all_networks.update(network_tracker.completed)
            all_networks.update(network_tracker.active)
            structures = list(all_clusters.values())
            best_structure = max(structures, key=lambda item: (item.structure_score, item.age_steps, item.particle_count), default=None)
            persistent = [cluster for cluster in structures if cluster.classification in {"PERSISTENT", "LONG_LIVED"} and cluster.tracker_persistence_score >= config.tracker_persistence_threshold]
            macros = [cluster for cluster in structures if cluster.classification == "MACRO_CLUSTER"]
            candidate_path = None
            if best_candidate is not None and best_candidate_state is not None:
                candidate_engine = SimulationEngine(genome, SimulationConfig(particle_count=config.particle_count, seed=universe_seed))
                candidate_engine.replace_state(best_candidate_state, best_candidate_step)
                candidate_path = save_structure_snapshot(candidate_engine, best_candidate, structures_root)
            best_network_candidate = best_network
            best_network_path = None
            if best_network_candidate is not None and best_network_state is not None:
                network_engine = SimulationEngine(genome, SimulationConfig(particle_count=config.particle_count, seed=universe_seed))
                network_engine.replace_state(best_network_state, best_network_step)
                best_network_path = save_network_snapshot(network_engine, best_network_candidate, config.result_root / "networks")
            network_structures = sum(network.classification == "NETWORK_STRUCTURE" for network in all_networks.values())
            network_candidates = sum(network.is_candidate for network in all_networks.values())
            best_network_overall = max(all_networks.values(), key=lambda network: (network.network_score, network.age_steps), default=None)
            result = SearchRunResult(run_number, genome.genome_hash, genome_seed, universe_seed, len(structures), len(persistent), len(macros), best_structure, best_candidate, candidate_path, simulation_steps=config.steps, network_structures=network_structures, network_candidates=network_candidates, best_network=best_network_overall, network_path=best_network_path)
        except Exception as error:  # preserve failure metadata without particle histories
            result = SearchRunResult(run_number, genome.genome_hash, genome_seed, universe_seed, 0, 0, 0, None, None, None, status="FAILED", error=str(error))
        row = _row_from_result(config, genome, result)
        _insert_row(database, row)
        results.append(result)
        existing_rows = _all_rows(database)
        total_steps = sum(int(item["simulation_steps"]) for item in existing_rows if item["status"] == "COMPLETED")
        progress(f"GENOME {run_number} / {config.runs}")
        progress(f"Structure score: {float(row['best_structure_score'] or 0.0):.4f}   Candidate: {'YES' if row['candidate'] else 'NO'}")
        if progress_state:
            progress_state(_progress_state(existing_rows, config, time.monotonic() - started, total_steps))
    existing_rows = _all_rows(database)
    _write_baseline_outputs(config, existing_rows)
    database.close()
    if progress_state:
        progress_state(_progress_state(existing_rows, config, time.monotonic() - started, total_steps))
    return results
