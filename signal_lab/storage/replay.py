"""Replayable snapshots for persistent structures."""

from __future__ import annotations

import json
from pathlib import Path

from signal_lab.experiment.clustering import ClusterObservation
from signal_lab.experiment.networks import NetworkObservation
from signal_lab.physics.engine import SimulationConfig, SimulationEngine
from signal_lab.physics.genome import Genome
from signal_lab.physics.particle import ParticleState


def save_structure_snapshot(engine: SimulationEngine, cluster: ClusterObservation, root: str | Path = "results/structures") -> Path:
    """Save genome, state, configuration, and raw cluster history without overwriting."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    base = root_path / f"structure_C{cluster.cluster_id}_step_{engine.step_count}"
    destination = base
    suffix = 2
    while destination.exists():
        destination = root_path / f"{base.name}_{suffix}"
        suffix += 1
    destination.mkdir()
    engine.genome.save(destination / "genome.json")
    engine.state.save(destination / "particle_state.npz")
    experiment = {
        "software_version": "0.1.0",
        "seed": engine.config.seed,
        "step": engine.step_count,
        "world": {"width": engine.config.width, "height": engine.config.height},
        "particle_count": engine.state.count,
        "cluster": cluster.to_dict(),
    }
    (destination / "structure.json").write_text(json.dumps(experiment, indent=2) + "\n", encoding="utf-8")
    return destination


def load_structure_snapshot(path: str | Path) -> tuple[SimulationEngine, dict[str, object]]:
    """Load a snapshot into a deterministic engine and return its metadata."""
    directory = Path(path)
    metadata = json.loads((directory / "structure.json").read_text(encoding="utf-8"))
    genome = Genome.load(directory / "genome.json")
    state = ParticleState.load(directory / "particle_state.npz")
    world = metadata.get("world", {})
    config = SimulationConfig(float(world.get("width", 1000.0)), float(world.get("height", 1000.0)), state.count, int(metadata.get("seed", 0)))
    engine = SimulationEngine(genome, config)
    engine.replace_state(state, int(metadata.get("step", 0)))
    return engine, metadata


def save_network_snapshot(engine: SimulationEngine, network: NetworkObservation, root: str | Path = "results/networks") -> Path:
    """Save a complete replay plus the network topology and raw metrics."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    base = root_path / f"network_N{network.network_id}_step_{engine.step_count}"
    destination = base
    suffix = 2
    while destination.exists():
        destination = root_path / f"{base.name}_{suffix}"
        suffix += 1
    destination.mkdir()
    engine.genome.save(destination / "genome.json")
    engine.state.save(destination / "particle_state.npz")
    metadata = {
        "software_version": "0.1.0",
        "seed": engine.config.seed,
        "step": engine.step_count,
        "world": {"width": engine.config.width, "height": engine.config.height},
        "particle_count": engine.state.count,
        "network": network.to_dict(),
    }
    (destination / "network.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return destination


def load_network_snapshot(path: str | Path) -> tuple[SimulationEngine, dict[str, object]]:
    """Load a saved network replay and return its metadata."""
    directory = Path(path)
    metadata = json.loads((directory / "network.json").read_text(encoding="utf-8"))
    genome = Genome.load(directory / "genome.json")
    state = ParticleState.load(directory / "particle_state.npz")
    world = metadata.get("world", {})
    config = SimulationConfig(float(world.get("width", 1000.0)), float(world.get("height", 1000.0)), state.count, int(metadata.get("seed", 0)))
    engine = SimulationEngine(genome, config)
    engine.replace_state(state, int(metadata.get("step", 0)))
    return engine, metadata
