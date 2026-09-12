"""Controlled Experiment-tab protocol for reproducible branch comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
import struct
import time
import zlib
from typing import Callable

import numpy as np

from signal_lab.experiment.clustering import ClusterTracker
from signal_lab.experiment.networks import NetworkTracker
from signal_lab.physics.engine import SimulationConfig, SimulationEngine
from signal_lab.physics.genome import Genome
from signal_lab.physics.particle import ParticleState


BRANCH_NAMES = ("CONTROL", "BIT-0", "BIT-1")
OUTCOMES = ("NO_CLEAR_RESPONSE", "LOCAL_RESPONSE", "LONG_RANGE_RESPONSE", "CHAOTIC_RESPONSE", "REPRODUCIBLE_CANDIDATE")


@dataclass
class MeasurementZone:
    name: str
    inner_multiple: float
    outer_multiple: float
    enabled: bool = True


@dataclass
class ExperimentSpec:
    name: str = "Untitled Experiment"
    experiment_id: str = "EXP_000001"
    reference_step: int = 0
    total_steps: int = 3000
    measurement_interval: int = 10
    warmup_steps: int = 0
    stop_automatically: bool = True
    target_type: str = "NO TARGET"
    target_id: str = ""
    network_id: str = ""
    node_id: str = ""
    custom_x: float = 500.0
    custom_y: float = 500.0
    target_radius: float = 30.0
    perturbation_type: str = "VELOCITY IMPULSE"
    impulse_magnitude: float = 0.5
    bit0_angle: float = -45.0
    bit1_angle: float = 45.0
    injection_step: int = 0
    zones: list[MeasurementZone] = field(default_factory=lambda: [MeasurementZone("Zone A", 1.0, 2.0), MeasurementZone("Zone B", 2.0, 3.0), MeasurementZone("Zone C", 3.0, 4.0), MeasurementZone("Zone D", 4.0, 5.0)])
    repeat_trials: int = 20
    repeat_position_noise_percent: float = 0.5
    repeat_velocity_noise_percent: float = 0.5

    def validate(self) -> None:
        if not self.name.strip() or not self.experiment_id.strip():
            raise ValueError("Experiment name and ID are required")
        if self.total_steps < 1 or self.measurement_interval < 1 or self.warmup_steps < 0:
            raise ValueError("Total steps and measurement interval must be positive; warmup cannot be negative")
        if self.target_radius <= 0 or self.impulse_magnitude < 0:
            raise ValueError("Target radius must be positive and impulse magnitude cannot be negative")
        if self.injection_step < 0 or self.injection_step > self.total_steps:
            raise ValueError("Injection step must be within the experiment window")
        if self.repeat_trials < 1:
            raise ValueError("Trials must be positive")


@dataclass
class ReferenceState:
    genome: Genome
    state: ParticleState
    seed: int
    step: int
    simulation_config: SimulationConfig

    def copy(self) -> "ReferenceState":
        genome = Genome.from_dict(self.genome.to_dict())
        return ReferenceState(genome, self.state.copy(), self.seed, self.step, SimulationConfig(self.simulation_config.width, self.simulation_config.height, self.simulation_config.particle_count, self.simulation_config.seed))


@dataclass
class BranchRun:
    name: str
    measurements: list[dict[str, object]]
    events: list[str]
    final_state: ParticleState
    final_step: int
    average_speed: float = 0.0
    cluster_count: int = 0
    target_integrity: float = 0.0
    largest_response_zone: str = "-"
    structure_score: float = 0.0
    network_score: float | None = None


@dataclass
class ExperimentRun:
    spec: ExperimentSpec
    reference: ReferenceState
    target_center: np.ndarray
    target_particle_ids: frozenset[int]
    branches: dict[str, BranchRun]
    summary: dict[str, object]
    events: list[dict[str, object]]


ProgressCallback = Callable[[str, dict[str, object]], None]


def capture_reference(engine: SimulationEngine) -> ReferenceState:
    """Take a deep immutable-by-convention copy of the complete simulator state."""
    genome = Genome.from_dict(engine.genome.to_dict())
    simulation_config = SimulationConfig(engine.config.width, engine.config.height, engine.state.count, engine.config.seed)
    return ReferenceState(genome, engine.state.copy(), engine.config.seed, engine.step_count, simulation_config)


def discover_targets(reference: ReferenceState) -> dict[str, object]:
    """Discover cluster and network choices from a reference without mutating it."""
    cluster_tracker = ClusterTracker(reference.simulation_config.width, reference.simulation_config.height, reference.genome.interaction_radius, min_cluster_size=5)
    clusters = cluster_tracker.update(reference.state, reference.step).clusters
    network_tracker = NetworkTracker(reference.simulation_config.width, reference.simulation_config.height, reference.genome.interaction_radius)
    networks = network_tracker.update(reference.state, clusters, reference.step)
    return {"clusters": clusters, "networks": networks}


def _target(reference: ReferenceState, spec: ExperimentSpec) -> tuple[np.ndarray, frozenset[int]]:
    discovered = discover_targets(reference)
    if spec.target_type == "CLUSTER" or spec.target_type == "STRUCTURE":
        cluster = next((item for item in discovered["clusters"] if str(item.cluster_id) == str(spec.target_id)), None)
        if cluster is not None:
            return cluster.centroid.copy(), cluster.particle_ids
    if spec.target_type == "NETWORK NODE":
        network = next((item for item in discovered["networks"] if str(item.network_id) == str(spec.network_id)), None)
        if network is not None:
            node = next((item for item in network.nodes if str(item.node_id) == str(spec.node_id)), None)
            if node is not None:
                return node.centroid.copy(), node.particle_ids
    if spec.target_type == "CUSTOM REGION":
        return np.asarray([spec.custom_x, spec.custom_y], dtype=np.float64), frozenset()
    return np.asarray([reference.simulation_config.width / 2.0, reference.simulation_config.height / 2.0], dtype=np.float64), frozenset()


def _distance_from_center(positions: np.ndarray, center: np.ndarray, width: float, height: float) -> np.ndarray:
    size = np.asarray([width, height], dtype=np.float64)
    delta = (positions - center + size / 2.0) % size - size / 2.0
    return np.linalg.norm(delta, axis=1)


def _spatial_entropy(positions: np.ndarray, width: float, height: float, bins: int = 8) -> float:
    if not len(positions):
        return 0.0
    grid, _, _ = np.histogram2d(positions[:, 0], positions[:, 1], bins=bins, range=((0, width), (0, height)))
    probabilities = grid.ravel() / len(positions)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log(probabilities)))


def _branch_measurement(engine: SimulationEngine, branch: str, step: int, zone: MeasurementZone, target_center: np.ndarray, target_ids: frozenset[int], cluster_count: int, structure_score: float, network_score: float | None) -> dict[str, object]:
    distances = _distance_from_center(engine.state.positions, target_center, engine.config.width, engine.config.height)
    radius = engine.genome.interaction_radius
    selection = (distances >= zone.inner_multiple * radius) & (distances < zone.outer_multiple * radius)
    indices = np.flatnonzero(selection)
    velocities = engine.state.velocities[indices]
    speeds = np.linalg.norm(velocities, axis=1)
    species_ratios = np.bincount(engine.state.species[indices], minlength=engine.genome.species_count).astype(np.float64)
    species_ratios /= max(float(len(indices)), 1.0)
    relative_position = engine.state.positions - target_center
    size = np.asarray([engine.config.width, engine.config.height], dtype=np.float64)
    relative_position = (relative_position + size / 2.0) % size - size / 2.0
    angular_momentum = float(np.sum(relative_position[indices, 0] * velocities[:, 1] - relative_position[indices, 1] * velocities[:, 0])) if len(indices) else 0.0
    target_count = sum(int(value in target_ids) for value in engine.state.ids) if target_ids else 0
    return {
        "branch": branch,
        "step": int(step),
        "zone": zone.name,
        "inner_radius": zone.inner_multiple * radius,
        "outer_radius": zone.outer_multiple * radius,
        "density": float(len(indices) / max(np.pi * ((zone.outer_multiple * radius) ** 2 - (zone.inner_multiple * radius) ** 2), 1e-12)),
        "particle_count": int(len(indices)),
        "average_velocity_x": float(np.mean(velocities[:, 0])) if len(indices) else 0.0,
        "average_velocity_y": float(np.mean(velocities[:, 1])) if len(indices) else 0.0,
        "average_speed": float(np.mean(speeds)) if len(indices) else 0.0,
        "speed_variance": float(np.var(speeds)) if len(indices) else 0.0,
        "species_ratios": json.dumps(species_ratios.tolist(), separators=(",", ":")),
        "spatial_entropy": _spatial_entropy(engine.state.positions[indices], engine.config.width, engine.config.height),
        "cluster_count": int(cluster_count),
        "angular_momentum": angular_momentum,
        "target_particle_count": int(target_count),
        "structure_score": float(structure_score),
        "network_score": network_score,
    }


def _apply_velocity_impulse(engine: SimulationEngine, target_center: np.ndarray, target_ids: frozenset[int], radius: float, magnitude: float, angle_degrees: float) -> None:
    distances = _distance_from_center(engine.state.positions, target_center, engine.config.width, engine.config.height)
    if target_ids:
        selection = np.asarray([int(value) in target_ids for value in engine.state.ids], dtype=bool) & (distances <= radius)
    else:
        selection = distances <= radius
    direction = np.asarray([np.cos(np.deg2rad(angle_degrees)), np.sin(np.deg2rad(angle_degrees))], dtype=np.float64)
    engine.state.velocities[selection] += direction * magnitude


def run_experiment(spec: ExperimentSpec, reference: ReferenceState, progress: ProgressCallback | None = None, pause_event: object | None = None) -> ExperimentRun:
    """Run CONTROL, BIT-0, and BIT-1 from exact copies of one reference."""
    spec.validate()
    target_center, target_ids = _target(reference, spec)
    branches: dict[str, BranchRun] = {}
    all_events: list[dict[str, object]] = []
    for branch in BRANCH_NAMES:
        genome = Genome.from_dict(reference.genome.to_dict())
        engine = SimulationEngine(genome, reference.simulation_config)
        engine.replace_state(reference.state, reference.step)
        cluster_tracker = ClusterTracker(engine.config.width, engine.config.height, genome.interaction_radius, min_cluster_size=5)
        network_tracker = NetworkTracker(engine.config.width, engine.config.height, genome.interaction_radius)
        measurements: list[dict[str, object]] = []
        branch_events: list[str] = []
        branch_start = reference.step
        end_step = branch_start + spec.total_steps
        next_measurement = branch_start
        while engine.step_count <= end_step:
            relative_step = engine.step_count - branch_start
            if relative_step == spec.injection_step and spec.perturbation_type == "VELOCITY IMPULSE" and branch != "CONTROL":
                angle = spec.bit0_angle if branch == "BIT-0" else spec.bit1_angle
                _apply_velocity_impulse(engine, target_center, target_ids, spec.target_radius, spec.impulse_magnitude, angle)
                branch_events.append(f"{branch} impulse injected at step {engine.step_count}")
            if engine.step_count >= next_measurement and (relative_step >= spec.warmup_steps or relative_step == 0):
                tracking = cluster_tracker.update(engine.state, engine.step_count)
                networks = network_tracker.update(engine.state, tracking.clusters, engine.step_count)
                selected_cluster = next((item for item in tracking.clusters if target_ids and target_ids.issubset(item.particle_ids)), None)
                structure_score = selected_cluster.structure_score if selected_cluster else max((item.structure_score for item in tracking.clusters), default=0.0)
                selected_network = next((item for item in networks if item.cluster_id == selected_cluster.cluster_id), None) if selected_cluster else None
                network_score = selected_network.network_score if selected_network else None
                for zone in spec.zones:
                    if zone.enabled:
                        measurements.append(_branch_measurement(engine, branch, engine.step_count, zone, target_center, target_ids, len(tracking.clusters), structure_score, network_score))
                next_measurement += spec.measurement_interval
                if progress:
                    progress(branch, {"step": relative_step, "total_steps": spec.total_steps, "measurements": len(measurements)})
            if engine.step_count >= end_step:
                break
            engine.step(min(spec.measurement_interval, end_step - engine.step_count))
            if pause_event is not None and getattr(pause_event, "is_set", lambda: False)():
                while getattr(pause_event, "is_set", lambda: False)():
                    time.sleep(0.05)
        final_measurements = [item for item in measurements if item["step"] == max(item["step"] for item in measurements)] if measurements else []
        branch_run = BranchRun(branch, measurements, branch_events, engine.state.copy(), engine.step_count, float(np.mean([item["average_speed"] for item in final_measurements])) if final_measurements else 0.0, int(final_measurements[0]["cluster_count"]) if final_measurements else 0, float(final_measurements[0]["target_particle_count"]) / max(len(target_ids), 1) if final_measurements and target_ids else 0.0, max(final_measurements, key=lambda item: float(item["density"]), default={}).get("zone", "-"), float(max((item["structure_score"] for item in final_measurements), default=0.0)), max((float(item["network_score"]) for item in final_measurements if item["network_score"] is not None), default=None))
        branches[branch] = branch_run
        all_events.extend({"branch": branch, "event": event} for event in branch_events)
    summary = compare_branches(spec, branches)
    return ExperimentRun(spec, reference.copy(), target_center.copy(), target_ids, branches, summary, all_events)


def compare_branches(spec: ExperimentSpec, branches: dict[str, BranchRun]) -> dict[str, object]:
    control = branches["CONTROL"].measurements
    bit0 = branches["BIT-0"].measurements
    bit1 = branches["BIT-1"].measurements
    pairs = []
    for first, second in zip(bit0, bit1):
        if first["zone"] == second["zone"]:
            separation = float(np.hypot(float(first["average_velocity_x"]) - float(second["average_velocity_x"]), float(first["average_velocity_y"]) - float(second["average_velocity_y"])))
            pairs.append((int(first["step"]), first["zone"], separation))
    max_pair = max(pairs, key=lambda item: item[2], default=(0, "-", 0.0))
    detectable = [item for item in pairs if item[2] > 1e-6]
    first_divergence = min((item[0] for item in detectable), default=None)
    duration = (max(item[0] for item in detectable) - first_divergence) if detectable and first_divergence is not None else 0
    control_pairs = []
    for row in bit0:
        match = next((item for item in control if item["step"] == row["step"] and item["zone"] == row["zone"]), None)
        if match:
            control_pairs.append(abs(float(row["average_speed"]) - float(match["average_speed"])))
    outcome = "NO_CLEAR_RESPONSE" if max_pair[2] <= 1e-6 else ("LOCAL_RESPONSE" if max_pair[1] in {"Zone A", "Zone B"} else "LONG_RANGE_RESPONSE")
    return {
        "experiment_complete": True,
        "reference_step": spec.reference_step,
        "target": {"type": spec.target_type, "id": spec.target_id, "network_id": spec.network_id, "node_id": spec.node_id},
        "perturbation": {"type": spec.perturbation_type, "magnitude": spec.impulse_magnitude, "bit0_angle": spec.bit0_angle, "bit1_angle": spec.bit1_angle, "injection_step": spec.injection_step},
        "total_steps": spec.total_steps,
        "bit0_vs_bit1_maximum_separation": {"distance": max_pair[2], "time": max_pair[0], "zone": max_pair[1]},
        "time_of_first_detectable_divergence": first_divergence,
        "response_duration": duration,
        "repeatability": "NOT_RUN",
        "control_deviation": float(np.mean(control_pairs)) if control_pairs else 0.0,
        "outcome_classification": outcome,
        "branch_separation_label": "Branch Separation",
    }


def _write_png(path: Path, positions: np.ndarray, width: int = 640, height: int = 420) -> None:
    image = bytearray(width * height * 3)
    if len(positions):
        xs = np.clip((positions[:, 0] / max(float(np.max(positions[:, 0])), 1.0) * (width - 1)).astype(int), 0, width - 1)
        ys = np.clip((positions[:, 1] / max(float(np.max(positions[:, 1])), 1.0) * (height - 1)).astype(int), 0, height - 1)
        for x, y in zip(xs, ys):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    px, py = x + dx, y + dy
                    if 0 <= px < width and 0 <= py < height:
                        index = (py * width + px) * 3
                        image[index:index + 3] = b"\x4f\xd2\x7d"
    raw = b"".join(b"\x00" + image[row * width * 3:(row + 1) * width * 3] for row in range(height))
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def save_experiment(run: ExperimentRun, root: str | Path = "results/experiments") -> Path:
    """Persist an experiment without overwriting an existing record."""
    run.spec.validate()
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    destination = root_path / run.spec.experiment_id
    if destination.exists():
        raise FileExistsError(f"Experiment {run.spec.experiment_id} already exists and will not be overwritten")
    destination.mkdir()
    run.reference.genome.save(destination / "genome.json")
    run.reference.state.save(destination / "reference_state.npz")
    metadata = {"spec": {**asdict(run.spec), "zones": [asdict(zone) for zone in run.spec.zones]}, "reference": {"seed": run.reference.seed, "step": run.reference.step, "simulation_config": asdict(run.reference.simulation_config)}, "target_center": run.target_center.tolist(), "target_particle_ids": sorted(run.target_particle_ids), "summary": run.summary}
    (destination / "experiment.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    (destination / "summary.json").write_text(json.dumps(run.summary, indent=2, default=str) + "\n", encoding="utf-8")
    (destination / "events.json").write_text(json.dumps(run.events, indent=2) + "\n", encoding="utf-8")
    for branch in BRANCH_NAMES:
        rows = run.branches[branch].measurements
        filename = branch.lower().replace("-", "") + "_measurements.csv"
        with (destination / filename).open("w", newline="", encoding="utf-8") as handle:
            fields = list(rows[0]) if rows else ["branch", "step", "zone"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    _write_png(destination / "preview.png", run.branches["BIT-1"].final_state.positions)
    return destination


def load_experiment(path: str | Path) -> tuple[ExperimentSpec, ReferenceState, dict[str, object]]:
    directory = Path(path)
    payload = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    raw_spec = dict(payload["spec"])
    raw_spec["zones"] = [MeasurementZone(**zone) for zone in raw_spec.get("zones", [])]
    spec = ExperimentSpec(**raw_spec)
    genome = Genome.load(directory / "genome.json")
    state = ParticleState.load(directory / "reference_state.npz")
    simulation = SimulationConfig(**payload["reference"]["simulation_config"])
    reference = ReferenceState(genome, state, int(payload["reference"]["seed"]), int(payload["reference"]["step"]), simulation)
    return spec, reference, payload
