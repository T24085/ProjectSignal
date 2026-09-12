"""Cohesion-aware persistent cluster detection for Project SIGNAL Phase 2B."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from signal_lab.physics.particle import ParticleState
from signal_lab.physics.spatial_hash import SpatialHash


def toroidal_centroid(positions: np.ndarray, width: float, height: float) -> np.ndarray:
    """Find a centroid that remains correct when a cluster crosses a boundary."""
    positions = np.asarray(positions, dtype=np.float64)
    result = np.empty(2, dtype=np.float64)
    for axis, size in enumerate((width, height)):
        angles = positions[:, axis] / size * (2.0 * np.pi)
        vector = np.mean(np.exp(1j * angles))
        if abs(vector) < 1e-12:
            reference = positions[0, axis]
            offsets = (positions[:, axis] - reference + size / 2.0) % size - size / 2.0
            result[axis] = (reference + np.mean(offsets)) % size
        else:
            result[axis] = (np.angle(vector) % (2.0 * np.pi)) / (2.0 * np.pi) * size
    return result


def toroidal_centroid_distance(a: np.ndarray, b: np.ndarray, width: float, height: float) -> float:
    """Return minimum-image distance between two centroids."""
    size = np.array([width, height], dtype=np.float64)
    delta = (np.asarray(b) - np.asarray(a) + size / 2.0) % size - size / 2.0
    return float(np.linalg.norm(delta))


def membership_overlap(previous_members: Iterable[int], current_members: Iterable[int]) -> float:
    """Return intersection-over-union for two permanent particle-ID sets."""
    previous = set(previous_members)
    current = set(current_members)
    union = previous | current
    return float(len(previous & current) / len(union)) if union else 0.0


def lifetime_class(age_steps: int) -> str:
    """Return the experimental lifetime label for a cluster age."""
    if age_steps < 100:
        return "TRANSIENT"
    if age_steps < 500:
        return "SHORT_LIVED"
    if age_steps < 2000:
        return "PERSISTENT"
    return "LONG_LIVED"


@dataclass
class ClusterEvent:
    step: int
    kind: str
    cluster_id: int
    related_ids: tuple[int, ...] = ()
    particle_count: int = 0

    @property
    def message(self) -> str:
        cluster = f"C{self.cluster_id}"
        if self.kind == "formed":
            return f"Cluster {cluster} formed (n={self.particle_count})"
        if self.kind == "dissolved":
            return f"Cluster {cluster} dissolved"
        if self.kind == "merged":
            related = ", ".join(f"C{value}" for value in self.related_ids)
            return f"Cluster {cluster} merged with {related}"
        if self.kind == "split":
            related = ", ".join(f"C{value}" for value in self.related_ids)
            return f"Cluster {cluster} split into {related}"
        if self.kind == "reached":
            return f"Cluster {cluster} reached {lifetime_class(self.particle_count)}"
        return f"Cluster {cluster} {self.kind}"


@dataclass
class ClusterObservation:
    """A tracked structure observation containing raw and derived metrics."""

    cluster_id: int
    birth_step: int
    last_seen_step: int
    particle_ids: frozenset[int]
    centroid: np.ndarray
    mean_velocity: np.ndarray
    species_distribution: np.ndarray
    radius: float
    persistence_score: float = 0.0
    membership_stability: float = 0.0
    size_stability: float = 0.0
    species_stability: float = 0.0
    shape_stability: float = 0.0
    motion_coherence: float = 0.0
    mean_neighbor_distance: float = 0.0
    neighbor_distance_std: float = 0.0
    local_neighbor_count_mean: float = 0.0
    local_neighbor_count_std: float = 0.0
    cluster_density: float = 0.0
    radius_of_gyration: float = 0.0
    diameter: float = 0.0
    compactness: float = 0.0
    average_degree: float = 0.0
    degree_variance: float = 0.0
    connected_component_count: int = 1
    largest_component_fraction: float = 1.0
    bridge_fraction: float = 0.0
    internal_motion: float = 0.0
    internal_motion_std: float = 0.0
    cohesion_score: float = 0.0
    identity_score: float = 0.0
    shape_score: float = 0.0
    dynamic_score: float = 0.0
    lifetime_score: float = 0.0
    structure_score: float = 0.0
    rg_cv: float = 0.0
    density_cv: float = 0.0
    count_cv: float = 0.0
    species_cv: float = 0.0
    connectivity_cv: float = 0.0
    history: list[dict[str, object]] = field(default_factory=list)

    @property
    def age_steps(self) -> int:
        return max(0, self.last_seen_step - self.birth_step)

    @property
    def particle_count(self) -> int:
        return len(self.particle_ids)

    @property
    def classification(self) -> str:
        return lifetime_class(self.age_steps)

    @property
    def is_macro_cluster(self) -> bool:
        return self.particle_count > 300

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "birth_step": self.birth_step,
            "last_seen_step": self.last_seen_step,
            "age_steps": self.age_steps,
            "particle_count": self.particle_count,
            "particle_ids": sorted(self.particle_ids),
            "centroid_x": float(self.centroid[0]),
            "centroid_y": float(self.centroid[1]),
            "mean_velocity_x": float(self.mean_velocity[0]),
            "mean_velocity_y": float(self.mean_velocity[1]),
            "species_distribution": self.species_distribution.tolist(),
            "radius": self.radius,
            "persistence_score": self.persistence_score,
            "cohesion_score": self.cohesion_score,
            "identity_score": self.identity_score,
            "shape_score": self.shape_score,
            "dynamic_score": self.dynamic_score,
            "lifetime_score": self.lifetime_score,
            "structure_score": self.structure_score,
            "membership_stability": self.membership_stability,
            "size_stability": self.size_stability,
            "species_stability": self.species_stability,
            "shape_stability": self.shape_stability,
            "motion_coherence": self.motion_coherence,
            "mean_neighbor_distance": self.mean_neighbor_distance,
            "neighbor_distance_std": self.neighbor_distance_std,
            "local_neighbor_count_mean": self.local_neighbor_count_mean,
            "local_neighbor_count_std": self.local_neighbor_count_std,
            "cluster_density": self.cluster_density,
            "radius_of_gyration": self.radius_of_gyration,
            "diameter": self.diameter,
            "compactness": self.compactness,
            "average_degree": self.average_degree,
            "degree_variance": self.degree_variance,
            "connected_component_count": self.connected_component_count,
            "largest_component_fraction": self.largest_component_fraction,
            "bridge_fraction": self.bridge_fraction,
            "internal_motion": self.internal_motion,
            "internal_motion_std": self.internal_motion_std,
            "rg_cv": self.rg_cv,
            "density_cv": self.density_cv,
            "count_cv": self.count_cv,
            "species_cv": self.species_cv,
            "connectivity_cv": self.connectivity_cv,
            "classification": self.classification,
            "macro_cluster": self.is_macro_cluster,
            "history": self.history,
        }


@dataclass
class TrackingResult:
    clusters: list[ClusterObservation]
    events: list[ClusterEvent]


class ClusterDetector:
    """Detect proximity components and calculate cohesion/connectivity metrics."""

    def __init__(self, width: float, height: float, link_radius: float, min_cluster_size: int = 5) -> None:
        self.width = float(width)
        self.height = float(height)
        self.link_radius = float(link_radius)
        self.min_cluster_size = int(min_cluster_size)
        self.spatial_hash = SpatialHash(width, height, link_radius)

    def detect(self, state: ParticleState) -> list[ClusterObservation]:
        """Detect connected components and retain raw cohesion measurements."""
        self.spatial_hash.rebuild(state.positions)
        size = np.array([self.width, self.height], dtype=np.float64)
        remaining = set(range(state.count))
        detected: list[ClusterObservation] = []
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            component = [seed]
            stack = [seed]
            while stack:
                index = stack.pop()
                for neighbor in self.spatial_hash.neighbors(state.positions[index]):
                    if neighbor not in remaining:
                        continue
                    delta = (state.positions[neighbor] - state.positions[index] + size / 2.0) % size - size / 2.0
                    if float(np.linalg.norm(delta)) <= self.link_radius:
                        remaining.remove(neighbor)
                        component.append(neighbor)
                        stack.append(neighbor)
            if len(component) < self.min_cluster_size:
                continue
            indices = np.asarray(component, dtype=np.int64)
            positions = state.positions[indices]
            velocities = state.velocities[indices]
            centroid = toroidal_centroid(positions, self.width, self.height)
            offsets = (positions - centroid + size / 2.0) % size - size / 2.0
            distances = np.linalg.norm(offsets, axis=1)
            radius_of_gyration = float(np.sqrt(np.mean(distances * distances)))
            radius = float(np.max(distances))
            diameter = _estimate_diameter(positions, self.width, self.height)
            graph = _build_local_graph(indices, state.positions, self.width, self.height, self.link_radius)
            edge_distances = []
            degrees = np.zeros(len(indices), dtype=np.float64)
            index_lookup = {int(index): offset for offset, index in enumerate(indices)}
            for global_index, neighbors in graph.items():
                local_index = index_lookup[global_index]
                degrees[local_index] = len(neighbors)
                for neighbor in neighbors:
                    if global_index < neighbor:
                        edge_distances.append(_toroidal_distance(state.positions[global_index], state.positions[neighbor], size))
            distances_array = np.asarray(edge_distances, dtype=np.float64)
            mean_neighbor_distance = float(np.mean(distances_array)) if len(distances_array) else self.link_radius
            neighbor_distance_std = float(np.std(distances_array)) if len(distances_array) else 0.0
            components = _graph_component_sizes(graph)
            articulation = _articulation_points(graph)
            largest_fraction = max(components, default=0) / max(len(indices), 1)
            bridge_fraction = len(articulation) / max(len(indices), 1)
            cluster_density = len(indices) / max(np.pi * max(radius_of_gyration, self.link_radius * 0.05) ** 2, 1e-12)
            compactness = min(1.0, max(0.0, np.sqrt(2.0) * radius_of_gyration / max(diameter, 1e-12)))
            speed = np.linalg.norm(velocities, axis=1)
            mean_speed = float(np.mean(speed))
            internal_velocities = velocities - np.mean(velocities, axis=0)
            internal_motion_values = np.linalg.norm(internal_velocities, axis=1)
            internal_motion = float(np.mean(internal_motion_values))
            internal_motion_std = float(np.std(internal_motion_values))
            motion_coherence = float(np.linalg.norm(np.sum(velocities, axis=0)) / max(np.sum(speed), 1e-12))
            species_count = int(np.max(state.species, initial=0)) + 1
            species_distribution = np.bincount(state.species[indices], minlength=species_count).astype(np.float64)
            species_distribution /= max(float(len(indices)), 1.0)
            cohesion_score = _cohesion_score(mean_neighbor_distance, degrees, compactness, largest_fraction, bridge_fraction, self.link_radius, len(components))
            dynamic_score = _dynamic_score(internal_motion, internal_motion_std, self.link_radius)
            detected.append(ClusterObservation(0, 0, 0, frozenset(int(value) for value in state.ids[indices]), centroid, np.mean(velocities, axis=0), species_distribution, radius, motion_coherence=motion_coherence, mean_neighbor_distance=mean_neighbor_distance, neighbor_distance_std=neighbor_distance_std, local_neighbor_count_mean=float(np.mean(degrees)), local_neighbor_count_std=float(np.std(degrees)), cluster_density=cluster_density, radius_of_gyration=radius_of_gyration, diameter=diameter, compactness=compactness, average_degree=float(np.mean(degrees)), degree_variance=float(np.var(degrees)), connected_component_count=len(components), largest_component_fraction=largest_fraction, bridge_fraction=bridge_fraction, internal_motion=internal_motion, internal_motion_std=internal_motion_std, cohesion_score=cohesion_score, dynamic_score=dynamic_score))
        return detected


class ClusterTracker:
    """Associate observations across frames and score temporal stability."""

    def __init__(self, width: float, height: float, link_radius: float, min_cluster_size: int = 5, minimum_overlap: float = 0.40) -> None:
        self.width = float(width)
        self.height = float(height)
        self.link_radius = float(link_radius)
        self.minimum_overlap = float(minimum_overlap)
        self.detector = ClusterDetector(width, height, link_radius, min_cluster_size)
        self.active: dict[int, ClusterObservation] = {}
        self.completed: dict[int, ClusterObservation] = {}
        self.next_cluster_id = 1
        self.last_events: list[ClusterEvent] = []

    def reset(self) -> None:
        self.active.clear()
        self.completed.clear()
        self.next_cluster_id = 1
        self.last_events = []

    def update(self, state: ParticleState, step: int) -> TrackingResult:
        raw_clusters = self.detector.detect(state)
        previous = dict(self.active)
        events: list[ClusterEvent] = []
        candidates_by_current: dict[int, list[tuple[float, float, int]]] = {}
        for current_index, current in enumerate(raw_clusters):
            candidates: list[tuple[float, float, int]] = []
            for cluster_id, prior in previous.items():
                overlap = membership_overlap(prior.particle_ids, current.particle_ids)
                distance = toroidal_centroid_distance(prior.centroid, current.centroid, self.width, self.height)
                if overlap >= self.minimum_overlap and distance <= self.link_radius * 3.0:
                    candidates.append((overlap, distance, cluster_id))
            candidates_by_current[current_index] = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))

        assigned_previous: set[int] = set()
        current_clusters: list[ClusterObservation] = []
        previous_to_current: dict[int, list[int]] = {}
        for current_index, raw in enumerate(raw_clusters):
            candidates = candidates_by_current[current_index]
            for candidate in candidates:
                previous_to_current.setdefault(candidate[2], []).append(current_index)
            cluster_id = next((candidate[2] for candidate in candidates if candidate[2] not in assigned_previous), None)
            if cluster_id is None:
                cluster_id = self.next_cluster_id
                self.next_cluster_id += 1
                observation = self._new_observation(raw, cluster_id, step)
                events.append(ClusterEvent(step, "formed", cluster_id, particle_count=observation.particle_count))
            else:
                prior = previous[cluster_id]
                assigned_previous.add(cluster_id)
                observation = self._continued_observation(prior, raw, step)
                if prior.classification != observation.classification and observation.classification in {"SHORT_LIVED", "PERSISTENT", "LONG_LIVED"}:
                    events.append(ClusterEvent(step, "reached", cluster_id, particle_count=observation.age_steps))
            current_clusters.append(observation)

        for cluster_id, indexes in previous_to_current.items():
            if len(indexes) > 1:
                child_ids = tuple(current_clusters[index].cluster_id for index in indexes if current_clusters[index].cluster_id != cluster_id)
                if child_ids:
                    events.append(ClusterEvent(step, "split", cluster_id, child_ids))
        for current_index, candidates in candidates_by_current.items():
            matching_previous = tuple(candidate[2] for candidate in candidates)
            if len(matching_previous) > 1:
                events.append(ClusterEvent(step, "merged", current_clusters[current_index].cluster_id, matching_previous))
        for cluster_id, prior in previous.items():
            if cluster_id not in assigned_previous:
                self.completed[cluster_id] = prior
                events.append(ClusterEvent(step, "dissolved", cluster_id))
        self.active = {observation.cluster_id: observation for observation in current_clusters}
        self.last_events = events
        return TrackingResult(current_clusters, events)

    def _new_observation(self, raw: ClusterObservation, cluster_id: int, step: int) -> ClusterObservation:
        raw.cluster_id = cluster_id
        raw.birth_step = step
        raw.last_seen_step = step
        raw.identity_score = 0.0
        raw.lifetime_score = 0.0
        raw.shape_score = _shape_score(raw)
        raw.structure_score = _structure_score(raw)
        raw.persistence_score = raw.structure_score
        raw.history = [_history_point(raw, step)]
        return raw

    def _continued_observation(self, prior: ClusterObservation, raw: ClusterObservation, step: int) -> ClusterObservation:
        overlap = membership_overlap(prior.particle_ids, raw.particle_ids)
        size_stability = max(0.0, 1.0 - abs(prior.particle_count - raw.particle_count) / max(prior.particle_count, raw.particle_count, 1))
        prior_species, current_species = _pad_distribution(prior.species_distribution, raw.species_distribution)
        species_stability = max(0.0, 1.0 - 0.5 * float(np.sum(np.abs(prior_species - current_species))))
        shape_stability = max(0.0, 1.0 - abs(prior.radius_of_gyration - raw.radius_of_gyration) / max(prior.radius_of_gyration, raw.radius_of_gyration, self.link_radius * 0.05))
        raw.cluster_id = prior.cluster_id
        raw.birth_step = prior.birth_step
        raw.last_seen_step = step
        raw.membership_stability = _running_average(prior.membership_stability, overlap, prior.age_steps)
        raw.size_stability = _running_average(prior.size_stability, size_stability, prior.age_steps)
        raw.species_stability = _running_average(prior.species_stability, species_stability, prior.age_steps)
        raw.shape_stability = _running_average(prior.shape_stability, shape_stability, prior.age_steps)
        raw.rg_cv = _coefficient_of_variation(prior.history, "radius_of_gyration", raw.radius_of_gyration)
        raw.density_cv = _coefficient_of_variation(prior.history, "cluster_density", raw.cluster_density)
        raw.count_cv = _coefficient_of_variation(prior.history, "particle_count", raw.particle_count)
        raw.species_cv = _species_cv(prior.history, raw.species_distribution)
        raw.connectivity_cv = _coefficient_of_variation(prior.history, "average_degree", raw.average_degree)
        raw.identity_score = _running_average(prior.identity_score, overlap, prior.age_steps)
        raw.history = prior.history
        raw.shape_score = _shape_score(raw)
        raw.lifetime_score = min(1.0, (step - prior.birth_step) / 2000.0)
        raw.cohesion_score = _cohesion_score(raw.mean_neighbor_distance, np.asarray([raw.local_neighbor_count_mean]), raw.compactness, raw.largest_component_fraction, raw.bridge_fraction, self.link_radius, raw.connected_component_count)
        raw.dynamic_score = _dynamic_score(raw.internal_motion, raw.internal_motion_std, self.link_radius)
        raw.structure_score = _structure_score(raw)
        raw.persistence_score = raw.structure_score
        raw.history = prior.history + [_history_point(raw, step)]
        return raw


def _history_point(cluster: ClusterObservation, step: int) -> dict[str, object]:
    return {"step": step, "particle_count": cluster.particle_count, "centroid_x": float(cluster.centroid[0]), "centroid_y": float(cluster.centroid[1]), "radius_of_gyration": cluster.radius_of_gyration, "cluster_density": cluster.cluster_density, "species_distribution": cluster.species_distribution.tolist(), "average_degree": cluster.average_degree, "structure_score": cluster.structure_score}


def _toroidal_distance(a: np.ndarray, b: np.ndarray, size: np.ndarray) -> float:
    delta = (np.asarray(b) - np.asarray(a) + size / 2.0) % size - size / 2.0
    return float(np.linalg.norm(delta))


def _estimate_diameter(positions: np.ndarray, width: float, height: float) -> float:
    if len(positions) < 2:
        return 0.0
    size = np.array([width, height], dtype=np.float64)
    sample = positions if len(positions) <= 500 else positions[np.linspace(0, len(positions) - 1, 500, dtype=int)]
    maximum = 0.0
    for index in range(len(sample)):
        deltas = (sample[index + 1:] - sample[index] + size / 2.0) % size - size / 2.0
        if len(deltas):
            maximum = max(maximum, float(np.max(np.linalg.norm(deltas, axis=1))))
    return maximum


def _build_local_graph(indices: np.ndarray, positions: np.ndarray, width: float, height: float, radius: float) -> dict[int, set[int]]:
    spatial = SpatialHash(width, height, radius)
    spatial.rebuild(positions)
    members = set(int(value) for value in indices)
    graph = {int(value): set() for value in indices}
    size = np.array([width, height], dtype=np.float64)
    for index in indices:
        global_index = int(index)
        for neighbor in spatial.neighbors(positions[global_index]):
            if neighbor not in members or neighbor == global_index:
                continue
            delta = (positions[neighbor] - positions[global_index] + size / 2.0) % size - size / 2.0
            if float(np.linalg.norm(delta)) <= radius:
                graph[global_index].add(int(neighbor))
    return graph


def _graph_component_sizes(graph: dict[int, set[int]]) -> list[int]:
    remaining = set(graph)
    sizes: list[int] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack = [root]
        size = 1
        while stack:
            for neighbor in graph[stack.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    size += 1
        sizes.append(size)
    return sizes


def _articulation_points(graph: dict[int, set[int]]) -> set[int]:
    """Tarjan articulation-point search on the cluster neighbor graph."""
    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    articulation: set[int] = set()
    counter = 0

    def visit(node: int) -> None:
        nonlocal counter
        counter += 1
        discovery[node] = low[node] = counter
        children = 0
        for neighbor in sorted(graph[node]):
            if neighbor not in discovery:
                parent[neighbor] = node
                children += 1
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
                if parent.get(node) is None and children > 1:
                    articulation.add(node)
                if parent.get(node) is not None and low[neighbor] >= discovery[node]:
                    articulation.add(node)
            elif neighbor != parent.get(node):
                low[node] = min(low[node], discovery[neighbor])

    for root in sorted(graph):
        if root not in discovery:
            parent[root] = None
            visit(root)
    return articulation


def _cohesion_score(mean_distance: float, degrees: np.ndarray, compactness: float, largest_fraction: float, bridge_fraction: float, link_radius: float, component_count: int) -> float:
    distance_score = float(np.exp(-mean_distance / max(link_radius, 1e-12)))
    degree_score = min(1.0, float(np.mean(degrees)) / 5.0)
    connectivity_score = largest_fraction * (1.0 if component_count == 1 else 1.0 / component_count)
    bridge_penalty = max(0.0, 1.0 - min(1.0, bridge_fraction * 8.0))
    compactness_score = min(1.0, compactness / 0.65)
    return float(np.clip(0.25 * distance_score + 0.20 * degree_score + 0.25 * compactness_score + 0.15 * connectivity_score + 0.15 * bridge_penalty, 0.0, 1.0))


def _dynamic_score(internal_motion: float, internal_motion_std: float, link_radius: float) -> float:
    if internal_motion <= 1e-12:
        return 0.0
    activity = min(1.0, internal_motion / max(link_radius * 0.02, 1e-12))
    dispersion = internal_motion_std / max(internal_motion, 1e-12)
    chaos_penalty = 1.0 / (1.0 + max(0.0, dispersion - 1.5))
    return float(np.clip(activity * chaos_penalty, 0.0, 1.0))


def _shape_score(cluster: ClusterObservation) -> float:
    temporal_cv = np.mean([cluster.rg_cv, cluster.density_cv, cluster.count_cv, cluster.species_cv, cluster.connectivity_cv])
    return float(np.clip(np.exp(-2.5 * temporal_cv) * cluster.shape_stability if cluster.history else 0.75, 0.0, 1.0))


def _structure_score(cluster: ClusterObservation) -> float:
    return float(np.clip(0.25 * cluster.cohesion_score + 0.25 * cluster.identity_score + 0.20 * cluster.shape_score + 0.15 * cluster.dynamic_score + 0.15 * cluster.lifetime_score, 0.0, 1.0))


def _coefficient_of_variation(history: list[dict[str, object]], key: str, current: float) -> float:
    values = [float(point[key]) for point in history if key in point] + [float(current)]
    mean = float(np.mean(values)) if values else 0.0
    return float(np.std(values) / max(abs(mean), 1e-12))


def _species_cv(history: list[dict[str, object]], current: np.ndarray) -> float:
    values = [np.asarray(point["species_distribution"], dtype=np.float64) for point in history if "species_distribution" in point]
    values.append(np.asarray(current, dtype=np.float64))
    padded = np.zeros((len(values), max(len(value) for value in values)))
    for row, value in enumerate(values):
        padded[row, :len(value)] = value
    return float(np.mean(np.std(padded, axis=0) / np.maximum(np.mean(padded, axis=0), 1e-12)))


def _running_average(previous: float, current: float, age: int) -> float:
    return current if age <= 0 else (previous * age + current) / (age + 1)


def _pad_distribution(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = max(len(a), len(b))
    return np.pad(a, (0, length - len(a))), np.pad(b, (0, length - len(b)))
