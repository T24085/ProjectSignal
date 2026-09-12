"""Deterministic internal graph and persistent network-structure detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from signal_lab.experiment.clustering import (
    ClusterObservation,
    _articulation_points,
    _clustering_coefficient,
    _graph_component_sizes,
    _shortest_path_metrics,
    membership_overlap,
    toroidal_centroid,
)
from signal_lab.experiment.config import StructureDetectionConfig, load_structure_config
from signal_lab.physics.particle import ParticleState
from signal_lab.physics.spatial_hash import SpatialHash


@dataclass
class ParticleGraph:
    """Raw neighbor graph and topology metrics for one connected cluster."""

    adjacency: dict[int, set[int]]
    edge_count: int
    connected_components: list[frozenset[int]]
    largest_component_fraction: float
    articulation_point_count: int
    articulation_point_fraction: float
    graph_diameter: int
    average_shortest_path_length: float
    clustering_coefficient: float

    @property
    def connected_component_count(self) -> int:
        return len(self.connected_components)

    @property
    def average_degree(self) -> float:
        return float(np.mean([len(value) for value in self.adjacency.values()])) if self.adjacency else 0.0

    @property
    def degree_variance(self) -> float:
        return float(np.var([len(value) for value in self.adjacency.values()])) if self.adjacency else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_count": self.edge_count,
            "connected_components": [sorted(component) for component in self.connected_components],
            "connected_component_count": self.connected_component_count,
            "largest_component_fraction": self.largest_component_fraction,
            "articulation_point_count": self.articulation_point_count,
            "articulation_point_fraction": self.articulation_point_fraction,
            "graph_diameter": self.graph_diameter,
            "average_shortest_path_length": self.average_shortest_path_length,
            "clustering_coefficient": self.clustering_coefficient,
            "average_degree": self.average_degree,
            "degree_variance": self.degree_variance,
        }


def build_particle_graph(state: ParticleState, particle_ids: Iterable[int], width: float, height: float, neighbor_radius: float) -> ParticleGraph:
    """Build a toroidal graph for the requested permanent particle IDs."""
    ids = frozenset(int(value) for value in particle_ids)
    id_to_index = {int(value): index for index, value in enumerate(state.ids)}
    indices = [id_to_index[value] for value in sorted(ids) if value in id_to_index]
    adjacency = {int(state.ids[index]): set() for index in indices}
    if not indices:
        return ParticleGraph(adjacency, 0, [], 0.0, 0, 0.0, 0, 0.0, 0.0)
    spatial = SpatialHash(width, height, neighbor_radius)
    spatial.rebuild(state.positions)
    size = np.asarray([width, height], dtype=np.float64)
    member_indices = set(indices)
    for index in indices:
        source_id = int(state.ids[index])
        for neighbor_index in spatial.neighbors(state.positions[index]):
            if neighbor_index not in member_indices or neighbor_index == index:
                continue
            delta = (state.positions[neighbor_index] - state.positions[index] + size / 2.0) % size - size / 2.0
            if float(np.linalg.norm(delta)) <= neighbor_radius:
                adjacency[source_id].add(int(state.ids[neighbor_index]))
    components = _components(adjacency)
    articulations = _articulation_points(adjacency)
    diameter, average_path = _shortest_path_metrics(adjacency)
    count = len(adjacency)
    return ParticleGraph(
        adjacency=adjacency,
        edge_count=sum(len(neighbors) for neighbors in adjacency.values()) // 2,
        connected_components=components,
        largest_component_fraction=max((len(component) for component in components), default=0) / max(count, 1),
        articulation_point_count=len(articulations),
        articulation_point_fraction=len(articulations) / max(count, 1),
        graph_diameter=diameter,
        average_shortest_path_length=average_path,
        clustering_coefficient=_clustering_coefficient(adjacency),
    )


def _components(adjacency: dict[int, set[int]]) -> list[frozenset[int]]:
    remaining = set(adjacency)
    components: list[frozenset[int]] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        component = {root}
        queue = [root]
        while queue:
            for neighbor in adjacency[queue.pop()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(frozenset(component))
    return components


@dataclass
class NetworkNode:
    node_id: int
    particle_ids: frozenset[int]
    centroid: np.ndarray
    radius: float
    density: float
    species_distribution: np.ndarray
    mean_velocity: np.ndarray
    birth_step: int
    last_seen_step: int
    identity_score: float = 0.0

    @property
    def particle_count(self) -> int:
        return len(self.particle_ids)

    @property
    def age_steps(self) -> int:
        return max(0, self.last_seen_step - self.birth_step)

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "particle_ids": sorted(self.particle_ids),
            "particle_count": self.particle_count,
            "centroid": [float(self.centroid[0]), float(self.centroid[1])],
            "radius": self.radius,
            "density": self.density,
            "species_distribution": self.species_distribution.tolist(),
            "mean_velocity": [float(self.mean_velocity[0]), float(self.mean_velocity[1])],
            "birth_step": self.birth_step,
            "last_seen_step": self.last_seen_step,
            "age": self.age_steps,
            "identity_score": self.identity_score,
        }


@dataclass
class NetworkEdge:
    edge_id: int
    node_a: int
    node_b: int
    path_particle_ids: frozenset[int]
    path_particle_count: int
    path_length: float
    minimum_width: float
    mean_density: float
    birth_step: int
    last_seen_step: int
    persistence: float = 0.0
    particle_turnover: float = 0.0

    @property
    def age_steps(self) -> int:
        return max(0, self.last_seen_step - self.birth_step)

    @property
    def identity_score(self) -> float:
        return max(0.0, 1.0 - self.particle_turnover)

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "node_a": self.node_a,
            "node_b": self.node_b,
            "path_particle_ids": sorted(self.path_particle_ids),
            "path_particle_count": self.path_particle_count,
            "path_length": self.path_length,
            "minimum_width": self.minimum_width,
            "mean_density": self.mean_density,
            "birth_step": self.birth_step,
            "last_seen_step": self.last_seen_step,
            "age": self.age_steps,
            "persistence": self.persistence,
            "particle_turnover": self.particle_turnover,
            "identity_score": self.identity_score,
        }


@dataclass
class NetworkObservation:
    network_id: int
    cluster_id: int
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]
    age_steps: int
    topology_persistence: float
    node_identity_stability: float
    edge_identity_stability: float
    degree_distribution: list[int]
    network_density: float
    largest_node_fraction: float
    graph_diameter: int
    average_shortest_path_length: float
    structural_cohesion: float
    dynamic_activity: float
    lifetime_score: float
    network_score: float
    classification: str
    persistent_node_count: int
    persistent_edge_count: int
    cluster_metrics: dict[str, object] = field(default_factory=dict)
    history: list[dict[str, object]] = field(default_factory=list)
    candidate: bool = False

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def is_candidate(self) -> bool:
        return self.candidate

    def to_dict(self) -> dict[str, object]:
        return {
            "network_id": self.network_id,
            "cluster_id": self.cluster_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "persistent_node_count": self.persistent_node_count,
            "persistent_edge_count": self.persistent_edge_count,
            "age": self.age_steps,
            "network_score": self.network_score,
            "classification": self.classification,
            "topology_persistence": self.topology_persistence,
            "node_identity_stability": self.node_identity_stability,
            "edge_identity_stability": self.edge_identity_stability,
            "degree_distribution": self.degree_distribution,
            "network_density": self.network_density,
            "largest_node_fraction": self.largest_node_fraction,
            "graph_diameter": self.graph_diameter,
            "average_shortest_path_length": self.average_shortest_path_length,
            "structural_cohesion": self.structural_cohesion,
            "dynamic_activity": self.dynamic_activity,
            "lifetime_score": self.lifetime_score,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "cluster_metrics": self.cluster_metrics,
            "history": self.history,
            "candidate": self.is_candidate,
        }


@dataclass
class NetworkEvent:
    step: int
    kind: str
    network_id: int
    node_id: int | None = None
    edge_id: int | None = None
    related_ids: tuple[int, ...] = ()

    @property
    def message(self) -> str:
        network = f"N{self.network_id}"
        if self.kind == "formed":
            return f"Network {network} formed"
        if self.kind == "node_persistent":
            return f"Node {network}.{self.node_id} formed"
        if self.kind == "edge_persistent":
            return f"Edge {network}.E{self.edge_id} formed between {network}.{self.related_ids[0]} and {network}.{self.related_ids[1]}"
        if self.kind == "edge_dissolved":
            return f"Edge {network}.E{self.edge_id} dissolved"
        if self.kind == "node_dissolved":
            return f"Node {network}.{self.node_id} dissolved"
        if self.kind == "topology_changed":
            return f"Network {network} topology changed"
        if self.kind == "became_persistent":
            return f"Network {network} became persistent"
        if self.kind == "dissolved":
            return f"Network {network} dissolved"
        return f"Network {network} {self.kind}"


class NetworkTracker:
    """Track dense nodes and pathway edges over cluster observations."""

    def __init__(self, width: float, height: float, interaction_radius: float, structure_config: StructureDetectionConfig | None = None) -> None:
        self.width = float(width)
        self.height = float(height)
        self.interaction_radius = float(interaction_radius)
        self.structure_config = structure_config or load_structure_config()
        self.graph_neighbor_radius = self.interaction_radius * self.structure_config.graph_neighbor_radius_ratio
        self.active: dict[int, NetworkObservation] = {}
        self.completed: dict[int, NetworkObservation] = {}
        self.next_node_id: dict[int, int] = {}
        self.next_edge_id: dict[int, int] = {}
        self.last_events: list[NetworkEvent] = []

    def reset(self) -> None:
        self.active.clear()
        self.completed.clear()
        self.next_node_id.clear()
        self.next_edge_id.clear()
        self.last_events = []

    def update(self, state: ParticleState, clusters: Iterable[ClusterObservation], step: int) -> list[NetworkObservation]:
        current: dict[int, NetworkObservation] = {}
        events: list[NetworkEvent] = []
        for cluster in clusters:
            graph = build_particle_graph(state, cluster.particle_ids, self.width, self.height, self.graph_neighbor_radius)
            raw_nodes = self._detect_nodes(state, cluster, graph)
            previous = self.active.get(cluster.cluster_id)
            nodes = self._track_nodes(raw_nodes, previous.nodes if previous else [], cluster.cluster_id, step)
            edges = self._build_edges(state, graph, nodes, previous.edges if previous else [], cluster.cluster_id, step)
            observation = self._observation(cluster, graph, nodes, edges, step)
            current[cluster.cluster_id] = observation
            events.extend(self._events(previous, observation, step))
        for network_id, previous in self.active.items():
            if network_id not in current:
                self.completed[network_id] = previous
                for edge in previous.edges:
                    if edge.age_steps >= self.structure_config.network_min_edge_age_steps:
                        events.append(NetworkEvent(step, "edge_dissolved", network_id, edge_id=edge.edge_id))
                for node in previous.nodes:
                    if node.age_steps >= self.structure_config.network_min_node_age_steps:
                        events.append(NetworkEvent(step, "node_dissolved", network_id, node_id=node.node_id))
                if previous.classification == "NETWORK_STRUCTURE":
                    events.append(NetworkEvent(step, "dissolved", network_id))
        self.active = current
        self.last_events = events
        return list(current.values())

    def _detect_nodes(self, state: ParticleState, cluster: ClusterObservation, graph: ParticleGraph) -> list[NetworkNode]:
        dense = {node for node, neighbors in graph.adjacency.items() if len(neighbors) >= self.structure_config.network_min_node_degree}
        components = _components({node: graph.adjacency[node] & dense for node in dense})
        id_to_index = {int(value): index for index, value in enumerate(state.ids)}
        nodes: list[NetworkNode] = []
        for component in components:
            if len(component) < self.structure_config.network_min_node_particle_count:
                continue
            indexes = [id_to_index[value] for value in component if value in id_to_index]
            positions = state.positions[indexes]
            centroid = toroidal_centroid(positions, self.width, self.height)
            size = np.asarray([self.width, self.height], dtype=np.float64)
            offsets = (positions - centroid + size / 2.0) % size - size / 2.0
            radius = float(np.max(np.linalg.norm(offsets, axis=1))) if len(offsets) else 0.0
            density = len(component) / max(np.pi * max(radius, self.graph_neighbor_radius * 0.1) ** 2, 1e-12)
            distribution = np.bincount(state.species[indexes], minlength=int(np.max(state.species, initial=0)) + 1).astype(np.float64)
            distribution /= max(float(len(indexes)), 1.0)
            nodes.append(NetworkNode(0, frozenset(component), centroid, radius, density, distribution, np.mean(state.velocities[indexes], axis=0), 0, 0))
        return nodes

    def _track_nodes(self, raw_nodes: list[NetworkNode], previous: list[NetworkNode], network_id: int, step: int) -> list[NetworkNode]:
        assigned: set[int] = set()
        next_id = self.next_node_id.get(network_id, 1)
        tracked: list[NetworkNode] = []
        for raw in sorted(raw_nodes, key=lambda node: min(node.particle_ids)):
            matches = [(membership_overlap(old.particle_ids, raw.particle_ids), old) for old in previous if old.node_id not in assigned]
            overlap, prior = max(matches, key=lambda item: (item[0], -item[1].node_id), default=(0.0, None))
            if prior is None or overlap < self.structure_config.minimum_membership_overlap:
                raw.node_id = next_id
                next_id += 1
                raw.birth_step = step
                raw.identity_score = 0.0
            else:
                assigned.add(prior.node_id)
                raw.node_id = prior.node_id
                raw.birth_step = prior.birth_step
                raw.identity_score = _running_average(prior.identity_score, overlap, prior.age_steps)
            raw.last_seen_step = step
            tracked.append(raw)
        self.next_node_id[network_id] = next_id
        return tracked

    def _build_edges(self, state: ParticleState, graph: ParticleGraph, nodes: list[NetworkNode], previous: list[NetworkEdge], network_id: int, step: int) -> list[NetworkEdge]:
        edges: list[NetworkEdge] = []
        node_members = {node.node_id: node.particle_ids for node in nodes}
        for first_index, first in enumerate(nodes):
            for second in nodes[first_index + 1:]:
                path = _shortest_path(graph.adjacency, first.particle_ids, second.particle_ids)
                if not path or any(any(value in node.particle_ids for value in path[1:-1]) for node in nodes if node.node_id not in {first.node_id, second.node_id}):
                    continue
                path_particles = frozenset(value for value in path if value not in first.particle_ids and value not in second.particle_ids)
                path_length = _path_length(path, state, self.width, self.height)
                widths = [len(graph.adjacency[value]) for value in path_particles]
                density = len(path_particles) / max(path_length * self.graph_neighbor_radius * 2.0, 1e-12)
                key = (min(first.node_id, second.node_id), max(first.node_id, second.node_id))
                prior = next((item for item in previous if (min(item.node_a, item.node_b), max(item.node_a, item.node_b)) == key), None)
                if prior is None:
                    edge_id = self.next_edge_id.get(network_id, 1)
                    self.next_edge_id[network_id] = edge_id + 1
                    edge = NetworkEdge(edge_id, key[0], key[1], path_particles, len(path_particles), path_length, float(min(widths, default=1)), density, step, step)
                else:
                    edge = NetworkEdge(prior.edge_id, key[0], key[1], path_particles, len(path_particles), path_length, float(min(widths, default=1)), density, prior.birth_step, step, _running_average(prior.persistence, 1.0, prior.age_steps), 1.0 - membership_overlap(prior.path_particle_ids, path_particles))
                edges.append(edge)
        return edges

    def _observation(self, cluster: ClusterObservation, graph: ParticleGraph, nodes: list[NetworkNode], edges: list[NetworkEdge], step: int) -> NetworkObservation:
        persistent_nodes = [node for node in nodes if node.age_steps >= self.structure_config.network_min_node_age_steps]
        persistent_edges = [edge for edge in edges if edge.age_steps >= self.structure_config.network_min_edge_age_steps]
        node_count = len(nodes)
        edge_count = len(edges)
        topology = float(np.mean([edge.persistence for edge in edges])) if edges else 0.0
        node_stability = float(np.mean([node.identity_score for node in nodes])) if nodes else 0.0
        edge_stability = float(np.mean([edge.identity_score for edge in edges])) if edges else 0.0
        node_graph = {node.node_id: set() for node in nodes}
        for edge in edges:
            node_graph[edge.node_a].add(edge.node_b)
            node_graph[edge.node_b].add(edge.node_a)
        graph_diameter, average_path = _shortest_path_metrics(node_graph)
        density = 2.0 * edge_count / max(node_count * (node_count - 1), 1)
        largest_node_fraction = max((node.particle_count for node in nodes), default=0) / max(cluster.particle_count, 1)
        persistent = len(persistent_nodes) >= self.structure_config.network_min_persistent_nodes and len(persistent_edges) >= self.structure_config.network_min_persistent_edges
        if persistent:
            classification = "NETWORK_STRUCTURE"
        elif cluster.size_class == "MACRO_CLUSTER":
            classification = "MACRO_AGGREGATE"
        else:
            classification = "COMPACT_STRUCTURE"
        score = float(np.clip(0.25 * topology + 0.20 * node_stability + 0.20 * edge_stability + 0.15 * cluster.cohesion_score + 0.10 * cluster.dynamic_score + 0.10 * cluster.lifetime_score, 0.0, 1.0))
        history = [{"step": step, "node_count": node_count, "edge_count": edge_count, "network_score": score, "topology_persistence": topology}]
        previous = self.active.get(cluster.cluster_id)
        if previous:
            history = previous.history + history
        candidate = classification == "NETWORK_STRUCTURE" and cluster.age_steps >= self.structure_config.network_candidate_min_age_steps and score >= self.structure_config.network_candidate_min_score
        return NetworkObservation(cluster.cluster_id, cluster.cluster_id, nodes, edges, cluster.age_steps, topology, node_stability, edge_stability, sorted(len(value) for value in node_graph.values()), density, largest_node_fraction, graph_diameter, average_path, cluster.cohesion_score, cluster.dynamic_score, cluster.lifetime_score, score, classification, len(persistent_nodes), len(persistent_edges), graph.to_dict() | {"cluster_size": cluster.particle_count, "cluster_size_class": cluster.size_class}, history, candidate)

    def _events(self, previous: NetworkObservation | None, current: NetworkObservation, step: int) -> list[NetworkEvent]:
        if previous is None:
            return [NetworkEvent(step, "formed", current.network_id)] if current.nodes else []
        events: list[NetworkEvent] = []
        previous_nodes = {node.node_id: node for node in previous.nodes}
        for node in current.nodes:
            if node.node_id not in previous_nodes and node.age_steps >= self.structure_config.network_min_node_age_steps:
                events.append(NetworkEvent(step, "node_persistent", current.network_id, node_id=node.node_id))
        previous_edges = {(edge.node_a, edge.node_b): edge for edge in previous.edges}
        current_edges = {(edge.node_a, edge.node_b): edge for edge in current.edges}
        for key, edge in current_edges.items():
            if edge.age_steps >= self.structure_config.network_min_edge_age_steps and (key not in previous_edges or previous_edges[key].age_steps < self.structure_config.network_min_edge_age_steps):
                events.append(NetworkEvent(step, "edge_persistent", current.network_id, edge_id=edge.edge_id, related_ids=key))
        for key, edge in previous_edges.items():
            if key not in current_edges and edge.age_steps >= self.structure_config.network_min_edge_age_steps:
                events.append(NetworkEvent(step, "edge_dissolved", current.network_id, edge_id=edge.edge_id))
        old_topology = {(edge.node_a, edge.node_b) for edge in previous.edges if edge.age_steps >= self.structure_config.network_min_edge_age_steps}
        new_topology = {(edge.node_a, edge.node_b) for edge in current.edges if edge.age_steps >= self.structure_config.network_min_edge_age_steps}
        if old_topology != new_topology:
            events.append(NetworkEvent(step, "topology_changed", current.network_id))
        if previous.classification != "NETWORK_STRUCTURE" and current.classification == "NETWORK_STRUCTURE":
            events.append(NetworkEvent(step, "became_persistent", current.network_id))
        if previous.classification == "NETWORK_STRUCTURE" and current.classification != "NETWORK_STRUCTURE":
            events.append(NetworkEvent(step, "dissolved", current.network_id))
        return events


def _shortest_path(adjacency: dict[int, set[int]], starts: Iterable[int], ends: Iterable[int]) -> list[int]:
    targets = set(ends)
    queue = list(sorted(set(starts)))
    parent: dict[int, int | None] = {node: None for node in queue}
    found = next((node for node in queue if node in targets), None)
    while queue and found is None:
        node = queue.pop(0)
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
                if neighbor in targets:
                    found = neighbor
                    break
    if found is None:
        return []
    path: list[int] = []
    while found is not None:
        path.append(found)
        found = parent[found]
    return list(reversed(path))


def _path_length(path: list[int], state: ParticleState, width: float, height: float) -> float:
    if len(path) < 2:
        return 0.0
    id_to_index = {int(value): index for index, value in enumerate(state.ids)}
    size = np.asarray([width, height], dtype=np.float64)
    total = 0.0
    for first, second in zip(path, path[1:]):
        delta = (state.positions[id_to_index[second]] - state.positions[id_to_index[first]] + size / 2.0) % size - size / 2.0
        total += float(np.linalg.norm(delta))
    return total


def _running_average(previous: float, current: float, age: int) -> float:
    return current if age <= 0 else (previous * age + current) / (age + 1)
