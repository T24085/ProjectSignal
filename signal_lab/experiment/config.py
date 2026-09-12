"""Configuration for detector calibration and Experiment #1 candidate rules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class StructureDetectionConfig:
    """All detector, classification, and primary-candidate thresholds."""

    observation_interval: int
    minimum_cluster_size: int
    minimum_membership_overlap: float
    tracker_persistence_threshold: float
    transient_max_age_steps: int
    short_lived_min_age_steps: int
    persistent_min_age_steps: int
    long_lived_min_age_steps: int
    macro_cluster_particle_count: int
    micro_cluster_max_particle_count: int
    small_structure_max_particle_count: int
    medium_structure_max_particle_count: int
    large_structure_max_particle_count: int
    graph_neighbor_radius_ratio: float
    network_min_node_particle_count: int
    network_min_node_degree: int
    network_min_node_age_steps: int
    network_min_edge_age_steps: int
    network_min_persistent_nodes: int
    network_min_persistent_edges: int
    network_candidate_min_age_steps: int
    network_candidate_min_score: float
    primary_min_particle_count: int
    primary_max_particle_count: int
    primary_min_age_steps: int
    primary_min_structure_score: float
    primary_min_cohesion: float
    primary_min_identity: float
    primary_min_shape: float
    primary_min_dynamics: float
    primary_max_bridge_fraction: float

    @classmethod
    def from_toml(cls, path: str | Path) -> "StructureDetectionConfig":
        with Path(path).open("rb") as handle:
            raw = tomllib.load(handle)
        section = raw["structure_detection"]
        classifications = section["classifications"]
        primary = section["primary_candidate"]
        return cls(
            observation_interval=int(section["observation_interval"]),
            minimum_cluster_size=int(section["minimum_cluster_size"]),
            minimum_membership_overlap=float(section["minimum_membership_overlap"]),
            tracker_persistence_threshold=float(section["tracker_persistence_threshold"]),
            transient_max_age_steps=int(classifications["transient_max_age_steps"]),
            short_lived_min_age_steps=int(classifications["short_lived_min_age_steps"]),
            persistent_min_age_steps=int(classifications["persistent_min_age_steps"]),
            long_lived_min_age_steps=int(classifications["long_lived_min_age_steps"]),
            macro_cluster_particle_count=int(classifications["macro_cluster_particle_count"]),
            micro_cluster_max_particle_count=int(section["size_classes"]["micro_cluster_max_particle_count"]),
            small_structure_max_particle_count=int(section["size_classes"]["small_structure_max_particle_count"]),
            medium_structure_max_particle_count=int(section["size_classes"]["medium_structure_max_particle_count"]),
            large_structure_max_particle_count=int(section["size_classes"]["large_structure_max_particle_count"]),
            graph_neighbor_radius_ratio=float(section["network"]["graph_neighbor_radius_ratio"]),
            network_min_node_particle_count=int(section["network"]["min_node_particle_count"]),
            network_min_node_degree=int(section["network"]["min_node_degree"]),
            network_min_node_age_steps=int(section["network"]["min_node_age_steps"]),
            network_min_edge_age_steps=int(section["network"]["min_edge_age_steps"]),
            network_min_persistent_nodes=int(section["network"]["min_persistent_nodes"]),
            network_min_persistent_edges=int(section["network"]["min_persistent_edges"]),
            network_candidate_min_age_steps=int(section["network"]["candidate_min_age_steps"]),
            network_candidate_min_score=float(section["network"]["candidate_min_score"]),
            primary_min_particle_count=int(primary["min_particle_count"]),
            primary_max_particle_count=int(primary["max_particle_count"]),
            primary_min_age_steps=int(primary["min_age_steps"]),
            primary_min_structure_score=float(primary["min_structure_score"]),
            primary_min_cohesion=float(primary["min_cohesion"]),
            primary_min_identity=float(primary["min_identity"]),
            primary_min_shape=float(primary["min_shape"]),
            primary_min_dynamics=float(primary["min_dynamics"]),
            primary_max_bridge_fraction=float(primary["max_bridge_fraction"]),
        )

    def classify(self, age_steps: int, particle_count: int) -> str:
        if particle_count > self.macro_cluster_particle_count:
            return "MACRO_CLUSTER"
        if age_steps < self.short_lived_min_age_steps:
            return "TRANSIENT"
        if age_steps < self.persistent_min_age_steps:
            return "SHORT_LIVED"
        if age_steps < self.long_lived_min_age_steps:
            return "PERSISTENT"
        return "LONG_LIVED"

    def size_class(self, particle_count: int) -> str:
        """Return the configured particle-count class."""
        if particle_count <= self.micro_cluster_max_particle_count:
            return "MICRO_CLUSTER"
        if particle_count <= self.small_structure_max_particle_count:
            return "SMALL_STRUCTURE"
        if particle_count <= self.medium_structure_max_particle_count:
            return "MEDIUM_STRUCTURE"
        if particle_count <= self.large_structure_max_particle_count:
            return "LARGE_STRUCTURE"
        return "MACRO_CLUSTER"

    def is_primary_candidate(self, cluster: object) -> bool:
        """Apply the configured Experiment #1 candidate gate to an observation."""
        return (
            self.primary_min_particle_count <= cluster.particle_count <= self.primary_max_particle_count
            and cluster.age_steps >= self.primary_min_age_steps
            and cluster.structure_score >= self.primary_min_structure_score
            and cluster.cohesion_score >= self.primary_min_cohesion
            and cluster.identity_score >= self.primary_min_identity
            and cluster.shape_score >= self.primary_min_shape
            and cluster.dynamic_score >= self.primary_min_dynamics
            and cluster.bridge_fraction <= self.primary_max_bridge_fraction
            and cluster.classification != "MACRO_CLUSTER"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_interval": self.observation_interval,
            "minimum_cluster_size": self.minimum_cluster_size,
            "minimum_membership_overlap": self.minimum_membership_overlap,
            "tracker_persistence_threshold": self.tracker_persistence_threshold,
            "classifications": {
                "transient_max_age_steps": self.transient_max_age_steps,
                "short_lived_min_age_steps": self.short_lived_min_age_steps,
                "persistent_min_age_steps": self.persistent_min_age_steps,
                "long_lived_min_age_steps": self.long_lived_min_age_steps,
                "macro_cluster_particle_count": self.macro_cluster_particle_count,
            },
            "size_classes": {
                "micro_cluster_max_particle_count": self.micro_cluster_max_particle_count,
                "small_structure_max_particle_count": self.small_structure_max_particle_count,
                "medium_structure_max_particle_count": self.medium_structure_max_particle_count,
                "large_structure_max_particle_count": self.large_structure_max_particle_count,
            },
            "network": {
                "graph_neighbor_radius_ratio": self.graph_neighbor_radius_ratio,
                "min_node_particle_count": self.network_min_node_particle_count,
                "min_node_degree": self.network_min_node_degree,
                "min_node_age_steps": self.network_min_node_age_steps,
                "min_edge_age_steps": self.network_min_edge_age_steps,
                "min_persistent_nodes": self.network_min_persistent_nodes,
                "min_persistent_edges": self.network_min_persistent_edges,
                "candidate_min_age_steps": self.network_candidate_min_age_steps,
                "candidate_min_score": self.network_candidate_min_score,
            },
            "primary_candidate": {
                "min_particle_count": self.primary_min_particle_count,
                "max_particle_count": self.primary_max_particle_count,
                "min_age_steps": self.primary_min_age_steps,
                "min_structure_score": self.primary_min_structure_score,
                "min_cohesion": self.primary_min_cohesion,
                "min_identity": self.primary_min_identity,
                "min_shape": self.primary_min_shape,
                "min_dynamics": self.primary_min_dynamics,
                "max_bridge_fraction": self.primary_max_bridge_fraction,
            },
        }


@lru_cache(maxsize=1)
def load_structure_config(path: str | Path | None = None) -> StructureDetectionConfig:
    config_path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / "configs" / "default.toml"
    return StructureDetectionConfig.from_toml(config_path)
