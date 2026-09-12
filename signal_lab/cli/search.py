"""Run the Phase 2A persistent-structure search."""

from __future__ import annotations

import argparse
from pathlib import Path

from signal_lab.search.runner import SearchConfig, run_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Search deterministic universes for persistent dynamic structures")
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--persistence-threshold", type=float, default=0.55)
    parser.add_argument("--observation-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--headless", action="store_true", help="accepted for compatibility; search is always headless")
    args = parser.parse_args()
    run_search(SearchConfig(runs=args.runs, steps=args.steps, particle_count=args.particles, min_cluster_size=args.min_cluster_size, persistence_threshold=args.persistence_threshold, observation_interval=args.observation_interval, seed=args.seed, result_root=args.results))


if __name__ == "__main__":
    main()
