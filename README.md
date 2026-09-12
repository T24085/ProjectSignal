# Project SIGNAL

Project SIGNAL is a reproducible research platform for testing whether very simple, local particle interactions can organize into structures that transmit a measurable perturbation.

> Project SIGNAL searches for reproducible emergent behavior. A high-scoring simulation is a research candidate, not evidence of a new law of physics, life, intelligence, or computation until independently analyzed and reproduced.

## Phase 1

This repository currently implements the Phase 1 deterministic simulation core:

- a 2D toroidal world;
- three-species particle states;
- asymmetric interaction genomes;
- smooth attraction/repulsion forces with universal core repulsion;
- a uniform-grid spatial hash;
- deterministic seeded initialization and reset;
- JSON genome and compressed particle-state persistence;
- a simple Tkinter real-time viewer.

Phase 2A/2B now adds persistent-structure detection, stable cluster identity tracking, cohesion-aware structure scoring, deterministic synthetic validation, replayable structure snapshots, and a headless random-search harness. Signal injection and large-scale genome search remain intentionally gated until the Phase 2B validation has been independently reviewed.

## Installation

Python 3.12+ is recommended. Phase 1 requires NumPy. Tkinter is included with most Windows Python distributions.

```powershell
py -3 -m pip install -r requirements.txt
```

For development tests:

```powershell
py -3 -m pip install -e ".[dev]"
```

## Run the simulator

```powershell
py -3 -m signal_lab.cli.simulate
```

The viewer starts paused. Use Start/Pause, Step, Reset, Randomize Genome, and the seed field to explore deterministic universes. Save Universe writes a genome and particle state to a selected directory; Load Universe restores both.

## Physics

Each particle has only an id, species, position, and velocity. For each nearby particle, the engine uses minimum-image toroidal distance. Within the core radius, particles receive universal repulsion. Outside the core, the signed interaction matrix value is smoothly shaped by a sine envelope. The matrix entry is directional: `A[i][j]` controls the force felt by species `i` from species `j`.

The spatial hash uses cells the size of the interaction radius and searches the current cell plus the eight neighboring cells. Neighbor traversal is sorted to keep results reproducible across runs.

The simulation batches all discovered directed neighbor forces into NumPy arrays each step. This preserves the force equations while avoiding a separate Python/NumPy call for every pair. Headless search also performs no rendering; use a larger `--observation-interval` when exploratory speed matters and reduce it for final measurements.

## Determinism

The initial state is generated from an explicit seed. Resetting an engine regenerates the same state and simulation steps are deterministic for a fixed genome, seed, and initial state. Save/load round trips preserve the complete particle arrays.

## Phase 2A structure search

Run the first headless search with:

```powershell
py -3 -m signal_lab.cli.search --runs 1000
```

For a shorter smoke run:

```powershell
py -3 -m signal_lab.cli.search --runs 1 --steps 20 --particles 50 --min-cluster-size 3
```

Search results are appended to `results/search_results.csv`. Persistent candidates are saved under `results/structures/` with their genome, seed, complete particle state, cluster measurements, and particle-count history. The dashboard's Search and Results buttons open the corresponding controls and result table; double-clicking a candidate loads its replay state into the simulator.

The batch launcher [run_search.bat](run_search.bat) exposes the same three primary experiment-size settings near the top of the file:

```bat
set RUNS=1000
set STEPS_PER_RUN=5000
set PARTICLES=1000
```

The Search window exposes these as editable `Genomes / runs`, `Steps per run`, and `Particles / universe` fields. Settings changed while a search is active apply to the next search after the current one is stopped or completed.

## Phase 2B validation

Run the deterministic synthetic validation suite before starting a large search:

```powershell
py -3 -m signal_lab.cli.validate_structures
```

The suite covers compact moving blobs, bridged blobs, random clouds, expanding clouds, frozen crystals, internal circulation, crossing structures, and toroidal boundary crossing. It reports raw cohesion, identity, shape, dynamics, connectivity, bridge, and temporal metrics.

## Scientific scope

Phase 1 provides mechanics for later experiments; it does not claim that visually interesting behavior is communication, computation, life, or novelty. Future phases must retain raw measurements, controls, seeds, configuration, and complete replays.
