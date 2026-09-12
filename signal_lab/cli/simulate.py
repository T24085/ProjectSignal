"""Launch the Phase 1 interactive simulator."""

from __future__ import annotations

import argparse

from signal_lab.ui.simulator_view import SimulatorView


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Project SIGNAL Phase 1 viewer")
    parser.add_argument("--seed", type=int, default=0, help="deterministic initial-state seed")
    parser.add_argument("--particles", type=int, default=1000, help="particle count")
    args = parser.parse_args()
    SimulatorView(seed=args.seed, particle_count=args.particles).run()


if __name__ == "__main__":
    main()
