"""Run the detector calibration suite and preserve its raw report."""

from __future__ import annotations

import argparse
from pathlib import Path

from signal_lab.experiment.validation import format_validation_report, run_validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the eight-case Project SIGNAL detector calibration suite")
    parser.add_argument("--report", type=Path, default=Path("validation_report.txt"))
    args = parser.parse_args()
    results = run_validation()
    report = format_validation_report(results)
    print(report, end="")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"Report written to {args.report}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
