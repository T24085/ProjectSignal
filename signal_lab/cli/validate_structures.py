"""Run the Phase 2B synthetic structure detector validation suite."""

from signal_lab.experiment.validation import format_validation_report, run_validation


def main() -> int:
    results = run_validation()
    print(format_validation_report(results), end="")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
