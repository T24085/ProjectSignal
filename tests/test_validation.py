import unittest

from signal_lab.experiment.validation import format_validation_report, run_validation


class ValidationSuiteTests(unittest.TestCase):
    def test_all_synthetic_validation_cases_pass(self):
        results = run_validation()
        self.assertEqual(len(results), 8)
        self.assertTrue(all(result.passed for result in results))

    def test_validation_report_is_human_readable(self):
        report = format_validation_report()
        self.assertIn("STRUCTURE DETECTOR VALIDATION", report)
        self.assertIn("8/8 VALIDATION TESTS PASSED", report)
        self.assertIn("Frozen crystal", report)
        self.assertIn("Toroidal boundary crossing", report)


if __name__ == "__main__":
    unittest.main()
