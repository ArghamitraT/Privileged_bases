"""User-run checks for paired-workflow completion/failure detection.

Conda environment: mrl_env_cuda12
Usage:
    python NonLinearPB/NonLinearPBExperiments/tests/helper_comparison.py
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_dsprites_comparison import run_training


class ComparisonTests(unittest.TestCase):
    def test_child_failure_is_not_reported_as_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("scripts.run_dsprites_comparison.subprocess.run") as process:
                process.return_value.returncode = 1
                with self.assertRaisesRegex(RuntimeError, "failed with exit 1"):
                    run_training("vae", ["python", "training.py"], Path(directory))

    def test_success_requires_completed_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "vae_runs/exprmnt_test"
            folder.mkdir(parents=True)
            status = folder / "status.json"
            status.write_text(json.dumps({"status": "failed"}))
            with patch("scripts.run_dsprites_comparison.subprocess.run") as process:
                process.return_value.returncode = 0
                with self.assertRaisesRegex(RuntimeError, "did not record successful"):
                    run_training("vae", ["python", "training.py"], root)
                status.write_text(json.dumps({"status": "completed"}))
                self.assertEqual(run_training("vae", ["python", "training.py"], root), folder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
