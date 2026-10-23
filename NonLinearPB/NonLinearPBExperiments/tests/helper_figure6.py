"""Validate factorial marginalization for Figure6 (user-run correctness test).

Conda environment: mrl_env_cuda12
Usage:
    python NonLinearPB/NonLinearPBExperiments/tests/helper_figure6.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_dsprites_figure6 import factor_responses, TRAVERSAL_VALUES


class Figure6Tests(unittest.TestCase):
    def test_marginalization_preserves_designated_factors(self):
        factors = np.indices((3, 6, 40, 32, 32)).reshape(5, -1).T
        means = np.column_stack((factors[:, 3], factors[:, 4],
                                 factors[:, 0] + factors[:, 1],
                                 factors[:, 0] + factors[:, 2])).astype(np.float32)
        responses = factor_responses(means)
        np.testing.assert_allclose(responses["position"][:, :, 0], np.repeat(np.arange(32)[:, None], 32, axis=1))
        np.testing.assert_allclose(responses["position"][:, :, 1], np.repeat(np.arange(32)[None, :], 32, axis=0))
        np.testing.assert_allclose(responses["scale"][:, :, 2], np.arange(3)[:, None] + np.arange(6)[None, :])
        np.testing.assert_allclose(responses["rotation"][:, :, 3], np.arange(3)[:, None] + np.arange(40)[None, :])

    def test_traversal_matches_five_rows_and_prior_range(self):
        np.testing.assert_array_equal(TRAVERSAL_VALUES, [3, 1.5, 0, -1.5, -3])


if __name__ == "__main__":
    unittest.main()
