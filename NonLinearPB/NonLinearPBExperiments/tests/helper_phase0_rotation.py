"""User-run checks for splits, task labels, probes and Experiment A bases.

Conda environment: mrl_env_cuda12
Usage:
    python NonLinearPB/NonLinearPBExperiments/tests/helper_phase0_rotation.py

Synthetic CPU data only, except test_labels_match_archive (reads the
dSprites annotations, not the images).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae_reproduction.tasks import (N_IMAGES, HEART, make_split, split_checksum, factor_labels,
                                    task_split, image_factors)
from vae_reproduction.probes import fit_linear_probe, greedy_coordinate_order, k_star
from vae_reproduction.bases import (initial_basis, fit_nested_rotation, fit_greedy_nested, pca_basis,
                                    lda_basis, permutation_basis, orthogonality_error, principal_angles)
from vae_reproduction.runs import DATA_PATH

CPU = torch.device("cpu")
FAST = {"max_epochs": 40, "batch_size": 512, "lr": 5e-2, "patience": 5}


def synthetic(n=3000, d=6, informative=4, classes=4, seed=0):
    """Labels depend only on coordinate `informative`; others are noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(classes, size=n)
    x = rng.normal(size=(n, d))
    x[:, informative] = 3.0 * y + 0.3 * rng.normal(size=n)
    return x.astype(np.float32), y


def splits(x, y):
    a, b = len(x) * 3 // 5, len(x) * 4 // 5
    return (x[:a], y[:a]), (x[a:b], y[a:b]), (x[b:], y[b:])


class SplitAndLabelTests(unittest.TestCase):
    def test_split_disjoint_complete_deterministic(self):
        split = make_split(0)
        joined = np.concatenate([split["train"], split["validation"], split["test"]])
        self.assertEqual(len(joined), N_IMAGES)
        self.assertEqual(len(np.unique(joined)), N_IMAGES)
        self.assertEqual(len(split["train"]), round(0.8 * N_IMAGES))
        self.assertEqual(split_checksum(split), split_checksum(make_split(0)))
        self.assertNotEqual(split_checksum(split), split_checksum(make_split(1)))

    def test_rotation_merge_and_heart_subset(self):
        indices = np.arange(N_IMAGES)
        raw = image_factors(indices)[:, 2]
        merged = factor_labels(indices, "rotation")
        self.assertTrue(np.all(merged[raw == 39] == 0))
        self.assertTrue(np.array_equal(merged[raw < 39], raw[raw < 39]))
        self.assertEqual(merged.max(), 38)
        hearts, labels = task_split("rotation_heart", indices)
        self.assertEqual(len(hearts), N_IMAGES // 3)
        self.assertTrue(np.all(image_factors(hearts)[:, 0] == HEART))
        self.assertEqual(len(labels), len(hearts))

    def test_labels_match_archive(self):
        if not DATA_PATH.exists():
            self.skipTest("dSprites archive not found")
        with np.load(DATA_PATH, allow_pickle=False) as archive:
            classes = archive["latents_classes"][:, 1:]
        sample = np.random.default_rng(0).choice(N_IMAGES, 5000, replace=False)
        for column, factor in enumerate(("shape", "scale", "rotation", "x", "y")):
            expected = classes[sample, column] % (39 if factor == "rotation" else 10 ** 6)
            self.assertTrue(np.array_equal(factor_labels(sample, factor), expected), factor)


class ProbeTests(unittest.TestCase):
    def test_full_linear_accuracy_invariant_to_rotation(self):
        x, y = synthetic()
        q = initial_basis("random3", x.shape[1]).astype(np.float32)
        train, val, test = splits(x, y)
        rotate = lambda s: (s[0] @ q, s[1])
        a = fit_linear_probe(train, val, test, 4, device=CPU)
        b = fit_linear_probe(rotate(train), rotate(val), rotate(test), 4, device=CPU)
        self.assertTrue(a["converged"] and b["converged"])
        self.assertAlmostEqual(a["test"]["accuracy"], b["test"]["accuracy"], delta=0.005)
        self.assertGreater(a["test"]["accuracy"], 0.95)

    def test_greedy_order_and_k_star(self):
        train, val, _ = splits(*synthetic())
        order, curve = greedy_coordinate_order(train, val, 4, device=CPU)
        self.assertEqual(order[0], 4)
        self.assertEqual(sorted(order), list(range(6)))
        self.assertEqual(k_star(curve, 0.02), 1)
        self.assertEqual(k_star([0.2, 0.5, 0.9, 0.91], 0.01), 3)


class BasisTests(unittest.TestCase):
    def test_closed_form_bases_orthonormal(self):
        x, y = synthetic()
        for q in (pca_basis(x)[0], lda_basis(x, y, 4)[0], permutation_basis([3, 0, 5, 1, 2, 4])):
            self.assertLess(orthogonality_error(q), 1e-8)
        lda, values = lda_basis(x, y, 4)
        self.assertLessEqual(len(values), 3)
        self.assertGreater(abs(lda[4, 0]), 0.99)  # first discriminant = informative axis

    def test_joint_mrl_moves_information_to_first_axis(self):
        train, val, test = splits(*synthetic())
        for init in ("identity", "random1"):
            q, heads, center, log = fit_nested_rotation(train, val, 4, range(1, 7), initial_basis(init, 6),
                                                        device=CPU, seed=0, config=FAST)
            self.assertLess(log["orthogonality_error"], 1e-5)
            self.assertGreater(abs(q[4, 0]), 0.95, init)
            fresh = fit_linear_probe((train[0] @ q[:, :1], train[1]), (val[0] @ q[:, :1], val[1]),
                                     (test[0] @ q[:, :1], test[1]), 4, device=CPU)
            self.assertGreater(fresh["test"]["accuracy"], 0.9)

    def test_greedy_nested_orthonormal_and_first_axis(self):
        train, val, _ = splits(*synthetic())
        q, log = fit_greedy_nested(train, val, 4, device=CPU, seed=0, config=FAST)
        self.assertEqual(q.shape, (6, 6))
        self.assertLess(log["orthogonality_error"], 1e-8)
        self.assertGreater(abs(q[4, 0]), 0.95)

    def test_principal_angles(self):
        q = initial_basis("random0", 6)
        np.testing.assert_allclose(principal_angles(q[:, :2], q[:, [1, 0]] * -1), 0, atol=1e-7)
        np.testing.assert_allclose(principal_angles(q[:, :1], q[:, 1:2]), [np.pi / 2], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
