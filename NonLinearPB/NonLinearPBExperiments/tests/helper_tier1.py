"""User-run checks for Tier 1 frozen-encoder target alignment (scripts/tier1_target_alignment.py).

Conda environment: mrl_env_cuda12
Usage:
    python NonLinearPB/NonLinearPBExperiments/tests/helper_tier1.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae_reproduction.bases import random_orthogonal
from vae_reproduction.probes import _scores
from vae_reproduction.tasks import N_IMAGES, image_factors, make_split
from scripts.tier1_target_alignment import (angle_scores, completed_basis, curve_record, equal_frequency_labels,
                                            geometry, make_targets, null_conventions, overlap, overlap_null,
                                            scatter_directions, select_jobs)


class TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.factors = image_factors(np.arange(N_IMAGES))
        cls.split = make_split(0, (0.8, 0.1, 0.1))

    def test_pure_angles_are_x_and_y(self):
        np.testing.assert_array_equal(angle_scores(self.factors, 0), self.factors[:, 3])
        np.testing.assert_array_equal(angle_scores(self.factors, 90), self.factors[:, 4])   # rounding removes cos(90) noise

    def test_bins_from_train_and_ties_go_up(self):
        scores = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 9], dtype=float)
        labels, edges = equal_frequency_labels(scores, np.arange(12), 6)
        np.testing.assert_array_equal(edges, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(labels[:12], np.repeat(np.arange(6), 2))
        self.assertEqual(labels[12], 5)                     # outside the train range: last class

    def test_six_targets_nearly_balanced(self):
        targets = make_targets(self.factors, self.split)
        self.assertEqual(list(targets), ["alpha0", "alpha30", "alpha60", "alpha90", "scale", "shape"])
        for name in ("alpha0", "alpha30", "alpha60", "alpha90"):
            share = np.bincount(targets[name]["labels"][self.split["train"]]) / len(self.split["train"])
            self.assertEqual(len(share), 6)
            self.assertLess(share.max() - share.min(), 0.04)
        self.assertFalse(np.array_equal(targets["alpha30"]["labels"], targets["alpha60"]["labels"]))


class GeometryTests(unittest.TestCase):
    def test_scatter_recovers_class_axis(self):
        rng = np.random.default_rng(0)
        labels = np.repeat(np.arange(6), 500)
        z = rng.normal(size=(3000, 10)) * .1
        z[:, 2] += labels                                   # classes separate along coordinate 3 only
        values, vectors, rank = scatter_directions(z, labels, 6)
        self.assertEqual(rank, 1)
        self.assertGreater(abs(vectors[2, 0]), .999)
        self.assertLessEqual(rank, 5)

    def test_overlap_and_a1(self):
        d = 10
        m = np.eye(d)[:, [0, 3]]
        o = overlap(m)
        self.assertEqual(o[0, 0], 1.0)                       # A(1)
        self.assertEqual(o[1, 1], .5)                        # ||P_2 M_2||^2 / 2
        self.assertEqual(o[3, 1], 1.0)
        g = geometry(m, np.array([2.0, 1.0]))
        self.assertEqual(g["depth_leading"], 1.0)
        self.assertAlmostEqual(g["depth_weighted"], (2 * 1 + 1 * 4) / 3)

    def test_null_mean_matches_analytic(self):
        mask = [1] * 4 + [0] * 6                             # active subspace = first four coordinates
        samples = overlap_null(2, mask, 4000, 0)
        k_active = np.minimum(np.arange(1, 11), 4)
        expected = k_active * 2 / (4 * np.minimum(np.arange(1, 11), 2))
        np.testing.assert_allclose(samples.mean(0)[:, 1], expected, atol=.02)

    def test_null_conventions_keep_rules_separate(self):
        masks = null_conventions({"variance_mask": [1, 0], "recon_gain_mask": [1, 1]}, 2)
        self.assertEqual(masks, {"all10": [1, 1], "variance": [1, 0], "recon_gain": [1, 1]})

    def test_completed_basis_is_orthogonal_and_keeps_lead(self):
        x = np.random.default_rng(0).normal(size=(500, 10))
        lead = random_orthogonal(10, 3)[:, :2]
        basis = completed_basis(lead, x)
        np.testing.assert_allclose(basis.T @ basis, np.eye(10), atol=1e-10)
        np.testing.assert_allclose(basis[:, :2], lead)


class ContributionTests(unittest.TestCase):
    def test_identity_is_one_and_best(self):
        from scripts.tier1_contribution import contributions, ordered_alignment
        self.assertAlmostEqual(ordered_alignment(np.eye(10)), 1.0)
        for seed in range(5):
            self.assertLess(ordered_alignment(contributions(random_orthogonal(10, seed))), 1.0)

    def test_contributions_are_doubly_stochastic(self):
        from scripts.tier1_contribution import contributions
        c = contributions(random_orthogonal(10, 5))
        np.testing.assert_allclose(c.sum(0), 1)
        np.testing.assert_allclose(c.sum(1), 1)

    def test_axis_angles(self):
        from scripts.tier1_contribution import axis_angles
        theta = axis_angles(np.array([[1, -1, .7071067811865476, -.7071067811865476, 0]]))
        np.testing.assert_allclose(theta[0], [0, 0, 45, 45, 90], atol=1e-6)    # sign-free

    def test_a_ordered_matches_formula(self):
        from scripts.tier1_contribution import ordered_alignment
        c = np.eye(10)[:, [2, 0, 1, 3, 4, 5, 6, 7, 8, 9]]       # target 1 = source 3, 2 = source 1, 3 = source 2
        w = 1 / np.arange(1, 11)
        expected = (w[2] * w[0] + w[0] * w[1] + w[1] * w[2] + (w[3:] ** 2).sum()) / (w ** 2).sum()
        self.assertAlmostEqual(ordered_alignment(c), expected)


class ProbeTests(unittest.TestCase):
    def test_balanced_accuracy(self):
        y = torch.tensor([0, 0, 0, 1])
        logits = torch.tensor([[1., 0], [1, 0], [1, 0], [1, 0]])   # always class 0
        scores = _scores(logits, y)
        self.assertAlmostEqual(scores["accuracy"], .75)
        self.assertAlmostEqual(scores["balanced_accuracy"], .5)

    def test_k_star_rule(self):
        record = curve_record([.2, .5, .8, .9, .9], [0] * 5, None, 6)
        normalized = record["normalized"]
        self.assertEqual(record["k_star"], next(k for k, v in enumerate(normalized, 1) if v >= .9 * normalized[-1]))
        self.assertTrue(record["k_star_informative"])
        self.assertFalse(curve_record([.17, .17], [0, 0], None, 6)["k_star_informative"])

    def test_default_jobs_skip_pilots_and_filter_beta(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            for name in ("recon_beta0_seed0", "posx_beta0_seed1_lam100", "posx_beta4_seed0_lam100_ep10",
                         "recon_beta4_seed0"):
                (Path(root) / name).mkdir()
                (Path(root) / name / "checkpoint_final.pt").touch()
            self.assertEqual(select_jobs(Path(root), None, [0.0]), ["posx_beta0_seed1_lam100", "recon_beta0_seed0"])


if __name__ == "__main__":
    unittest.main()
