"""User-run checks for supervised/deterministic MRL pretraining and its per-job analysis.

Conda environment: mrl_env_cuda12
Usage:
    python NonLinearPB/NonLinearPBExperiments/tests/helper_pretrain.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae_reproduction.model import VAE, PrefixHeads, masked_prefix, mrl_pretrain_loss, posterior_kl, prefix_masks
from scripts.analyze_pretrain import activity, correlation_ratio, latest_analysis
from scripts.run_dsprites_pretrain import jobs_for


class PretrainLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = VAE(hidden_dim=8)
        self.x = torch.randint(0, 2, (3, 4096)).float()

    def test_masks_keep_prefixes(self):
        masks = prefix_masks(10, "cpu")
        z = torch.randn(2, 10)
        for k in range(1, 11):
            torch.testing.assert_close(z * masks[k - 1], masked_prefix(z, k), rtol=0, atol=0)

    def test_beta0_is_deterministic_and_matches_per_prefix_bce(self):
        out1 = mrl_pretrain_loss(self.model, self.x, 0.0)
        out2 = mrl_pretrain_loss(self.model, self.x, 0.0)
        torch.testing.assert_close(out1["loss"], out2["loss"], rtol=0, atol=0)
        self.assertEqual(out1["kl"].item(), 0.0)
        mean, _ = self.model.encode(self.x)
        manual = torch.stack([F.binary_cross_entropy_with_logits(
            self.model.decoder(masked_prefix(mean, k)), self.x, reduction="none").sum(1).mean()
            for k in range(1, 11)])
        torch.testing.assert_close(out1["prefix_recon"], manual)
        torch.testing.assert_close(out1["loss"], manual.mean())

    def test_kl_is_full_latent_once(self):
        torch.manual_seed(1)
        out = mrl_pretrain_loss(self.model, self.x, 4.0)
        mean, logvar = self.model.encode(self.x)
        torch.testing.assert_close(out["kl"], posterior_kl(mean, logvar).sum())
        torch.testing.assert_close(out["loss"], out["recon"] + 4.0 * out["kl"])

    def test_heads_read_only_their_prefix(self):
        heads = PrefixHeads(10, 32)
        z = torch.randn(4, 10, requires_grad=True)
        heads(z)[2].sum().backward()            # prefix k = 3
        self.assertTrue(torch.all(z.grad[:, 3:] == 0))
        self.assertTrue(torch.any(z.grad[:, :3] != 0))

    def test_supervised_term_is_one_at_uniform_prediction(self):
        heads = PrefixHeads(10, 32)
        for head in heads.heads:
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
        labels = torch.randint(0, 32, (3,))
        out = mrl_pretrain_loss(self.model, self.x, 0.0, heads, labels, 100.0, math.log(32))
        self.assertAlmostEqual(out["sup"].item(), 1.0, places=5)
        torch.testing.assert_close(out["loss"], out["recon"] + 100.0 * out["sup"])


class AnalysisTests(unittest.TestCase):
    def test_correlation_ratio(self):
        labels = np.repeat(np.arange(4), 50)
        values = np.stack([labels.astype(float), np.random.default_rng(0).normal(size=200)], 1)
        eta = correlation_ratio(values, labels)
        self.assertAlmostEqual(eta[0], 1.0)
        self.assertLess(eta[1], 0.1)

    def test_recon_gain_activity_counts_beta0_usage(self):
        means = np.random.default_rng(0).normal(size=(100, 3)) * np.array([1, 1e-3, 1])
        recon = np.array([100.0, 50.0, 49.9])
        out = activity(means.astype(np.float32), np.zeros_like(means, dtype=np.float32), recon, 0.0)
        self.assertEqual(out["recon_gain_mask"], [1, 1, 0])
        self.assertEqual(out["variance_mask"], [1, 0, 1])
        self.assertNotIn("kl_mask", out)

    def test_latest_analysis_prefers_newest_folder(self):
        with tempfile.TemporaryDirectory() as root:
            job = Path(root) / "recon_beta0_seed0"
            for stamp, value in (("2026_01_01__00_00_00", 1), ("2026_02_01__00_00_00", 2)):
                (job / f"analysis_{stamp}").mkdir(parents=True)
                (job / f"analysis_{stamp}" / "analysis.json").write_text(json.dumps({job.name: {"v": value}}))
            self.assertEqual(latest_analysis(job), {"v": 2})
            self.assertIsNone(latest_analysis(Path(root) / "missing"))

    def test_short_runs_get_epoch_suffix(self):
        from experiments.dsprites_pretrain import job_name
        self.assertEqual(job_name("posx", 4.0, 0, 100.0), "posx_beta4_seed0_lam100")
        self.assertEqual(job_name("posx", 4.0, 0, 100.0, 10), "posx_beta4_seed0_lam100_ep10")
        self.assertEqual(job_name("recon", 0.0, 1, 0.0), "recon_beta0_seed1")

    def test_queue_jobs(self):
        class Args:
            stages, betas, seeds, pilot_lambdas, lambda_sup = ["pilot", "recon", "posx"], [0.0, 4.0], [0, 1], [10.0], 100.0
        jobs = jobs_for(Args)
        self.assertEqual(len(jobs), 1 + 4 + 4)
        self.assertIn("--lambda-sup", jobs[-1])
        self.assertNotIn("--lambda-sup", jobs[1])


if __name__ == "__main__":
    unittest.main()
