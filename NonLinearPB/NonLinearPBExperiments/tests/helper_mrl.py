"""User-run reconstruction-MRL loss, resume, ordering, and reporting checks.

Conda environment: mrl_env_cuda12
Usage:
    python NonLinearPB/NonLinearPBExperiments/tests/helper_mrl.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae_reproduction.model import VAE, masked_prefix, mrl_vae_loss
from experiments.dsprites_vae_reproduction import train_model, build_optimizer, report
from scripts.plot_dsprites_figure6 import coordinate_order
from scripts.run_dsprites_mrl_matrix import aggregate, CONDITIONS


class MRLTests(unittest.TestCase):
    def test_masks_and_common_sample(self):
        z = torch.randn(2, 10, requires_grad=True)
        torch.testing.assert_close(masked_prefix(z, 10), z, rtol=0, atol=0)
        masked_prefix(z, 3).sum().backward()
        torch.testing.assert_close(z.grad[:, :3], torch.ones(2, 3))
        torch.testing.assert_close(z.grad[:, 3:], torch.zeros(2, 7))
        model = VAE(hidden_dim=8)
        inputs, encoder_calls = [], []
        a = model.encoder.register_forward_hook(lambda *args: encoder_calls.append(1))
        b = model.decoder.register_forward_pre_hook(lambda _, args: inputs.append(args[0].detach().clone()))
        x = torch.randint(0, 2, (2, 4096)).float()
        loss, recon, kl, _, prefixes = mrl_vae_loss(model, x, 4)
        a.remove()
        b.remove()
        self.assertEqual(len(encoder_calls), 1)
        self.assertEqual(len(inputs), 10)
        for k, actual in enumerate(inputs, 1):
            torch.testing.assert_close(actual, masked_prefix(inputs[-1], k), rtol=0, atol=0)
        torch.testing.assert_close(loss, recon + 4 * kl)
        torch.testing.assert_close(recon, prefixes.mean())
        # Scalar BCE averaging differs from averaging logits before BCE.
        target = torch.zeros(2, 4)
        logits = [torch.zeros_like(target), torch.full_like(target, 4)]
        f = torch.nn.functional.binary_cross_entropy_with_logits
        self.assertGreater(abs(sum(f(v, target) for v in logits).item() / 2 - f(sum(logits) / 2, target).item()), .1)

    def test_sequential_gradients_match_single_objective_backward(self):
        model = VAE(hidden_dim=8)
        sequential = VAE(hidden_dim=8)
        sequential.load_state_dict(model.state_dict())
        x = torch.randint(0, 2, (2, 4096)).float()
        torch.manual_seed(37)
        loss, *_ = mrl_vae_loss(model, x, 4)
        loss.backward()
        torch.manual_seed(37)
        actual, *_ = mrl_vae_loss(sequential, x, 4, backward=True)
        torch.testing.assert_close(actual, loss.detach())
        for a, b in zip(model.parameters(), sequential.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=1e-5)

    def test_mrl_resume_and_one_step_per_batch(self):
        torch.manual_seed(71)
        images = torch.randint(0, 2, (6, 4096), dtype=torch.uint8)
        config = {"seed": 71, "epochs": 2, "batch_size": 3, "lr": .01,
                  "num_workers": 0, "latent_dim": 10, "hidden_dim": 8,
                  "beta": 4, "checkpoint_every": 1, "image_limit": 6,
                  "dataset_sha256": "synthetic", "reconstruction_mrl": True}
        with tempfile.TemporaryDirectory() as directory:
            source, recovered = Path(directory) / "source", Path(directory) / "recovered"
            source.mkdir()
            recovered.mkdir()
            model = VAE(hidden_dim=8)
            optimizer = build_optimizer(model, config)
            with patch("experiments.dsprites_vae_reproduction.build_optimizer", return_value=optimizer):
                with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    history = train_model(model, images, config, torch.device("cpu"), source)
                    self.assertEqual(step.call_count, 4)
            restored = VAE(hidden_dim=8)
            resume = {**config, "resume_checkpoint": str(source / "checkpoint_epoch_1.pt")}
            other = train_model(restored, images, resume, torch.device("cpu"), recovered)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(history[-1]["prefix_reconstruction_nll"], other[-1]["prefix_reconstruction_nll"])
            with self.assertRaisesRegex(ValueError, "mismatch for reconstruction_mrl"):
                train_model(restored, images, {**resume, "reconstruction_mrl": False}, torch.device("cpu"), recovered)

    def test_mrl_figure_keeps_native_indices(self):
        variance = np.arange(10, 0, -1)
        native, kind = coordinate_order({"reconstruction_mrl": True}, variance)
        np.testing.assert_array_equal(native, np.arange(10))
        self.assertEqual(kind, "native")
        ordinary, kind = coordinate_order({}, variance)
        np.testing.assert_array_equal(ordinary, np.arange(9, -1, -1))
        self.assertEqual(kind, "variance")
        with self.assertRaisesRegex(ValueError, "native coordinate"):
            coordinate_order({"reconstruction_mrl": True}, variance, "variance")

    def test_all_nine_scores_retained_and_conditions_separate(self):
        config = {"seeds": [0, 1, 2], "classifier_seeds": [0, 1, 2], "smoke": False}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {condition: {} for condition in CONDITIONS}
            for ci, (condition, (beta, mrl)) in enumerate(CONDITIONS.items()):
                for seed in config["seeds"]:
                    run = root / condition / str(seed)
                    run.mkdir(parents=True)
                    scores = [{"beta": beta, "reconstruction_mrl": mrl, "model_seed": seed,
                               "classifier_seed": c, "accuracy": .1 + .1 * ci + .02 * seed + .01 * c}
                              for c in config["classifier_seeds"]]
                    (run / "metric_scores.json").write_text(json.dumps(scores))
                    manifest[condition][str(seed)] = {"run_directory": str(run)}
            records, summary = aggregate(root, manifest, config)
            self.assertEqual(len(records), 36)
            for ci, condition in enumerate(CONDITIONS):
                self.assertEqual(summary[condition]["count"], 9)
                self.assertAlmostEqual(summary[condition]["all_mean"], .13 + .1 * ci)
            combined = root / "combined"
            combined.mkdir()
            report(records, combined, False, discard_bottom_half=False)
            stats = json.loads((combined / "metric_summary.json").read_text())
            self.assertEqual(len(stats), 4)
            for value in stats.values():
                self.assertEqual(value["count"], 9)
                self.assertEqual(value["score_filtering"], "none")
                self.assertFalse(any("best_half" in key for key in value))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
