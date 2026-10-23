"""CPU correctness tests for VAE loss, checkpointing and factor metric.

Conda environment: mrl_env_cuda12
Usage:
    python NonLinearPB/NonLinearPBExperiments/tests/run_tests.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae_reproduction.data import FACTOR_SIZES, TARGET_COLUMNS, factor_indices
from vae_reproduction.model import VAE, vae_loss, model_from_config
from vae_reproduction.metric import sample_pair_indices, metric_examples, fit_classifier
from experiments.dsprites_vae_reproduction import (
    save_checkpoint, train_model, capacity_at_step, build_optimizer, report,
)


class ReproductionTests(unittest.TestCase):
    def test_factor_index_order_and_pair_constraints(self):
        factors = np.indices(tuple(FACTOR_SIZES)).reshape(5, -1).T
        np.testing.assert_array_equal(factor_indices(factors), np.arange(737280))
        targets = np.repeat(np.arange(4), 32)
        left, right = sample_pair_indices(targets, 64, np.random.default_rng(5))
        a, b = factors[left], factors[right]
        for target, column in enumerate(TARGET_COLUMNS):
            mask = targets == target
            np.testing.assert_array_equal(a[mask, :, column], b[mask, :, column])
            self.assertGreater(np.unique(a[mask, :, column]).size, 1)
            for nuisance in set(range(5)) - {column}:
                self.assertTrue(np.any(a[mask, :, nuisance] != b[mask, :, nuisance]))

    def test_loss_units_and_analytic_kl(self):
        x = torch.zeros(3, 4096)
        mean = torch.ones(3, 10)
        logvar = torch.zeros_like(mean)
        loss, recon, kl, coordinates = vae_loss(torch.zeros_like(x), x, mean, logvar, 4)
        self.assertAlmostEqual(recon.item(), 4096 * np.log(2), places=2)
        self.assertAlmostEqual(kl.item(), 5)
        self.assertAlmostEqual(loss.item(), recon.item() + 20, places=3)
        torch.testing.assert_close(coordinates, torch.full((10,), 0.5))

    def test_reparameterization_and_checkpoint_reload(self):
        torch.manual_seed(4)
        model = VAE(hidden_dim=16)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
        x = torch.rand(3, 4096)
        logits, mean, logvar = model(x)
        vae_loss(logits, x, mean, logvar, 4)[0].backward()
        self.assertGreater(model.mean.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.logvar.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        generator = torch.Generator().manual_seed(4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, optimizer, generator, 1, {"seed": 4})
            state = torch.load(path, map_location="cpu", weights_only=False)
            copy = VAE(hidden_dim=16)
            copy.load_state_dict(state["model"])
            torch.testing.assert_close(model.encode(x)[0], copy.encode(x)[0])
            self.assertEqual(state["epoch"], 1)
            self.assertIn("loader_rng", state)
            self.assertTrue(state["optimizer"]["state"])

    def test_default_architecture_and_optional_checkpoint_loading(self):
        torch.manual_seed(7)
        baseline = VAE(hidden_dim=16)
        torch.manual_seed(7)
        explicit = VAE(hidden_dim=16, decoder_activation="tanh", initialization="default")
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
        cfg = {"latent_dim": 10, "hidden_dim": 16,
               "decoder_activation": "relu", "initialization": "xavier"}
        variant = model_from_config(cfg)
        for layer in variant.modules():
            if isinstance(layer, torch.nn.Linear):
                self.assertTrue(torch.all(layer.bias == 0))
        x = torch.rand(2, 4096)
        restored = model_from_config(cfg)
        restored.load_state_dict(variant.state_dict())
        z = variant.encode(x)[0]
        torch.testing.assert_close(variant.decoder(z), restored.decoder(z))
        self.assertEqual(sum(isinstance(layer, torch.nn.ReLU) for layer in variant.decoder), 3)
        self.assertEqual(sum(isinstance(layer, torch.nn.Tanh) for layer in model_from_config(
            {"latent_dim": 10, "hidden_dim": 16}).decoder), 3)

    def test_capacity_loss_reduction_and_gradient_direction(self):
        x = torch.zeros(2, 4096)
        # Image KLs 0 and 2; batch-mean KL=1 exactly matches capacity=1.
        mean = torch.tensor([[0.0], [2.0]], requires_grad=True)
        logvar = torch.zeros_like(mean)
        loss, recon, kl, _ = vae_loss(torch.zeros_like(x), x, mean, logvar, 4,
                                     objective="capacity", capacity=1, gamma=7)
        self.assertEqual(kl.item(), 1)
        torch.testing.assert_close(loss, recon)
        for target, expected_sign in ((0.0, 1), (2.0, -1)):
            mean.grad = None
            loss = vae_loss(torch.zeros_like(x), x, mean, logvar, 4,
                            objective="capacity", capacity=target, gamma=7)[0]
            loss.backward()
            self.assertEqual(int(torch.sign(mean.grad[1, 0])), expected_sign)
        cfg = {"capacity_max": 25, "capacity_steps": 100000}
        self.assertEqual(capacity_at_step(cfg, 0), 0)
        self.assertEqual(capacity_at_step(cfg, 50000), 12.5)
        self.assertEqual(capacity_at_step(cfg, 200000), 25)
        optimizer = build_optimizer(VAE(hidden_dim=16), {"optimizer": "adam", "lr": 0.0005})
        self.assertIsInstance(optimizer, torch.optim.Adam)

    def test_capacity_report_has_no_fixed_beta_reference(self):
        scores = [{"beta": 4.0, "objective": "capacity", "model_seed": 0,
                   "classifier_seed": 0, "accuracy": 0.5}]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report(scores, folder, True)
            import json
            summary = json.loads((folder / "metric_summary.json").read_text())
            self.assertIsNone(summary["capacity"]["paper_reference"])
            self.assertIn("capacity-controlled", (folder / "results_summary.txt").read_text())

    def test_metric_ground_truth_and_shuffled_labels(self):
        factors = np.indices(tuple(FACTOR_SIZES)).reshape(5, -1).T
        means = (factors[:, TARGET_COLUMNS] / (FACTOR_SIZES[list(TARGET_COLUMNS)] - 1)).astype(np.float32)
        train = metric_examples(means, 1000, 64, 11)
        validation = metric_examples(means, 400, 64, 12)
        test = metric_examples(means, 1000, 64, 13)
        self.assertFalse(np.array_equal(train[0], test[0]))
        self.assertTrue(np.all(np.abs(test[0][np.arange(1000), test[1]]) < 1e-7))
        config = {"lr": 0.01, "batch_size": 128, "epochs": 150, "patience": 20, "min_delta": 1e-4}
        result, _, _ = fit_classifier(train, validation, test, 0, config)
        self.assertGreater(result["accuracy"], 0.95)
        rng = np.random.default_rng(19)
        shuffled = [(x, rng.permutation(y)) for x, y in (train, validation, test)]
        result, _, _ = fit_classifier(*shuffled, 0, config)
        self.assertTrue(0.17 < result["accuracy"] < 0.33)

    def test_resume_matches_uninterrupted_training(self):
        torch.manual_seed(23)
        images = torch.randint(0, 2, (9, 4096), dtype=torch.uint8)
        config = {"seed": 23, "epochs": 2, "batch_size": 3, "lr": 0.01,
                  "num_workers": 0, "latent_dim": 10, "hidden_dim": 16,
                  "beta": 4, "checkpoint_every": 1, "image_limit": 9,
                  "dataset_sha256": "synthetic-test"}
        with tempfile.TemporaryDirectory() as directory:
            source, resumed = Path(directory) / "source", Path(directory) / "resumed"
            source.mkdir()
            resumed.mkdir()
            model = VAE(hidden_dim=16)
            train_model(model, images, config, torch.device("cpu"), source)
            restored = VAE(hidden_dim=16)
            resumed_config = {**config, "resume_checkpoint": str(source / "checkpoint_epoch_1.pt")}
            train_model(restored, images, resumed_config, torch.device("cpu"), resumed)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

    def test_capacity_adam_resume_preserves_schedule_and_rejects_changes(self):
        torch.manual_seed(31)
        images = torch.randint(0, 2, (6, 4096), dtype=torch.uint8)
        config = {"seed": 31, "epochs": 2, "batch_size": 3, "lr": 0.0005,
                  "num_workers": 0, "latent_dim": 10, "hidden_dim": 16,
                  "beta": 4, "checkpoint_every": 1, "image_limit": 6,
                  "dataset_sha256": "synthetic-test", "objective": "capacity",
                  "optimizer": "adam", "initialization": "xavier",
                  "decoder_activation": "relu", "capacity_max": 2,
                  "capacity_steps": 3, "gamma": 10}
        with tempfile.TemporaryDirectory() as directory:
            source, resumed = Path(directory) / "source", Path(directory) / "resumed"
            source.mkdir()
            resumed.mkdir()
            model = model_from_config(config)
            history = train_model(model, images, config, torch.device("cpu"), source)
            restored = model_from_config(config)
            resumed_config = {**config, "resume_checkpoint": str(source / "checkpoint_epoch_1.pt")}
            resumed_history = train_model(restored, images, resumed_config, torch.device("cpu"), resumed)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(history[-1]["global_step"], 4)
            self.assertEqual(history[-1]["capacity_target_mean"], resumed_history[-1]["capacity_target_mean"])
            with self.assertRaisesRegex(ValueError, "mismatch for capacity_steps"):
                train_model(model_from_config(config), images,
                            {**resumed_config, "capacity_steps": 4}, torch.device("cpu"), resumed)

    def test_legacy_checkpoint_without_variant_options_resumes(self):
        torch.manual_seed(41)
        images = torch.randint(0, 2, (6, 4096), dtype=torch.uint8)
        config = {"seed": 41, "epochs": 2, "batch_size": 3, "lr": 0.01,
                  "num_workers": 0, "latent_dim": 10, "hidden_dim": 16,
                  "beta": 4, "checkpoint_every": 1, "image_limit": 6,
                  "dataset_sha256": "synthetic-test"}
        with tempfile.TemporaryDirectory() as directory:
            source, resumed = Path(directory) / "source", Path(directory) / "resumed"
            source.mkdir()
            resumed.mkdir()
            model = VAE(hidden_dim=16)
            train_model(model, images, config, torch.device("cpu"), source)
            checkpoint = source / "checkpoint_epoch_1.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state.pop("global_step")
            torch.save(state, checkpoint)
            explicit = {**config, "objective": "beta", "optimizer": "adagrad",
                        "initialization": "default", "decoder_activation": "tanh",
                        "resume_checkpoint": str(checkpoint)}
            restored = model_from_config(explicit)
            train_model(restored, images, explicit, torch.device("cpu"), resumed)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
