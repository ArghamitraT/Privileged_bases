"""User-run checks for Experiment 1 (hierarchical CNN-AE / MRL-AE): model, loss, init, metrics, queue.

Conda environment: mrl_env_cuda12
Usage:
    python NonLinearPB/NonLinearPBExperiments/tests/helper_hier_exp1.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from hier_exp1.metrics import PAIR_TYPES, auc, containment, pair_indices, shuffled_containment
from hier_exp1.model import CNNAE, LayerNorm2d, ReconClassifier, objective_loss, prefix_mse
from hier_exp1_train import build_model
from scripts.run_hier_exp1 import jobs_for


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = CNNAE()
        self.x = torch.rand(4, 1, 64, 64)

    def test_shapes(self):
        z = self.model.encode(self.x)
        self.assertEqual(tuple(z.shape), (4, 10))
        xhat = self.model.decode(z)
        self.assertEqual(tuple(xhat.shape), (4, 1, 64, 64))
        self.assertNotIsInstance(self.model.decoder[-1], torch.nn.Sigmoid)       # linear output (MSE)
        self.assertEqual({k: v.shape[-1] for k, v in ReconClassifier()(self.x).items()},
                         {"coarse": 3, "mid_node": 9, "leaf_id": 27})

    def test_layernorm_is_channelwise(self):
        ln = LayerNorm2d(8)
        h = torch.randn(2, 8, 5, 5)
        expected = F.layer_norm(h.permute(0, 2, 3, 1), (8,)).permute(0, 3, 1, 2)
        torch.testing.assert_close(ln(h), expected)
        torch.testing.assert_close(ln(h).mean(dim=1), torch.zeros(2, 5, 5), atol=1e-5, rtol=0)

    def test_latent_has_no_activation(self):
        self.assertIsInstance(self.model.encoder[-1], torch.nn.Linear)

    def test_prefix_mse_matches_manual_decoding(self):
        z = self.model.encode(self.x)
        got = prefix_mse(self.model, self.x, z)
        for k in range(1, 11):
            zk = torch.cat([z[:, :k], torch.zeros(4, 10 - k)], dim=1)
            manual = ((self.model.decode(zk) - self.x) ** 2).mean(dim=(1, 2, 3))
            torch.testing.assert_close(got[k - 1], manual, rtol=1e-5, atol=1e-7)

    def test_objectives(self):
        per_k = prefix_mse(self.model, self.x)
        torch.testing.assert_close(objective_loss(self.model, self.x, "mrl"), per_k.mean())
        torch.testing.assert_close(objective_loss(self.model, self.x, "ae"), per_k[-1].mean(), rtol=1e-5, atol=1e-7)

    def test_same_seed_same_init_across_objectives(self):
        a, b, c = build_model(3).state_dict(), build_model(3).state_dict(), build_model(4).state_dict()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertFalse(all(torch.equal(a[key], c[key]) for key in a))


class MetricTests(unittest.TestCase):
    def test_containment(self):
        coarse = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(containment(np.array([0, 0, 1, 1, 2, 2, 3, 3]), coarse), 1.0)     # perfect split
        self.assertEqual(containment(coarse, coarse), 1.0)                                  # unchanged
        self.assertEqual(containment(np.array([0, 1, 0, 1, 0, 1, 0, 1]), coarse), 0.5)     # cuts across

    def test_shuffle_keeps_sizes_and_is_lower(self):
        rng = np.random.default_rng(0)
        coarse = np.repeat([0, 1, 2], 300)
        fine = np.repeat(np.arange(9), 100)
        sh = shuffled_containment(fine, coarse, 20, rng)
        self.assertLess(sh.mean(), 0.6)
        self.assertGreater(sh.mean(), 1 / 3)
        self.assertEqual(containment(fine, coarse), 1.0)

    def test_pair_types(self):
        rng = np.random.default_rng(0)
        c = np.repeat([0, 1, 2], 90)
        m = np.tile(np.repeat([0, 1, 2], 30), 3)
        f = np.tile(np.repeat([0, 1, 2], 10), 9)
        labels = {"coarse": c, "mid_node": 3 * c + m, "leaf_id": 9 * c + 3 * m + f}
        pairs = pair_indices(labels, 500, rng)
        self.assertEqual(set(pairs), set(PAIR_TYPES))
        i, j = pairs["same_leaf"]
        self.assertTrue((labels["leaf_id"][i] == labels["leaf_id"][j]).all() and (i != j).all())
        i, j = pairs["same_mid_diff_leaf"]
        self.assertTrue(((labels["mid_node"][i] == labels["mid_node"][j]) & (labels["leaf_id"][i] != labels["leaf_id"][j])).all())
        i, j = pairs["same_coarse_diff_mid"]
        self.assertTrue(((c[i] == c[j]) & (labels["mid_node"][i] != labels["mid_node"][j])).all())
        i, j = pairs["diff_coarse"]
        self.assertTrue((c[i] != c[j]).all())
        self.assertTrue(all(len(v[0]) == 500 for v in pairs.values()))

    def test_auc_matches_sklearn(self):
        rng = np.random.default_rng(1)
        lo, hi = rng.normal(0, 1, 300).round(1), rng.normal(0.5, 1, 200).round(1)
        expected = roc_auc_score(np.r_[np.zeros(300), np.ones(200)], np.r_[lo, hi])
        self.assertAlmostEqual(auc(lo, hi), expected, places=10)


class DiagnosticsTests(unittest.TestCase):
    def _train(self, record, folder=None, steps=6):
        from hier_exp1.diagnostics import Recorder
        torch.manual_seed(0)
        model = build_model(0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        data = torch.rand(steps * 4, 1, 64, 64)
        rec = Recorder(model, opt, folder, warmup=2, factor=0.0) if record else None   # factor 0: every step "spikes"
        for step in range(steps):
            idx = torch.arange(step * 4, step * 4 + 4)
            x = data[idx]
            if rec:
                rec.before_step(step, 1, idx)
                per_kb = prefix_mse(model, x)
                loss = per_kb.mean()
                per = per_kb.detach().mean(dim=1)
            else:
                per, loss = None, objective_loss(model, x, "mrl")
            opt.zero_grad()
            loss.backward()
            if rec:
                rec.after_backward()
            opt.step()
            if rec:
                rec.after_step(loss, per, x)
        if rec:
            rec.end_epoch(1, steps, torch.Generator().manual_seed(0), torch.arange(10))
            rec.close()
        return model.state_dict()

    def test_recording_does_not_change_training(self):
        import tempfile
        plain = self._train(False)
        with tempfile.TemporaryDirectory() as d:
            recorded = self._train(True, d)
        for key in plain:
            torch.testing.assert_close(plain[key], recorded[key], rtol=0, atol=0)

    def test_outputs_and_spike_dump(self):
        import csv as _csv, tempfile
        with tempfile.TemporaryDirectory() as d:
            self._train(True, d, steps=24)                     # spike detection needs 20 losses of history
            with open(Path(d) / "steps.csv") as fh:
                rows = list(_csv.DictReader(fh))
            self.assertEqual(len(rows), 24)
            self.assertIn("loss_k10", rows[0])
            self.assertNotEqual(rows[0]["act_dec_up3_frac_dead_units"], "")          # step 0 measures activations
            self.assertTrue((Path(d) / "epochs" / "epoch_001.pt").exists())
            self.assertTrue((Path(d) / "order" / "epoch_001.npy").exists())
            spikes = sorted(Path(d).glob("spike_step*"))
            self.assertEqual(len(spikes), 1)                                         # one dump, then POST steps recorded
            pre = sorted(spikes[0].glob("pre_*.pt"))
            self.assertGreaterEqual(len(pre), 1)
            state = torch.load(pre[-1], weights_only=True)
            self.assertEqual(set(state), {"step", "epoch", "weights", "adam", "batch_idx"})
            from hier_exp1.diagnostics import POST
            self.assertEqual(len(list(spikes[0].glob("post_*.pt"))), POST)


class WandbTests(unittest.TestCase):
    def test_logging_does_not_change_training_and_grid_shape(self):
        import tempfile
        from hier_exp1.wandb_log import GRID_KS, GRID_SCALE, WandbLogger
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_SILENT"] = "true"
        val_x = (torch.rand(27, 1, 64, 64) * 255).to(torch.uint8)
        mid = np.repeat(np.arange(9), 3)

        def train(log, folder=None):
            torch.manual_seed(0)
            model = build_model(0)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            data = torch.rand(16, 1, 64, 64)
            wb = WandbLogger("test", "t", "g", {}, Path(folder), model, val_x, mid) if log else None
            for step in range(4):
                loss = objective_loss(model, data[step * 4:(step + 1) * 4], "mrl")
                opt.zero_grad()
                loss.backward()
                opt.step()
                if wb:
                    wb.log_step({"step": step, "epoch": 1, "loss": float(loss), "update_norm_encoder": 0.1,
                                 "weight_norm_encoder": 2.0}, [3] if step == 3 else [])
            if wb:
                wb.log_epoch({"epoch": 1, "train_loss": 0.1}, 1, 1e-3, 1.0, torch.device("cpu"))
                grid = wb._grid()
                wb.finish("finished", best_epoch=1)
                return model.state_dict(), grid
            return model.state_dict(), None

        plain, _ = train(False)
        with tempfile.TemporaryDirectory() as d:
            logged, grid = train(True, d)
        for key in plain:
            torch.testing.assert_close(plain[key], logged[key], rtol=0, atol=0)
        self.assertEqual(grid.shape, (9 * 66 * GRID_SCALE, (1 + len(GRID_KS)) * 66 * GRID_SCALE))


class PruneTests(unittest.TestCase):
    def test_epoch_checkpoints_pruned(self):
        import tempfile
        from hier_exp1.diagnostics import KEEP_EVERY_EPOCHS, KEEP_LAST_EPOCHS, Recorder
        model = build_model(0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder(model, opt, d)
            n = KEEP_EVERY_EPOCHS + 3
            for e in range(1, n + 1):
                rec.end_epoch(e, e, torch.Generator().manual_seed(0), torch.arange(4))
            kept = sorted(int(p.stem.split("_")[1]) for p in (Path(d) / "epochs").glob("epoch_*.pt"))
            expected = sorted({e for e in range(1, n + 1) if e > n - KEEP_LAST_EPOCHS or e % KEEP_EVERY_EPOCHS == 0})
            self.assertEqual(kept, expected)
            self.assertEqual(len(list((Path(d) / "order").glob("*.npy"))), n)      # batch orders all kept


class SaveAllTests(unittest.TestCase):
    def test_step_weights_initial_state_and_no_pruning(self):
        import tempfile
        from hier_exp1.diagnostics import KEEP_EVERY_EPOCHS, Recorder
        torch.manual_seed(0)
        model = build_model(0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        data = torch.rand(12, 1, 64, 64)
        with tempfile.TemporaryDirectory() as d:
            rec = Recorder(model, opt, d, save_all=True)
            rec.save_initial(torch.Generator().manual_seed(0))
            for step in range(3):
                idx = torch.arange(step * 4, step * 4 + 4)
                rec.before_step(step, 1, idx)
                loss = objective_loss(model, data[idx], "mrl")
                opt.zero_grad(); loss.backward(); rec.after_backward(); opt.step()
                rec.after_step(loss, None, data[idx])
            for e in range(1, KEEP_EVERY_EPOCHS + 4):
                rec.end_epoch(e, 3, torch.Generator().manual_seed(0), torch.arange(4))
            rec.close()
            files = sorted(p.name for p in (Path(d) / "step_weights").glob("*.pt"))
            self.assertEqual(files, ["upd_000001.pt", "upd_000002.pt", "upd_000003.pt"])
            last = torch.load(Path(d) / "step_weights" / "upd_000003.pt", weights_only=True)
            for k, v in model.state_dict().items():                               # weights after the last update
                torch.testing.assert_close(last[k], v.cpu(), rtol=0, atol=0)
            fresh = build_model(0)
            fresh.load_state_dict(last)                                           # loads straight into a model
            init = torch.load(Path(d) / "epochs" / "epoch_000.pt", weights_only=True)
            for k, v in build_model(0).state_dict().items():
                torch.testing.assert_close(init["model"][k], v, rtol=0, atol=0)
            self.assertEqual(len(list((Path(d) / "epochs").glob("epoch_*.pt"))), KEEP_EVERY_EPOCHS + 4)   # 0..53, none pruned

    def test_launcher_passes_save_all(self):
        import tempfile
        clf = Path(tempfile.mkdtemp())
        (clf / "metrics.json").write_text("{}")
        jobs = jobs_for(Path("/tmp/none/s"), [0], clf, False, objectives=["mrl"], diagnostics=True, save_all=True)
        tr = next(j for j in jobs if j.name == "train_mrl_seed0")
        self.assertIn("--save-all", tr.cmd)
        ev = next(j for j in jobs if j.name == "eval_mrl_seed0")
        self.assertNotIn("--save-all", ev.cmd)


class NoiseTests(unittest.TestCase):
    def test_noise_only_on_kept_entries_and_off_by_default(self):
        torch.manual_seed(0)
        model = CNNAE()
        x = torch.rand(3, 1, 64, 64)
        seen = []
        real = model.decode
        model.decode = lambda z: (seen.append(z.detach().clone()), real(z))[1]
        z = model.encode(x)
        prefix_mse(model, x, z, noise_sigma=0.5)
        codes = seen[-1].view(10, 3, 10)
        for k in range(1, 11):
            self.assertTrue(torch.all(codes[k - 1][:, k:] == 0))                 # tail stays exactly zero
            self.assertFalse(torch.allclose(codes[k - 1][:, :k], z[:, :k]))      # kept entries are perturbed
        prefix_mse(model, x, z)
        clean = seen[-1].view(10, 3, 10)
        torch.testing.assert_close(clean[9], z)                                   # sigma 0: no noise at all
        model.decode = real
        torch.testing.assert_close(objective_loss(model, x, "ae"), objective_loss(model, x, "ae", 0.0))

    def test_bounded_model(self):
        a, b = build_model(2).state_dict(), build_model(2, bounded=True).state_dict()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)           # tanh adds no parameters
        z = build_model(2, bounded=True).encode(torch.rand(4, 1, 64, 64) * 50)
        self.assertTrue(bool((z.abs() <= 1).all()))

    def test_sweep_jobs(self):
        import tempfile
        from scripts.run_hier_exp1 import jobs_for
        clf = Path(tempfile.mkdtemp())
        (clf / "metrics.json").write_text("{}")                                    # existing frozen classifier
        jobs = jobs_for(Path("/tmp/none/sigma_0.5"), [0], clf, False, noise_sigma=0.5, prefix="sigma_0.5/")
        self.assertNotIn("sigma_0.5/classifier", {j.name for j in jobs})
        names = {j.name for j in jobs}
        self.assertIn("sigma_0.5/train_mrl_seed0", names)
        ev = next(j for j in jobs if j.name == "sigma_0.5/eval_mrl_seed0")
        self.assertEqual(ev.deps, {"sigma_0.5/train_mrl_seed0"})
        tr = next(j for j in jobs if j.name == "sigma_0.5/train_mrl_seed0")
        self.assertEqual(tr.cmd[tr.cmd.index("--noise-sigma") + 1], "0.5")
        self.assertNotIn("--bounded", tr.cmd)
        mrl_only = {j.name for j in jobs_for(Path("/tmp/none/s"), [0], clf, False, objectives=["mrl"], noise_sigma=0.2)}
        self.assertFalse(any("_ae_" in n for n in mrl_only))
        self.assertIn("eval_random_seed0", mrl_only)


class QueueTests(unittest.TestCase):
    def test_train_only_diagnostics(self):
        jobs = jobs_for(Path("/tmp/none_run"), [1, 2], Path("/tmp/none_classifier"), fast=False,
                        objectives=["mrl"], train_only=True, diagnostics=True)
        self.assertEqual([j.name for j in jobs], ["train_mrl_seed1", "train_mrl_seed2"])
        self.assertTrue(all("--diagnostics" in j.cmd for j in jobs))

    def test_jobs_and_dependencies(self):
        run, clf = Path("/tmp/none_run"), Path("/tmp/none_classifier")
        jobs = {j.name: j for j in jobs_for(run, [0, 1], clf, fast=True)}
        self.assertIn("classifier", jobs)
        self.assertEqual(sum(n.startswith("train_") for n in jobs), 4)
        self.assertEqual(sum(n.startswith("eval_") for n in jobs), 6)
        self.assertEqual(jobs["eval_mrl_seed1"].deps, {"classifier", "train_mrl_seed1"})
        self.assertEqual(jobs["eval_random_seed0"].deps, {"classifier"})
        self.assertTrue({n for n in jobs if n != "aggregate"} <= jobs["aggregate"].deps)
        self.assertEqual(list(jobs)[1], "train_mrl_seed0")                  # longest jobs queued first


if __name__ == "__main__":
    unittest.main(verbosity=1)
