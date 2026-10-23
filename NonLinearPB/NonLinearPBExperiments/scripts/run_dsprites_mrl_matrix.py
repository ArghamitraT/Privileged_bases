"""Four-condition, three-seed fixed-beta reconstruction-MRL comparison.

Conda environment: mrl_env_cuda12
Usage (from code/; always launch through detached tmux):
    bash NonLinearPB/NonLinearPBExperiments/scripts/run_dsprites_mrl_matrix_tmux.sh --smoke
    bash NonLinearPB/NonLinearPBExperiments/scripts/run_dsprites_mrl_matrix_tmux.sh

Full default: 4 conditions x 3 model seeds x 3 classifier seeds; no filtering.
Smoke: the same matrix, TWO epochs on 8192 training images, 16 metric pairs,
1000/400/1000 metric examples, classifier cap40. Scores are execution diagnostics.
No full experiment is launched merely by importing or installing this script.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = CODE_ROOT.parent
ENTRY = Path(__file__).parents[1] / "experiments/dsprites_vae_reproduction.py"
FIGURE = Path(__file__).with_name("plot_dsprites_figure6.py")
CONDITIONS = {"vae": (1, False), "beta_vae": (4, False),
              "mrl_vae": (1, True), "mrl_beta_vae": (4, True)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def training_command(condition, seed, device, leaf, config):
    beta, mrl = CONDITIONS[condition]
    command = [sys.executable, "-u", str(ENTRY), "--betas", str(beta),
               "--seeds", str(seed), "--epochs", str(config["epochs"]),
               "--classifier-seeds", *map(str, config["classifier_seeds"]),
               "--classifier-epochs", str(config["classifier_epochs"]),
               "--batch-size", "256", "--objective", "beta", "--optimizer", "adagrad",
               "--lr", "0.01", "--initialization", "default", "--decoder-activation", "tanh",
               "--skip-untrained", "--no-score-filtering", "--device", device,
               "--output-root", str(leaf)]
    if mrl:
        command.append("--mrl-reconstruction")
    if config["smoke"]:
        command.append("--fast")
    return command


def train_one(condition, seed, device, root, config):
    leaf = root / "conditions" / condition / f"seed_{seed}"
    leaf.mkdir(parents=True, exist_ok=False)
    command = training_command(condition, seed, device, leaf, config)
    write_json(leaf / "launch.json", {"command": command, "condition": condition, "model_seed": seed})
    log = leaf / "console.log"
    print(f"Starting {condition}, seed {seed}: {shlex.join(command)}", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=CODE_ROOT, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{condition}/seed{seed}: exit {result.returncode}; see {log}")
    folders = sorted((leaf / "test_runs" if config["smoke"] else leaf).glob("exprmnt_*"))
    if len(folders) != 1:
        raise RuntimeError(f"Expected one run for {condition}/seed{seed}")
    run = folders[0]
    if json.loads((run / "status.json").read_text())["status"] != "completed":
        raise RuntimeError(f"Incomplete evaluation: {run}")
    beta, mrl = CONDITIONS[condition]
    model = run / f"{'mrl_' if mrl else ''}beta_{beta}_seed_{seed}"
    history = [json.loads(line) for line in (model / "training.jsonl").read_text().splitlines()]
    actual_config = json.loads((run / "config.json").read_text())
    expected_updates = config["epochs"] * ((actual_config["training_images"] + 255) // 256)
    if history[-1]["global_step"] != expected_updates:
        raise RuntimeError(f"Unexpected optimizer-update count: {model}")
    if len(history) != config["epochs"] or history[-1]["epoch"] != config["epochs"]:
        raise RuntimeError(f"Wrong epoch budget: {model}")
    for row in history:
        expected_loss = row["reconstruction_nll"] + beta * row["kl"]
        if abs(expected_loss - row["loss"]) > max(0.01, abs(expected_loss) * 1e-5):
            raise RuntimeError(f"Loss arithmetic mismatch: {model}")
        if mrl:
            prefixes = row["prefix_reconstruction_nll"]
            if len(prefixes) != 10 or abs(statistics.mean(prefixes) - row["reconstruction_nll"]) > 0.01:
                raise RuntimeError(f"Prefix averaging mismatch: {model}")
            if row["full_reconstruction_nll"] != prefixes[-1]:
                raise RuntimeError(f"Full prefix mismatch: {model}")
    print(f"Completed {condition}, seed {seed}: {run}", flush=True)
    return {"run_directory": str(run), "checkpoint": str(model / "checkpoint_final.pt"),
            "model_directory": str(model), "condition": condition, "model_seed": seed}


def aggregate(root, manifest, config):
    records = []
    for condition in CONDITIONS:
        for seed in config["seeds"]:
            entry = manifest[condition][str(seed)]
            scores = json.loads((Path(entry["run_directory"]) / "metric_scores.json").read_text())
            expected = set(config["classifier_seeds"])
            if len(scores) != len(expected) or {r["classifier_seed"] for r in scores} != expected:
                raise RuntimeError(f"Missing/duplicate classifier fits: {condition}, seed {seed}")
            beta, mrl = CONDITIONS[condition]
            if any(r["beta"] != beta or r.get("reconstruction_mrl", False) != mrl or r["model_seed"] != seed for r in scores):
                raise RuntimeError(f"Wrong score provenance: {condition}, seed {seed}")
            records.extend({**r, "condition": condition} for r in scores)
    folder = root / "aggregate"
    folder.mkdir()
    write_json(folder / "all_36_scores.json", records)
    model_rows, tables, summary = [], [], {}
    for condition in CONDITIONS:
        means = []
        for seed in config["seeds"]:
            scores = [r for r in records if r["condition"] == condition and r["model_seed"] == seed]
            mean = statistics.mean(r["accuracy"] for r in scores)
            means.append(mean)
            model_rows.append({"condition": condition, "model_seed": seed, "mean_accuracy": mean})
            tables.append({"condition": condition, "model_seed": seed,
                           **{f"classifier_seed_{r['classifier_seed']}": r["accuracy"] for r in scores}})
        scores = [r for r in records if r["condition"] == condition]
        summary[condition] = {"all_mean": statistics.mean(r["accuracy"] for r in scores),
                              "count": len(scores), "model_seed_means": means,
                              "model_seed_sd": statistics.stdev(means) if len(means) > 1 else None,
                              "score_filtering": "none"}
    differences = []
    for mrl, ordinary in (("mrl_vae", "vae"), ("mrl_beta_vae", "beta_vae")):
        for seed in config["seeds"]:
            means = {r["condition"]: r["mean_accuracy"] for r in model_rows if r["model_seed"] == seed}
            differences.append({"comparison": f"{mrl} - {ordinary}", "model_seed": seed,
                                "accuracy_difference": means[mrl] - means[ordinary]})
    for name, rows in (("per_condition_model_classifier_tables.csv", tables),
                       ("per_model_means.csv", model_rows), ("paired_model_seed_differences.csv", differences)):
        with (folder / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(folder / "summary.json", summary)
    lines = ["SMOKE ONLY: two epochs, subset training, reduced metric settings; scores are not scientific results."
             if config["smoke"] else "Four-condition reconstruction-MRL comparison; all scores retained."]
    for condition, stats in summary.items():
        lines.append(f"{condition}: mean={100 * stats['all_mean']:.2f}%, n={stats['count']}; no discarded scores")
    lines.append("See aggregate/ for every raw score and nested model/classifier tables; figure6/ for each seed.")
    (root / "results_summary.txt").write_text("\n".join(lines) + "\n")
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--classifier-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--classifier-epochs", type=int)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "files/results")
    args = parser.parse_args()
    if args.smoke and args.epochs not in (None, 2):
        parser.error("Smoke runs are restricted to exactly two epochs")
    epochs = args.epochs if args.epochs is not None else (2 if args.smoke else 100)
    classifier_epochs = args.classifier_epochs if args.classifier_epochs is not None else (40 if args.smoke else 2000)
    if epochs < 1 or classifier_epochs < 1 or len(set(args.seeds)) != len(args.seeds) or len(set(args.classifier_seeds)) != len(args.classifier_seeds):
        parser.error("Positive epoch budgets and unique seeds are required")
    config = {"smoke": args.smoke, "seeds": args.seeds, "classifier_seeds": args.classifier_seeds,
              "epochs": epochs, "classifier_epochs": classifier_epochs, "score_filtering": "none",
              "prefixes": list(range(1, 11)), "mrl_sampling": "one_full_sample_masked",
              "optimizer": "adagrad", "lr": 0.01, "initialization": "default", "decoder_activation": "tanh"}
    started = time.monotonic()
    stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    base = args.output_root / "test_runs" if args.smoke else args.output_root
    root = base / f"exprmnt_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    print(f"Matrix directory: {root}", flush=True)
    write_json(root / "config.json", config)
    (root / "experiment_description.log").write_text(__doc__ + "\n" + json.dumps(config, indent=2) + "\n")
    shutil.copytree(CODE_ROOT, root / "code_snapshot", ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "*.pyc", ".DS_Store", "figure", "reproduction_outputs"))
    manifest = {condition: {} for condition in CONDITIONS}
    figures = {}
    try:
        for seed in args.seeds:
            for family, pair in (("ordinary", ("vae", "beta_vae")), ("matryoshka", ("mrl_vae", "mrl_beta_vae"))):
                write_json(root / "status.json", {"status": "running", "stage": "training_and_evaluation", "family": family, "model_seed": seed})
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = {condition: pool.submit(train_one, condition, seed, f"cuda:{device}", root, config)
                               for device, condition in enumerate(pair)}
                    for condition, future in futures.items():
                        manifest[condition][str(seed)] = future.result()
                        write_json(root / "manifest.json", {"models": manifest, "figures": figures})
                output = root / "figure6" / family / f"seed_{seed}"
                output.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", str(FIGURE), "--vae-checkpoint", manifest[pair[0]][str(seed)]["checkpoint"],
                           "--beta-checkpoint", manifest[pair[1]][str(seed)]["checkpoint"], "--device", "cuda:0", "--output", str(output)]
                if args.smoke:
                    command.append("--allow-smoke")
                with (output.parent / f"seed_{seed}.log").open("w") as handle:
                    subprocess.run(command, cwd=CODE_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
                provenance = json.loads((output / "provenance.json").read_text())
                for model in provenance["models"]:
                    if model["config"].get("reconstruction_mrl", False) and model["coordinate_order_zero_based"] != list(range(10)):
                        raise RuntimeError("MRL Figure 6 coordinate order was changed")
                figures[f"{family}_seed_{seed}"] = str(output)
                write_json(root / "manifest.json", {"models": manifest, "figures": figures})
                print(f"Figure 6 complete: {family}, seed {seed}", flush=True)
        aggregate(root, manifest, config)
        write_json(root / "status.json", {"status": "completed", "model_count": 4 * len(args.seeds),
                   "classifier_fit_count": 4 * len(args.seeds) * len(args.classifier_seeds), "combined_figure_count": len(figures)})
        print((root / "results_summary.txt").read_text(), flush=True)
        print(f"Matrix complete: {root}", flush=True)
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        (root / "runtime.txt").write_text(f"{time.monotonic() - started:.3f} seconds\n")


if __name__ == "__main__":
    main()
