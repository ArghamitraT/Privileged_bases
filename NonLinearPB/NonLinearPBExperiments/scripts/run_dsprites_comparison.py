"""Train a paired VAE/candidate, evaluate both, then generate Figure 6 A/B.

Conda environment: mrl_env_cuda12
Usage (from code/; launch through tmux rather than a foreground terminal):
    bash NonLinearPB/NonLinearPBExperiments/scripts/run_dsprites_comparison_tmux.sh --profile repo
    bash NonLinearPB/NonLinearPBExperiments/scripts/run_dsprites_comparison_tmux.sh --profile paper
    bash NonLinearPB/NonLinearPBExperiments/scripts/run_dsprites_comparison_tmux.sh --profile training-only

Profiles: repo = Adam/Xavier/ReLU with beta1 VAE and capacity candidate;
paper = existing Adagrad/default/Tanh with beta1 and beta4;
training-only = Adam/default/Tanh with beta1 and capacity candidate.
Existing metric calculations/settings and diagnostic conventions are unchanged.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = CODE_ROOT.parent
ENTRY = Path(__file__).parents[1] / "experiments/dsprites_vae_reproduction.py"
FIGURE = Path(__file__).with_name("plot_dsprites_figure6.py")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def training_commands(profile, run_dir, epochs, seed, classifier_epochs):
    common = [sys.executable, "-u", str(ENTRY), "--epochs", str(epochs),
              "--seeds", str(seed), "--classifier-epochs", str(classifier_epochs),
              "--batch-size", "256"]
    if profile in ("repo", "training-only"):
        common += ["--optimizer", "adam"]
    if profile == "repo":
        common += ["--initialization", "xavier", "--decoder-activation", "relu"]
    candidate = (["--betas", "4"] if profile == "paper" else
                 ["--objective", "capacity", "--gamma", "1000",
                  "--capacity-max", "25", "--capacity-steps", "100000"])
    return {
        "vae": common + ["--betas", "1", "--device", "cuda:0",
                          "--output-root", str(run_dir / "vae_runs")],
        "beta_vae": common + candidate + ["--skip-untrained", "--device", "cuda:1",
                                          "--output-root", str(run_dir / "beta_vae_runs")],
    }


def run_training(name, command, root):
    log = root / f"{name}.log"
    print(f"Starting {name}: {shlex.join(command)}", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                cwd=CODE_ROOT, check=False)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit {result.returncode}; see {log}")
    folders = sorted((root / f"{name}_runs").glob("exprmnt_*"))
    if len(folders) != 1:
        raise RuntimeError(f"Expected one {name} run, found {len(folders)}")
    folder = folders[0]
    if json.loads((folder / "status.json").read_text())["status"] != "completed":
        raise RuntimeError(f"{name} did not record successful evaluation")
    print(f"Completed {name}: {folder}", flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("repo", "paper", "training-only"), default="repo")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classifier-epochs", type=int, default=2000)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "files/results")
    args = parser.parse_args()
    if args.epochs < 1 or args.classifier_epochs < 1:
        parser.error("Epoch budgets must be positive")
    started = time.monotonic()
    stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    root = args.output_root / f"exprmnt_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    print(f"Comparison directory: {root}", flush=True)
    commands = training_commands(args.profile, root, args.epochs, args.seed, args.classifier_epochs)
    config = {**vars(args), "output_root": str(args.output_root), "commands": commands,
              "note": "Paired seed-0-style comparison; identical existing metric and Figure 6 evaluation. Each model run contains its own config, environment, code snapshot, and diagnostics."}
    write_json(root / "config.json", config)
    (root / "experiment_description.log").write_text(json.dumps(config, indent=2) + "\n")
    write_json(root / "status.json", {"status": "running", "stage": "training_and_evaluation"})
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {name: pool.submit(run_training, name, cmd, root) for name, cmd in commands.items()}
            folders = {name: future.result() for name, future in futures.items()}
        vae_checkpoint = folders["vae"] / f"beta_1_seed_{args.seed}/checkpoint_final.pt"
        candidate_tag = "beta_4" if args.profile == "paper" else "capacity"
        candidate_checkpoint = folders["beta_vae"] / f"{candidate_tag}_seed_{args.seed}/checkpoint_final.pt"
        write_json(root / "status.json", {"status": "running", "stage": "figure6", "model_runs": {k: str(v) for k, v in folders.items()}})
        figure_command = [sys.executable, "-u", str(FIGURE), "--vae-checkpoint", str(vae_checkpoint),
                          "--beta-checkpoint", str(candidate_checkpoint), "--device", "cuda:0",
                          "--output", str(root / "figure6")]
        with (root / "figure6.log").open("w") as handle:
            subprocess.run(figure_command, stdout=handle, stderr=subprocess.STDOUT,
                           cwd=CODE_ROOT, check=True)
        summary = []
        for name, folder in folders.items():
            summary += [f"{name}: {folder}", (folder / "results_summary.txt").read_text()]
        summary += [f"Figure 6 A/B: {root / 'figure6'}"]
        (root / "results_summary.txt").write_text("\n".join(summary) + "\n")
        write_json(root / "manifest.json", {"model_runs": {k: str(v) for k, v in folders.items()},
                   "vae_checkpoint": str(vae_checkpoint), "candidate_checkpoint": str(candidate_checkpoint),
                   "figure6_directory": str(root / "figure6"), "figure_command": figure_command})
        write_json(root / "status.json", {"status": "completed"})
        print((root / "results_summary.txt").read_text(), flush=True)
        print(f"All training/evaluation/Figure 6 complete: {root}", flush=True)
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        (root / "runtime.txt").write_text(f"{time.monotonic() - started:.3f} seconds\n")


if __name__ == "__main__":
    main()
