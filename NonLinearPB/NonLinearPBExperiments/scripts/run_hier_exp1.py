"""Queue all of Experiment 1 across GPUs: classifier, 10 trainings, 15 evaluations, aggregation.

Conda environment: mrl_env_cuda12
Usage (from code/; normally via run_hier_exp1_tmux.sh):
    python NonLinearPB/NonLinearPBExperiments/scripts/run_hier_exp1.py
    python NonLinearPB/NonLinearPBExperiments/scripts/run_hier_exp1.py --fast
    python NonLinearPB/NonLinearPBExperiments/scripts/run_hier_exp1.py --run-dir /path/to/exprmnt_x   # resume: skips finished jobs
    python NonLinearPB/NonLinearPBExperiments/scripts/run_hier_exp1.py --objectives mrl --seeds 1 2 --train-only --diagnostics

Spec: NonLinearPB/plans/hierarchical_Exp1.md. Jobs start as soon as their
dependencies finish (an evaluation needs its model and the classifier) and fit
in GPU memory: each GPU has MEMORY_GB, each job reserves its measured JOB_GB. An
MRL run saturates one RTX 3090 and needs ~15 GB, so at most one runs per GPU; AE runs,
the classifier and evaluations fill the rest. MRL jobs start first (longest).
The classifier is trained into <run-dir>/classifier/ (only if missing), so every
model in the run is scored by the same frozen classifier; --classifier-dir reuses
an existing one instead.
Per-job logs go to <run-dir>/logs/. A job whose output exists is skipped.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from hier_exp1.data import OUTPUT_ROOT, dataset_record
from hier_exp1.cifar import dataset_record as cifar_record

# ==============================================================================
# CONFIG
# ==============================================================================
SEEDS = [0, 1, 2, 3, 4]
GPUS = [0, 1]
# Measured peak reserved memory per job (2026-11-02) + ~0.5 GB CUDA context each; RTX 3090 = 23.7 GiB.
MEMORY_GB = 23.0
JOB_GB = {"train_mrl": 15.5, "train_pair": 15.5, "train_ae": 4.0, "classifier": 3.0, "evaluate": 3.0, "aggregate": 0.0}
PLAN = PACKAGE.parent / "plans/hierarchical_Exp1.md"
EXPERIMENTS = PACKAGE / "experiments"
# ==============================================================================


class Job:
    def __init__(self, name, kind, cmd, done_file, deps=(), gpu=True):
        self.name, self.kind, self.cmd, self.done_file, self.deps, self.gpu = name, kind, cmd, done_file, set(deps), gpu


def jobs_for(run_dir, seeds, classifier_dir, fast, objectives=("mrl", "ae"), train_only=False, diagnostics=False, lr=None, epochs=None, wandb=False,
             noise_sigma=0.0, bounded=False, prefix="", save_all=False, epoch_weights=False, grid_every=None, latent_dim=None, dataset=None):
    """Jobs for one run folder. prefix (e.g. "sigma_0.5/") keeps job names unique when several
    folders share one queue; noise_sigma/bounded go to training (and bounded also to evaluation)."""
    py = [sys.executable, "-u"]
    extra = (["--fast"] if fast else []) + (["--bounded"] if bounded else [])
    train_extra = (["--lr", f"{lr:g}"] if lr is not None else []) + (["--epochs", str(epochs)] if epochs is not None else []) + (["--wandb"] if wandb else []) \
        + (["--noise-sigma", f"{noise_sigma:g}"] if noise_sigma else []) + (["--save-all"] if save_all else []) \
        + (["--epoch-weights"] if epoch_weights else []) + (["--grid-every", str(grid_every)] if grid_every else []) \
        + (["--latent-dim", str(latent_dim)] if latent_dim else []) \
        + (["--dataset", dataset] if dataset else [])
    jobs = []
    if train_only:
        for objective in objectives:
            for s in seeds:
                jobs.append(Job(f"train_{objective}_seed{s}", f"train_{objective}",
                                py + [str(EXPERIMENTS / "hier_exp1_train.py"), "--objective", objective, "--seed", str(s),
                                      "--run-dir", str(run_dir)] + extra + train_extra + (["--diagnostics"] if diagnostics else []),
                                run_dir / "checkpoints" / objective / f"seed_{s}" / "summary.json"))
        return _prefixed(jobs, prefix)
    if not (classifier_dir / "metrics.json").exists():
        jobs.append(Job("classifier", "classifier",
                        py + [str(EXPERIMENTS / "hier_exp1_classifier.py"), "--output", str(classifier_dir)] + extra,
                        classifier_dir / "metrics.json"))
    for objective in [o for o in ("mrl", "ae") if o in objectives]:
        for s in seeds:
            jobs.append(Job(f"train_{objective}_seed{s}", f"train_{objective}",
                            py + [str(EXPERIMENTS / "hier_exp1_train.py"), "--objective", objective, "--seed", str(s),
                                  "--run-dir", str(run_dir)] + extra + train_extra + (["--diagnostics"] if diagnostics else []),
                            run_dir / "checkpoints" / objective / f"seed_{s}" / "summary.json"))
    clf_dep = {"classifier"} if jobs and jobs[0].name == "classifier" else set()
    for objective in [o for o in ("ae", "mrl") if o in objectives] + ["random"]:
        for s in seeds:
            deps = set(clf_dep) | ({f"train_{objective}_seed{s}"} if objective != "random" else set())
            jobs.append(Job(f"eval_{objective}_seed{s}", "evaluate",
                            py + [str(EXPERIMENTS / "hier_exp1_evaluate.py"), "--objective", objective, "--seed", str(s),
                                  "--run-dir", str(run_dir), "--classifier", str(classifier_dir)] + extra,
                            run_dir / "probes" / f"status_{objective}_seed{s}.json", deps))
    jobs.append(Job("aggregate", "aggregate",
                    py + [str(PACKAGE / "scripts" / "hier_exp1_aggregate.py"), "--run-dir", str(run_dir)],
                    run_dir / "summary.csv", {j.name for j in jobs}, gpu=False))
    return _prefixed(jobs, prefix)


def _prefixed(jobs, prefix):
    if prefix:
        for j in jobs:
            j.name, j.deps = prefix + j.name, {prefix + d for d in j.deps}
    return jobs


def make_run_dir(fast):
    root = OUTPUT_ROOT / ("test_runs" if fast else "")
    run_dir = root / f"exprmnt_{datetime.now():%Y_%m_%d__%H_%M_%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(PACKAGE, run_dir / "code_snapshot/NonLinearPBExperiments",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(PLAN, run_dir / "code_snapshot" / PLAN.name)
    return run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--gpus", type=int, nargs="+", default=GPUS)
    ap.add_argument("--memory-gb", type=float, default=MEMORY_GB)
    ap.add_argument("--classifier-dir", type=Path)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--objectives", nargs="+", choices=("mrl", "ae", "pair"), default=["mrl", "ae"],
                    help="pair ({z1}, {z1, zj} codes) is training-only: use with --train-only")
    ap.add_argument("--train-only", action="store_true", help="only the training jobs (no classifier, evaluation, aggregate)")
    ap.add_argument("--diagnostics", action="store_true", help="pass --diagnostics to every training job (step log, spike dumps, stop on collapse)")
    ap.add_argument("--lr", type=float, help="Adam learning rate for every training job (default: the trainer's)")
    ap.add_argument("--wandb", action="store_true", help="mirror training logs to Weights & Biases (login from ~/.netrc)")
    ap.add_argument("--epochs", type=int, help="training epochs for every training job (default: the trainer's, 100)")
    ap.add_argument("--noise-sigmas", type=float, nargs="+",
                    help="training-noise sweep: one subfolder sigma_<s>/ per value inside one experiment folder, one shared queue")
    ap.add_argument("--bounded", action="store_true", help="tanh on z (all models, incl. untrained controls)")
    ap.add_argument("--save-all", action="store_true",
                    help="pass --save-all to every training job: every epoch state incl. epoch 0, weights after every "
                         "update (~3.3 MB each), test codes every epoch, W&B recon grid every epoch (implies --diagnostics)")
    ap.add_argument("--epoch-weights", action="store_true",
                    help="every training job saves weights at the init and after every epoch (~3.3 MB each)")
    ap.add_argument("--latent-dim", type=int, help="size of z for every training job (default: the trainer's, 10); use with --train-only")
    ap.add_argument("--dataset", choices=("hier", "cifar100"), help="training data (trainer default: hier); cifar100 needs --train-only")
    ap.add_argument("--grid-every", type=int, help="W&B recon/grid every N epochs (trainer default 5)")
    ap.add_argument("--ae-noise-sigmas", type=float, nargs="+",
                    help="sweep: train AE only at these sigmas (default: at every sigma, if ae is in --objectives)")
    args = ap.parse_args()
    if args.save_all:
        args.diagnostics = True
    if "pair" in args.objectives and not args.train_only:
        ap.error("--objectives pair is for training only (evaluation has no pair codes); add --train-only")
    if args.dataset == "cifar100" and not args.train_only:
        ap.error("--dataset cifar100 is for training only (evaluation and classifier are for the synthetic data); add --train-only")
    if args.latent_dim and not args.train_only:
        ap.error("--latent-dim is for training only (evaluation assumes a 10-d code); add --train-only")
    if args.noise_sigmas and not args.classifier_dir:
        ap.error("--noise-sigmas needs --classifier-dir (reuse a frozen classifier so all sigmas are scored alike)")
    run_dir = args.run_dir or make_run_dir(args.fast)
    classifier_dir = args.classifier_dir or run_dir / "classifier"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    if args.noise_sigmas:
        jobs = []
        for sg in args.noise_sigmas:
            sub = run_dir / f"sigma_{sg:g}"
            sub.mkdir(parents=True, exist_ok=True)
            objs = [o for o in args.objectives if o != "ae" or args.ae_noise_sigmas is None or sg in args.ae_noise_sigmas]
            jobs += jobs_for(sub, args.seeds, classifier_dir, args.fast, objs, args.train_only, args.diagnostics,
                             args.lr, args.epochs, args.wandb, noise_sigma=sg, bounded=args.bounded, prefix=f"sigma_{sg:g}/",
                             save_all=args.save_all, epoch_weights=args.epoch_weights, grid_every=args.grid_every, latent_dim=args.latent_dim, dataset=args.dataset)
        order = {"train_mrl": 0, "train_ae": 1}                     # longest jobs first across all sigmas
        jobs.sort(key=lambda j: order.get(j.kind, 2))
    else:
        jobs = jobs_for(run_dir, args.seeds, classifier_dir, args.fast, args.objectives, args.train_only, args.diagnostics,
                        args.lr, args.epochs, args.wandb, bounded=args.bounded, save_all=args.save_all,
                        epoch_weights=args.epoch_weights, grid_every=args.grid_every, latent_dim=args.latent_dim, dataset=args.dataset)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps({
            "experiment": "Experiment 1: CNN-AE vs CNN-MRL-AE on hierarchical v2", "plan": str(PLAN),
            "seeds": args.seeds, "objectives": ["ae", "mrl"], "controls": ["random (untrained init, same seed)"],
            "architecture": "small residual CNN-AE, 10-d deterministic latent, shared prefix decoder (hier_exp1/model.py)",
            "training": f"Adam {args.lr if args.lr is not None else 1e-3:g}, batch 256, {args.epochs or 100} epochs, best validation own objective, TF32, torch.compile",
            "evaluation": "plan section 5D; settings in experiments/hier_exp1_evaluate.py CONFIG",
            "classifier_dir": None if args.train_only else str(classifier_dir), "data": (cifar_record() if args.dataset == "cifar100" else dataset_record()), "dataset": args.dataset or "hier", "fast": args.fast,
            "objectives_trained": args.objectives, "train_only": args.train_only, "diagnostics": args.diagnostics, "lr": args.lr, "epochs": args.epochs, "wandb": args.wandb, "noise_sigmas": args.noise_sigmas, "ae_noise_sigmas": args.ae_noise_sigmas, "bounded": args.bounded, "save_all": args.save_all,
            "epoch_weights": args.epoch_weights, "grid_every": args.grid_every, "latent_dim": args.latent_dim,
            "gpus": args.gpus, "memory_gb_per_gpu": args.memory_gb, "job_gb": JOB_GB, "command": sys.argv}, indent=2) + "\n")
    with open(run_dir / "launches.jsonl", "a") as fh:          # every launch into this folder, incl. resumes
        fh.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "command": sys.argv,
                             "lr": args.lr, "epochs": args.epochs, "train_only": args.train_only,
                             "diagnostics": args.diagnostics, "wandb": args.wandb, "objectives": args.objectives,
                             "seeds": args.seeds, "noise_sigmas": args.noise_sigmas, "bounded": args.bounded,
                             "save_all": args.save_all, "epoch_weights": args.epoch_weights,
                             "grid_every": args.grid_every, "latent_dim": args.latent_dim, "dataset": args.dataset}) + "\n")

    done = {j.name for j in jobs if j.done_file.exists()}
    pending = [j for j in jobs if j.name not in done]
    failed, running = set(), {}                       # process -> (job, gpu, log)
    print(f"Run directory: {run_dir}\n{len(pending)} jobs to run ({len(done)} already done) on GPUs {args.gpus}", flush=True)
    while pending or running:
        for proc, (job, gpu, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[proc]
                ok = proc.returncode == 0
                (done if ok else failed).add(job.name)
                status = 'done' if ok else ('CRASH DETECTED (stopped by diagnostics)' if proc.returncode == 3 else f'FAILED ({proc.returncode})')
                print(f"{time.strftime('%H:%M:%S')} {status}: {job.name}"
                      + (f" [cuda:{gpu}]" if gpu is not None else ""), flush=True)
        blocked = [j for j in pending if j.deps & failed]
        for j in blocked:
            pending.remove(j)
            failed.add(j.name)
            print(f"{time.strftime('%H:%M:%S')} SKIPPED (dependency failed): {j.name}", flush=True)
        used = {g: sum(JOB_GB[j.kind] for j, gg, _ in running.values() if gg == g) for g in args.gpus}
        for job in list(pending):
            if not job.deps <= done:
                continue
            gpu = None
            if job.gpu:
                fits = [g for g in args.gpus if used[g] + JOB_GB[job.kind] <= args.memory_gb]
                if not fits:
                    continue
                gpu = min(fits, key=lambda g: used[g])
                used[gpu] += JOB_GB[job.kind]
            log_path = run_dir / "logs" / f"{job.name}.log"              # sweep: logs/sigma_<s>/<job>.log
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w")
            cmd = job.cmd + (["--device", f"cuda:{gpu}"] if gpu is not None else [])
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(PACKAGE.parents[1]),
                                    env={**os.environ, "PYTHONHASHSEED": "0",
                                         "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
            running[proc] = (job, gpu, log)
            pending.remove(job)
            print(f"{time.strftime('%H:%M:%S')} start: {job.name}" + (f" [cuda:{gpu}]" if gpu is not None else ""), flush=True)
        time.sleep(5)
    print(f"All jobs finished; {len(failed)} failed: {sorted(failed)}\nRun directory: {run_dir}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
