"""Queue dsprites_pretrain.py jobs across GPUs, several per GPU, into one experiment folder.

Conda environment: mrl_env_cuda12
Usage (from code/; normally via run_dsprites_pretrain_tmux.sh):
    python NonLinearPB/NonLinearPBExperiments/scripts/run_dsprites_pretrain.py --stages pilot recon
    python NonLinearPB/NonLinearPBExperiments/scripts/run_dsprites_pretrain.py --stages posx --lambda-sup 30 --run-dir /path/to/exprmnt_x
    python NonLinearPB/NonLinearPBExperiments/scripts/run_dsprites_pretrain.py --stages pilot --fast

Stages: pilot = S_posX, beta 4, seed 0, 10 epochs, one job per --pilot-lambdas
value; recon = S_recon for every beta x seed; posx = S_posX for every beta x
seed at --lambda-sup. Jobs start in that order; each GPU runs --per-gpu jobs
at once. One process saturates an RTX 3090 for this model, so --per-gpu > 1
only fills gaps between steps. Per-job logs go to <run-dir>/logs/.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vae_reproduction.runs import OUTPUT_ROOT, make_run_dir, snapshot

# ==============================================================================
# CONFIG
# ==============================================================================
TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "experiments/dsprites_pretrain.py"
BETAS = [0.0, 1.0, 4.0]
SEEDS = [0, 1]
PILOT_LAMBDAS = [10.0, 30.0, 100.0, 300.0]
PILOT_BETA = 4.0
PILOT_EPOCHS = 10
GPUS = [0, 1]
PER_GPU = 3
# ==============================================================================


def jobs_for(args):
    jobs = []
    for stage in args.stages:
        if stage == "pilot":
            jobs += [["--objective", "posx", "--beta", f"{PILOT_BETA:g}", "--seed", "0",
                      "--lambda-sup", f"{lam:g}", "--epochs", str(PILOT_EPOCHS)]
                     for lam in args.pilot_lambdas]
        elif stage == "recon":
            jobs += [["--objective", "recon", "--beta", f"{b:g}", "--seed", str(s)]
                     for b in args.betas for s in args.seeds]
        elif stage == "posx":
            jobs += [["--objective", "posx", "--beta", f"{b:g}", "--seed", str(s),
                      "--lambda-sup", f"{args.lambda_sup:g}"]
                     for b in args.betas for s in args.seeds]
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=("pilot", "recon", "posx"), required=True)
    parser.add_argument("--lambda-sup", type=float)
    parser.add_argument("--pilot-lambdas", type=float, nargs="+", default=PILOT_LAMBDAS)
    parser.add_argument("--betas", type=float, nargs="+", default=BETAS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--gpus", type=int, nargs="+", default=GPUS)
    parser.add_argument("--per-gpu", type=int, default=PER_GPU)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    if "posx" in args.stages and (args.lambda_sup is None or args.lambda_sup <= 0):
        parser.error("stage posx needs --lambda-sup > 0 (choose it from the pilot)")
    run_dir = args.run_dir or make_run_dir(OUTPUT_ROOT, args.fast)
    if args.run_dir is None:
        snapshot(run_dir)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    pending = jobs_for(args)
    extra = ["--run-dir", str(run_dir)] + (["--tf32"] if args.tf32 else []) + (["--fast"] if args.fast else [])
    print(f"Run directory: {run_dir}\n{len(pending)} jobs on GPUs {args.gpus}, {args.per_gpu} per GPU", flush=True)
    running, failed = {}, []                      # process -> (gpu, name, log handle)
    while pending or running:
        for process, (gpu, name, log) in list(running.items()):
            if process.poll() is not None:
                log.close()
                del running[process]
                status = "done" if process.returncode == 0 else f"FAILED ({process.returncode})"
                if process.returncode:
                    failed.append(name)
                print(f"{time.strftime('%H:%M:%S')} {status}: {name} [cuda:{gpu}]", flush=True)
        for gpu in args.gpus:
            while pending and sum(g == gpu for g, _, _ in running.values()) < args.per_gpu:
                job = pending.pop(0)
                name = "_".join(job[1::2])
                log = (run_dir / "logs" / f"{name}.log").open("w")
                process = subprocess.Popen([sys.executable, "-u", str(TRAIN_SCRIPT), *job,
                                            "--device", f"cuda:{gpu}", *extra],
                                           stdout=log, stderr=subprocess.STDOUT)
                running[process] = (gpu, name, log)
                print(f"{time.strftime('%H:%M:%S')} start: {name} [cuda:{gpu}]", flush=True)
        time.sleep(5)
    print(f"All jobs finished; {len(failed)} failed: {failed}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
