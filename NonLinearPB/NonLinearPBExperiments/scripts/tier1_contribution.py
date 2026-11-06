"""Source-coordinate contributions to learned target directions (Tier 1, Method 2 only).

Conda environment: mrl_env_cuda12
Usage (from code/):
    python NonLinearPB/NonLinearPBExperiments/scripts/tier1_contribution.py --run-dir /path/to/exprmnt_x
    python NonLinearPB/NonLinearPBExperiments/scripts/tier1_contribution.py --run-dir /path/to/exprmnt_x --jobs recon_beta0_seed0

Post-hoc, CPU only, no fitting. Reads each job's newest tier1_alignment_<stamp>/
(arrays.npz: the learned Q_T per target and restart). tier1_target_alignment.py also calls contribution_job()
on its own folder when a job finishes. Standalone runs write to
<job>/tier1_contribution_<stamp>/ and, for several jobs, <run-dir>/tier1_contribution_combined_<stamp>/.

For each Q_T (columns q_j = target directions, rows i = source coordinates):
  C[i, j] = Q[i, j]^2  contribution of source coordinate i to target direction j
           (rows and columns each sum to 1; identity = source order kept,
           permutation = pure reordering, ~0.1 everywhere = random rotation);
  A_ordered = sum_ij v_j w_i C[i, j] / sum_j v_j w_j with w_i = 1/i, v_j = 1/j
           (1 for the identity; maximal over orthogonal Q there);
  theta[i, j] = arccos|Q[i, j]| in degrees: angle between source axis e_i and
           target axis q_j as lines (column signs of Q_T are arbitrary, so the
           sign-free angle is used); 0 = same axis, 90 = unrelated;
           cos^2 theta = C. theta[j, j] = angle between PB_s(j) and PB_t(j).
Null: A_ordered of NULL_DRAWS random orthogonal matrices.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonlinearpb_matplotlib")

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(CODE_ROOT))
from utility import save_runtime
from vae_reproduction.bases import random_orthogonal
from vae_reproduction.runs import write_json

# ==============================================================================
# CONFIG
# ==============================================================================
NULL_DRAWS = 1000
NULL_SEED = 2026
# ==============================================================================


def position_weights(d):
    """w_i = 1/i (source) and v_j = 1/j (target), as in the A_ordered definition."""
    return 1 / np.arange(1, d + 1)


def contributions(q):
    return np.asarray(q, dtype=np.float64) ** 2


def axis_angles(q):
    """theta[i, j] = arccos|Q[i, j]| (degrees): angle between source axis i and target axis j, as lines."""
    return np.degrees(np.arccos(np.clip(np.abs(np.asarray(q, dtype=np.float64)), 0, 1)))


def ordered_alignment(c):
    w = v = position_weights(len(c))
    return float(w @ c @ v / (w @ v))


def null_distribution(d, draws=NULL_DRAWS, seed=NULL_SEED):
    a = [ordered_alignment(contributions(random_orthogonal(d, seed + i))) for i in range(draws)]
    return {"A_ordered": {"mean": float(np.mean(a)), "p05": float(np.quantile(a, .05)),
                          "p95": float(np.quantile(a, .95))}}


def target_record(qs):
    restarts = [{"C": contributions(q).tolist(), "A_ordered": ordered_alignment(contributions(q)),
                 "theta_deg": axis_angles(q).tolist(), "theta_diagonal_deg": np.diag(axis_angles(q)).tolist()}
                for q in qs]
    values = np.array([r["A_ordered"] for r in restarts])
    theta = np.array([r["theta_deg"] for r in restarts])
    return {"restarts": restarts,
            "A_ordered": {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())},
            "C_mean": np.mean([r["C"] for r in restarts], axis=0).tolist(),
            "theta_mean_deg": theta.mean(0).tolist(),
            "theta_diagonal_mean_deg": np.diag(theta.mean(0)).tolist()}


def newest_tier1(job_dir):
    folders = sorted((p for p in Path(job_dir).glob("tier1_alignment_*")
                      if not p.name.endswith("_smoke") and (p / "tier1.json").exists()),
                     key=lambda p: p.name, reverse=True)
    return folders[0] if folders else None


def contribution_job(tier1_folder, out_folder, stamp):
    """Contribution analysis of one job's Tier 1 result; writes into out_folder; returns the result."""
    tier1_folder, out_folder = Path(tier1_folder), Path(out_folder)
    job = tier1_folder.parent.name
    tier1 = json.loads((tier1_folder / "tier1.json").read_text())[job]
    arrays = np.load(tier1_folder / "arrays.npz")
    targets = list(tier1["targets"])
    result = {"objective": tier1["objective"], "beta": tier1["beta"], "seed": tier1["seed"],
              "source": str(tier1_folder), "targets": {}}
    for name in targets:
        result["targets"][name] = target_record(arrays[f"{name}_Q"])
    d = len(arrays[f"{targets[0]}_Q"][0])
    result["null"] = null_distribution(d)
    write_json(out_folder / "contribution.json", {job: result})
    matrix_figures(job, result, out_folder, stamp)
    signed_q_figure(job, {name: arrays[f"{name}_Q"] for name in targets}, result, out_folder, stamp)
    angle_figure(job, result, out_folder, stamp)
    (out_folder / "contribution_summary.txt").write_text(summary_text({job: result}))
    return result


# ------------------------------------------------------------------- outputs ---
def matrix_figures(job, result, folder, stamp):
    targets = list(result["targets"])
    d = len(result["targets"][targets[0]]["C_mean"])
    ticks = np.arange(d)
    labels = [str(i + 1) for i in ticks]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    for ax, name in zip(axes.flat, targets):
        t = result["targets"][name]
        c = np.array(t["C_mean"])
        ax.imshow(c, cmap="viridis", vmin=0, vmax=1)
        for i in range(d):
            for j in range(d):
                ax.text(j, i, f"{c[i, j]:.2f}".removeprefix("0"), ha="center", va="center",
                        fontsize=5.5, color="white" if c[i, j] < .5 else "black")
        ax.set_title(f"{name}: A_ordered {t['A_ordered']['mean']:.2f}", fontsize=10)
        ax.set_xticks(ticks, labels, fontsize=7)
        ax.set_yticks(ticks, labels, fontsize=7)
        ax.set_xlabel("target direction j (column of Q_T)", fontsize=8)
        ax.set_ylabel("source coordinate i", fontsize=8)
    fig.suptitle(f"{job}: contribution C[i, j] = Q_T[i, j]^2, mean over restarts "
                 f"(columns and rows sum to 1)", fontsize=12)
    fig.tight_layout()
    fig.savefig(folder / f"contribution_mean_{job}_{stamp}.png", dpi=150)
    plt.close(fig)

    restarts = len(result["targets"][targets[0]]["restarts"])
    fig, axes = plt.subplots(len(targets), restarts, figsize=(2.3 * restarts, 2.3 * len(targets)), squeeze=False)
    for row, name in enumerate(targets):
        for col, r in enumerate(result["targets"][name]["restarts"]):
            ax = axes[row][col]
            ax.imshow(np.array(r["C"]), cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"A_ordered {r['A_ordered']:.2f}", fontsize=7)
            if col == 0:
                ax.set_ylabel(f"{name}\nsource i", fontsize=8)
            if row == len(targets) - 1:
                ax.set_xlabel(f"restart {col}: target j", fontsize=7)
    fig.suptitle(f"{job}: C per restart", fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / f"contribution_restarts_{job}_{stamp}.png", dpi=130)
    plt.close(fig)


def angle_figure(job, result, folder, stamp):
    """theta[i, j] = arccos|Q_T[i, j]| in degrees, mean over restarts (sign-free, so averaging is valid)."""
    targets = list(result["targets"])
    d = len(result["targets"][targets[0]]["theta_mean_deg"])
    ticks, labels = np.arange(d), [str(i + 1) for i in range(d)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    for ax, name in zip(axes.flat, targets):
        t = result["targets"][name]
        theta = np.array(t["theta_mean_deg"])
        image = ax.imshow(theta, cmap="magma", vmin=0, vmax=90)
        for i in range(d):
            for j in range(d):
                ax.text(j, i, f"{theta[i, j]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if theta[i, j] < 50 else "black")
        diagonal = " ".join(f"{v:.0f}" for v in t["theta_diagonal_mean_deg"][:3])
        ax.set_title(f"{name}: A_ordered {t['A_ordered']['mean']:.2f}; diagonal j=1..3: {diagonal} deg", fontsize=9)
        ax.set_xticks(ticks, labels, fontsize=7)
        ax.set_yticks(ticks, labels, fontsize=7)
        ax.set_xlabel("target axis j (PB_t)", fontsize=8)
        ax.set_ylabel("source axis i (PB_s)", fontsize=8)
    fig.colorbar(image, ax=axes, shrink=.6, label="angle (deg): 0 = same axis, 90 = unrelated")
    fig.suptitle(f"{job}: angle between source axis i and learned target axis j, arccos|Q_T[i, j]|, "
                 f"mean over restarts", fontsize=12)
    fig.savefig(folder / f"angles_{job}_{stamp}.png", dpi=150)
    plt.close(fig)


def signed_q_figure(job, qs, result, folder, stamp):
    """The learned Q_T itself (signed, not squared), every restart: rows targets, columns restarts.

    Column signs are arbitrary (q_j and -q_j are the same direction), so Q is
    never averaged over restarts.
    """
    targets = list(qs)
    restarts, d = len(qs[targets[0]]), qs[targets[0]].shape[1]
    ticks, labels = np.arange(d), [str(i + 1) for i in range(d)]
    fig, axes = plt.subplots(len(targets), restarts, figsize=(3.4 * restarts, 3.5 * len(targets)), squeeze=False)
    for row, name in enumerate(targets):
        for col in range(restarts):
            ax = axes[row][col]
            q = qs[name][col]
            ax.imshow(q, cmap="RdBu_r", vmin=-1, vmax=1)
            for i in range(d):
                for j in range(d):
                    ax.text(j, i, f"{q[i, j]:.2f}".replace("0.", "."), ha="center", va="center", fontsize=4.2,
                            color="white" if abs(q[i, j]) > .6 else "black")
            ax.set_title(f"{name}, restart {col}: A_ordered "
                         f"{result['targets'][name]['restarts'][col]['A_ordered']:.2f}", fontsize=8)
            ax.set_xticks(ticks, labels, fontsize=6)
            ax.set_yticks(ticks, labels, fontsize=6)
            if row == len(targets) - 1:
                ax.set_xlabel("target direction j", fontsize=7)
            if col == 0:
                ax.set_ylabel("source coordinate i", fontsize=7)
    fig.suptitle(f"{job}: learned Q_T (signed; column sign is arbitrary)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(folder / f"signed_Q_{job}_{stamp}.png", dpi=150)
    plt.close(fig)


def combined_figure(results, folder, stamp):
    jobs = list(results)
    targets = list(results[jobs[0]]["targets"])
    x = np.arange(len(targets))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    null = results[jobs[0]]["null"]["A_ordered"]
    ax.axhspan(null["p05"], null["p95"], color="0.85", label="random rotation 5–95%")
    for k, job in enumerate(jobs):
        s = [results[job]["targets"][t]["A_ordered"] for t in targets]
        offset = (k - (len(jobs) - 1) / 2) * .06
        ax.errorbar(x + offset, [v["mean"] for v in s],
                    yerr=[[v["mean"] - v["min"] for v in s], [v["max"] - v["mean"] for v in s]],
                    fmt="o", markersize=3, linewidth=.8, label=job)
    ax.set_xticks(x, targets)
    ax.set_ylabel("A_ordered (1 = source order kept)")
    ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_title("Method 2: A_ordered (w_i = 1/i, v_j = 1/j), mean over restarts, bars: min–max")
    fig.tight_layout()
    fig.savefig(folder / f"contribution_scores_{stamp}.png", dpi=150)
    plt.close(fig)


def summary_text(results):
    null = next(iter(results.values()))["null"]["A_ordered"]
    jobs = list(results)
    targets = list(results[jobs[0]]["targets"])
    lines = ["Method 2 (learned Q_T): A_ordered = sum_ij v_j w_i Q_ij^2 / sum_j v_j w_j, w_i = 1/i, v_j = 1/j.",
             "Mean [min, max] over restarts. 1 = source order kept.",
             f"Random rotation: mean {null['mean']:.3f} [5%: {null['p05']:.3f}, 95%: {null['p95']:.3f}]", "",
             f"{'job':26s} " + " ".join(f"{t:18s}" for t in targets)]
    for job in jobs:
        lines.append(f"{job:26s} " + " ".join(
            f"{v['mean']:.2f} [{v['min']:.2f},{v['max']:.2f}]   "
            for v in (results[job]["targets"][t]["A_ordered"] for t in targets)))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--jobs", nargs="+")
    args = parser.parse_args()
    started = time.perf_counter()
    stamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    jobs = args.jobs or sorted(p.name for p in args.run_dir.iterdir()
                               if p.is_dir() and newest_tier1(p) is not None)
    if not jobs:
        parser.error("No jobs with a Tier 1 result")
    results = {}
    for job in jobs:
        source = newest_tier1(args.run_dir / job)
        if source is None:
            parser.error(f"No Tier 1 result for {job}")
        folder = args.run_dir / job / f"tier1_contribution_{stamp}"
        folder.mkdir()
        results[job] = contribution_job(source, folder, stamp)
        save_runtime(str(folder), time.perf_counter() - started)
        print(f"{job}: contributions -> {folder}", flush=True)
    if len(results) > 1:
        folder = args.run_dir / f"tier1_contribution_combined_{stamp}"
        folder.mkdir()
        write_json(folder / "contribution.json", results)
        combined_figure(results, folder, stamp)
        (folder / "contribution_summary.txt").write_text(summary_text(results))
        (folder / "experiment_description.log").write_text(__doc__ + f"\nJobs: {jobs}\n")
        save_runtime(str(folder), time.perf_counter() - started)
        print(f"Combined: {folder}", flush=True)


if __name__ == "__main__":
    main()
