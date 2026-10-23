"""Experiment A: orthogonal bases of fixed features, learned and closed-form.

Conda environment: mrl_env_cuda12
Usage:
    Import the basis builders from the Experiment A entry point (and later
    from Experiment C's C0+Q control).

A basis is a d x d orthonormal matrix Q; prefix k uses features x @ Q[:, :k]
(the rows of x are r(x) transposed, so this is Q^T r). Learned rotations use
Q = Q0 expm(S - S^T), which is exactly orthogonal for any S; S starts at zero,
so Q0 is the initialization (identity or a random orthogonal matrix).
"""

import copy

import numpy as np
from scipy import linalg
import torch
from torch import nn
from torch.nn import functional as F

ROTATION_CONFIG = {"lr": 1e-2, "batch_size": 8192, "max_epochs": 100,
                   "patience": 10, "min_delta": 1e-5}


def random_orthogonal(d, seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def initial_basis(name, d):
    """'identity' or 'random<N>' (seeded random orthogonal matrix)."""
    if name == "identity":
        return np.eye(d)
    if name.startswith("random"):
        return random_orthogonal(d, int(name[len("random"):]))
    raise ValueError(f"Unknown basis initialization {name}")


def orthogonality_error(q):
    return float(np.abs(q.T @ q - np.eye(q.shape[1])).max())


class Rotation(nn.Module):
    def __init__(self, q0):
        super().__init__()
        self.register_buffer("q0", torch.as_tensor(q0, dtype=torch.float32))
        self.raw = nn.Parameter(torch.zeros(len(q0), len(q0)))

    def forward(self):
        return self.q0 @ torch.linalg.matrix_exp(self.raw - self.raw.T)


class Direction(nn.Module):
    """Unit vector inside span(complement): complement @ a / |a|."""

    def __init__(self, complement, seed):
        super().__init__()
        self.register_buffer("complement", torch.as_tensor(complement, dtype=torch.float32))
        start = np.random.default_rng(seed).standard_normal(complement.shape[1])
        self.raw = nn.Parameter(torch.as_tensor(start, dtype=torch.float32))

    def forward(self):
        return self.complement @ (self.raw / self.raw.norm())


def _device_split(split, device):
    x, y = split
    return (torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(y), dtype=torch.int64, device=device))


def _train(modules, loss_fn, train, validation, *, device, seed, config):
    """Adam minibatches; keep the state with lowest validation loss."""
    x_train, y_train = train
    x_val, y_val = validation
    parameters = [p for module in modules for p in module.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=config["lr"])
    generator = torch.Generator().manual_seed(seed)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, config["max_epochs"] + 1):
        for batch in torch.randperm(len(x_train), generator=generator).split(config["batch_size"]):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss_fn(x_train[batch], y_train[batch]).backward()
            optimizer.step()
        with torch.no_grad():
            value = float(loss_fn(x_val, y_val))
        if not np.isfinite(value):
            raise RuntimeError("Nonfinite validation loss in basis training")
        history.append({"epoch": epoch, "validation_loss": value})
        if value < best - config["min_delta"]:
            best, stale = value, 0
            best_state = [copy.deepcopy(module.state_dict()) for module in modules]
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    for module, state in zip(modules, best_state):
        module.load_state_dict(state)
    return {"epochs": len(history), "best_validation_loss": best,
            "reached_epoch_limit": len(history) == config["max_epochs"] and stale < config["patience"],
            "history": history}


def fit_nested_rotation(train, validation, classes, prefixes, q0, *, device, seed, config=None):
    """Joint rotation + independent linear head per prefix; mean prefix CE.

    prefixes=range(1, d+1) is the joint-MRL Q; prefixes=[k] is the per-k
    oracle (only Q[:, :k] is meaningful). Returns Q, head accuracies, log.
    """
    config = {**ROTATION_CONFIG, **(config or {})}
    torch.manual_seed(seed)
    train, validation = _device_split(train, device), _device_split(validation, device)
    center = train[0].mean(0)
    prefixes = list(prefixes)
    rotation = Rotation(q0).to(device)
    heads = nn.ModuleList(nn.Linear(k, classes) for k in prefixes).to(device)

    def loss_fn(x, y):
        z = (x - center) @ rotation()
        return torch.stack([F.cross_entropy(head(z[:, :k]), y) for head, k in zip(heads, prefixes)]).mean()

    log = _train([rotation, heads], loss_fn, train, validation, device=device, seed=seed, config=config)
    with torch.no_grad():
        q = rotation().cpu().double().numpy()
    log["orthogonality_error"] = orthogonality_error(q)
    log["prefixes"] = prefixes
    return q, heads, center, log


@torch.no_grad()
def head_accuracies(q, heads, prefixes, center, split, device):
    """Accuracy of the trained prefix heads (not the fresh evaluation probes)."""
    x, y = _device_split(split, device)
    z = (x - center) @ torch.as_tensor(q, dtype=torch.float32, device=device)
    return {int(k): float((head(z[:, :k]).argmax(1) == y).float().mean())
            for head, k in zip(heads, prefixes)}


def fit_greedy_nested(train, validation, classes, *, device, seed, config=None):
    """Add one orthogonal direction at a time; earlier directions stay frozen.

    Step j trains a unit direction in the orthogonal complement of the first
    j-1 directions together with a fresh linear head on all j coordinates.
    The last direction is the one-dimensional complement.
    """
    config = {**ROTATION_CONFIG, **(config or {})}
    torch.manual_seed(seed)
    train_t, validation_t = _device_split(train, device), _device_split(validation, device)
    center = train_t[0].mean(0)
    d = train_t[0].shape[1]
    directions, logs = np.zeros((d, 0)), []
    for j in range(1, d + 1):
        complement = linalg.null_space(directions.T) if j > 1 else np.eye(d)
        if complement.shape[1] == 1:
            directions = np.hstack([directions, complement])
            break
        direction = Direction(complement, seed * 1000 + j).to(device)
        head = nn.Linear(j, classes).to(device)
        fixed = torch.as_tensor(directions, dtype=torch.float32, device=device)

        def loss_fn(x, y):
            basis = torch.cat([fixed, direction()[:, None]], dim=1)
            return F.cross_entropy(head((x - center) @ basis), y)

        log = _train([direction, head], loss_fn, train_t, validation_t, device=device,
                     seed=seed * 1000 + j, config=config)
        log.pop("history")
        logs.append({"step": j, **log})
        with torch.no_grad():
            directions = np.hstack([directions, direction().cpu().double().numpy()[:, None]])
    q, _ = np.linalg.qr(directions)  # removes float drift; spans are unchanged
    q = q * np.sign(np.sum(q * directions, axis=0))
    return q, {"steps": logs, "orthogonality_error": orthogonality_error(q)}


def pca_basis(x):
    """Eigenvectors of the train covariance, largest variance first."""
    values, vectors = np.linalg.eigh(np.cov(np.asarray(x, dtype=np.float64), rowvar=False))
    order = np.argsort(values)[::-1]
    return _fix_signs(vectors[:, order]), values[order]


def lda_basis(x, y, classes, regularization=1e-6):
    """Fisher discriminant directions made Euclidean-orthonormal.

    Rule: solve Sb v = lambda Sw v; keep the top min(classes-1, d) directions
    with nonzero eigenvalue; Gram-Schmidt (QR) them in eigenvalue order, which
    keeps every leading span; fill the rest with PCA of the features inside the
    orthogonal complement. Sw gets a ridge regularization*trace(Sw)/d.
    """
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[1]
    mean = x.mean(0)
    within, between = np.zeros((d, d)), np.zeros((d, d))
    for label in range(classes):
        members = x[y == label]
        if len(members) == 0:
            continue
        class_mean = members.mean(0)
        deviation = members - class_mean
        within += deviation.T @ deviation
        between += len(members) * np.outer(class_mean - mean, class_mean - mean)
    within, between = within / len(x), between / len(x)
    within += regularization * np.trace(within) / d * np.eye(d)
    values, vectors = linalg.eigh(between, within)
    order = np.argsort(values)[::-1][:min(classes - 1, d)]
    order = order[values[order] > 1e-10 * max(values.max(), 1e-300)]
    lead, r = np.linalg.qr(vectors[:, order])
    lead = lead * np.sign(np.diag(r))
    if lead.shape[1] < d:
        complement = linalg.null_space(lead.T)
        inside = (x - mean) @ complement
        values_c, vectors_c = np.linalg.eigh(np.cov(inside, rowvar=False).reshape(complement.shape[1], -1))
        lead = np.hstack([lead, complement @ vectors_c[:, np.argsort(values_c)[::-1]]])
    return _fix_signs(lead), values[order]


def permutation_basis(order):
    q = np.zeros((len(order), len(order)))
    q[order, np.arange(len(order))] = 1.0
    return q


def _fix_signs(q):
    """Largest-magnitude entry of each column positive (display convention only)."""
    rows = np.abs(q).argmax(axis=0)
    return q * np.sign(q[rows, np.arange(q.shape[1])])


def principal_angles(a, b):
    """Principal angles (radians, ascending) between column spans of a and b."""
    return np.sort(linalg.subspace_angles(a, b))
