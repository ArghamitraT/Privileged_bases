"""Evaluation probes on fixed features: linear (L-BFGS) and small MLP.

Conda environment: mrl_env_cuda12
Usage:
    Import fit_linear_probe, fit_mlp_probe and greedy_coordinate_order.

The linear probe is multinomial logistic regression on train-centered
features, fitted full-batch by L-BFGS to a gradient tolerance (no epoch cap),
with a Frobenius weight penalty. Centering and the penalty are both invariant
to orthogonal feature changes, so full-dimensional accuracy is the same for
every orthogonal basis up to optimization tolerance.
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LINEAR_CONFIG = {"l2": 1e-5, "max_iter": 1000, "tolerance_grad": 1e-6,
                 "tolerance_change": 1e-10, "history_size": 20}
MLP_CONFIG = {"hidden": 256, "lr": 1e-3, "batch_size": 4096, "max_epochs": 50,
              "patience": 5}


def _tensors(split, device):
    x, y = split
    return (torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(y), dtype=torch.int64, device=device))


@torch.no_grad()
def _scores(logits, y):
    """Accuracy, cross-entropy, and balanced accuracy (mean recall over classes present in y)."""
    correct = (logits.argmax(1) == y).float()
    present = torch.unique(y)
    recall = torch.stack([correct[y == c].mean() for c in present])
    return {"accuracy": float(correct.mean()), "balanced_accuracy": float(recall.mean()),
            "ce": float(F.cross_entropy(logits, y))}


def _confusion(logits, y, classes):
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (y.cpu().numpy(), logits.argmax(1).cpu().numpy()), 1)
    return matrix.tolist()


def fit_linear_probe(train, validation, test, classes, *, device, config=None,
                     confusion=False):
    """Splits are (features, labels); test may be None."""
    config = {**LINEAR_CONFIG, **(config or {})}
    x_train, y_train = _tensors(train, device)
    if x_train.ndim != 2 or x_train.shape[1] == 0:
        raise ValueError("Linear probe needs a nonempty feature matrix")
    center = x_train.mean(0)
    x_train = x_train - center
    weight = torch.zeros(x_train.shape[1], classes, device=device, requires_grad=True)
    bias = torch.zeros(classes, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias], lr=1, max_iter=config["max_iter"],
        tolerance_grad=config["tolerance_grad"], tolerance_change=config["tolerance_change"],
        history_size=config["history_size"], line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(x_train @ weight + bias, y_train) + config["l2"] * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    closure()
    gradient = torch.cat([weight.grad.flatten(), bias.grad]).abs().max().item()
    iterations = optimizer.state[weight]["n_iter"]
    result = {"iterations": int(iterations), "max_abs_gradient": gradient,
              "converged": bool(iterations < config["max_iter"]),
              "train": _scores((x_train @ weight + bias).detach(), y_train)}
    for name, split in (("validation", validation), ("test", test)):
        if split is None:
            continue
        x, y = _tensors(split, device)
        logits = ((x - center) @ weight + bias).detach()
        result[name] = _scores(logits, y)
        if confusion and name == "test":
            result["test_confusion"] = _confusion(logits, y, classes)
    return result


def fit_mlp_probe(train, validation, test, classes, *, device, seed, config=None):
    """Two-hidden-layer ReLU MLP on standardized features; best validation accuracy."""
    config = {**MLP_CONFIG, **(config or {})}
    torch.manual_seed(seed)
    x_train, y_train = _tensors(train, device)
    x_val, y_val = _tensors(validation, device)
    center, scale = x_train.mean(0), x_train.std(0).clamp_min(1e-6)
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], config["hidden"]), nn.ReLU(),
        nn.Linear(config["hidden"], config["hidden"]), nn.ReLU(),
        nn.Linear(config["hidden"], classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    generator = torch.Generator().manual_seed(seed)
    standardize = lambda x: (x - center) / scale
    best, best_state, stale, epochs = -1.0, None, 0, 0
    for epochs in range(1, config["max_epochs"] + 1):
        model.train()
        for batch in torch.randperm(len(x_train), generator=generator).split(config["batch_size"]):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(standardize(x_train[batch])), y_train[batch]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            accuracy = _scores(model(standardize(x_val)), y_val)["accuracy"]
        if accuracy > best:
            best, best_state, stale = accuracy, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    result = {"epochs": epochs, "reached_epoch_limit": epochs == config["max_epochs"] and stale < config["patience"]}
    with torch.no_grad():
        for name, split in (("train", train), ("validation", validation), ("test", test)):
            if split is not None:
                x, y = _tensors(split, device)
                result[name] = _scores(model(standardize(x)), y)
    return result


def greedy_coordinate_order(train, validation, classes, *, device, config=None):
    """Forward selection: add the coordinate with best validation linear-probe accuracy.

    Ties go to the lowest coordinate index. Returns the order and the
    validation accuracy after each addition.
    """
    x_train, y_train = train
    x_val, y_val = validation
    chosen, remaining, curve = [], list(range(x_train.shape[1])), []
    while remaining:
        best = None
        for candidate in remaining:
            columns = chosen + [candidate]
            score = fit_linear_probe((x_train[:, columns], y_train), (x_val[:, columns], y_val),
                                     None, classes, device=device, config=config)
            accuracy = score["validation"]["accuracy"]
            if best is None or accuracy > best[0]:
                best = (accuracy, candidate)
        chosen.append(best[1])
        remaining.remove(best[1])
        curve.append(best[0])
    return chosen, curve


def fit_linear_regression(train, validation, test):
    """Least squares with intercept; R^2 = 1 - SSE/SST pooled over target columns.

    Splits are (features, targets) with targets n x m. SST uses each split's own
    target mean, so R^2 can be negative on held-out data.
    """
    x_train, t_train = (np.asarray(v, dtype=np.float64) for v in train)
    design = lambda x: np.hstack([np.asarray(x, dtype=np.float64), np.ones((len(x), 1))])
    coefficients = np.linalg.lstsq(design(x_train), t_train, rcond=None)[0]
    result = {}
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        if split is None:
            continue
        x, t = (np.asarray(v, dtype=np.float64) for v in split)
        residual = t - design(x) @ coefficients
        result[name] = float(1 - np.sum(residual ** 2) / np.sum((t - t.mean(0)) ** 2))
    return result


def greedy_order_r2(train, validation):
    """Forward selection by validation R^2 of linear regression; ties to lowest index."""
    chosen, remaining, curve = [], list(range(train[0].shape[1])), []
    while remaining:
        best = None
        for candidate in remaining:
            columns = chosen + [candidate]
            score = fit_linear_regression((train[0][:, columns], train[1]),
                                          (validation[0][:, columns], validation[1]), None)["validation"]
            if best is None or score > best[0]:
                best = (score, candidate)
        chosen.append(best[1])
        remaining.remove(best[1])
        curve.append(best[0])
    return chosen, curve


def k_star(curve, epsilon):
    """Smallest k (1-based) with curve[k-1] >= curve[-1] - epsilon."""
    target = curve[-1] - epsilon
    return next(k for k, value in enumerate(curve, 1) if value >= target)
