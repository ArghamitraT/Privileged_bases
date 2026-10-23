"""Higgins held-fixed-factor metric; labels identify factors, not their values.

Conda environment: mrl_env_cuda12
Usage:
    Import metric_examples and fit_classifier from the reproduction entry point.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import FACTOR_SIZES, TARGET_COLUMNS, factor_indices


def sample_pair_indices(targets, pairs, rng):
    """Resample the fixed value for each pair, with other factors independent."""
    shape = (len(targets), pairs, 5)
    left = rng.integers(FACTOR_SIZES, size=shape)
    right = rng.integers(FACTOR_SIZES, size=shape)
    for target, column in enumerate(TARGET_COLUMNS):
        mask = targets == target
        right[mask, :, column] = left[mask, :, column]
    return factor_indices(left), factor_indices(right)


def metric_examples(means, count, pairs, seed):
    if count <= 0 or count % 4 or pairs <= 0:
        raise ValueError("Metric count must be positive and divisible by four; pairs positive")
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(4), count // 4)
    rng.shuffle(labels)
    features = np.empty((count, means.shape[1]), dtype=np.float32)
    for start in range(0, count, 256):
        stop = min(start + 256, count)
        left, right = sample_pair_indices(labels[start:stop], pairs, rng)
        features[start:stop] = np.abs(means[left] - means[right]).mean(axis=1)
    return features, labels.astype(np.int64)


def fit_classifier(train, validation, test, seed, config):
    """Fit CPU linear softmax, choose by validation CE, evaluate test once."""
    torch.manual_seed(seed)
    classifier = nn.Linear(train[0].shape[1], 4)
    optimizer = torch.optim.Adagrad(classifier.parameters(), lr=config["lr"])
    x_train, y_train = map(torch.from_numpy, train)
    x_val, y_val = map(torch.from_numpy, validation)
    x_test, y_test = map(torch.from_numpy, test)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state, stale = float("inf"), None, 0
    history = []
    for epoch in range(config["epochs"]):
        order = torch.randperm(len(x_train), generator=generator)
        classifier.train()
        for batch in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
        classifier.eval()
        with torch.no_grad():
            val_loss = F.cross_entropy(classifier(x_val), y_val).item()
        if not np.isfinite(val_loss):
            raise RuntimeError("Nonfinite metric classifier loss")
        history.append({"epoch": epoch + 1, "validation_ce": val_loss})
        if val_loss < best_loss - config["min_delta"]:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in classifier.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    classifier.load_state_dict(best_state)
    with torch.no_grad():
        predictions = classifier(x_test).argmax(dim=1).numpy()
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (test[1], predictions), 1)
    result = {
        "classifier_seed": seed,
        "accuracy": float(np.mean(predictions == test[1])),
        "per_factor_accuracy": (confusion.diagonal() / confusion.sum(axis=1)).tolist(),
        "confusion_matrix": confusion.tolist(),
        "epochs": len(history), "best_validation_ce": best_loss,
        "reached_epoch_limit": len(history) == config["epochs"],
    }
    return result, classifier.state_dict(), history
