# 3 hours full but we'll see

import numpy as np
import explib


def merge_grids(*grids):
    return sorted(list(set.union(*[set(grid) for grid in grids])))


EXPERIMENTS = []

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar100",
        "model": "resnet50",
        "batch_size": b_size,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
        },
        "drop_last": True,
        "final_reruns": True,
    }
    for alpha in np.logspace(-8, 1, num=10, base=10)
    for seed in range(5)
    for b_size in [2048, 4096, 8192]
]

EXPERIMENTS_ADAM = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar100",
        "model": "resnet50",
        "batch_size": b_size,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0.9,
            "b2": 0.999,
        },
        "drop_last": True,
        "final_reruns": True,
    }
    for alpha in np.logspace(-8, 1, num=10, base=10)
    for seed in range(5)
    for b_size in [2048, 4096, 8192]
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)
EXPERIMENTS.extend(EXPERIMENTS_ADAM)

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar10",
        "model": "resnet34",
        "batch_size": b_size,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
        },
        "drop_last": True,
        "final_reruns": True,
    }
    for alpha in merge_grids(
        np.logspace(-4, 0, num=5, base=10), np.logspace(-2, 0, num=6, base=10)
    )
    for seed in range(5)
    for b_size in [2048, 4096, 8192]
]

EXPERIMENTS_ADAM = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar10",
        "model": "resnet34",
        "batch_size": b_size,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0.9,
            "b2": 0.999,
        },
        "drop_last": True,
        "final_reruns": True,
    }
    for alpha in np.logspace(-6, 1, num=8, base=10)
    for seed in range(5)
    for b_size in [2048, 4096, 8192]
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)
EXPERIMENTS.extend(EXPERIMENTS_ADAM)


if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        descr="all experiments", experiments=EXPERIMENTS
    )
