# 6 hours normal gpu

import numpy as np
import explib


def merge_grids(*grids):
    return sorted(list(set.union(*[set(grid) for grid in grids])))


EXPERIMENTS = []

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "mnist",
        "model": "lenet5",
        "batch_size": b_size,
        "max_epoch": 600,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
        },
        "drop_last": True,
        "final_reruns": True,
    }
    for alpha in np.logspace(-6, 2, num=9, base=10)
    for seed in range(5)
    for b_size in [60000]
]

EXPERIMENTS_ADAM = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "mnist",
        "model": "lenet5",
        "batch_size": b_size,
        "max_epoch": 600,
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
    for alpha in np.logspace(-6, 2, num=9, base=10)
    for seed in range(5)
    for b_size in [60000]
]
EXPERIMENTS.extend(EXPERIMENTS_SGD)
EXPERIMENTS.extend(EXPERIMENTS_ADAM)

if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        descr="all experiments", experiments=EXPERIMENTS
    )
