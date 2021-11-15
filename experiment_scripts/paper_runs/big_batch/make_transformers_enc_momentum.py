# 6 hours 32 gpu

import numpy as np
import explib


def merge_grids(*grids):
    return sorted(list(set.union(*[set(grid) for grid in grids])))


EXPERIMENTS = []

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["ppl"],
        "dataset": "wikitext2",
        "model": "transformer_encoder",
        "model_args": {
            "tgt_len": 35,
        },
        "batch_size": 256,
        "max_epoch": 1000,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
            "momentum": 0.9,
        },
        "drop_last": True,
        "accumulate_steps": step,
        "final_reruns": True,
    }
    for alpha in merge_grids(
        np.logspace(-4, 1, num=6, base=10), np.logspace(-2, 0, num=6, base=10)
    )
    for seed in range(5)
    for step in [232]
]

EXPERIMENTS_ADAM = [
    {
        "loss_func": "logloss",
        "metrics": ["ppl"],
        "dataset": "wikitext2",
        "model": "transformer_encoder",
        "model_args": {
            "tgt_len": 35,
        },
        "batch_size": 256,
        "max_epoch": 1000,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0,
            "b2": 0.999,
        },
        "drop_last": True,
        "accumulate_steps": step,
        "final_reruns": True,
    }
    for alpha in merge_grids(
        np.logspace(-6, 1, num=8, base=10), np.logspace(-4, -2, num=6, base=10)
    )
    for seed in range(5)
    for step in [232]
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)
EXPERIMENTS.extend(EXPERIMENTS_ADAM)

if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        descr="all experiments", experiments=EXPERIMENTS
    )
