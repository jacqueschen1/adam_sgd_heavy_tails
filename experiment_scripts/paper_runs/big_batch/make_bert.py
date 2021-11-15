# 6 hours full gpu

import numpy as np
import explib


def merge_grids(*grids):
    return sorted(list(set.union(*[set(grid) for grid in grids])))


EXPERIMENTS = []

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["f1", "exact_match"],
        "dataset": "squad",
        "model": "bert_base_pretrained",
        "model_args": {
            "tgt_len": 384,
            "doc_stride": 128,
        },
        "batch_size": 24,
        "max_epoch": 7,
        "seed": seed,
        "trained_norms": False,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
        },
        "drop_last": True,
        "accumulate_steps": step,
        "final_reruns": True,
    }
    for seed in range(5)
    for alpha in merge_grids(
        np.logspace(-4, 1, num=6, base=10), np.logspace(-2, 0, num=6, base=10)
    )
    for step in [32, 64, 128]
]

EXPERIMENTS_ADAM = [
    {
        "loss_func": "logloss",
        "metrics": ["f1", "exact_match"],
        "dataset": "squad",
        "model": "bert_base_pretrained",
        "model_args": {
            "tgt_len": 384,
            "doc_stride": 128,
        },
        "batch_size": 24,
        "max_epoch": 7,
        "seed": seed,
        "trained_norms": False,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0.9,
            "b2": 0.999,
        },
        "drop_last": True,
        "accumulate_steps": step,
        "final_reruns": True,
    }
    for seed in range(5)
    for alpha in [3e-5, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    for step in [32, 64, 128]
]

EXPERIMENTS.extend(EXPERIMENTS_ADAM)
EXPERIMENTS.extend(EXPERIMENTS_SGD)

if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        descr="all experiments", experiments=EXPERIMENTS
    )
