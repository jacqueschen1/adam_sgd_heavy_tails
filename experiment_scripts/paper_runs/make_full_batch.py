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
        "batch_size": 17000,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
            "momentum": m,
        },
        "full_batch": True,
    }
    for seed in range(5)
    for alpha in merge_grids(
        np.logspace(-5, 2, num=8, base=10), np.logspace(-2, 0, num=6, base=10)
    )
    for m in [0.9]
]
EXPERIMENTS.extend(EXPERIMENTS_SGD)

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar100",
        "model": "resnet50",
        "batch_size": 17000,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
        },
        "full_batch": True,
    }
    for seed in range(5)
    for alpha in merge_grids(
        np.logspace(-5, 2, num=8, base=10), np.logspace(-2, 0, num=6, base=10)
    )
]

EXPERIMENTS_ADAM = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar100",
        "model": "resnet50",
        "batch_size": 17000,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0.9,
            "b2": 0.999,
        },
        "full_batch": True,
    }
    for seed in range(5)
    for alpha in merge_grids(
        np.logspace(-6, 2, num=9, base=10), np.logspace(-4, -2, num=6, base=10)
    )
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)
EXPERIMENTS.extend(EXPERIMENTS_ADAM)

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar10",
        "model": "resnet34",
        "batch_size": 20000,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
            "momentum": m,
        },
        "full_batch": True,
    }
    for seed in range(5)
    for alpha in [
        1e-06,
        6.8129206905796085e-06,
        4.641588833612782e-05,
        0.00031622776601683794,
        0.001,
        0.0021544346900318843,
        0.00631,
        0.01,
        0.014677992676220704,
        0.02511886431509579,
        0.06309573444801933,
        0.1,
        0.15848931924611143,
        0.1585,
        0.3981071705534973,
        1.0,
    ]
    for m in [0.9]
]
EXPERIMENTS.extend(EXPERIMENTS_SGD)

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar10",
        "model": "resnet34",
        "batch_size": 20000,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
        },
        "full_batch": True,
    }
    for seed in range(5)
    for alpha in [
        1e-06,
        6.8129206905796085e-06,
        4.641588833612782e-05,
        0.00031622776601683794,
        0.001,
        0.0021544346900318843,
        0.00631,
        0.01,
        0.014677992676220704,
        0.02511886431509579,
        0.06309573444801933,
        0.1,
        0.15848931924611143,
        0.1585,
        0.3981071705534973,
        1.0,
    ]
]

EXPERIMENTS_ADAM = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar10",
        "model": "resnet34",
        "batch_size": 20000,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": b1,
            "b2": 0.999,
        },
        "full_batch": True,
    }
    for seed in range(5)
    for alpha in [
        1e-06,
        6.8129206905796085e-06,
        4.641588833612782e-05,
        0.00031622776601683794,
        0.0021544346900318843,
        0.014677992676220704,
        0.1,
    ]
    for b1 in [0, 0.9]
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)
EXPERIMENTS.extend(EXPERIMENTS_ADAM)

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "mnist",
        "model": "lenet5",
        "batch_size": 60000,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
            "momentum": m,
        },
        "full_batch": True,
    }
    for seed in range(5)
    for alpha in np.logspace(-6, 2, num=9, base=10)
    for m in [0.9]
]
EXPERIMENTS.extend(EXPERIMENTS_SGD)

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "mnist",
        "model": "lenet5",
        "batch_size": 60000,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
        },
        "full_batch": True,
    }
    for seed in range(5)
    for alpha in np.logspace(-6, 2, num=9, base=10)
]

EXPERIMENTS_ADAM = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "mnist",
        "model": "lenet5",
        "batch_size": 60000,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0.9,
            "b2": 0.999,
        },
        "full_batch": True,
    }
    for seed in range(5)
    for alpha in np.logspace(-6, 2, num=9, base=10)
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)
EXPERIMENTS.extend(EXPERIMENTS_ADAM)

EXPERIMENTS_ADAM = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar100",
        "model": "resnet50",
        "batch_size": 17000,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0,
            "b2": 0.999,
        },
        "full_batch": True,
    }
    for seed in range(5)
    for alpha in merge_grids(
        np.logspace(-6, 2, num=9, base=10), np.logspace(-4, -2, num=6, base=10)
    )
]

EXPERIMENTS.extend(EXPERIMENTS_ADAM)

EXPERIMENTS_ADAM = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "mnist",
        "model": "lenet5",
        "batch_size": 60000,
        "max_epoch": 200,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0,
            "b2": 0.999,
        },
        "full_batch": True,
    }
    for seed in range(5)
    for alpha in np.logspace(-6, 2, num=9, base=10)
]

EXPERIMENTS.extend(EXPERIMENTS_ADAM)

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
        "max_epoch": 80,
        "seed": seed,
        "trained_norms": False,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
            "momentum": 0.9,
        },
        "full_batch": True,
        "accumulate_steps": step,
    }
    for seed in range(5)
    for alpha in np.logspace(-4, 1, num=6, base=10)
    for step in [256]
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
        "max_epoch": 80,
        "seed": seed,
        "trained_norms": False,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0,
            "b2": 0.999,
        },
        "full_batch": True,
        "accumulate_steps": step,
    }
    for seed in range(5)
    for alpha in [3e-5, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    for step in [256]
]

EXPERIMENTS.extend(EXPERIMENTS_ADAM)
EXPERIMENTS.extend(EXPERIMENTS_SGD)


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
        "max_epoch": 80,
        "seed": seed,
        "trained_norms": False,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
        },
        "full_batch": True,
        "accumulate_steps": step,
    }
    for seed in range(5)
    for alpha in np.logspace(-7, 1, num=9, base=10)
    for step in [256]
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
        "max_epoch": 80,
        "seed": seed,
        "trained_norms": False,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0.9,
            "b2": 0.999,
        },
        "full_batch": True,
        "accumulate_steps": step,
    }
    for seed in range(5)
    for alpha in [3e-5, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    for step in [256]
]

EXPERIMENTS.extend(EXPERIMENTS_ADAM)
EXPERIMENTS.extend(EXPERIMENTS_SGD)

if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        descr="full batch experiments", experiments=EXPERIMENTS
    )
