import numpy as np
import explib


def merge_grids(*grids):
    return sorted(list(set.union(*[set(grid) for grid in grids])))


EXPERIMENTS = []

# SGD and Adam
EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "mnist",
        "model": "lenet5",
        "batch_size": 128,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
            "momentum": m,
        },
    }
    for alpha in np.logspace(-6, 2, num=9, base=10)
    for seed in range(5)
    for m in [0.9]
]
EXPERIMENTS.extend(EXPERIMENTS_SGD)

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "mnist",
        "model": "lenet5",
        "batch_size": 128,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
        },
    }
    for alpha in np.logspace(-6, 2, num=9, base=10)
    for seed in range(5)
]

EXPERIMENTS_ADAM = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "mnist",
        "model": "lenet5",
        "batch_size": 128,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": b1,
            "b2": 0.999,
        },
    }
    for alpha in np.logspace(-6, 2, num=9, base=10)
    for seed in range(5)
    for b1 in [0, 0.9]
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)
EXPERIMENTS.extend(EXPERIMENTS_ADAM)

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar10",
        "model": "resnet34",
        "batch_size": 128,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
            "momentum": 0,
        },
    }
    for alpha in merge_grids(
        np.logspace(-5, 2, num=8, base=10), np.logspace(-2, 0, num=6, base=10)
    )
    for seed in range(5)
]

EXPERIMENTS_SGD_M = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar10",
        "model": "resnet34",
        "batch_size": 128,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
            "momentum": 0.9,
        },
    }
    for alpha in merge_grids(
        np.logspace(-5, 3, num=9, base=10), np.logspace(-2, -0, num=6, base=10)
    )
    for seed in range(5)
]

EXPERIMENTS_ADAM = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar10",
        "model": "resnet34",
        "batch_size": 128,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0.9,
            "b2": 0.999,
        },
    }
    for alpha in merge_grids(
        np.logspace(-6, 2, num=9, base=10), np.logspace(-4, -2, num=6, base=10)
    )
    for seed in range(5)
]

EXPERIMENTS_ADAM_M = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar10",
        "model": "resnet34",
        "batch_size": 128,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0,
            "b2": 0.999,
        },
    }
    for alpha in merge_grids(
        np.logspace(-6, 3, num=10, base=10), np.logspace(-4, -2, num=6, base=10)
    )
    for seed in range(5)
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)
EXPERIMENTS.extend(EXPERIMENTS_SGD_M)
EXPERIMENTS.extend(EXPERIMENTS_ADAM)
EXPERIMENTS.extend(EXPERIMENTS_ADAM_M)

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar100",
        "model": "resnet50",
        "batch_size": 128,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
        },
    }
    for alpha in merge_grids(
        np.logspace(-5, 2, num=8, base=10), np.logspace(-2, 0, num=6, base=10)
    )
    for seed in range(5)
]

EXPERIMENTS_SGD_M = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar100",
        "model": "resnet50",
        "batch_size": 128,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
            "momentum": 0.9,
        },
    }
    for alpha in merge_grids(
        np.logspace(-5, 3, num=9, base=10), np.logspace(-2, -0, num=6, base=10)
    )
    for seed in range(5)
]

EXPERIMENTS_ADAM = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar100",
        "model": "resnet50",
        "batch_size": 128,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0.9,
            "b2": 0.999,
        },
    }
    for alpha in merge_grids(
        np.logspace(-6, 2, num=9, base=10), np.logspace(-4, -2, num=6, base=10)
    )
    for seed in range(5)
]

EXPERIMENTS_ADAM_M = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar100",
        "model": "resnet50",
        "batch_size": 128,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0,
            "b2": 0.999,
        },
    }
    for alpha in merge_grids(
        np.logspace(-6, 3, num=10, base=10), np.logspace(-4, -2, num=6, base=10)
    )
    for seed in range(5)
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)
EXPERIMENTS.extend(EXPERIMENTS_SGD_M)
EXPERIMENTS.extend(EXPERIMENTS_ADAM)
EXPERIMENTS.extend(EXPERIMENTS_ADAM_M)

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["ppl"],
        "dataset": "wikitext2",
        "model": "transformer_encoder",
        "model_args": {
            "tgt_len": 35,
        },
        "batch_size": 64,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
            "momentum": m,
        },
    }
    for alpha in merge_grids(
        np.logspace(-7, 1, num=9, base=10), np.logspace(-2, 0, num=6, base=10)
    )
    for seed in range(5)
    for m in [0.9]
]
EXPERIMENTS.extend(EXPERIMENTS_SGD)

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["ppl"],
        "dataset": "wikitext2",
        "model": "transformer_encoder",
        "model_args": {
            "tgt_len": 35,
        },
        "batch_size": 64,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
        },
    }
    for alpha in merge_grids(
        np.logspace(-7, 1, num=9, base=10), np.logspace(-2, 0, num=6, base=10)
    )
    for seed in range(5)
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
        "batch_size": 64,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0.9,
            "b2": 0.999,
        },
    }
    for alpha in merge_grids(
        np.logspace(-6, 0, num=7, base=10), np.logspace(-4, -2, num=6, base=10)
    )
    for seed in range(5)
]

EXPERIMENTS_ADAM_M = [
    {
        "loss_func": "logloss",
        "metrics": ["ppl"],
        "dataset": "wikitext2",
        "model": "transformer_encoder",
        "model_args": {
            "tgt_len": 35,
        },
        "batch_size": 64,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0,
            "b2": 0.999,
        },
    }
    for alpha in merge_grids(
        np.logspace(-5, 0, num=6, base=10), np.logspace(-5, 0, num=10, base=10)
    )
    for seed in range(5)
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)
EXPERIMENTS.extend(EXPERIMENTS_ADAM)
EXPERIMENTS.extend(EXPERIMENTS_ADAM_M)

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["ppl"],
        "dataset": "ptb",
        "model": "transformer_xl",
        "model_args": {
            "n_layer": 6,
            "d_model": 512,
            "n_head": 8,
            "d_head": 64,
            "d_inner": 2048,
            "dropout": 0.1,
            "dropatt": 0.0,
            "tgt_len": 128,
            "mem_len": 128,
        },
        "batch_size": 64,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
            "momentum": m,
        },
    }
    for alpha in merge_grids(
        np.logspace(-6, 1, num=8, base=10), np.logspace(-3, 1, num=6, base=10)
    )
    for seed in range(5)
    for m in [0.9]
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)
EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["ppl"],
        "dataset": "ptb",
        "model": "transformer_xl",
        "model_args": {
            "n_layer": 6,
            "d_model": 512,
            "n_head": 8,
            "d_head": 64,
            "d_inner": 2048,
            "dropout": 0.1,
            "dropatt": 0.0,
            "tgt_len": 128,
            "mem_len": 128,
        },
        "batch_size": 64,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
        },
    }
    for alpha in merge_grids(
        np.logspace(-6, 1, num=8, base=10), np.logspace(-3, 1, num=6, base=10)
    )
    for seed in range(5)
]

EXPERIMENTS_ADAM = [
    {
        "loss_func": "logloss",
        "metrics": ["ppl"],
        "dataset": "ptb",
        "model": "transformer_xl",
        "model_args": {
            "n_layer": 6,
            "d_model": 512,
            "n_head": 8,
            "d_head": 64,
            "d_inner": 2048,
            "dropout": 0.1,
            "dropatt": 0.0,
            "tgt_len": 128,
            "mem_len": 128,
        },
        "batch_size": 64,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0.9,
            "b2": 0.999,
        },
    }
    for alpha in merge_grids(
        np.logspace(-6, 1, num=8, base=10), np.logspace(-4, -3, num=4, base=10)
    )
    for seed in range(5)
]

EXPERIMENTS_ADAM_M = [
    {
        "loss_func": "logloss",
        "metrics": ["ppl"],
        "dataset": "ptb",
        "model": "transformer_xl",
        "model_args": {
            "n_layer": 6,
            "d_model": 512,
            "n_head": 8,
            "d_head": 64,
            "d_inner": 2048,
            "dropout": 0.1,
            "dropatt": 0.0,
            "tgt_len": 128,
            "mem_len": 128,
        },
        "batch_size": 64,
        "max_epoch": 100,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": 0,
            "b2": 0.999,
        },
    }
    for alpha in merge_grids(
        np.logspace(-5, 0, num=6, base=10), np.logspace(-5, 0, num=10, base=10)
    )
    for seed in range(5)
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)
EXPERIMENTS.extend(EXPERIMENTS_ADAM)
EXPERIMENTS.extend(EXPERIMENTS_ADAM_M)

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
        "max_epoch": 4,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
            "momentum": m,
        },
    }
    for seed in range(5)
    for alpha in np.logspace(-7, 1, num=9, base=10)
    for m in [0.9]
]

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
        "max_epoch": 4,
        "seed": seed,
        "opt": {
            "name": "SGD",
            "alpha": alpha,
        },
    }
    for seed in range(5)
    for alpha in np.logspace(-7, 1, num=9, base=10)
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
        "max_epoch": 4,
        "seed": seed,
        "opt": {
            "name": "Adam",
            "alpha": alpha,
            "b1": b1,
            "b2": 0.999,
        },
    }
    for seed in range(5)
    for alpha in [3e-5, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    for b1 in [0, 0.9]
]

EXPERIMENTS.extend(EXPERIMENTS_ADAM)
EXPERIMENTS.extend(EXPERIMENTS_SGD)


if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        descr="all training experiments", experiments=EXPERIMENTS
    )
