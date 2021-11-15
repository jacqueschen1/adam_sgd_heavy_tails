import explib


EXPERIMENTS = []

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "mnist",
        "model": "lenet5",
        "batch_size": 128,
        "max_epoch": 0,
        "seed": 0,
        "opt": {
            "name": "SGD",
            "alpha": 0.1,
        },
        "init_noise_norm": True,
    }
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar10",
        "model": "resnet34",
        "batch_size": 128,
        "max_epoch": 0,
        "seed": 4,
        "opt": {
            "name": "SGD",
            "alpha": 0.1,
        },
        "init_noise_norm": True,
    }
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)

EXPERIMENTS_SGD = [
    {
        "loss_func": "logloss",
        "metrics": ["accuracy"],
        "dataset": "cifar100",
        "model": "resnet50",
        "batch_size": 128,
        "max_epoch": 0,
        "seed": 0,
        "opt": {
            "name": "SGD",
            "alpha": 0.1,
        },
        "init_noise_norm": True,
    }
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
        "max_epoch": 0,
        "seed": 0,
        "opt": {
            "name": "SGD",
            "alpha": 0.1,
        },
        "init_noise_norm": True,
    }
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
        "max_epoch": 0,
        "seed": 0,
        "opt": {
            "name": "SGD",
            "alpha": 0.1,
        },
        "init_noise_norm": True,
    }
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
        "max_epoch": 0,
        "seed": 0,
        "opt": {
            "name": "SGD",
            "alpha": 0.1,
        },
        "init_noise_norm": True,
    }
]

EXPERIMENTS.extend(EXPERIMENTS_SGD)


if __name__ == "__main__":
    explib.expmaker.experiment_maker_cli(
        descr="all training experiments", experiments=EXPERIMENTS
    )
