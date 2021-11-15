import json

from . import dataset, expmaker, logging, model, optim
from .experiment import Experiment

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment runner")
    parser.add_argument(
        "experiment_file",
        type=str,
        help="Experiment file",
        default=None,
    )
    parser.add_argument("workspace_path", type=str, help="Workspace path", default=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode, won't create wandb logs",
        default=False,
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print logs", default=False
    )
    parser.add_argument("--gpu", nargs="?", type=str, default="cuda", help="GPU name")
    parser.add_argument(
        "--trained_norms",
        action="store_true",
        help="Calculate noise norm at epoch 1 and at max_epoch",
        default=False,
    )
    args = parser.parse_args()

    if args.experiment_file is None:
        raise ValueError
    with open(args.experiment_file, "r") as fp:
        exp_dict = json.load(fp)
        experiment_hash = args.experiment_file.split("/")[-1].strip(".json")
        exp = Experiment(
            exp_dict,
            args.workspace_path,
            experiment_hash,
            args.debug,
            args.verbose,
            args.gpu,
            args.trained_norms,
        )
        exp.run()
