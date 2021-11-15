import logging
import os
from pathlib import Path
import wandb
from dotenv import load_dotenv


def init(workspace_dir, experiment_hash, dataset_name, debug, verbose):
    """Initialize the logging"""

    load_dotenv()
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')

    logs_path = os.path.join(workspace_dir, dataset_name, experiment_hash, "logs")
    Path(logs_path).mkdir(parents=True, exist_ok=True)

    file_path = os.path.join(logs_path, "logs")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=file_path,
        filemode="a+",
    )

    if verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)

    if not debug:
        wandb_path = os.path.join(workspace_dir, dataset_name, experiment_hash)
        wandb.init(project=WANDB_PROJECT, dir=wandb_path)

    def logging_helper(dict, commit=True):
        if not debug:
            wandb.log(dict, commit=commit)

        logging.info(dict)

    return logging_helper
