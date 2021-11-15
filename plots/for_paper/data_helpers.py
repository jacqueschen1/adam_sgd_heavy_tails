"""
If called: a script to download the initial data dump
If imported: helper functions to load data
"""
import functools

import pdb
import qparse
from tqdm import tqdm
import os
import pandas as pd
import wandb
import json
from pandas.io.json import json_normalize
import ast

api = wandb.Api()

# %%
# Constants and magic strings
#

ALL_RUNS_SUMMARY_FILE = "runs_summary.csv"
ALL_RUNS_INIT_FILE = "runs_init.csv"
DATA_FOLDER = "data"
WANDB_PROJECT = "jacqueschen1/test-runs-adam-sgd"

MAX_EPOCH = "max_epoch"
DATASET = "dataset"
EPOCH = "epoch"
OPT = "opt"
K_ID = "id"
K_SS = "alpha"
OPT_NAME = "opt_name"
OPT_C = "c"
TRAIN_LOSS = "training_loss"
SEED = "seed"
MODEL = "model"
BATCH_SIZE = "batch_size"
TIMESTAMP = "_timestamp"
TRAIN_ACC = "train_accuracy"
TRAIN_PPL = "train_ppl"
ARMIJO_STEP_SIZE = "step_size"
GRAD_NORM = "grad_norm"
INIT_DIFF = "init_diff"
BIGGER_KERNEL = "bigger_kernel"
FULL_BATCH = "full_batch"
OPT_B1 = "b1"
OPT_MOMENTUM = "momentum"
DROP_LAST = "drop_last"
ACC_STEP = "accumulate_steps"
F_ONE = "train_exact_f1"
EXACT_MATCH = "train_exact_match"
HASH = "hash"
AVERAGE_LOSS = "average_training_loss"

name_to_label = {
    "mnist": "MNIST",
    "resnet50": "ResNet50",
    "resnet34": "ResNet34",
    "transformer_encoder": "Trans-Enc",
    "transformer_xl": "Transformer-XL",
    "bert_base_pretrained": "BERT",
    "wikitext2": "Wikitext2",
    "ptb": "PTB",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "lenet5": "LeNet5",
    "squad": "SQuAD",
}


def load_csv():
    return pd.read_csv(ALL_RUNS_SUMMARY_FILE, header=0, squeeze=True)


def clean_data(df):
    new_df = df["opt"].apply(lambda s: pd.Series(ast.literal_eval(s) if s == s else {}))
    new_df.rename(columns={"name": "opt_name"}, inplace=True)

    df = pd.concat([df, new_df], axis=1)

    new_df = df["model_args"].apply(
        lambda s: pd.Series(ast.literal_eval(s) if s == s else {})
    )

    df = pd.concat([df, new_df], axis=1)

    int_columns = [MAX_EPOCH, EPOCH, BATCH_SIZE, TIMESTAMP]
    float_columns = []
    string_columns = [DATASET, TRAIN_LOSS]
    for int_col in int_columns:
        df[int_col].apply(lambda x: int(x) if x == x else "")

    for string_col in string_columns:
        df[string_col].fillna("", inplace=True)
        df = df[df[string_col] != ""]

    df[TRAIN_ACC] = df[TRAIN_ACC] * 100

    return df


def select_group(group, df):
    return df[df[K_GROUP] == group]


@functools.lru_cache(maxsize=100, typed=False)
def get_data():
    df = load_csv()
    df = clean_data(df)
    return df


@functools.lru_cache(maxsize=None, typed=False)
def get_run(id, data_type=TRAIN_LOSS):
    """
    Returns the data from a particular run, including loss/iteration (sampled).
    Downloads the data from wandb if this particular run is new.
    Saves the result in DATA_FOLDER.
    """
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    file_path = os.path.join(DATA_FOLDER, id + "_" + data_type + ".csv")

    if not os.path.isfile(file_path):
        run = api.run(WANDB_PROJECT + "/" + id)
        df = run.history(samples=500, keys=[data_type], pandas=(True))
        df.to_csv(file_path)

    df = pd.read_csv(file_path, header=0, squeeze=True)

    if (data_type == TRAIN_ACC) and len(df) > 0:
        df[TRAIN_ACC] = df[TRAIN_ACC] * 100

    return df


def download_init_values():
    """
    Downloads the value of the loss (and other stats) at the first step logged for each run.
    """
    runs_df = get_data()

    id_list = []
    firststeps = []
    for id in tqdm(runs_df[K_ID]):
        run = api.run(WANDB_PROJECT + "/" + id)
        for step in run.scan_history(page_size=100):
            firststeps.append(step)
            print(step)
            break
        id_list.append(run.id)

    steps_df = pd.DataFrame.from_records(firststeps)
    name_df = pd.DataFrame({"id": id_list})

    all_df = pd.concat([name_df, steps_df], axis=1)

    all_df.to_csv(ALL_RUNS_INIT_FILE)


def download_all_runs_summary():
    """
    Download a summary of all runs on the wandb project
    """
    runs = api.runs(WANDB_PROJECT, per_page=1000)

    summary_list = []
    config_list = []
    name_list = []
    id_list = []
    tags_list = []
    state_list = []
    for run in tqdm(runs):
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items()})
        name_list.append(run.name)
        id_list.append(run.id)
        state_list.append(run.state)
        tags_list.append(run.tags)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_tags_df = pd.DataFrame(
        {"name": name_list, "tags": tags_list, "state": state_list, "id": id_list}
    )
    all_df = pd.concat([name_tags_df, config_df, summary_df], axis=1)

    all_df.to_csv(ALL_RUNS_SUMMARY_FILE)


tags_to_checkfuncs = {
    "old_runs": lambda run, summary: (summary.get("_timestamp") <= 1624365143),
    "testing_big_batch_with_arbitrary_stepsize": lambda run, summary: (
        (summary.get("opt.alpha") == 0.4)
    ),
    "old_full_batch_transformer_encoder_ptb": lambda run, summary: (
        (summary.get("full_batch", False) == True)
        and (
            (summary.get("dataset") == "ptb")
            or (summary.get("model") == "transformer_encoder")
        )
        and summary.get("_timestamp") <= 1629129016
    ),
    "old_full_batch_squad": lambda run, summary: (
        (summary.get("full_batch", False) == True)
        and (summary.get("dataset") in ["squad"])
        and summary.get("_timestamp") <= 1629340576
    ),
    "old_squad_momentum": lambda run, summary: (
        (
            (
                (summary.get("opt.name") == "SGD")
                and summary.get("opt.momentum", None) == 0.9
            )
            or (
                (summary.get("opt.name") == "Adam") and summary.get("opt.b1", None) == 0
            )
        )
        and (summary.get("dataset") in ["squad"])
        and summary.get("_timestamp") <= 1628662010
    ),
    "old_test_runs": lambda run, summary: (
        not (summary.get("dataset") in ["squad"])
        and int(summary.get("max_epoch")) <= 10
    ),
    "old_increasing_batch_size": lambda run, summary: (
        (summary.get("drop_last", False) == True)
        and (summary.get("final_reruns", False) == False)
    ),
    "ran_only_for_one_epoch_but_didnt_crash_probably_messing_around": lambda run, summary: (
        (summary.get("epoch") == 0.0)
        and (run.state == "finished")
        and (summary.get("training_error", False) == False)
    ),
    "old_squad_not_big_batch": lambda run, summary: (
        (summary.get("drop_last", False) == False)
        and (summary.get("dataset") in ["squad"])
        and (
            (
                summary.get("_timestamp") <= 1631176793
                and (summary.get("full_batch", False) == False)
            )
            or (
                summary.get("opt.alpha", 0) > 9.9
                and summary.get("opt.momentum", None) == 0.9
                and (summary.get("opt.name") == "SGD")
                and (summary.get("full_batch", False) == True)
            )
        )
    ),
    "short_full_batch_runs": lambda run, summary: (
        (summary.get("drop_last", False) == True)
        and (summary.get("final_reruns", False) == True)
        and (
            (
                (summary.get("dataset") in ["mnist"])
                and (summary.get("batch_size") == 60000)
                and (summary.get("max_epoch") == 200)
            )
            or (
                (summary.get("dataset") in ["ptb"])
                and (summary.get("accumulate_steps") == 112)
                and (summary.get("max_epoch") == 200)
            )
            or (
                (summary.get("dataset") in ["wikitext2"])
                and (summary.get("accumulate_steps") == 232)
                and (summary.get("max_epoch") == 200)
            )
        )
    ),
}


def update_tags():
    def update_tags_for(run):
        if run.state == "running":
            return False

        updated_run = False
        old_tags = list([tag for tag in run.tags])
        try:
            for tag, check_func in tags_to_checkfuncs.items():
                summary = run.summary._json_dict
                if len(summary) == 0:
                    continue

                if check_func(run, summary):
                    if tag not in run.tags:
                        run.tags.append(tag)
                        updated_run = True
                else:
                    if tag in run.tags:
                        run.tags.remove(tag)
                        updated_run = True
        except:
            pdb.set_trace()

        if updated_run:
            print(f"Updating runs. {old_tags} -> {run.tags}")
            run.update()

    runs = api.runs(WANDB_PROJECT, per_page=1000)
    for run in tqdm(runs):
        update_tags_for(run)


if __name__ == "__main__":
    args = qparse.qparse(
        descr="Download data from Wandb",
        args=[],
        flags=[
            ("summary", "Download the summary of all runs in the project"),
            ("init", "Download data about each run at initialization"),
            ("update_tags", "Update tags, mark old runs"),
        ],
    )

    if args.summary:
        download_all_runs_summary()
    if args.init:
        download_init_values()
    if args.update_tags:
        update_tags()
