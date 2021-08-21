"""
If called: a script to download the initial data dump
If imported: helper functions to load data
"""
import qparse
import pdb
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
    # print(df[OPT])

    for string_col in string_columns:
        df[string_col].fillna("", inplace=True)
        df = df[df[string_col] != ""]

    return df


def select_group(group, df):
    return df[df[K_GROUP] == group]


def get_data():
    df = load_csv()
    df = clean_data(df)
    # df = select_group(group, df)
    return df


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
        # print(file_path)
        # print(df)

    return pd.read_csv(file_path, header=0, squeeze=True)


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
    runs = api.runs(WANDB_PROJECT, per_page=500)

    summary_list = []
    config_list = []
    name_list = []
    id_list = []
    for run in tqdm(runs):
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items()})

        name_list.append(run.name)
        id_list.append(run.id)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({"name": name_list, "id": id_list})
    all_df = pd.concat([name_df, config_df, summary_df], axis=1)

    all_df.to_csv(ALL_RUNS_SUMMARY_FILE)


if __name__ == "__main__":
    args = qparse.qparse(
        descr="Download data from Wandb",
        args=[],
        flags=[
            ("summary", "Download the summary of all runs in the project"),
            ("init", "Download data about each run at initialization"),
        ],
    )

    if args.summary:
        download_all_runs_summary()
    if args.init:
        download_init_values()
