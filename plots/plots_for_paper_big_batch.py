from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import data_helpers as h
import argparse
import fplt
from data_helpers import name_to_label

# Plotting constants and magic strings
CS = ["#0077bb", "#cc3311", "#ddaa33", "#228833", "#aa3377", "#ee7733", "#66ccee"]
LIGHT_CS = ["#6699cc", "#fa6450", "#eecc66", "#ccddaa", "#efdfff", "#ee8866", "#99ddff"]
BEST_PLOT = "best_run"
METRIC_VS_SS = "vs_ss"
TRAIN_METRIC = "train_metric"
TIME_VS_METRIC = "time_vs_metric"

dataset_sizes = {
    "mnist": 60000,
    "cifar10": 50000,
    "cifar100": 50000,
    "wikitext2_transformer_encoder": 59676,
    "ptb": 7263,
    "wikitext2": 13925,
    "squad": 87714,
}


def gen_best_run_plot(
    model,
    dataset,
    fig,
    axes,
    timestamp=0,
    batch_size=-1,
    acc_step=-1,
    metric=h.TRAIN_LOSS,
    legend=False,
    axis_labels=False,
    full_batch=False,
    big_batch=False,
    iteration_limit=-1,
):
    """Generates data metrics vs. step size plots."""
    print(model, dataset, batch_size, metric)
    df = h.get_data()

    if dataset == "squad":
        df = df[(df[h.EPOCH] > 1)]
    else:
        df = df[(df[h.MAX_EPOCH] > 10)]

    if timestamp != 0:
        df = df[df[h.TIMESTAMP] > timestamp]

    if batch_size > 0:
        df = df[df[h.BATCH_SIZE] == batch_size]

    if acc_step > 0:
        df = df[df[h.ACC_STEP] == acc_step]

    SELECT_PATTERN = (df[h.DATASET] == dataset) & (df[h.MODEL] == model)
    if big_batch:
        if dataset == "wikitext2" and model == "transformer_xl":
            max_epoch = 100
        elif dataset == "squad":
            max_epoch = 7
        else:
            max_epoch = 200
        SELECT_PATTERN = (
            (SELECT_PATTERN)
            & (df[h.DROP_LAST] == True)
            & (df[h.MAX_EPOCH] == max_epoch)
        )
    else:
        SELECT_PATTERN = (SELECT_PATTERN) & (df[h.DROP_LAST].isnull())
    if full_batch:
        SELECT_PATTERN = (SELECT_PATTERN) & (df[h.FULL_BATCH])

    sgd_data = df[
        SELECT_PATTERN
        & (df[h.OPT_NAME] == "SGD")
        & ((df[h.OPT_MOMENTUM].isnull()) | (df[h.OPT_MOMENTUM] == 0))
    ]
    adam_data = df[SELECT_PATTERN & (df[h.OPT_NAME] == "Adam") & (df[h.OPT_B1] == 0.9)]
    num_iterations_per_epoch = 0
    correction_term = 1
    if big_batch:
        key = dataset
        if model == "transformer_encoder":
            key = dataset + "_" + model
        num_iterations_per_epoch = dataset_sizes[key] // abs(batch_size * acc_step)
        # scale iteration limit based on batch size
        iteration_limit = math.ceil(iteration_limit / (num_iterations_per_epoch))
        print(num_iterations_per_epoch)
        print(iteration_limit)

        if dataset != "squad":
            # correction term for improper logging of training loss
            correction_term = dataset_sizes[key] // batch_size
            correction_term = correction_term / num_iterations_per_epoch

    plot_best_run(
        axes,
        "SGD",
        sgd_data,
        metric,
        0,
        iteration_limit=iteration_limit,
        num_iterations_per_epoch=num_iterations_per_epoch,
        correction_term=correction_term,
    )
    plot_best_run(
        axes,
        "Adam",
        adam_data,
        metric,
        1,
        iteration_limit=iteration_limit,
        num_iterations_per_epoch=num_iterations_per_epoch,
        correction_term=correction_term,
    )

    if metric != h.TRAIN_ACC and metric != h.F_ONE:
        axes.set_yscale("log")
    if axis_labels:
        y_metric = "Training Accuracy"
        axes.set_xlabel("Epoch") if not big_batch else axes.set_xlabel("Num Iterations")
        if metric == h.TRAIN_PPL:
            y_metric = "Training PPL"
        elif metric == h.TRAIN_LOSS:
            y_metric = "Training Loss"
        elif metric == h.F_ONE:
            y_metric = "Training F1"
        axes.set_ylabel(
            "$\\bf {} \ {}$ \n {}".format(
                name_to_label[model], name_to_label[dataset], y_metric
            )
        )

    if big_batch:
        axes.set_title(
            str(abs(batch_size * acc_step)),
            pad=2,
        )
    else:
        axes.set_title(
            "{}, {}".format(model.replace("_", "-").capitalize(), dataset.capitalize()),
            pad=2,
        )

    if not axis_labels:
        axes.tick_params(labelleft=False)

    handles, labels = axes.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    loc = "lower right"
    if model == "transformer_xl" or model == "transformer_encoder":
        loc = "upper right"
    if metric == h.TRAIN_LOSS:
        if dataset != "mnist":
            loc = "upper right"
        else:
            loc = "lower right"

    legend = axes.legend(
        by_label.values(), by_label.keys(), fontsize=6, markerscale=6, loc=loc
    )
    [line.set_linewidth(1) for line in legend.get_lines()]


def plot_best_run(
    axes,
    label,
    data,
    metric,
    idx,
    hyperparam=h.K_SS,
    iteration_limit=-1,
    num_iterations_per_epoch=-1,
    correction_term=1,
):
    """Plots best run given a subset of the data"""
    ids = list(data[h.K_ID])
    runs = {}
    for id in tqdm(ids):
        run = h.get_run(id, data_type=metric)
        if iteration_limit > 0:
            run = run[run.index <= iteration_limit]
        runs[id] = run

    if len(ids) > 0:
        hyperparam, run_data, time_steps, _ = gen_best_runs_for_metric(
            ids, runs, data, metric, hyperparam=hyperparam
        )
        mean_loss = []
        max_loss = []
        min_loss = []
        for step in time_steps:
            datas = run_data[run_data["_step"] == step]
            mean_loss.append(datas[metric].mean() * correction_term)
            max_loss.append(datas[metric].max() * correction_term)
            min_loss.append(datas[metric].min() * correction_term)

        # Fix the time step logging so that we start at 0
        i = 2 if metric == h.TRAIN_LOSS else 3
        time_steps = [time_step - i for time_step in time_steps]

        if num_iterations_per_epoch > 0:
            for i in range(len(time_steps)):
                if time_steps[i] != 1:
                    time_steps[i] = num_iterations_per_epoch * time_steps[i]

        axes.plot(
            time_steps,
            mean_loss,
            "-",
            color=CS[idx],
            label=label + " " + str(hyperparam),
            markersize=1,
        )
        axes.fill_between(time_steps, min_loss, max_loss, color=LIGHT_CS[idx])


def gen_best_runs_for_metric(ids, run_for_ids, summary_data, metric, hyperparam=h.K_SS):
    """
    Finds the best step size and data from runs with that step size.
    Also returns the starting metric value (the value at epoch 0) for that run.
    """
    step_sizes = list(summary_data[hyperparam].unique())
    step_to_use = None
    curr_loss = math.inf
    bigger_better = (
        metric == h.TRAIN_ACC or metric == h.F_ONE or metric == h.EXACT_MATCH
    )
    if bigger_better:
        curr_loss = -math.inf
    for step_size in step_sizes:
        df_row = summary_data.loc[summary_data[hyperparam] == step_size]
        ids = list(df_row[h.K_ID])
        vals = []
        for id in ids:
            run_metric = run_for_ids[id]
            vals.append(run_metric[metric].iloc[-1])
        if hyperparam == h.OPT_C and step_size < 0.1:
            continue
        mean_loss = np.array(vals).mean()
        if mean_loss < curr_loss and not bigger_better:
            curr_loss = mean_loss
            step_to_use = step_size
        if mean_loss > curr_loss and bigger_better:
            curr_loss = mean_loss
            step_to_use = step_size

    curr_id = []

    for seed in range(5):
        df_row = summary_data.loc[
            (summary_data[h.SEED] == seed) & (summary_data[hyperparam] == step_to_use)
        ]
        if df_row[h.K_ID].size != 0:
            for i in range(df_row[h.K_ID].size):
                id = df_row[h.K_ID].iloc[i]
                if len(run_for_ids[id]) > 3:
                    curr_id.append(id)

    print(step_to_use)
    print(curr_id)
    runs_list = [run_for_ids[id] for id in curr_id]
    run_data = pd.concat(runs_list)

    run_data[metric].fillna("", inplace=True)
    run_data = run_data[(run_data[metric] != "")]

    time_steps = sorted(list(run_data["_step"].unique()))

    datas = run_data[run_data["_step"] == time_steps[0]]
    max_start_loss = datas[metric].mean()

    return step_to_use, run_data, time_steps, max_start_loss


def gen_step_size_vs_metric_plot(
    model,
    dataset,
    fig,
    axes,
    timestamp=0,
    batch_size=-1,
    acc_step=-1,
    metric=h.TRAIN_LOSS,
    legend=False,
    axis_labels=False,
    full_batch=False,
    big_batch=False,
):
    """Generates data metrics vs. step size plots."""
    print(model, dataset, metric, batch_size)
    df = h.get_data()
    if dataset == "squad":
        df = df[(df[h.EPOCH] > 1)]
    else:
        df = df[(df[h.MAX_EPOCH] > 10)]

    if timestamp != 0:
        df = df[df[h.TIMESTAMP] > timestamp]

    max_ss = df[h.K_SS].max()
    min_ss = df[h.K_SS].min()

    if batch_size > 0:
        df = df[df[h.BATCH_SIZE] == batch_size]

    if acc_step > 0:
        df = df[df[h.ACC_STEP] == acc_step]

    unique_ss = list(df[h.K_SS].unique())
    # fig, axes = plt.subplots(1, 3, figsize=(40, 15))

    SELECT_PATTERN = (df[h.DATASET] == dataset) & (df[h.MODEL] == model)
    if big_batch:
        SELECT_PATTERN = (SELECT_PATTERN) & (df[h.DROP_LAST] == True)
    else:
        SELECT_PATTERN = (SELECT_PATTERN) & (df[h.DROP_LAST].isnull())
    if full_batch:
        SELECT_PATTERN = (SELECT_PATTERN) & (df[h.FULL_BATCH])

    sgd_data = df[
        SELECT_PATTERN & (df[h.OPT_NAME] == "SGD") & (df[h.OPT_MOMENTUM].isnull())
    ]
    adam_data = df[SELECT_PATTERN & (df[h.OPT_NAME] == "Adam") & (df[h.OPT_B1] == 0.9)]
    adam_no_momentum_data = df[
        SELECT_PATTERN & (df[h.OPT_NAME] == "Adam") & (df[h.OPT_B1] == 0)
    ]
    sgd_momentum_data = df[
        SELECT_PATTERN & (df[h.OPT_NAME] == "SGD") & (df[h.OPT_MOMENTUM] == 0.9)
    ]
    ssd_data = df[
        SELECT_PATTERN & (df[h.OPT_NAME] == "Signum") & (df[h.OPT_MOMENTUM] == 0)
    ]
    mssd_data = df[
        SELECT_PATTERN & (df[h.OPT_NAME] == "Signum") & (df[h.OPT_MOMENTUM] == 0.9)
    ]
    adam_ids = list(adam_data[h.K_ID])

    runs_for_ids_adam = {}

    for id in tqdm(adam_ids):
        runs_for_ids_adam[id] = h.get_run(id, data_type=metric)

    _, _, _, max_start_loss = gen_best_runs_for_metric(
        adam_ids, runs_for_ids_adam, adam_data, metric
    )

    if metric == h.TRAIN_ACC or metric == h.F_ONE or metric == h.EXACT_MATCH:
        if dataset == "mnist" or dataset == "cifar10":
            max_start_loss = 0.1
        if dataset == "cifar100":
            max_start_loss = 0.01
        if dataset == "squad":
            max_start_loss = 1
    axes.axhline(y=max_start_loss, color=CS[2], linestyle="-", label="Initial Value")

    plot_ss_vs_metric(
        sgd_data, axes, "SGD", metric, 0, unique_ss, max_start_loss, model
    )
    plot_ss_vs_metric(
        adam_data, axes, "Adam", metric, 1, unique_ss, max_start_loss, model
    )

    if big_batch:
        axes.set_title(
            "{}, {} {}".format(
                model.replace("_", "-").capitalize(),
                dataset.capitalize(),
                abs(batch_size * acc_step),
            ),
            pad=10,
        )
    else:
        axes.set_title(
            "{}, {}".format(model.capitalize(), dataset.capitalize()), pad=10
        )
    # if legend:
    #     handles, labels = axes.get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     axes.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=36)
    axes.set_xscale("log")
    if metric == h.TRAIN_PPL or metric == h.TRAIN_LOSS:
        axes.set_yscale("log")
    axes.set_xlim(min_ss, max_ss)
    # print(axes[0].get_ylim())
    # axes.set_ylim(max_start_loss + 1)
    if axis_labels:
        axes.set_xlabel("Step Size")

        if metric == h.TRAIN_PPL:
            axes.set_ylabel("Training PPL")
        elif metric == h.TRAIN_LOSS:
            axes.set_ylabel("Training Loss")
        else:
            axes.set_ylabel("Training Accuracy")

    if metric != h.TRAIN_PPL:
        axes.yaxis.set_major_locator(plt.MaxNLocator(5))


def plot_ss_vs_metric(data, axes, label, metric, idx, unique_ss, max_start_loss, model):
    steps, mean, min_val, max_val, diverged = [], [], [], [], []
    bigger_better = (
        metric == h.TRAIN_ACC or metric == h.F_ONE or metric == h.EXACT_MATCH
    )
    for ss in sorted(unique_ss):
        non_diverge = data[(data[h.K_SS] == ss)]
        if len(non_diverge) != 0:
            # print(len(non_diverge))
            # print(non_diverge[metric])
            if (
                model == "transformer_encoder"
                and label == "SGD"
                and ss == 10 ** -4
                and metric == h.TRAIN_LOSS
            ):
                # Remove outlier from crash
                non_diverge = data[(data[h.K_SS] == ss) & (data[metric] < 9)]

            # print(non_diverge[metric])
            mean_loss = non_diverge[metric].mean()
            # print(mean_loss)
            print(max_start_loss)
            max_loss = non_diverge[metric].max() - mean_loss
            if math.isnan(mean_loss):
                mean_loss = math.inf
            min_loss = mean_loss - non_diverge[metric].min()

            if (
                (not bigger_better and (mean_loss >= max_start_loss - 0.01))
                or (bigger_better and (mean_loss <= max_start_loss + 0.01))
                # or (non_diverge[h.EPOCH].mean() < non_diverge[h.MAX_EPOCH].mean() - 1)
            ):
                axes.scatter(
                    ss,
                    max_start_loss,
                    facecolors="none",
                    linewidth=5.0,
                    edgecolors=CS[idx],
                    label=label + " Diverged",
                    s=160,
                )

                diverged.append(ss)
            else:
                steps.append(ss)
                mean.append(mean_loss)
                min_val.append(min_loss)
                max_val.append(max_loss)
    axes.errorbar(
        steps,
        mean,
        yerr=[min_val, max_val],
        fmt="o-",
        color=CS[idx],
        label=label,
        markersize=1,
    )

    if len(steps) != 0:

        len_diverged = len(diverged)
        for i in range(len_diverged):
            curr = diverged[i]
            if curr < steps[0] and (
                (i < len_diverged - 1 and diverged[i + 1] > steps[0])
                or i == len_diverged - 1
            ):
                to_plot_x = [curr, steps[0]]
                to_plot_y = [max_start_loss, mean[0]]
                axes.plot(to_plot_x, to_plot_y, "--", color=CS[idx])
            if curr > steps[-1]:
                to_plot_x = [curr, steps[-1]]
                to_plot_y = [max_start_loss, mean[-1]]
                axes.plot(to_plot_x, to_plot_y, "--", color=CS[idx])
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        default=None,
        type=str,
        help="the model used, or comma separated list of models",
    )
    parser.add_argument(
        "dataset",
        default=None,
        type=str,
        help="the dataset, or comma separated list of datasets",
    )
    parser.add_argument(
        "--timestamp",
        default=0,
        type=int,
        help="only get runs with timestamp after given timestamp",
    )
    parser.add_argument(
        "--batch_size",
        default="-1",
        type=str,
        help="the batch_size, or comma separated list of batch_size",
    )
    parser.add_argument(
        "--acc_step",
        default="-1",
        type=str,
        help="the accumulate_steps, or comma separated list of accumulate_steps",
    )
    parser.add_argument(
        "--metric", default=h.TRAIN_LOSS, type=str, help="metric for plot"
    )
    parser.add_argument("--plot_type", default=BEST_PLOT, type=str, help="type of plot")
    parser.add_argument(
        "--logged_step",
        action="store_true",
        help="true if logged every step instead of epoch",
    )
    parser.add_argument("--full_batch", action="store_true", help="get full batch runs")
    parser.add_argument("--big_batch", action="store_true", help="get big batch runs")

    args = parser.parse_args()

    assert not (args.full_batch and args.big_batch)

    if "," in args.dataset:
        datasets = args.dataset.split(",")
        models = args.model.split(",")
        batch_sizes = args.batch_size.split(",")
        acc_steps = args.acc_step.split(",")
    else:
        datasets = [args.dataset]
        models = [args.model]
        batch_sizes = [args.batch_size]
        acc_steps = [args.acc_step]

    batch_sizes = [int(batch_size) for batch_size in batch_sizes]
    acc_steps = [int(acc_step) for acc_step in acc_steps]
    if len(acc_steps) < len(batch_sizes):
        acc_steps = [-1 for i in range(len(batch_sizes))]

    length = len(models)
    plt.rcParams.update({"font.size": 8})
    plt.rcParams["lines.linewidth"] = 1
    plt.rc("ytick", labelsize=7)
    plt.rc("xtick", labelsize=7)
    fig, axes = plt.subplots(1, length, figsize=(7, 5.5 / (60 / 20)))

    # Iteration limit will help limit the plot sizes so all the runs of different batch sizes are visible
    if args.big_batch and len(datasets) > 1:
        if datasets[0] != "squad":
            key = datasets[2]
            if models[2] == "transformer_encoder":
                key = datasets[2] + "_" + models[2]
            num_iterations_per_epoch = dataset_sizes[key] // abs(
                batch_sizes[2] * acc_steps[2]
            )
            iteration_limit = 200 * num_iterations_per_epoch
        else:
            key = datasets[1]
            num_iterations_per_epoch = dataset_sizes[key] // abs(
                batch_sizes[1] * acc_steps[1]
            )
            iteration_limit = 80 * num_iterations_per_epoch
    else:
        iteration_limit = -1

    for i in range(length):
        y = i // 3
        x = i % 3
        print(x, y, i)
        if args.metric == TRAIN_METRIC:
            if datasets[i] in ["cifar10", "cifar100", "mnist"]:
                metric = h.TRAIN_ACC
            elif datasets[i] in ["squad"]:
                metric = h.F_ONE
            else:
                metric = h.TRAIN_PPL
        else:
            metric = args.metric

        if args.plot_type == BEST_PLOT:
            gen_best_run_plot(
                models[i],
                datasets[i],
                fig,
                axes[i],
                timestamp=args.timestamp,
                batch_size=batch_sizes[i],
                acc_step=acc_steps[i],
                metric=metric,
                legend=y == 0 and x == 1,
                axis_labels=i == 0,
                full_batch=args.full_batch,
                big_batch=args.big_batch,
                iteration_limit=iteration_limit,
            )

        if args.plot_type == METRIC_VS_SS:
            gen_step_size_vs_metric_plot(
                models[i],
                datasets[i],
                fig,
                axes[i],
                timestamp=args.timestamp,
                batch_size=batch_sizes[i],
                acc_step=acc_steps[i],
                metric=metric,
                legend=i == length - 1,
                axis_labels=x == 0,
                full_batch=args.full_batch,
                big_batch=args.big_batch,
            )

    if args.plot_type == BEST_PLOT and args.big_batch:

        x_lim_max = axes[0].get_xlim()[1]

        if models[0] == "bert_base_pretrained":
            x_lim_max = 450

        y_lim = [math.inf, -math.inf]
        for i in range(length):
            y = i // 3
            x = i % 3
            y_lim = [
                min(y_lim[0], axes[i].get_ylim()[0]),
                max(y_lim[1], axes[i].get_ylim()[1]),
            ]
        for i in range(length):
            y = i // 3
            x = i % 3
            curr = axes[i].get_xlim()
            axes[i].set_xlim(curr[0], x_lim_max)

            print(axes[i].get_xlim())
            axes[i].set_ylim(y_lim[0], min(y_lim[1], 10 ** 7))

    for i in range(length):
        fplt.hide_frame(axes[i])

    if args.big_batch:
        fig.suptitle("Minibatch Size", y=0.95)

    fig.tight_layout()
    print("save")
    prefix = ""
    if args.full_batch:
        prefix = "full_batch_"
    if args.big_batch:
        prefix = "big_batch_{}_{}_".format(models[0], datasets[0])
    plt.savefig("{}plot_paper_{}_{}.pdf".format(prefix, args.plot_type, args.metric))

    plt.close()
