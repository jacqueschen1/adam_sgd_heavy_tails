import math
import os
import pdb
from pathlib import Path
import matplotlib as mpl

import data_helpers as h
from data_helpers import name_to_label
import argparse
import pandas as pd
import numpy as np

LINESTYLE_TO_DIVERGENCE = "dotted"

PT_YELLOW = "#DDAA33"
PT_RED = "#BB5566"
PT_BLUE = "#004488"
CS = [PT_YELLOW, PT_RED, PT_BLUE, PT_YELLOW, PT_RED, PT_BLUE]

# CS = ["#0077bb", "#cc3311", "#ddaa33", "#228833", "#aa3377", "#ee7733", "#66ccee"]
# CS = ["#0077bb", "#cc3311", "#33bbee", "#ee3377", "#aa3377", "#ee7733", "#ddaa33"]
# LIGHT_CS = ["#6699cc", "#fa6450", "#88ccee", "#ee99aa", "#efdfff", "#ee8866", "#99ddff"]
# LIGHT_CS = ["#6699cc", "#fa6450", "#eecc66", "#ccddaa", "#efdfff", "#ee8866", "#99ddff"]
BEST_PLOT = "best_run"
METRIC_VS_SS = "vs_ss"
TRAIN_METRIC = "train_metric"
TRAIN_LOSS = "training_loss"
TIME_VS_METRIC = "time_vs_metric"


LABEL_ADAM_NM = "Adam-NM"
LABEL_ADAM = "Adam"
LABEL_SGD_M = "SGD+M"
LABEL_SGD = "SGD"
LABEL_INITIAL_VALUE = "Initial Value"
LABEL_STEP_SIZE = "Step-size"
LABEL_EPOCH = "Epoch"
LABEL_ITER = "Num. Iterations"

LABEL_TO_STYLE = {
    LABEL_ADAM: {"color": PT_RED, "linestyle": "-"},
    LABEL_ADAM_NM: {"color": PT_RED, "linestyle": "--"},
    LABEL_SGD_M: {"color": PT_BLUE, "linestyle": "-"},
    LABEL_SGD: {"color": PT_BLUE, "linestyle": "--"},
}
COLOR_INITIAL_VALUE = PT_YELLOW

metric_to_text = {
    h.TRAIN_PPL: "Perplexity",
    h.TRAIN_LOSS: "Loss",
    h.F_ONE: "F1 Score",
    h.TRAIN_ACC: "Accuracy",
}

dataset_sizes = {
    "mnist": 60000,
    "cifar10": 50000,
    "cifar100": 50000,
    "wikitext2_transformer_encoder": 59648,
    "ptb": 7263,
    "wikitext2": 13925,
    "squad": 87714,
}


def cli():
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
    parser.add_argument("--full_batch", action="store_true", help="get full batch runs")
    parser.add_argument("--big_batch", action="store_true", help="get big batch runs")
    parser.add_argument("--momentum", action="store_true")

    return parser


def init_plt_style(plt):
    plt.rcParams.update({"font.size": 8})
    plt.rcParams["lines.linewidth"] = 1
    plt.rc("ytick", labelsize=7)
    plt.rc("xtick", labelsize=7)


def process_args(args):
    if args.big_batch and args.full_batch:
        raise ValueError("Big batch and Full batch can't be on at the same time.")

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
        acc_steps = [-1 for _ in range(len(batch_sizes))]

    assert len(datasets) == len(models)

    length = len(datasets)

    return datasets, models, batch_sizes, acc_steps, length


def select_metric(metric_type, dataset):
    if metric_type == TRAIN_METRIC:
        if dataset in ["cifar10", "cifar100", "mnist"]:
            metric = h.TRAIN_ACC
        elif dataset in ["squad"]:
            metric = h.F_ONE
        else:
            metric = h.TRAIN_PPL
    elif metric_type == TRAIN_LOSS:
        metric = TRAIN_LOSS
    else:
        raise ValueError(f"Metric type unknown: {metric_type}")
    return metric


def common_ss_vs_metric_errorbar_and_points(
    axes, diverged, label, max_start_loss, max_val, mean, min_val, steps
):
    color = LABEL_TO_STYLE[label]["color"]
    axes.errorbar(
        steps,
        mean,
        yerr=[min_val, max_val],
        fmt="o",
        linestyle=LABEL_TO_STYLE[label]["linestyle"],
        color=color,
        label=label,
        markersize=1,
    )

    axes.fill_between(
        steps,
        np.array(mean) - np.array(min_val),
        np.array(mean) + np.array(max_val),
        color=color,
        alpha=0.1,
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
                axes.plot(
                    to_plot_x, to_plot_y, linestyle=LINESTYLE_TO_DIVERGENCE, color=color
                )
            if curr > steps[-1]:
                to_plot_x = [curr, steps[-1]]
                to_plot_y = [max_start_loss, mean[-1]]
                axes.plot(
                    to_plot_x, to_plot_y, linestyle=LINESTYLE_TO_DIVERGENCE, color=color
                )
                break


def prepare_best_runs(curr_id, metric, run_for_ids, step_to_use):
    print(curr_id, step_to_use)
    runs_list = [run_for_ids[run_id] for run_id in curr_id]
    run_data = pd.concat(runs_list)
    run_data[metric].fillna("", inplace=True)
    run_data = run_data[(run_data[metric] != "")]
    time_steps = sorted(list(run_data["_step"].unique()))
    datas = run_data[run_data["_step"] == time_steps[0]]
    max_start_loss = datas[metric].mean()
    return step_to_use, run_data, time_steps, max_start_loss


def get_data_summary_for_metric(metric, run_data, time_steps):
    mean_loss = []
    max_loss = []
    min_loss = []
    for step in time_steps:
        datas = run_data[run_data["_step"] == step]
        mean_loss.append(datas[metric].mean())
        max_loss.append(datas[metric].max())
        min_loss.append(datas[metric].min())
    return max_loss, mean_loss, min_loss


def make_legend(axes):
    handles, labels = axes.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = axes.legend(by_label.values(), by_label.keys(), fontsize=6, markerscale=4)
    [line.set_linewidth(2) for line in legend.get_lines()]


def get_runs_for_ids_and_metric(adam_ids, metric):
    runs_for_ids_adam = {}
    for run_id in adam_ids:
        runs_for_ids_adam[run_id] = h.get_run(run_id, data_type=metric)
    return runs_for_ids_adam


def special_max_start_loss(max_start_loss, metric):
    if metric == h.TRAIN_ACC or metric == h.F_ONE or metric == h.EXACT_MATCH:
        max_start_loss = 0
    return max_start_loss


def save_figure(args, datasets, models, plt):
    if args.big_batch:
        file_basename = make_file_basename(args, datasets[0], models[0])
    else:
        file_basename = make_file_basename(args, None, None)

    output_dir = "output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, file_basename + ".pdf")

    print(f"Saving [{path}]")
    plt.savefig(path)
    plt.close()


def save_data(args, model, dataset, dfs, batchsize=None):
    file_basename = make_file_basename(args, dataset, model)

    file_basename = f"{file_basename}_{dataset}_{model}"
    if batchsize is not None:
        file_basename += f"_{batchsize}"

    output_dir = "output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for key, df in dfs.items():
        df.to_csv(f"{output_dir}/{file_basename}_{key}.csv")

    summary_file = f"{output_dir}/{file_basename}_summary.txt"

    def sci_not_abbr(x):
        if math.isinf(x):
            return "inf"
        return f"10e{math.ceil(math.log10(x))}"

    with open(summary_file, "w") as fh:
        for key, df in dfs.items():
            fh.write(f"{key}\n")
            fh.write("\n")
            try:
                for ss in sorted(list(df[h.K_SS].unique())):
                    subset_ss = df[df[h.K_SS] == ss]
                    fh.write(f"    {ss}: ")
                    for (seed, loss, epoch) in zip(
                        list(subset_ss[h.SEED]),
                        list(subset_ss[h.TRAIN_LOSS]),
                        list(subset_ss[h.EPOCH]),
                    ):
                        fh.write(f"({seed},{epoch},{sci_not_abbr(loss)}) ")
                    fh.write("\n")
            except Exception as e:
                print(e)
                pdb.set_trace()
            fh.write("\n")


def make_file_basename(args, dataset=None, model=None):
    prefix = ""
    if args.full_batch:
        prefix = "full_batch_"
    if args.big_batch:
        prefix = "big_batch_{}_{}_".format(dataset, model)
    postfix = "_mom" if args.momentum else ""
    return f"{prefix}{args.plot_type}_{args.metric}{postfix}"


def latex_sci_notation(x: float):
    scinot_str = "{:.1e}".format(x)
    num, exp = scinot_str.split("e")
    return "$" + num + r"\cdot" + "10^{" + exp + "}" + "$"


def select_best_stepsize(hyperparam, run_for_ids, step_sizes, summary_data):
    step_to_use = None
    curr_loss = math.inf
    for step_size in step_sizes:
        df_row = summary_data.loc[summary_data[hyperparam] == step_size]
        ids = list(df_row[h.K_ID])
        vals = []
        for run_id in ids:
            run_metric = run_for_ids[run_id]
            if len(run_metric) > 0:
                vals.append(run_metric[h.TRAIN_LOSS].iloc[-1])
        if hyperparam == h.OPT_C and step_size < 0.1:
            continue
        max_loss = np.array(vals).max()
        if max_loss < curr_loss:
            curr_loss = max_loss
            step_to_use = step_size
    return step_to_use


def title_format(model, dataset):
    return f"{name_to_label[model]}, {name_to_label[dataset]}"


def format_problem_and_metric(dataset, metric, model, short=True):
    if short:
        return "$\\bf {}$ \n {}".format(name_to_label[dataset], metric_to_text[metric])
    else:
        return "$\\bf {} - {}$ \n {}".format(
            name_to_label[model], name_to_label[dataset], metric_to_text[metric]
        )


def iter_limit_and_per_epoch(
    acc_step, batch_size, big_batch, dataset, iteration_limit, model
):
    num_iterations_per_epoch = 0
    if big_batch:
        key = dataset
        if model == "transformer_encoder":
            key = dataset + "_" + model
        num_iterations_per_epoch = dataset_sizes[key] // abs(batch_size * acc_step)
        # scale iteration limit based on batch size
        iteration_limit = math.ceil(iteration_limit / num_iterations_per_epoch)
        print(num_iterations_per_epoch)
        print(iteration_limit)
    return iteration_limit, num_iterations_per_epoch


def get_runs_helper(ids, data, dataset, iteration_limit, metric):
    """
    Workaround to use runs that log TRAIN_LOSS and AVERAGE_LOSS.

    TODO: Why do we need this and what is this doing
    """
    output = {}
    for run_id in ids:
        data_point = data[data[h.K_ID] == run_id]
        if not pd.isnull(data_point[h.AVERAGE_LOSS].item()):
            if metric == h.TRAIN_LOSS:
                run_train_loss = h.get_run(str(run_id), data_type=h.TRAIN_LOSS)
                run_average_loss = h.get_run(str(run_id), data_type=h.AVERAGE_LOSS)
                run = run_average_loss.copy() - [0, 1, 0]
                begin = run_train_loss.loc[run_train_loss["_step"] == 3]
                run.rename(columns={h.AVERAGE_LOSS: h.TRAIN_LOSS}, inplace=True)
                run = pd.concat([begin, run])
            else:
                run = h.get_run(str(run_id), data_type=metric)
                if dataset != "squad":
                    run = run.iloc[1:, :] - [1, 1, 0]
                else:
                    begin = run.loc[run["_step"] == 3]
                    run = pd.concat([begin, run.iloc[1:, :] - [0, 1, 0]])
        else:
            run = h.get_run(str(run_id), data_type=metric)

        if iteration_limit > 0:
            run = run[run.index <= iteration_limit]
        output[run_id] = run
    return output


def get_value_at_end_of_run_for_ids(ids, metric, iteration_limit=0):
    """Returns a list of the loss at the iteration_limit of each run in ids
    (if set, end otherwise)"""
    vals = []
    for run_id in ids:
        run = h.get_run(run_id, data_type=metric)
        if len(run) == 0:
            vals.append(math.inf)
            break
        if iteration_limit > 0:
            run = run[run.index <= iteration_limit]
        vals.append(run[metric].iloc[-1])
    vals = np.array(vals)
    return vals


def tag_yaxis(ax, increment=1):
    """
    Tags in logscale with increments. For .5, will give
        10**0, 10**.5, 10**1, ...
    for increment = 1.0,
        10**0, 10**1, 10**2, ...
    """
    if type(ax.yaxis._scale) != mpl.scale.LogScale:
        return

    ymin, ymax = ax.get_ylim()
    if ymax < 10 and ymax > 9:
        ymax = 11
    first_tick = np.ceil(np.log10(ymin) / increment) * increment
    last_tick = np.floor(np.log10(ymax) / increment) * increment

    def half_step_range(start, stop):
        r = start
        while r <= stop:
            yield r
            r += increment

    y = list(half_step_range(first_tick, last_tick))
    y = np.array(y)
    y = np.power(10, y)
    x = []
    if len(y) >= 5:
        for i in range(0, len(y), 2):
            # breakpoint()
            x.append(y[i].item())
    else:
        x = y
    # breakpoint()
    ax.set_yticks(x)


FIGSIZE_4x1 = (7, 1.2)
FIGSIZE_4x1_short = (7, 1)
FIGSIZE_3x2 = (7, 3.0)
FIGSIZE_3x2_short = (7, 2.8)
