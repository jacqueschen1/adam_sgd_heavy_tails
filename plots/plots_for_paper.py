from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import math
import data_helpers as h
import argparse
import fplt
from data_helpers import name_to_label


# Plotting constants and magic strings
CS = ["#0077bb", "#cc3311", "#33bbee", "#ee3377", "#aa3377", "#ee7733", "#ddaa33"]
LIGHT_CS = ["#6699cc", "#fa6450", "#88ccee", "#ee99aa", "#efdfff", "#ee8866", "#99ddff"]
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
    xaxis_labels=False,
    yaxis_labels=False,
    full_batch=False,
    big_batch=False,
    iteration_limit=-1,
):
    """Plots the metrics for best runs for each optimizer."""
    print(model, dataset, batch_size, metric)
    df = h.get_data()
    df = df[(df[h.EPOCH] == df[h.MAX_EPOCH] - 1)]

    if dataset == "squad":
        df = df[(df[h.EPOCH] > 1)]
    else:
        df = df[(df[h.MAX_EPOCH] > 10)]

    # These timestamp overrides ensures that's it's getting the proper runs for full batch that had
    # the random seed change what subset of the dataset it trains on
    if full_batch and (model == "transformer_encoder" or dataset == "ptb"):
        timestamp = 1629129016
    if dataset == "squad":
        if full_batch:
            timestamp = 1629340576

    if timestamp != 0:
        df = df[df[h.TIMESTAMP] > timestamp]
    if batch_size > 0:
        df = df[df[h.BATCH_SIZE] == batch_size]
    if acc_step > 0:
        df = df[df[h.ACC_STEP] == acc_step]

    SELECT_PATTERN = (df[h.DATASET] == dataset) & (df[h.MODEL] == model)
    if full_batch:
        SELECT_PATTERN = (SELECT_PATTERN) & (df[h.FULL_BATCH])
    else:
        SELECT_PATTERN = (
            (SELECT_PATTERN)
            & (df[h.DROP_LAST].isnull())
            & ((df[h.FULL_BATCH] == False) | (df[h.FULL_BATCH].isnull()))
        )

    sgd_data = df[
        SELECT_PATTERN
        & (df[h.OPT_NAME] == "SGD")
        & ((df[h.OPT_MOMENTUM].isnull()) | (df[h.OPT_MOMENTUM] == 0))
    ]
    adam_data = df[SELECT_PATTERN & (df[h.OPT_NAME] == "Adam") & (df[h.OPT_B1] == 0.9)]

    # if not full batch, make sure it's getting the bert runs with bugs fixed
    if dataset == "squad":
        SELECT_PATTERN = SELECT_PATTERN & (df[h.TIMESTAMP] > 1628662010)
    adam_no_momentum_data = df[
        SELECT_PATTERN & (df[h.OPT_NAME] == "Adam") & (df[h.OPT_B1] == 0)
    ]
    sgd_momentum_data = df[
        SELECT_PATTERN & (df[h.OPT_NAME] == "SGD") & (df[h.OPT_MOMENTUM] == 0.9)
    ]
    correction_term = 1

    if metric == h.TRAIN_ACC:
        # display as percentage
        correction_term *= 100

    if full_batch and dataset == "squad":
        iteration_limit = 80

    plot_best_run(
        axes,
        "NM-Adam",
        adam_no_momentum_data,
        metric,
        3,
        correction_term=correction_term,
        diff_style=True,
    )
    plot_best_run(
        axes,
        "Adam",
        adam_data,
        metric,
        1,
        iteration_limit=iteration_limit,
        correction_term=correction_term,
    )
    plot_best_run(
        axes, "M-SGD", sgd_momentum_data, metric, 2, correction_term=correction_term
    )

    plot_best_run(
        axes,
        "SGD",
        sgd_data,
        metric,
        0,
        iteration_limit=iteration_limit,
        correction_term=correction_term,
        diff_style=True,
    )

    if big_batch:
        axes.set_xscale("log")
        axes.minorticks_off()

    if metric != h.TRAIN_ACC and metric != h.F_ONE:
        axes.set_yscale("log")
        axes.minorticks_off()
    if yaxis_labels:
        if metric == h.TRAIN_PPL:
            axes.set_ylabel("Training PPL", labelpad=0.2)
        elif metric == h.TRAIN_LOSS:
            axes.set_ylabel("Training Loss", labelpad=0.2)
        elif metric == h.F_ONE:
            axes.set_ylabel("Training F1", labelpad=-1)
        else:
            axes.set_ylabel("Training Accuracy", labelpad=0.2)

    if xaxis_labels:
        axes.set_xlabel("Epoch") if not big_batch else axes.set_xlabel("Num Iterations")

    if not xaxis_labels and not yaxis_labels and full_batch:
        axes.tick_params(labelleft=False)

    if dataset == "squad" and metric == h.F_ONE:
        axes.set_ylabel("Training F1", labelpad=-1)

    if metric == h.TRAIN_ACC and full_batch:
        axes.set_ylim(0, 100)

    if big_batch:
        axes.set_title(
            "{}, {} {}".format(
                name_to_label[model],
                name_to_label[dataset],
                abs(batch_size * acc_step),
            ),
            pad=2,
        )
    else:
        axes.set_title(
            "{}, {}".format(
                name_to_label[model],
                name_to_label[dataset],
            ),
            pad=2,
        )

    if legend:
        handles, labels = axes.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend = axes.legend(
            by_label.values(), by_label.keys(), fontsize=7, markerscale=8, ncol=2
        )
        [line.set_linewidth(2) for line in legend.get_lines()]


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
    override=None,
    diff_style=False,
):
    """Plots best run given a subset of the data"""
    ids = list(data[h.K_ID])
    runs = {}
    for id in tqdm(ids):
        run = h.get_run(str(id), data_type=metric)
        if iteration_limit > 0:
            run = run[run.index <= iteration_limit]
        runs[id] = run

    if len(ids) > 0:
        hyperparam, run_data, time_steps, _ = gen_best_runs_for_metric(
            runs, data, metric, hyperparam=hyperparam, override=override
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
        i = 3 if metric == h.TRAIN_LOSS else 4
        if iteration_limit != -1:
            i -= 1
        time_steps = [time_step - i for time_step in time_steps]

        if num_iterations_per_epoch > 0:
            for i in range(len(time_steps)):
                if time_steps[i] != 1:
                    time_steps[i] = num_iterations_per_epoch * time_steps[i]

        axes.plot(
            time_steps,
            mean_loss,
            "--" if diff_style else "-",
            color=CS[idx],
            label=label,  # + " " + str(hyperparam),
            markersize=1,
        )
        axes.fill_between(time_steps, min_loss, max_loss, color=LIGHT_CS[idx])


def gen_best_runs_for_metric(
    run_for_ids, summary_data, metric, hyperparam=h.K_SS, override=None
):
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
        mean_loss = df_row[metric].mean()
        if mean_loss < curr_loss and not bigger_better:
            curr_loss = mean_loss
            step_to_use = step_size
        if mean_loss > curr_loss and bigger_better:
            curr_loss = mean_loss
            step_to_use = step_size

    curr_id = []

    if override is not None:
        step_to_use = override

    for seed in range(5):
        df_row = summary_data.loc[
            (summary_data[h.SEED] == seed) & (summary_data[hyperparam] == step_to_use)
        ]
        if df_row[h.K_ID].size != 0:
            for i in range(df_row[h.K_ID].size):
                id = df_row[h.K_ID].iloc[i]
                if len(run_for_ids[id]) > 3:
                    curr_id.append(id)
                    continue
    print(step_to_use)
    print(curr_id)
    runs_list = [run_for_ids[id] for id in curr_id]
    run_data = pd.concat(runs_list)
    run_data[metric].fillna("", inplace=True)
    run_data = run_data[(run_data[metric] != "")]

    time_steps = sorted(list(run_data["_step"].unique()))

    datas = run_data[run_data["_step"] == time_steps[0]]
    start_metric_value = datas[metric].mean()

    return step_to_use, run_data, time_steps, start_metric_value


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
    xaxis_labels=False,
    yaxis_labels=False,
    full_batch=False,
    big_batch=False,
    momentum=False,
):
    """Generates data metrics vs. step size plots."""
    print(model, dataset, metric, batch_size)
    df = h.get_data()
    if dataset == "squad":
        if not big_batch and not full_batch:
            df = df[(df[h.MAX_EPOCH] == 4)]
        else:
            df = df[(df[h.MAX_EPOCH] > 4)]
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

    SELECT_PATTERN = (df[h.DATASET] == dataset) & (df[h.MODEL] == model)
    if full_batch:
        SELECT_PATTERN = (SELECT_PATTERN) & (df[h.FULL_BATCH])
    else:
        SELECT_PATTERN = (
            (SELECT_PATTERN)
            & (df[h.DROP_LAST].isnull())
            & ((df[h.FULL_BATCH] == False) | (df[h.FULL_BATCH].isnull()))
        )

    sgd_data = df[
        SELECT_PATTERN
        & (df[h.OPT_NAME] == "SGD")
        & ((df[h.OPT_MOMENTUM].isnull()) | (df[h.OPT_MOMENTUM] == 0))
    ]
    adam_data = df[SELECT_PATTERN & (df[h.OPT_NAME] == "Adam") & (df[h.OPT_B1] == 0.9)]
    if dataset == "squad":
        SELECT_PATTERN = SELECT_PATTERN & (df[h.TIMESTAMP] > 1628662010)
    adam_no_momentum_data = df[
        SELECT_PATTERN & (df[h.OPT_NAME] == "Adam") & (df[h.OPT_B1] == 0)
    ]
    sgd_momentum_data = df[
        SELECT_PATTERN & (df[h.OPT_NAME] == "SGD") & (df[h.OPT_MOMENTUM] == 0.9)
    ]
    adam_ids = list(adam_data[h.K_ID])

    runs_for_ids_adam = {}

    for id in tqdm(adam_ids):
        runs_for_ids_adam[id] = h.get_run(id, data_type=metric)

    # Get the metric value for the start of the run
    _, _, _, start_metric_value = gen_best_runs_for_metric(
        runs_for_ids_adam, adam_data, metric
    )

    correction_term = 1

    if metric == h.TRAIN_ACC:
        # display as percentage
        correction_term *= 100

    if metric == h.TRAIN_ACC or metric == h.F_ONE or metric == h.EXACT_MATCH:
        if dataset == "mnist" or dataset == "cifar10":
            start_metric_value = 0.1
        if dataset == "cifar100":
            start_metric_value = 0.01
        if dataset == "squad":
            start_metric_value = 1
    axes.axhline(
        y=start_metric_value * correction_term,
        color=CS[6],
        linestyle="-",
        label="Initial Value",
    )

    if momentum:
        plot_ss_vs_metric(
            adam_no_momentum_data,
            axes,
            "NM-Adam",
            metric,
            3,
            unique_ss,
            start_metric_value,
            model,
            correction_term=correction_term,
        )
        plot_ss_vs_metric(
            sgd_momentum_data,
            axes,
            "M-SGD",
            metric,
            2,
            unique_ss,
            start_metric_value,
            model,
            correction_term=correction_term,
        )
    else:
        plot_ss_vs_metric(
            sgd_data,
            axes,
            "SGD",
            metric,
            0,
            unique_ss,
            start_metric_value,
            model,
            correction_term=correction_term,
        )
        plot_ss_vs_metric(
            adam_data,
            axes,
            "Adam",
            metric,
            1,
            unique_ss,
            start_metric_value,
            model,
            correction_term=correction_term,
        )

    axes.set_title("{}, {}".format(model.capitalize(), dataset.capitalize()), pad=5)
    axes.set_xscale("log")
    if metric == h.TRAIN_PPL or metric == h.TRAIN_LOSS:
        axes.set_yscale("log")
    axes.set_xlim(min_ss, max_ss)
    axes.minorticks_off()
    if yaxis_labels:
        if metric == h.TRAIN_PPL:
            axes.set_ylabel("Training PPL", labelpad=0.5)
        elif metric == h.TRAIN_LOSS:
            axes.set_ylabel("Training Loss", labelpad=0.5)
        elif metric == h.F_ONE:
            axes.set_ylabel("Training F1", labelpad=0.5)
        else:
            axes.set_ylabel("Training Accuracy", labelpad=0.5)

    if xaxis_labels:
        axes.set_xlabel("Epoch") if not big_batch else axes.set_xlabel("Num Iterations")
    if dataset == "squad" and metric == h.F_ONE:
        axes.set_ylabel("Training F1", labelpad=-1)
    if metric != h.TRAIN_PPL:
        axes.yaxis.set_major_locator(plt.MaxNLocator(5))

    if metric == h.TRAIN_ACC:
        axes.set_ylim(0, 105)
        if not yaxis_labels and not xaxis_labels:
            axes.tick_params(labelleft=False)

    if legend:
        handles, labels = axes.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend = axes.legend(
            by_label.values(), by_label.keys(), fontsize=6, markerscale=4
        )
        [line.set_linewidth(2) for line in legend.get_lines()]


def plot_ss_vs_metric(
    data,
    axes,
    label,
    metric,
    idx,
    unique_ss,
    start_metric_value,
    model,
    correction_term=1,
):
    """Plots metric vs step size for the given data."""
    steps, mean, min_val, max_val, diverged = [], [], [], [], []
    bigger_better = (
        metric == h.TRAIN_ACC or metric == h.F_ONE or metric == h.EXACT_MATCH
    )
    for ss in sorted(unique_ss):
        non_diverge = data[(data[h.K_SS] == ss)]
        if len(non_diverge) != 0:
            if (
                model == "transformer_encoder"
                and label == "SGD"
                and ss == 10 ** -4
                and metric == h.TRAIN_LOSS
            ):
                # Remove outlier from crash
                non_diverge = data[(data[h.K_SS] == ss) & (data[metric] < 9)]

            # if non_diverge
            non_diverge = non_diverge[
                non_diverge[h.EPOCH] >= non_diverge[h.MAX_EPOCH].mean() - 1
            ]

            mean_loss = non_diverge[metric].mean()
            max_loss = non_diverge[metric].max() - mean_loss
            if math.isnan(mean_loss):
                mean_loss = math.inf
            min_loss = mean_loss - non_diverge[metric].min()

            if (not bigger_better and (mean_loss >= start_metric_value - 0.01)) or (
                bigger_better and (mean_loss <= start_metric_value + 0.01)
            ):
                # Plot diverged points
                axes.scatter(
                    ss,
                    start_metric_value * correction_term,
                    facecolors="none",
                    linewidth=1,
                    edgecolors=CS[idx],
                    # label=label + " Diverged",
                    s=2,
                )
                diverged.append(ss)
            else:
                steps.append(ss)
                mean.append(mean_loss * correction_term)
                min_val.append(min_loss * correction_term)
                max_val.append(max_loss * correction_term)

    axes.errorbar(
        steps,
        mean,
        yerr=[min_val, max_val],
        fmt="o-",
        color=CS[idx],
        label=label,
        markersize=1,
    )

    # Connect diverged points with other points
    if len(steps) != 0:
        len_diverged = len(diverged)
        for i in range(len_diverged):
            curr = diverged[i]
            if curr < steps[0] and (
                (i < len_diverged - 1 and diverged[i + 1] > steps[0])
                or i == len_diverged - 1
            ):
                to_plot_x = [curr, steps[0]]
                to_plot_y = [start_metric_value * correction_term, mean[0]]
                axes.plot(to_plot_x, to_plot_y, "--", color=CS[idx])
            if curr > steps[-1]:
                to_plot_x = [curr, steps[-1]]
                to_plot_y = [start_metric_value * correction_term, mean[-1]]
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
    parser.add_argument(
        "--momentum",
        action="store_true",
        help="plot momentum alternative optimizers for vs_ss plots",
    )

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

    assert len(datasets) == len(models)

    length = len(datasets)
    plt.rcParams.update({"font.size": 8})
    plt.rcParams["lines.linewidth"] = 1
    plt.rc("ytick", labelsize=7)
    plt.rc("xtick", labelsize=7)
    fig, axes = plt.subplots(
        int(math.ceil(length / 3)), 3, figsize=(7, 9.5 / (60 / 20))
    )

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
                axes[y, x],
                timestamp=args.timestamp,
                batch_size=batch_sizes[i],
                acc_step=acc_steps[i],
                metric=metric,
                legend=y == 0 and x == 0,
                xaxis_labels=y == 1,
                yaxis_labels=x == 0,
                full_batch=args.full_batch,
                big_batch=args.big_batch,
            )

        if args.plot_type == METRIC_VS_SS:
            gen_step_size_vs_metric_plot(
                models[i],
                datasets[i],
                fig,
                axes[y, x],
                timestamp=args.timestamp,
                batch_size=batch_sizes[i],
                acc_step=acc_steps[i],
                metric=metric,
                legend=y == 0 and x == 0,
                xaxis_labels=y == 1,
                yaxis_labels=x == 0,
                full_batch=args.full_batch,
                big_batch=args.big_batch,
                momentum=args.momentum,
            )

    if args.full_batch:
        ylim = axes[1, 1].get_ylim()
        axes[1, 1].set_ylim(ylim[0], min(ylim[1], 10 ** 7))

    fplt.hide_frame(*axes[0])
    fplt.hide_frame(*axes[1])
    if args.plot_type == METRIC_VS_SS:
        fig.tight_layout()
    else:
        fig.tight_layout(h_pad=1.5, w_pad=-0.5)
    print("save")

    prefix = ""
    if args.full_batch:
        prefix = "full_batch_"
    if args.big_batch:
        prefix = "new_log_big_batch_{}_{}_".format(models[0], datasets[0])
    plt.savefig(
        "{}plot_paper_{}_{}{}.pdf".format(
            prefix, args.plot_type, args.metric, "_momen" if args.momentum else ""
        ),  # bbox_inches="tight", pad_inches=0.5
    )

    plt.close()
