"""
Used to generate the plots for the increasing batch size experiments
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt
import math
import data_helpers as h
import fplt
import data_selection
import plotting_common as plth
from plotting_common import (
    BEST_PLOT,
    METRIC_VS_SS,
    dataset_sizes,
    LABEL_ADAM,
    LABEL_SGD,
    LABEL_TO_STYLE,
    get_value_at_end_of_run_for_ids,
    tag_yaxis,
)


# %%
# Plotting constants and magic strings


def gen_best_run_plot(
    args,
    model,
    dataset,
    ax,
    batch_size=-1,
    acc_step=-1,
    metric=h.TRAIN_LOSS,
    yaxis_labels=False,
    legend=False,
    iteration_limit=-1,
):
    big_batch = True
    full_batch = False
    (adam_data, sgd_data, adam_nm_data, sgd_m_data,) = data_selection.select_runs(
        acc_step, batch_size, big_batch, dataset, full_batch, metric, model
    )

    iteration_limit, num_iterations_per_epoch = plth.iter_limit_and_per_epoch(
        acc_step, batch_size, big_batch, dataset, iteration_limit, model
    )

    plot_best_run(
        ax,
        LABEL_SGD,
        sgd_data,
        metric,
        iteration_limit=iteration_limit,
        num_iterations_per_epoch=num_iterations_per_epoch,
    )
    plot_best_run(
        ax,
        LABEL_ADAM,
        adam_data,
        metric,
        iteration_limit=iteration_limit,
        num_iterations_per_epoch=num_iterations_per_epoch,
    )

    if metric in [h.TRAIN_ACC, h.F_ONE]:
        ax.set_ylim([50, 102.5])
    else:
        ax.set_yscale("log")
    if yaxis_labels:
        ax.set_ylabel(plth.format_problem_and_metric(dataset, metric, model))

    ax.set_title(str(abs(batch_size * acc_step)), pad=2)

    if not yaxis_labels:
        ax.tick_params(labelleft=False, which="both")

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        loc = "lower right"
        if model == "transformer_xl" or model == "transformer_encoder":
            loc = "upper right"
        if metric == h.TRAIN_LOSS:
            if dataset != "mnist":
                loc = "upper right"
            else:
                loc = "lower right"

        legend = ax.legend(
            by_label.values(), by_label.keys(), fontsize=6, markerscale=6, loc=loc
        )
        [line.set_linewidth(1) for line in legend.get_lines()]

    plth.save_data(
        args,
        model,
        dataset,
        {
            "adam": adam_data,
            "sgd": sgd_data,
            "adam_nm": adam_nm_data,
            "sgd_m": sgd_m_data,
        },
        batch_size,
    )


def plot_best_run(
    ax,
    label,
    data,
    metric,
    hyperparam=h.K_SS,
    iteration_limit=-1,
    num_iterations_per_epoch=-1,
):
    ids = list(data[h.K_ID])
    color = LABEL_TO_STYLE[label]["color"]

    runs_metric = None
    runs_training_loss = {}
    for run_id in ids:
        run = h.get_run(run_id, data_type=h.TRAIN_LOSS)
        if iteration_limit > 0:
            run = run[run.index <= iteration_limit]
        runs_training_loss[run_id] = run

    if metric != h.TRAIN_LOSS:
        runs_metric = {}
        for run_id in ids:
            run = h.get_run(run_id, data_type=metric)
            begin = run.loc[run["_step"] == 3]
            run = pandas.concat([begin, run.iloc[1:, :] - [0, 1, 0]])
            if iteration_limit > 0:
                run = run[run.index <= iteration_limit]
            runs_metric[run_id] = run

    if len(ids) > 0:
        hyperparam, run_data, time_steps, _ = gen_best_runs_for_metric(
            runs_training_loss,
            data,
            metric,
            actual_metric_data=runs_metric,
            hyperparam=hyperparam,
        )
        max_loss, mean_loss, min_loss = plth.get_data_summary_for_metric(
            metric, run_data, time_steps
        )
        i = 2
        time_steps = [time_step - i for time_step in time_steps]

        if num_iterations_per_epoch > 0:
            for i in range(len(time_steps)):
                if time_steps[i] != 1:
                    time_steps[i] = num_iterations_per_epoch * time_steps[i]

        ax.plot(
            time_steps,
            mean_loss,
            linestyle=plth.LABEL_TO_STYLE[label]["linestyle"],
            color=color,
            label=label + " " + plth.latex_sci_notation(hyperparam),
            markersize=1,
        )
        ax.fill_between(time_steps, min_loss, max_loss, color=color, alpha=0.2)


def gen_best_runs_for_metric(
    run_for_ids, summary_data, metric, hyperparam=h.K_SS, actual_metric_data=None
):

    step_sizes = list(summary_data[hyperparam].unique())

    step_to_use = plth.select_best_stepsize(
        hyperparam, run_for_ids, step_sizes, summary_data
    )

    curr_id = []

    for seed in range(5):
        df_row = summary_data.loc[
            (summary_data[h.SEED] == seed) & (summary_data[hyperparam] == step_to_use)
        ]
        if df_row[h.K_ID].size != 0:
            for i in range(df_row[h.K_ID].size):
                run_id = df_row[h.K_ID].iloc[i]
                if len(run_for_ids[run_id]) > 3:
                    curr_id.append(run_id)

    data_to_use = actual_metric_data if actual_metric_data is not None else run_for_ids
    return plth.prepare_best_runs(curr_id, metric, data_to_use, step_to_use)


def gen_step_size_vs_metric_plot(
    args,
    model,
    dataset,
    ax,
    batch_size=-1,
    acc_step=-1,
    metric=h.TRAIN_LOSS,
    legend=False,
    yaxis_labels=False,
    iteration_limit=-1,
):
    big_batch = True
    full_batch = False
    (adam_data, sgd_data, adam_nm_data, sgd_m_data,) = data_selection.select_runs(
        acc_step, batch_size, big_batch, dataset, full_batch, metric, model
    )
    max_ss, min_ss, unique_ss = data_selection.find_step_size_range(
        acc_step, batch_size, big_batch, dataset, full_batch, metric, model
    )

    adam_ids = list(adam_data[h.K_ID])

    max_start_loss = None
    if not (metric == h.TRAIN_ACC or metric == h.F_ONE or metric == h.EXACT_MATCH):
        runs_for_ids_adam = plth.get_runs_for_ids_and_metric(adam_ids, metric)
        runs_for_ids_adam_training_loss = plth.get_runs_for_ids_and_metric(
            adam_ids, h.TRAIN_LOSS
        )
        _, _, _, max_start_loss = gen_best_runs_for_metric(
            runs_for_ids_adam_training_loss,
            adam_data,
            metric,
            actual_metric_data=runs_for_ids_adam,
        )

    iteration_limit, _ = plth.iter_limit_and_per_epoch(
        acc_step, batch_size, big_batch, dataset, iteration_limit, model
    )

    if max_start_loss:
        ax.axhline(
            y=max_start_loss,
            color=plth.COLOR_INITIAL_VALUE,
            linestyle="-",
            label=plth.LABEL_INITIAL_VALUE,
        )
    else:
        max_start_loss = 0

    plot_ss_vs_metric(
        sgd_data,
        ax,
        LABEL_SGD,
        metric,
        unique_ss,
        max_start_loss,
        iteration_limit=iteration_limit,
    )
    plot_ss_vs_metric(
        adam_data,
        ax,
        LABEL_ADAM,
        metric,
        unique_ss,
        max_start_loss,
        iteration_limit=iteration_limit,
    )

    ax.set_title(str(abs(batch_size * acc_step)), pad=2)

    ax.set_xscale("log")
    if metric not in [h.TRAIN_ACC, h.F_ONE]:
        ax.set_yscale("log")
        if metric == h.TRAIN_LOSS:
            tag_yaxis(ax, increment=1)
    else:
        ax.set_ylim(0, 105)
    ax.set_xlim(min_ss, max_ss)

    if yaxis_labels:
        ax.set_ylabel(plth.format_problem_and_metric(dataset, metric, model))
    else:
        ax.set_yticklabels([], minor=False)

    ax.set_yticklabels([], minor=True)
    ax.set_yticks([], minor=True)

    # if metric != h.TRAIN_PPL:
    #     ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    if legend:
        plth.make_legend(ax)

    plth.save_data(
        args,
        model,
        dataset,
        {
            "adam": adam_data,
            "sgd": sgd_data,
            "adam_nm": adam_nm_data,
            "sgd_m": sgd_m_data,
        },
        batch_size,
    )


def plot_ss_vs_metric(
    data, ax, label, metric, unique_ss, max_start_loss, iteration_limit=-1
):
    color = LABEL_TO_STYLE[label]["color"]
    steps, mean, min_val, max_val, diverged = [], [], [], [], []
    bigger_better = (
        metric == h.TRAIN_ACC or metric == h.F_ONE or metric == h.EXACT_MATCH
    )
    for ss in sorted(unique_ss):
        non_diverge = data[(data[h.K_SS] == ss)]
        if len(non_diverge) != 0:

            ids = list(non_diverge[h.K_ID])

            vals = get_value_at_end_of_run_for_ids(ids, metric, iteration_limit)

            if np.isnan(vals).any():
                mean_loss = math.inf
            else:
                mean_loss = vals.mean()

            if (not bigger_better and (mean_loss >= max_start_loss - 0.01)) or (
                bigger_better and (mean_loss <= max_start_loss + 0.01)
            ):
                ax.scatter(
                    ss,
                    max_start_loss,
                    facecolors="none",
                    linewidth=1,
                    edgecolors=color,
                    s=2,
                )

                diverged.append(ss)
            else:
                max_loss = vals.max()
                if not bigger_better:
                    max_loss = min(max_loss, max_start_loss)
                max_loss = max_loss - mean_loss
                min_loss = mean_loss - vals.min()
                steps.append(ss)
                mean.append(mean_loss)
                min_val.append(min_loss)
                max_val.append(max_loss)

    plth.common_ss_vs_metric_errorbar_and_points(
        ax, diverged, label, max_start_loss, max_val, mean, min_val, steps
    )


def main(args):

    assert (not args.full_batch) and args.big_batch

    datasets, models, batch_sizes, acc_steps, length = plth.process_args(args)

    plth.init_plt_style(plt)

    figsize = (
        plth.FIGSIZE_4x1_short if args.plot_type == BEST_PLOT else plth.FIGSIZE_4x1
    )
    fig, axes = plt.subplots(1, length, figsize=figsize)

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
        metric = plth.select_metric(args.metric, datasets[i])

        if args.plot_type == BEST_PLOT:
            gen_best_run_plot(
                args,
                models[i],
                datasets[i],
                axes[i],
                batch_size=batch_sizes[i],
                acc_step=acc_steps[i],
                metric=metric,
                yaxis_labels=i == 0,
                iteration_limit=iteration_limit,
            )

        if args.plot_type == METRIC_VS_SS:
            gen_step_size_vs_metric_plot(
                args,
                models[i],
                datasets[i],
                axes[i],
                batch_size=batch_sizes[i],
                acc_step=acc_steps[i],
                metric=metric,
                legend=False,
                yaxis_labels=i == 0,
                iteration_limit=iteration_limit,
            )

    if args.plot_type == BEST_PLOT and args.big_batch:

        x_lim_max = axes[0].get_xlim()[1]

        if models[0] == "bert_base_pretrained":
            x_lim_max = 450

        y_lim = [math.inf, -math.inf]
        for i in range(length):
            y_lim = [
                min(y_lim[0], axes[i].get_ylim()[0]),
                max(y_lim[1], axes[i].get_ylim()[1]),
            ]
        for i in range(length):
            curr = axes[i].get_xlim()
            axes[i].set_xlim(curr[0], x_lim_max)
            axes[i].set_ylim(y_lim[0], min(y_lim[1], 10 ** 7))

    fplt.normalize_y_axis(*axes)

    fplt.hide_frame(*axes)

    fig.tight_layout(pad=0.5)
    plth.save_figure(args, datasets, models, plt)


if __name__ == "__main__":
    main(plth.cli().parse_args())
