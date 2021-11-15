"""
Used to generate the plots for the plots for the standard and large batch experiments
"""

import matplotlib.pyplot as plt
import math

import pandas
import numpy as np
import data_helpers as h
import fplt
import data_selection
import plotting_common as plth
from plotting_common import (
    BEST_PLOT,
    METRIC_VS_SS,
    LABEL_ADAM_NM,
    LABEL_ADAM,
    LABEL_SGD_M,
    LABEL_SGD,
    LABEL_TO_STYLE,
    get_runs_helper,
    tag_yaxis,
)


# %%
# Plotting constants and magic strings
MAXIMUM_Y_VALUE = 10 ** 7


def gen_best_run_plot(
    args,
    model,
    dataset,
    ax,
    batch_size=-1,
    acc_step=-1,
    metric=h.TRAIN_LOSS,
    legend=False,
    xaxis_labels=False,
    yaxis_labels=False,
    full_batch=False,
):
    big_batch = False
    (adam_data, sgd_data, adam_nm_data, sgd_m_data,) = data_selection.select_runs(
        acc_step, batch_size, big_batch, dataset, full_batch, metric, model
    )

    plot_best_run(ax, LABEL_ADAM_NM, adam_nm_data, metric, dataset)
    plot_best_run(ax, LABEL_ADAM, adam_data, metric, dataset)
    plot_best_run(ax, LABEL_SGD_M, sgd_m_data, metric, dataset)
    plot_best_run(ax, LABEL_SGD, sgd_data, metric, dataset)

    if metric not in [h.TRAIN_ACC, h.F_ONE]:
        ax.set_yscale("log")
        ax.minorticks_off()
        # if metric == h.TRAIN_LOSS:
        # tag_yaxis(ax, increment=1)
    if yaxis_labels:
        ax.set_ylabel(plth.metric_to_text[metric])

    if xaxis_labels:
        ax.set_xlabel(plth.LABEL_ITER if full_batch else plth.LABEL_EPOCH)

    if dataset == "squad" and metric == h.F_ONE:
        ax.set_ylabel(plth.metric_to_text[metric], labelpad=-1)

    if full_batch:
        if metric in [h.TRAIN_ACC, h.F_ONE]:
            ax.set_ylim(0, 105)
    else:
        if metric in [h.TRAIN_ACC, h.F_ONE]:
            ax.set_ylim(50, 102.5)

    ax.set_title(plth.title_format(model, dataset), pad=2)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend = ax.legend(
            by_label.values(),
            by_label.keys(),
            fontsize=7,
            markerscale=8,
            ncol=2,
            frameon=False,
        )
        [line.set_linewidth(2) for line in legend.get_lines()]

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
    )


def plot_best_run(
    ax,
    label,
    data,
    metric,
    dataset,
    hyperparam=h.K_SS,
    iteration_limit=-1,
    num_iterations_per_epoch=-1,
):
    color = LABEL_TO_STYLE[label]["color"]
    ids = list(data[h.K_ID])
    runs_training_loss = get_runs_helper(
        ids, data, dataset, iteration_limit, h.TRAIN_LOSS
    )
    runs_metric = None
    if metric != h.TRAIN_LOSS:
        runs_metric = get_runs_helper(ids, data, dataset, iteration_limit, metric)

    if len(ids) > 0:
        hyperparam, run_data, time_steps, _ = gen_best_runs_for_metric(
            runs_training_loss,
            data,
            metric,
            hyperparam=hyperparam,
            actual_metric_data=runs_metric,
        )
        max_loss, mean_loss, min_loss = plth.get_data_summary_for_metric(
            metric, run_data, time_steps
        )

        i = 3 if metric == h.TRAIN_LOSS else 4
        if iteration_limit != -1:
            i -= 1
        if dataset == "squad" and metric != h.TRAIN_LOSS:
            i -= 1

        time_steps = [time_step - i for time_step in time_steps]

        if num_iterations_per_epoch > 0:
            for i in range(len(time_steps)):
                if time_steps[i] != 1:
                    time_steps[i] = num_iterations_per_epoch * time_steps[i]

        ls = LABEL_TO_STYLE[label]["linestyle"]
        ax.plot(
            time_steps,
            mean_loss,
            linestyle=ls,
            color=color,
            label=label,
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
                    continue

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
    xaxis_labels=False,
    yaxis_labels=False,
    full_batch=False,
    momentum=False,
):
    big_batch = False
    (adam_data, sgd_data, adam_nm_data, sgd_m_data,) = data_selection.select_runs(
        acc_step, batch_size, big_batch, dataset, full_batch, metric, model
    )
    max_ss, min_ss, unique_ss = data_selection.find_step_size_range(
        acc_step, batch_size, big_batch, dataset, full_batch, metric, model
    )
    adam_ids = list(adam_data[h.K_ID])
    if not (metric == h.TRAIN_ACC or metric == h.F_ONE or metric == h.EXACT_MATCH):
        runs_for_ids_adam = get_runs_helper(adam_ids, adam_data, dataset, 0, metric)
        runs_for_ids_adam_training_loss = get_runs_helper(
            adam_ids, adam_data, dataset, 0, h.TRAIN_LOSS
        )
        _, _, _, max_start_loss = gen_best_runs_for_metric(
            runs_for_ids_adam_training_loss,
            adam_data,
            metric,
            actual_metric_data=runs_for_ids_adam,
        )

        ax.axhline(
            y=max_start_loss,
            color=plth.COLOR_INITIAL_VALUE,
            linestyle="-",
            label=plth.LABEL_INITIAL_VALUE,
        )
    else:
        max_start_loss = 0

    if momentum:
        plot_ss_vs_metric(
            adam_nm_data, ax, LABEL_ADAM_NM, metric, unique_ss, max_start_loss
        )
        plot_ss_vs_metric(
            sgd_m_data, ax, LABEL_SGD_M, metric, unique_ss, max_start_loss
        )
    else:
        plot_ss_vs_metric(sgd_data, ax, LABEL_SGD, metric, unique_ss, max_start_loss)
        plot_ss_vs_metric(adam_data, ax, LABEL_ADAM, metric, unique_ss, max_start_loss)

    ax.set_title(plth.title_format(model, dataset), pad=5)

    ax.set_xscale("log")
    if metric == h.TRAIN_PPL or metric == h.TRAIN_LOSS:
        ax.set_yscale("log")
        if metric == h.TRAIN_LOSS:
            tag_yaxis(ax, increment=1)
    ax.set_xlim(min_ss, max_ss)
    ax.minorticks_off()

    if yaxis_labels:
        ax.set_ylabel(plth.metric_to_text[metric], labelpad=0.5)

    if xaxis_labels:
        ax.set_xlabel(plth.LABEL_STEP_SIZE)

    if metric == h.F_ONE:
        ax.set_ylabel(plth.metric_to_text[metric], labelpad=-1)
    # if metric != h.TRAIN_PPL:
    #     ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    if metric == h.TRAIN_ACC:
        ax.set_ylim(0, 105)

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
    )


def plot_ss_vs_metric(data, axes, label, metric, unique_ss, max_start_loss):
    color = LABEL_TO_STYLE[label]["color"]
    steps, mean, min_val, max_val, diverged = [], [], [], [], []
    bigger_better = (
        metric == h.TRAIN_ACC or metric == h.F_ONE or metric == h.EXACT_MATCH
    )
    for ss in sorted(unique_ss):
        non_diverge = data[(data[h.K_SS] == ss)]
        if not non_diverge.empty:
            print("greater" if len(non_diverge) > 3 else "", ss)

            if non_diverge.empty:
                continue

            if metric == h.TRAIN_LOSS:
                vals = []
                for _, row in non_diverge.iterrows():
                    if pandas.isnull(row[h.AVERAGE_LOSS]):
                        vals.append(row[h.TRAIN_LOSS])
                    else:
                        vals.append(row[h.AVERAGE_LOSS])
                vals = np.asarray(vals)
            else:
                vals = non_diverge[metric]

            if np.isnan(vals).any():
                mean_loss = math.inf
            else:
                mean_loss = vals.mean()

            if (not bigger_better and (mean_loss >= max_start_loss - 0.01)) or (
                bigger_better and (mean_loss <= max_start_loss + 0.01)
            ):
                axes.scatter(
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
        axes, diverged, label, max_start_loss, max_val, mean, min_val, steps
    )


def main(args):
    assert not (args.full_batch and args.big_batch)
    assert not args.big_batch

    datasets, models, batch_sizes, acc_steps, length = plth.process_args(args)

    plth.init_plt_style(plt)

    figsize = (
        plth.FIGSIZE_3x2_short if args.plot_type == METRIC_VS_SS else plth.FIGSIZE_3x2
    )
    fig, axes = plt.subplots(int(math.ceil(length / 3)), 3, figsize=figsize)

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
                axes[y, x],
                batch_size=batch_sizes[i],
                acc_step=acc_steps[i],
                metric=metric,
                legend=y == 0 and x == 0,
                xaxis_labels=y == 1,
                yaxis_labels=x == 0,
                full_batch=args.full_batch,
            )

        if args.plot_type == METRIC_VS_SS:
            gen_step_size_vs_metric_plot(
                args,
                models[i],
                datasets[i],
                axes[y, x],
                batch_size=batch_sizes[i],
                acc_step=acc_steps[i],
                metric=metric,
                legend=False,
                xaxis_labels=y == 1,
                yaxis_labels=x == 0,
                full_batch=args.full_batch,
                momentum=args.momentum,
            )

    if args.full_batch:
        ylim = axes[1, 1].get_ylim()
        axes[1, 1].set_ylim(ylim[0], min(ylim[1], MAXIMUM_Y_VALUE))

    fplt.hide_frame(*axes[0])
    fplt.hide_frame(*axes[1])

    if args.plot_type == METRIC_VS_SS:
        fig.tight_layout(pad=0.5)
    else:
        fig.tight_layout(pad=0.5, h_pad=1.5, w_pad=-0.5)

    plth.save_figure(args, datasets, models, plt)


if __name__ == "__main__":
    main(plth.cli().parse_args())
