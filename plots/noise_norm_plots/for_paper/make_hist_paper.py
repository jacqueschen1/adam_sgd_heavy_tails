import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.stats import levy_stable, norm
import fplt
import math
import random
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter

norms_file_names = [
    "lenet5_mnist",
    "resnet34_cifar10",
    "resnet50_cifar100",
    "transformer_encoder_wikitext2",
    "transformer_xl_ptb",
    "bert_base_pretrained_squad",
    "resnet50_imagenet",
]
bins = [60, 70, 80, "auto", 60, 1000]
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
    "imagenet": "ImageNet",
}

alphas = [2.0, 2.0, 2.0, 2.0, 1.60, 1.18, 1.05, 1.5]
x_ticks = [
    [0.02, 0.06],
    [2500, 3000],
    [100000, 150000],
    [98, 102],
    [0.10, 0.15, 0.2],
    [0.2, 0.3, 0.4],
    [],
    [0.1, 0.11, 0.12],
]


def produce_histogram_log(axes, norms, num_bins, file_name, axis_labels=True):
    hist, bins, _ = axes.hist(norms, bins=100, density=True)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), num_bins)
    axes.clear()
    # axes.hist(norms, bins=logbins, density=True)
    manual_histograms(axes, norms, logbins)
    file_name = file_name.split("_")
    if len(file_name) > 2:
        model = "_".join(file_name[:-1])
        dataset = file_name[-1]
    else:
        model = file_name[0]
        dataset = file_name[1]
    print(model)

    axes.set_title("{}, {}".format(name_to_label[model], name_to_label[dataset]))
    axes.set_xscale("log")

    if axis_labels:
        axes.set_xlabel("Noise norm")
        axes.set_ylabel("Density")


def manual_histograms(ax, data, bins):
    """ """
    heights = []
    for i in range(len(bins) - 1):
        bin_lo, bin_hi = bins[i], bins[i + 1]
        data_larger_than_bin_lo = data[bin_lo <= data]
        data_in_bin = data_larger_than_bin_lo[data_larger_than_bin_lo < bin_hi]
        heights.append(data_in_bin.size / len(data))

    ax.bar(bins[:-1], heights, width=np.diff(bins), bottom=0, align="edge")
    return


def produce_histogram(axes, norms, num_bins, file_name, axis_labels=True):
    hist, bins, _ = axes.hist(norms, bins=100, density=True)
    bins = np.linspace((bins[0]), (bins[-1]), num_bins)
    axes.clear()

    manual_histograms(axes, norms, bins)
    file_name = file_name.split("_")
    if len(file_name) > 2:
        model = "_".join(file_name[:-1])
        dataset = file_name[-1]
    else:
        model = file_name[0]
        dataset = file_name[1]

    axes.set_title("{}, {}".format(name_to_label[model], name_to_label[dataset]))
    if axis_labels:
        axes.set_xlabel("Noise norm")
        axes.set_ylabel("Density")


def produce_stable(axes, alpha, beta):
    r = levy_stable.rvs(alpha, beta, size=300, loc=0.1, scale=0.001)
    # r = np.abs(r)
    hist, bins, _ = axes.hist(r, bins=100, density=True)
    bins = np.linspace((bins[0]), (bins[-1]), 40)
    axes.clear()

    manual_histograms(axes, r, bins)

    #    axes.hist(r, bins=logbins, density=True)
    axes.set_title("Synthetic Levy-stable")
    axes.set_xscale("log")


def produce_normal(axes):
    r = norm.rvs(size=1000, loc=100, scale=1.0)
    r = np.abs(r)

    hist, bins, _ = axes.hist(r, bins=10)
    bins = np.linspace((bins[0]), (bins[-1]), 25)
    axes.clear()

    # axes.hist(r, bins=logbins)
    manual_histograms(axes, r, bins)
    axes.set_title("Synthetic Gaussian")
    axes.set_xscale("log")


if __name__ == "__main__":
    np.random.seed(1)
    np.seterr(all="raise")

    random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet", action="store_true")
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 8})
    plt.rcParams["lines.linewidth"] = 2
    plt.rc("ytick", labelsize=8)
    plt.rc("xtick", labelsize=8)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(7, 7 / (60 / 20)))
    for i in range(8):
        x = i % 4
        y = i // 4

        if i == 3:
            produce_normal(axes[y, x])
            axes[y, x].yaxis.set_major_locator(MaxNLocator(3))
        elif i == 7:
            produce_stable(axes[y, x], 1.5, 0.9)
            axes[y, x].yaxis.set_major_locator(MaxNLocator(3))

        else:
            index = i
            if i > 3:
                index = i - 1

            norms = np.load("numpy/" + norms_file_names[index] + ".npy")
            num_bins = 25
            if i == 6:
                num_bins = 150
            # if i == 5:
            #     produce_histogram_log(axes[y, x], norms, num_bins, norms_file_names[i], axis_labels=x == 0)
            # else:
            if i == 6:
                produce_histogram_log(
                    axes[y, x],
                    norms[:-1],
                    num_bins,
                    norms_file_names[index],
                    axis_labels=x == 0,
                )
            else:
                produce_histogram(
                    axes[y, x],
                    norms[:-1],
                    num_bins,
                    norms_file_names[index],
                    axis_labels=True,
                )

        axes[y, x].set_xticks([], minor=True)
        axes[y, x].set_xticklabels([], minor=True)

        if x != 3:
            label = "$\\alpha_{\\mathrm{MLE}} = "
        else:
            label = "$\\alpha = "

        axes[y, x].text(
            0.85,
            0.7,
            label + str(alphas[i]) + "$",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[y, x].transAxes,
        )

        if i == 6:
            x_major = LogLocator(base=10.0, numticks=5)
            axes[y, x].xaxis.set_major_locator(x_major)
            x_minor = LogLocator(
                base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10
            )
            axes[y, x].xaxis.set_minor_locator(x_minor)
            axes[y, x].xaxis.set_minor_formatter(NullFormatter())

        else:
            axes[y, x].set_xticks(x_ticks[i], minor=False)
            axes[y, x].set_xticklabels(x_ticks[i], minor=False)

        axes[y, x].yaxis.set_major_locator(MaxNLocator(3))

    fplt.hide_frame(*axes[0])
    fplt.hide_frame(*axes[1])

    for i in range(8):
        x = i % 4
        y = i // 4

        if x == 0:
            if y == 0:
                axes[y, x].set_ylabel("$\\bf Vision$ \n Density")
            else:
                axes[y, x].set_ylabel("$\\bf NLP$ \n Density")

        else:
            axes[y, x].set_ylabel("")
            axes[y, x].yaxis.set_ticklabels([])
        if y == 1:
            axes[y, x].set_xlabel("Error$^2$")
        else:
            axes[y, x].set_xlabel("")

    fig.tight_layout()

    plt.savefig("final_fig_for_paper.pdf")
