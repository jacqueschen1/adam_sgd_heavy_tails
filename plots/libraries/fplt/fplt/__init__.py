from functools import reduce

from .colors import *
from . import easings
from .animations import *
from .misc import *
from datetime import datetime
from matplotlib.dates import date2num, DateConverter, num2date
from matplotlib.container import ErrorbarContainer
from datetime import datetime
from math import atan2, degrees
import warnings


def circle_points(x, y, r, density=100):
    """The (cartesian) coordinates for a circle of radius r centered at (x, y)."""
    thetas = np.linspace(0, 2 * np.pi, density)
    xs = x + r * np.cos(thetas)
    ys = y + r * np.sin(thetas)
    return xs, ys


def ellipse_points(center, A, scaling=1.0, density=100):
    """Returns the cartesian coordinates of an ellipse centered at center
    with shape given by the matrix A.

    Corresponds to the level ``f(x) = 1`` for ``f(x) = (x-c)^T A (x-c)``.
    """
    CIRCLEGRID = np.linspace(0, 2 * np.pi, density)
    A = np.linalg.inv(A) * scaling
    xs = A[0, 0] * np.sin(CIRCLEGRID) + A[0, 1] * np.cos(CIRCLEGRID) + center[0]
    ys = A[1, 1] * np.cos(CIRCLEGRID) + A[1, 0] * np.sin(CIRCLEGRID) + center[1]
    return xs, ys


def plot_circle(ax, x, y, r, density=100, **kwargs):
    """
    Plots a circle of radius r on ax, centered at (x, y).
    Additional key-value arguments are passed to ax.plot
    """
    thetas = np.linspace(0, 2 * np.pi, density)
    xs = x + r * np.cos(thetas)
    ys = y + r * np.sin(thetas)
    ax.plot(xs, ys, **kwargs)


def enable_latex(matplotlib, packages=None):
    """
    Enables latex for the given matplotlib instance.

    A list of package names can be passed to be loaded in the preamble.
    Defaults to loading amsmath.
    """
    if packages is None:
        packages = ["amsmath"]

    preamble = [r"\usepackage{%s}" % package for package in packages]

    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["text.latex.preamble"] = [preamble]


def set_font_size(matplotlib, size=18, family=None):
    """
    Sets the font size (default: 18) and family (default: "serif")
    """
    font = {"size": size}
    if family is not None:
        font["family"] = family
    matplotlib.rc("font", **font)


def hide_frame(*axes, top=True, right=True, left=False, bottom=False):
    for ax in axes:
        ax.spines["top"].set_visible(not top)
        ax.spines["right"].set_visible(not right)
        ax.spines["left"].set_visible(not left)
        ax.spines["bottom"].set_visible(not bottom)


def hide_all_frame(*axes):
    hide_frame(*axes, top=True, right=True, left=True, bottom=True)


def hide_ticklabels(*axes, x=True, y=True):
    for ax in axes:
        if x:
            ax.set_xticklabels([], minor=True)
            ax.set_xticklabels([], minor=False)
        if y:
            ax.set_yticklabels([], minor=True)
            ax.set_yticklabels([], minor=False)


def hide_ticks(*axes, x=True, y=True):
    for ax in axes:
        if x:
            ax.set_xticks([], minor=True)
            ax.set_xticks([], minor=False)
        if y:
            ax.set_yticks([], minor=True)
            ax.set_yticks([], minor=False)


def clean_ax(*axes):
    """Hide ticks and ticklabels on all axes"""
    for ax in axes:
        hide_ticklabels(ax)
        hide_ticks(ax)


def strip_axes(axes):
    hide_ticks(axes)
    hide_labels(axes)


def hide_labels(*axes, x=True, y=True):
    for ax in axes:
        if x:
            ax.set_xlabel("")
        if y:
            ax.set_ylabel("")


def save(fig, name, tight=True, transparent=False):
    fig.savefig(name, bbox_inches="tight" if tight else None, transparent=transparent)


def save_current(plt, name):
    plt.gcf().savefig(name + ".png", transparent=True)


def axis(ax, x=True, y=True):
    if x:
        ax.axhline(0, color="k", alpha=0.2)
    if y:
        ax.axvline(0, color="k", alpha=0.2)


def make_grid(fig, nrows=1, ncols=1, **kwargs):
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, **kwargs)
    if nrows == 1 or ncols == 1:
        axes = []
        for i in range(nrows):
            for j in range(ncols):
                axes.append(fig.add_subplot(gs[i, j]))
    else:
        axes = []
        for i in range(nrows):
            axes.append([])
            for j in range(ncols):
                axes[i].append(fig.add_subplot(gs[i, j]))
    return axes


def equalize_xy_axes(*axes):
    for ax in axes:
        axlimits = [*ax.get_xlim(), *ax.get_ylim()]
        minlim, maxlim = np.min(axlimits), np.max(axlimits)
        ax.set_xlim([minlim, maxlim])
        ax.set_ylim([minlim, maxlim])


def normalize_y_axis(*axes):
    miny, maxy = np.inf, -np.inf
    for ax in axes:
        y1, y2 = ax.get_ylim()
        miny = np.min([miny, y1])
        maxy = np.max([maxy, y2])
    for ax in axes:
        ax.set_ylim([miny, maxy])


# Label line with line2D label data
def labelLine(
    line,
    x,
    label=None,
    align=True,
    drop_label=False,
    manual_rotation=0,
    ydiff=0.0,
    **kwargs
):
    """Label a single matplotlib line at position x

    Parameters
    ----------
    line : matplotlib.lines.Line
       The line holding the label
    x : number
       The location in data unit of the label
    label : string, optional
       The label to set. This is inferred from the line by default
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent calls to e.g. legend
       do not use it anymore.
    kwargs : dict, optional
       Optional arguments passed to ax.text
    """
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    mask = np.isfinite(ydata)
    if mask.sum() == 0:
        raise Exception("The line %s only contains nan!" % line)

    # Find first segment of xdata containing x
    if len(xdata) == 2:
        i = 0
        xa = min(xdata)
        xb = max(xdata)
    else:
        for i, (xa, xb) in enumerate(zip(xdata[:-1], xdata[1:])):
            if min(xa, xb) <= x <= max(xa, xb):
                break
        else:
            raise Exception("x label location is outside data range!")

    def x_to_float(x):
        """Make sure datetime values are properly converted to floats."""
        return date2num(x) if isinstance(x, datetime) else x

    xfa = x_to_float(xa)
    xfb = x_to_float(xb)
    ya = ydata[i]
    yb = ydata[i + 1]
    y = ya + (yb - ya) * (x_to_float(x) - xfa) / (xfb - xfa)

    if not (np.isfinite(ya) and np.isfinite(yb)):
        warnings.warn(
            (
                "%s could not be annotated due to `nans` values. "
                "Consider using another location via the `x` argument."
            )
            % line,
            UserWarning,
        )
        return

    if not label:
        label = line.get_label()

    if drop_label:
        line.set_label(None)

    if align:
        # Compute the slope and label rotation
        screen_dx, screen_dy = ax.transData.transform(
            (xfa, ya)
        ) - ax.transData.transform((xfb, yb))
        rotation = (degrees(atan2(screen_dy, screen_dx)) + 90) % 180 - 90
    else:
        rotation = manual_rotation

    # Set a bunch of keyword arguments
    if "color" not in kwargs:
        kwargs["color"] = line.get_color()

    if ("horizontalalignment" not in kwargs) and ("ha" not in kwargs):
        kwargs["ha"] = "center"

    if ("verticalalignment" not in kwargs) and ("va" not in kwargs):
        kwargs["va"] = "center"

    if "backgroundcolor" not in kwargs:
        kwargs["backgroundcolor"] = ax.get_facecolor()

    if "clip_on" not in kwargs:
        kwargs["clip_on"] = True

    if "zorder" not in kwargs:
        kwargs["zorder"] = 2.5

    ax.text(x, y + ydiff, label, rotation=rotation, **kwargs)


def make_full_axis(fig):
    gs = fig.add_gridspec(nrows=1, ncols=1, left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax = fig.add_subplot(gs[0, 0])
    hide_ticks(ax)
    hide_ticklabels(ax)
    hide_all_frame(ax)
    return ax


def make_figure_list(n, scaling=4, marg=0.1):
    fig = mpl.pyplot.figure(figsize=(n * scaling, scaling))
    h_marg = marg
    W = (1.0 - (n + 1) * h_marg) / n
    H = n * W
    v_marg = (1.0 - H) / 2
    axes = [fig.add_axes([(i + 1) * h_marg + i * W, v_marg, W, H]) for i in range(n)]
    return fig, axes


def fullscreen(plt):
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()


def load_style(mpl, stylename="anim"):
    styles = {
        "anim": {
            "figure.facecolor": "#F3F3F3",
            "axes.facecolor": "#F3F3F3",
            "axes.prop_cycle": mpl.cycler(color=COLORS["CC"]),
            "image.cmap": "YlOrBr",
        },
        "paper": {
            "axes.prop_cycle": mpl.cycler(color=COLORS["CC"]),
            "image.cmap": "YlOrBr",
        },
    }
    for k, v in styles[stylename].items():
        mpl.rcParams[k] = v


def set_alpha(ax, alpha):
    """

    :param ax:
    :return:
    """
    ax.patch.set_alpha(alpha)
    for line in ax.lines:
        line.set_alpha(alpha)
    for spine in ax.spines.values():
        spine.set_alpha(alpha)
    ax.tick_params("both", colors=(0, 0, 0, alpha))


def aspectratio(fig):
    s = fig.get_size_inches()
    return s[0] / s[1]


def make_linear_curve_in_logspace(ax):
    xlims = ax.get_xlim()
    from_b10_x, to_b10_x = [
        np.ceil(np.log10(xlims[0])),
        np.floor(np.log10(xlims[1])),
    ]
    x_b10_powers = np.linspace(from_b10_x, to_b10_x, int(to_b10_x - from_b10_x) + 1)

    ylims = ax.get_ylim()
    from_b10_y, to_b10_y = [
        np.ceil(np.log10(ylims[0])),
        np.floor(np.log10(ylims[1])),
    ]
    y_b10_powers = np.linspace(
        from_b10_y,
        to_b10_y + to_b10_x - from_b10_x,
        int(to_b10_y - from_b10_y + to_b10_x - from_b10_x) + 1,
    )

    for x in x_b10_powers:
        ax.axvline(10 ** x, alpha=0.5, color="gray", linestyle="--")
    for y in y_b10_powers:
        ax.axline(
            [10 ** from_b10_x, 10 ** y],
            [10 ** to_b10_x, 10 ** (y - (to_b10_x))],
            alpha=0.5,
            color="gray",
            linestyle="--",
        )
