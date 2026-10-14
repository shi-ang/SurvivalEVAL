from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import is_color_like

from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike


def _validate_plot_indices(
    indices: int | list[int],
    size: int,
    input_name: str = "curve_indices",
    *,
    one_based: bool = False,
) -> np.ndarray:
    """Validate a nonempty 1D selection, preserving its order and repetitions.

    Return integer indices using the caller's zero-based sample or one-based
    cause convention. Reject noninteger selections and out-of-range entries.
    """
    indices = np.atleast_1d(np.asarray(indices))
    if (
        indices.ndim != 1
        or indices.size == 0
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        raise ValueError(
            f"{input_name} must be an integer or a nonempty 1D sequence of integers."
        )
    start = int(one_based)
    if np.any((indices < start) | (indices >= size + start)):
        raise IndexError(
            f"{input_name} must be between {start} and {size + start - 1}."
        )
    return indices


def _plot_curves(
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    *,
    title: str,
    x_label: str,
    y_label: str,
    color: str | list[str] | None = None,
    x_lim: float | tuple[float, float] | None = None,
    y_lim: float | tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    line_styles: list[str] | None = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw selected curves in one Matplotlib call with a consistent local style.

    Time/curve values have their point axis last; a 1D array may be shared by
    all lines. Colors may be shared or specified per line. Optional line styles
    distinguish samples in CIF plots unless kwargs supplies linestyle or ls.
    Reuse ax when supplied, and return its figure and axes without showing them.
    Other kwargs are passed to Axes.plot; label overrides the generated labels.
    """
    colors = None
    if color is not None and not is_color_like(color):
        if (
            not isinstance(color, (list, tuple))
            or len(color) != len(labels)
            or not all(is_color_like(value) for value in color)
        ):
            raise ValueError(
                "color must be a valid color or one valid color per plotted curve."
            )
        colors, color = color, None
    if "linewidth" not in kwargs and "lw" not in kwargs:
        kwargs["linewidth"] = 2.2
    kwargs.setdefault("alpha", 0.9)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    lines = ax.plot(x.T, y.T, color=color, label=kwargs.pop("label", labels), **kwargs)
    if colors is not None:
        for line, line_color in zip(lines, colors):
            line.set_color(line_color)
    if line_styles is not None and "linestyle" not in kwargs and "ls" not in kwargs:
        for line, line_style in zip(lines, line_styles):
            line.set_linestyle(line_style)

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    if not any(ax.get_title(loc=loc) for loc in ("left", "center", "right")):
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_axisbelow(True)
    ax.grid(True, color="#dbe1e8", linewidth=0.7, alpha=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=9)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            legend_labels,
            frameon=False,
            fontsize=9,
            ncol=2 if len(handles) > 5 else 1,
        )
    return ax.figure, ax


def pp_plot(
    obs: NumericArrayLike,
    exp: NumericArrayLike,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    color: str = "blue",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Generate a probability-probability (P-P) plot given observed and expected probabilities.

    Parameters
    ----------
    obs: NumericArrayLike
        Observed probabilities.
    exp: NumericArrayLike
        Expected probabilities. Generally, you would want the expected probabilities to be sorted in ascending order.
    xlim: tuple, optional
        Limits for the x-axis.
    ylim: tuple, optional
        Limits for the y-axis.
    color: str, default: 'blue'
        Color for the prediction line.
    **kwargs
        Additional keyword arguments for customizing the plot axes.

    Returns
    -------
    fig: plt.Figure
        The figure object of the P-P plot.
    axe: plt.Axes
        The axes object of the P-P plot.
    """
    assert len(obs) == len(
        exp
    ), "Observed and expected probabilities must have the same length."

    fig, axe = plt.subplots()
    axe.plot([0, 1], [0, 1], "--", label="Ideal", color="grey", linewidth=2)
    axe.plot(
        np.asarray(exp),
        np.asarray(obs),
        "o-",
        markersize=8,
        label="Prediction",
        color=color,
        linewidth=2,
    )
    if xlim is not None:
        axe.set_xlim(xlim)
    if ylim is not None:
        axe.set_ylim(ylim)
    axe.set_xlabel("Expected Probability")
    axe.set_ylabel("Observed Probability")
    axe.legend()
    for key, value in kwargs.items():
        plt.setp(axe, **{key: value})
    fig.tight_layout()
    return fig, axe
