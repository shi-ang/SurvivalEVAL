import matplotlib.pyplot as plt
import numpy as np

from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike


def pp_plot(
    obs: NumericArrayLike,
    exp: NumericArrayLike,
    xlim: tuple = None,
    ylim: tuple = None,
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
