"""Shared prediction and plotting support for survival-curve evaluators."""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike
from SurvivalEVAL.Evaluations.util import (
    align_curve_and_time_coordinates,
    check_and_convert,
    predict_mean_st,
    predict_median_st,
    predict_multi_probs_from_curve,
    predict_prob_from_curve,
    predict_rmst,
    survival_to_quantile,
    validate_time_point,
    validate_time_points,
    zero_padding,
)
from SurvivalEVAL.Evaluations.util_plots import _plot_curves, _validate_plot_indices


class _BaseSurvivalEvaluator:
    """Handle survival-curve predictions independently of observed outcomes."""

    def __init__(
        self,
        pred_survs: NumericArrayLike,
        time_coordinates: NumericArrayLike,
        n_samples: int,
        predict_time_method: str = "Median",
        interpolation: str = "Linear",
    ) -> None:
        """
        Initialize curves for the sample count validated by the outcome evaluator.

        Parameters
        ----------
        pred_survs : NumericArrayLike
            Survival probabilities with shape (n_times,) or (n_samples, n_times).
        time_coordinates : NumericArrayLike
            Time coordinates matching the last axis of the predicted curves.
        n_samples : int
            Number of samples that the predicted curves must describe.
        predict_time_method : str
            Point prediction method, "Median", "Mean", or "RMST", case-insensitive.
        interpolation : str
            Curve interpolation method, "Linear" or "Pchip".
        """
        self._n_samples = n_samples
        self.set_prediction_inputs(pred_survs, time_coordinates)

        predict_time_method = predict_time_method.lower()
        if predict_time_method == "median":
            self.predict_time_method = predict_median_st
        elif predict_time_method == "mean":
            self.predict_time_method = predict_mean_st
        elif predict_time_method == "rmst":
            self.predict_time_method = predict_rmst
        else:
            raise TypeError(
                "Please enter one of 'Median', 'Mean', or 'RMST' for calculating predicted survival time."
            )

        self.interpolation = interpolation

    @staticmethod
    def _validate_prediction_sample_count(
        pred_survs: np.ndarray,
        time_coordinates: np.ndarray,
        n_samples: int,
    ) -> None:
        """
        Validate that prediction inputs have one row per testing sample.

        Parameters
        ----------
        pred_survs : ndarray
            Predicted survival curves, either shared across samples or one row
            per sample.
        time_coordinates : ndarray
            Time coordinates, either shared across samples or one row per sample.
        n_samples : int
            Expected number of testing samples.

        Raises
        ------
        ValueError
            If a 2-D prediction input has a row count different from
            ``n_samples``.
        """
        sample_counts = {}
        if pred_survs.ndim == 2:
            sample_counts["pred_survs"] = pred_survs.shape[0]
        if time_coordinates.ndim == 2:
            sample_counts["time_coordinates"] = time_coordinates.shape[0]

        mismatched_counts = {
            name: count for name, count in sample_counts.items() if count != n_samples
        }
        if mismatched_counts:
            count_details = ", ".join(
                f"{name} has {count}" for name, count in mismatched_counts.items()
            )
            raise ValueError(
                "The number of prediction rows must match the number of testing "
                f"samples ({n_samples}); {count_details}."
            )

    @staticmethod
    def _validate_prediction_curve_inputs(
        pred_survs: np.ndarray,
        time_coordinates: np.ndarray,
    ) -> None:
        """
        Validate survival-curve probability and time-grid invariants.

        Parameters
        ----------
        pred_survs : ndarray
            Predicted survival probabilities after any zero-padding.
        time_coordinates : ndarray
            Time coordinates after any zero-padding.

        Raises
        ------
        ValueError
            If time coordinates are not strictly increasing along each row, if
            survival probabilities are outside ``[0, 1]``, or if survival curves
            increase over time.
        """
        if np.any(np.diff(time_coordinates, axis=-1) <= 0):
            raise ValueError(
                "Each row of time_coordinates must be strictly increasing."
            )
        if np.any((pred_survs < 0) | (pred_survs > 1)):
            raise ValueError("Survival probabilities must be between 0 and 1.")
        if np.any(np.diff(pred_survs, axis=-1) > 1e-8):
            raise ValueError("Survival probabilities must be nonincreasing.")

    def set_prediction_inputs(
        self,
        pred_survs: NumericArrayLike,
        time_coordinates: NumericArrayLike,
    ) -> None:
        """
        Reset predicted survival curves and their time coordinates together.

        Prefer this method when updating both inputs together, because the
        pair is validated, zero-padded, and cached-property invalidated once.
        Property setters remain available for changing only one input.

        Parameters
        ----------
        pred_survs : NumericArrayLike
            Survival probabilities with shape (n_times,) or (n_samples, n_times).
        time_coordinates : NumericArrayLike
            Time coordinates matching the last axis of the predicted curves.
        """
        pred_survs = check_and_convert(pred_survs)
        time_coordinates = check_and_convert(time_coordinates)

        ndim_surv = pred_survs.ndim
        ndim_time = time_coordinates.ndim
        if ndim_time == 1 and ndim_surv == 1:
            raise TypeError(
                "At least one of 'pred_survs' or 'time_coordinates' must be a 2D array."
            )

        self._validate_prediction_sample_count(
            pred_survs, time_coordinates, self._n_samples
        )
        pred_survs, time_coordinates = zero_padding(pred_survs, time_coordinates)
        self._validate_prediction_curve_inputs(pred_survs, time_coordinates)
        self._pred_survs, self._time_coordinates = pred_survs, time_coordinates
        self.ndim_surv = self._pred_survs.ndim
        self.ndim_time = self._time_coordinates.ndim
        self._clear_cache()

    @property
    def pred_survs(self) -> np.ndarray:
        """
        Return the stored survival curves.

        Returns
        -------
        ndarray
            Stored survival probabilities, shared across samples or with one row per sample.
        """
        return self._pred_survs

    @pred_survs.setter
    def pred_survs(self, val: NumericArrayLike) -> None:
        """
        Replace and validate the stored survival curves.

        Parameters
        ----------
        val : NumericArrayLike
            Replacement survival probabilities matching the stored grid.
        """
        print("Setter called. Resetting predicted curves for this evaluator.")
        pred_survs = check_and_convert(val)
        # The stored grid may already include a zero-time column inserted by
        # zero_padding. Accept raw replacement curves by adding the matching
        # survival-at-zero column before validating curve/grid lengths.
        if self._time_coordinates.shape[-1] == pred_survs.shape[-1] + 1 and np.all(
            self._time_coordinates[..., 0] == 0
        ):
            pred_survs = np.concatenate(
                (
                    np.ones((*pred_survs.shape[:-1], 1), dtype=pred_survs.dtype),
                    pred_survs,
                ),
                axis=-1,
            )
        self.set_prediction_inputs(pred_survs, self._time_coordinates)

    @property
    def time_coordinates(self) -> np.ndarray:
        """
        Return the stored curve time coordinates.

        Returns
        -------
        ndarray
            Stored time coordinates corresponding to the predicted curves.
        """
        return self._time_coordinates

    @time_coordinates.setter
    def time_coordinates(self, val: NumericArrayLike) -> None:
        """
        Replace and validate the time coordinates for the stored curves.

        Parameters
        ----------
        val : NumericArrayLike
            Replacement time coordinates matching the stored curves.
        """
        print("Setter called. Resetting time coordinates for this evaluator.")
        time_coordinates = check_and_convert(val)
        # Symmetric one-sided update: if curves are already zero-padded and
        # the replacement grid is raw, prepend time zero so the pair remains
        # aligned before the shared validation path runs.
        if self._pred_survs.shape[-1] == time_coordinates.shape[-1] + 1 and not np.all(
            time_coordinates[..., 0] == 0
        ):
            time_coordinates = np.concatenate(
                (
                    np.zeros(
                        (*time_coordinates.shape[:-1], 1),
                        dtype=time_coordinates.dtype,
                    ),
                    time_coordinates,
                ),
                axis=-1,
            )
        self.set_prediction_inputs(self._pred_survs, time_coordinates)

    @property
    def interpolation(self) -> str:
        """
        Curve interpolation method; changing it invalidates cached event times.

        Returns
        -------
        str
            Interpolation method used when evaluating curves between their time coordinates.
        """
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value: str) -> None:
        """
        Set the curve interpolation method and invalidate cached predictions.

        Parameters
        ----------
        value : str
            Curve interpolation method, "Linear" or "Pchip".
        """
        if not isinstance(value, str) or value.lower() not in {"linear", "pchip"}:
            raise ValueError("interpolation must be 'Linear' or 'Pchip'.")
        self._interpolation = value.capitalize()
        self._clear_cache()

    @cached_property
    def predicted_event_times(self) -> np.ndarray:
        """
        Predict and cache event times using the configured curve summary.

        Returns
        -------
        ndarray
            One event-time point prediction per sample using the configured prediction method.
        """
        return self.predict_time_from_curve(self.predict_time_method)

    def _clear_cache(self) -> None:
        # See how to clear cache in functools:
        # https://docs.python.org/3/library/functools.html#functools.cached_property
        # https://stackoverflow.com/questions/62662564/how-do-i-clear-the-cache-from-cached-property-decorator
        """Invalidate cached event-time predictions."""
        self.__dict__.pop("predicted_event_times", None)

    def predict_time_from_curve(
        self,
        predict_method: Callable[[np.ndarray, np.ndarray, str], float | np.ndarray],
    ) -> np.ndarray:
        """
        Predict survival time from survival curves.

        Parameters
        ----------
        predict_method : callable
            Function accepting (survival_curves, time_coordinates, interpolation) and returning event times.
            There are three build-in methods: 'predict_median_st', 'predict_mean_st', and 'predict_rmst'.
            'predict_median_st' uses the median of the survival curve as the predicted survival time.
            'predict_mean_st' uses the expected time of the survival curve as the predicted survival time.
            'predict_rmst' uses the restricted mean survival time of the survival curve as the predicted survival time.

        Returns
        -------
        predicted_times : ndarray, shape = (n_samples, )
            Predicted survival time for each sample.
        """
        if (
            (predict_method is not predict_mean_st)
            and (predict_method is not predict_median_st)
            and (predict_method is not predict_rmst)
        ):
            error = (
                "Prediction method must be 'predict_mean_st', 'predict_median_st', or 'predict_rmst'"
                f"got '{predict_method.__name__}' instead"
            )
            raise TypeError(error)

        predicted_times = predict_method(
            self._pred_survs, self._time_coordinates, self.interpolation
        )
        return predicted_times

    def predict_probability_from_curve(
        self,
        target_time: float | np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the survival probability at a given time point from a predicted curve.

        Each predicted curve will only have one corresponding probability.
        Note that this method is different from the 'predict_multi_probabilities_from_curve' method,
        which predicts the multiple probabilities at multiple time points from a predicted curve.

        Parameters
        ----------
        target_time : float | ndarray, shape = (n_samples, )
            Time point(s) at which the survival probability is to be predicted.
            If float or int, the same time point is used for all samples. If
            array-like, each sample will have its own target time. The length
            of the array must be the same as the number of samples.

        Returns
        -------
        predicted_survival_probability : ndarray, shape = (n_samples, )
            Predicted survival probabilities at the target time point(s).
        """
        if self.ndim_surv == 2 and self.ndim_time == 1:
            n_samples = self._pred_survs.shape[0]
        elif self.ndim_surv == 1 and self.ndim_time == 2:
            n_samples = self._time_coordinates.shape[0]
        elif self.ndim_surv == 2 and self.ndim_time == 2:
            assert (
                self._pred_survs.shape[0] == self._time_coordinates.shape[0]
            ), "The number of samples in pred_survs and time_coordinates must be the same"
            n_samples = self._pred_survs.shape[0]
        else:
            raise TypeError("Dimensional error")

        if np.isscalar(target_time):
            target_time = validate_time_point(
                target_time,
                input_name="target_time",
            ) * np.ones(n_samples, dtype=self._time_coordinates.dtype)
        elif isinstance(target_time, np.ndarray):
            target_time = validate_time_points(target_time, input_name="target_time")
            if target_time.shape[0] != n_samples:
                raise ValueError(
                    "target_time must have the same length as the number of samples."
                )
        else:
            error = (
                "Target time must be a float, int, or 1D array, "
                f"got '{type(target_time)}' instead"
            )
            raise TypeError(error)

        predict_probs = np.empty(n_samples, dtype=self._pred_survs.dtype)

        if self.ndim_surv == 2 and self.ndim_time == 1:
            for i, curve in enumerate(self._pred_survs):
                predict_probs[i] = predict_prob_from_curve(
                    curve, self.time_coordinates, target_time[i], self.interpolation
                )
        elif self.ndim_surv == 1 and self.ndim_time == 2:
            for i, times in enumerate(self._time_coordinates):
                predict_probs[i] = predict_prob_from_curve(
                    self._pred_survs, times, target_time[i], self.interpolation
                )
        elif self.ndim_surv == 2 and self.ndim_time == 2:
            for i in range(n_samples):
                predict_probs[i] = predict_prob_from_curve(
                    self._pred_survs[i, :],
                    self._time_coordinates[i, :],
                    target_time[i],
                    self.interpolation,
                )
        else:
            raise TypeError("Dimensional error")

        return predict_probs

    def predict_multi_probabilities_from_curve(
        self,
        target_times: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the survival probability at multiple time points from the predicted curve.

        Parameters
        ----------
        target_times : ndarray, shape = (n_target_times)
            Time points at which the survival probabilities are to be predicted.

        Returns
        -------
        prob_mat : ndarray, shape = (n_samples, n_target_times)
            Predicted survival probabilities at the target time points.
        """
        target_times = validate_time_points(target_times, input_name="target_times")

        if self.ndim_surv == 2 and self.ndim_time == 1:
            n_samples = self._pred_survs.shape[0]
            prob_mat = np.empty(
                (n_samples, target_times.shape[0]), dtype=self._pred_survs.dtype
            )

            for i, curve in enumerate(self._pred_survs):
                prob_mat[i] = predict_multi_probs_from_curve(
                    curve, self.time_coordinates, target_times, self.interpolation
                )
        elif self.ndim_surv == 1 and self.ndim_time == 2:
            n_samples = self._time_coordinates.shape[0]
            prob_mat = np.empty(
                (n_samples, target_times.shape[0]), dtype=self._pred_survs.dtype
            )

            for i, times in enumerate(self._time_coordinates):
                prob_mat[i] = predict_multi_probs_from_curve(
                    self._pred_survs, times, target_times, self.interpolation
                )
        elif self.ndim_surv == 2 and self.ndim_time == 2:
            n_samples = self._pred_survs.shape[0]
            prob_mat = np.empty(
                (n_samples, target_times.shape[0]), dtype=self._pred_survs.dtype
            )

            for i in range(n_samples):
                prob_mat[i] = predict_multi_probs_from_curve(
                    self._pred_survs[i, :],
                    self._time_coordinates[i, :],
                    target_times,
                    self.interpolation,
                )
        else:
            raise TypeError("Dimensional error")

        return prob_mat

    def predict_multi_hazards_from_curve(self, target_times: np.ndarray) -> np.ndarray:
        """
        Predict piecewise constant hazard rates at multiple target times.

        Parameters
        ----------
        target_times : ndarray, shape = (n_target_times)
            Time points at which the hazard rates are to be predicted.

        Returns
        -------
        hazard_mat : ndarray, shape = (n_samples, n_target_times)
            Predicted hazard rates at the target time points.
        """
        target_times = validate_time_points(target_times, input_name="target_times")

        if self._pred_survs.shape[-1] < 2:
            raise ValueError(
                "At least two time points are required to estimate hazards."
            )

        n_samples = self._n_samples
        hazard_mat = np.empty((n_samples, target_times.shape[0]), dtype=float)
        for i in range(n_samples):
            hazard_mat[i] = self._predict_risks_from_curve(
                i, target_times, risks="hazard"
            )

        return hazard_mat

    def _predict_risks_from_curve(
        self,
        sample_index: int,
        target_times: np.ndarray,
        risks: str,
    ) -> np.ndarray:
        """
        Predict one sample's risks without allocating a matrix of scores.

        Parameters
        ----------
        sample_index : int
            Zero-based index of the sample whose curve is evaluated.
        target_times : ndarray
            One-dimensional array of non-negative evaluation times.
        risks : str
            "survival" returns negative survival probabilities; "hazard" returns interval hazard rates.

        Returns
        -------
        ndarray
            One risk score per target time for the selected sample.
        """
        target_times = validate_time_points(target_times, input_name="target_times")
        survival_curve = (
            self._pred_survs[sample_index] if self.ndim_surv == 2 else self._pred_survs
        )
        time_grid = (
            self._time_coordinates[sample_index]
            if self.ndim_time == 2
            else self._time_coordinates
        )
        if risks == "survival":
            predicted_risks = np.asarray(
                predict_multi_probs_from_curve(
                    survival_curve, time_grid, target_times, self.interpolation
                ),
                dtype=self._pred_survs.dtype,
            )
            np.negative(predicted_risks, out=predicted_risks)
            return predicted_risks

        if time_grid.size < 2:
            raise ValueError(
                "At least two time points are required to estimate hazards."
            )
        if np.any(target_times > time_grid[-1]):
            raise ValueError(
                "target_times must not exceed the largest time coordinate "
                "when estimating hazard rates."
            )

        interval_indices = np.searchsorted(time_grid[1:], target_times, side="left")
        interval_indices = np.clip(interval_indices, 0, time_grid.size - 2)
        eps = 1e-12
        left_survival = np.clip(survival_curve[interval_indices], eps, None)
        right_survival = np.clip(survival_curve[interval_indices + 1], eps, None)
        interval_widths = time_grid[interval_indices + 1] - time_grid[interval_indices]
        return -np.log(right_survival / left_survival) / interval_widths

    def predict_interval(
        self,
        quantile_range: tuple[float, float] | None = None,
        cov_level: float | None = None,
    ) -> np.ndarray:
        """
        Predict the survival interval from the predicted survival curve.

        The interval means that the event time will fall between the lower and upper bounds
        with a certain probability (coverage level).

        Parameters
        ----------
        quantile_range : tuple[float, float]
            The lower and upper quantiles to define the prediction interval.
            If `cov_level` is also provided, it must equal the difference
            between the upper and lower quantiles.
        cov_level : float, default=None
            The coverage level to define the prediction interval.
            If `quantile_range` is also provided, it must equal the difference
            between the upper and lower quantiles.

        Returns
        -------
        intervals : ndarray, shape = (n_samples, 2)
            The predicted survival intervals for each sample.
        """
        if (quantile_range is not None) and (cov_level is not None):
            lower_quantile, upper_quantile = quantile_range
            if not 0 < lower_quantile < upper_quantile < 1:
                raise ValueError("Quantiles must be between 0 and 1 and lower < upper.")
            if not 0 < cov_level < 1:
                raise ValueError("Coverage level must be between 0 and 1.")
            if not np.isclose(upper_quantile - lower_quantile, cov_level):
                raise ValueError(
                    "The difference between upper and lower quantiles must "
                    "equal the coverage level."
                )
        elif (quantile_range is not None) and (cov_level is None):
            lower_quantile, upper_quantile = quantile_range
            if not 0 < lower_quantile < upper_quantile < 1:
                raise ValueError("Quantiles must be between 0 and 1 and lower < upper.")
        elif (quantile_range is None) and (cov_level is not None):
            if not 0 < cov_level < 1:
                raise ValueError("Coverage level must be between 0 and 1.")
            lower_quantile = (1 - cov_level) / 2
            upper_quantile = 1 - lower_quantile
        else:
            raise ValueError("Please provide either 'quantile_range' or 'cov_level'.")

        pred_survs, time_coordinates = align_curve_and_time_coordinates(
            self._pred_survs, self._time_coordinates
        )

        interval_pred = survival_to_quantile(
            pred_survs,
            time_coordinates,
            quantile_levels=[lower_quantile, upper_quantile],
            interpolate=self.interpolation,
        )
        return interval_pred

    def _select_plot_inputs(
        self,
        curve_indices: int | list[int],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Select requested samples, keeping shared 1D inputs without broadcasting.

        Return survival curves, their time coordinates and sample labels in
        selection order. Copy only selected rows of sample-specific inputs.

        Parameters
        ----------
        curve_indices : int | list[int]
            Zero-based sample indices to select, in the requested order.

        Returns
        -------
        tuple[ndarray, ndarray, list[str]]
            Selected survival curves, matching time grids, and sample labels in selection order.
        """
        indices = _validate_plot_indices(curve_indices, self._n_samples)
        return (
            self._pred_survs[indices] if self.ndim_surv == 2 else self._pred_survs,
            (
                self._time_coordinates[indices]
                if self.ndim_time == 2
                else self._time_coordinates
            ),
            [f"Sample {i}" for i in indices],
        )

    def plot_survival_curves(
        self,
        curve_indices: int | list[int],
        color: str | list[str] | None = None,
        x_lim: float | tuple[float, float] | None = None,
        y_lim: float | tuple[float, float] | None = None,
        x_label: str = "Time",
        y_label: str = "Survival probability",
        *,
        ax: plt.Axes | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot selected survival curves on the stored prediction grids.

        Parameters
        ----------
        curve_indices : int or list of int
            Zero-based sample indices, plotted in the supplied order.
        color : str or list of str, optional
            One shared color or one color per selected curve. Defaults to the
            Matplotlib color cycle.
        x_lim, y_lim : float or tuple of float, optional
            Axis limits. Survival probabilities default to the range [0, 1.02].
        x_label, y_label : str
            Axis labels, defaulting to Time and Survival probability.
        ax : matplotlib.axes.Axes, optional
            Existing axes to reuse. Otherwise create a new figure.
        **kwargs : Any
            Matplotlib line options, including linewidth, linestyle, alpha and
            label. A label sequence overrides the generated sample labels.

        Returns
        -------
        fig, ax : tuple
            Figure and axes, without displaying the plot. Stored values are
            connected directly; no interpolation or extrapolation is performed.
        """
        curves, times, labels = self._select_plot_inputs(curve_indices)
        return _plot_curves(
            times,
            curves,
            labels,
            title="Predicted survival",
            x_label=x_label,
            y_label=y_label,
            color=color,
            x_lim=(0, float(np.max(times))) if x_lim is None else x_lim,
            y_lim=(0, 1.02) if y_lim is None else y_lim,
            ax=ax,
            **kwargs,
        )

    def plot_quantile_curves(
        self,
        curve_indices: int | list[int],
        color: str | list[str] | None = None,
        x_lim: float | tuple[float, float] | None = None,
        y_lim: float | tuple[float, float] | None = None,
        x_label: str = "Quantile",
        y_label: str = "Time",
        *,
        ax: plt.Axes | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot event-time quantiles for selected samples on the stored grid.

        Parameters
        ----------
        curve_indices : int or list of int
            Zero-based sample indices, plotted in the supplied order.
        color : str or list of str, optional
            One shared color or one color per selected curve.
        x_lim, y_lim : float or tuple of float, optional
            Axis limits. Quantiles default to [0, 1].
        x_label, y_label : str
            Axis labels, defaulting to Quantile and Time.
        ax : matplotlib.axes.Axes, optional
            Existing axes to reuse. Otherwise create a new figure.
        **kwargs : Any
            Matplotlib line options; see plot_survival_curves.

        Returns
        -------
        fig, ax : tuple
            Figure and axes, without displaying the plot. Plot time against
            quantile 1 - S(t) at the stored coordinates. Unreached quantiles are
            not extrapolated, and probabilities are computed only for the
            selected curves.
        """
        curves, times, labels = self._select_plot_inputs(curve_indices)
        return _plot_curves(
            1 - curves,
            times,
            labels,
            title="Predicted event-time quantiles",
            x_label=x_label,
            y_label=y_label,
            color=color,
            x_lim=(0, 1) if x_lim is None else x_lim,
            y_lim=(0, float(np.max(times)) * 1.02) if y_lim is None else y_lim,
            ax=ax,
            **kwargs,
        )
