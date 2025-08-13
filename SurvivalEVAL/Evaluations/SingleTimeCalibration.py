import numpy as np
import pandas as pd
from scipy.stats import chi2
from lifelines import CoxPHFitter
from patsy import dmatrix  # For spline basis matrix
import matplotlib.pyplot as plt  # For plotting

from SurvivalEVAL.Evaluations.custom_types import Numeric, NumericArrayLike
from SurvivalEVAL.Evaluations.util import check_and_convert, predict_prob_from_curve
from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeier, TurnbullEstimator


def one_calibration(
        preds: np.ndarray,
        event_time: np.ndarray,
        event_indicator: np.ndarray,
        target_time: Numeric,
        num_bins: int = 10,
        binning_strategy: str = "C",
        method: str = "DN"
) -> (float, list, list):
    """
    Compute the one calibration score for a given set of predictions and true event times.

    Parameters
    ----------
    preds: np.ndarray
        The predicted probabilities of experiencing the event at the time of interest.
    event_time: np.ndarray
        The true event times.
    event_indicator: np.ndarray
        The indicator of whether the event is observed or not.
    target_time: Numeric
        The time of interest.
    num_bins: int
        The number of bins to divide the predictions into.
    binning_strategy: str
        The strategy to bin the predictions. The options are: "C" (default), and "H".
        C-statistics means the predictions are divided into equal-sized bins based on the predicted probabilities.
        H-statistics means the predictions are divided into equal-increment bins from 0 to 1.
    method: str
        The method to handle censored patients. The options are: "DN" (default), and "Uncensored".
        "Uncensored" method simply removes the censored patients, and uses the standard Hosmer-Lemeshow test.
        "DN" method uses the D'Agostino-Nam method, which uses the Kaplan-Meier estimate of the survival function
        to compute the average observed probabilities in each bin.

    Returns
    -------
    score: float
        The one calibration score.
    observed_probabilities: list
        The observed probabilities in each bin.
    expected_probabilities: list
        The expected probabilities in each bin.
    """
    if binning_strategy == "C":
        sorted_idx = np.argsort(-preds)
        sorted_predictions = preds[sorted_idx]
        sorted_event_time = event_time[sorted_idx]
        sorted_event_indicator = event_indicator[sorted_idx]

        binned_event_time = np.array_split(sorted_event_time, num_bins)
        binned_event_indicator = np.array_split(sorted_event_indicator, num_bins)
        binned_predictions = np.array_split(sorted_predictions, num_bins)
    elif binning_strategy == "H":
        # Create bins from 0 to 1 with equal increments
        bin_edges = np.linspace(0, 1, num_bins + 1)
        binned_event_time = []
        binned_event_indicator = []
        binned_predictions = []

        for i in range(num_bins):
            # Get the indices of predictions that fall into the current bin
            bin_mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
            binned_event_time.append(event_time[bin_mask])
            binned_event_indicator.append(event_indicator[bin_mask])
            binned_predictions.append(preds[bin_mask])
    else:
        error = "Please enter one of 'C','H' for binning_strategy."
        raise TypeError(error)

    hl_statistics = 0
    observed_probabilities = []
    expected_probabilities = []

    for b in range(num_bins):
        # mean_prob = np.mean(binned_predictions[b])
        bin_size = len(binned_event_time[b])

        if bin_size == 0:
            # This is for H-statistics binning strategy,
            # If a bin has no data, skip it
            continue

        # For Uncensored method, we simply remove the censored patients,
        # for D'Agostina-Nam method, we will use 1-KM(t) as the observed probability.
        if method == "Uncensored":
            filter_idx = ~((binned_event_time[b] < target_time) & (binned_event_indicator[b] == 0))
            mean_prob = np.mean(binned_predictions[b][filter_idx])
            event_count = sum(binned_event_time[b][filter_idx] < target_time)
            event_probability = event_count / bin_size
            hl_statistics += (event_count - bin_size * mean_prob) ** 2 / (
                    bin_size * mean_prob * (1 - mean_prob))
        elif method == "DN":
            mean_prob = np.mean(binned_predictions[b])
            km_model = KaplanMeier(binned_event_time[b], binned_event_indicator[b])
            event_probability = 1 - km_model.predict(target_time)
            hl_statistics += (bin_size * event_probability - bin_size * mean_prob) ** 2 / (bin_size * mean_prob * (1 - mean_prob))
        else:
            error = "Please enter one of 'Uncensored','DN' for method."
            raise TypeError(error)
        observed_probabilities.append(event_probability)
        expected_probabilities.append(mean_prob)

    # recalculate the number of bins as the number of bins with data
    num_bins = len(observed_probabilities)
    degree_of_freedom = num_bins - 1 if (num_bins <= 15 and method == "DN") else num_bins - 2
    if degree_of_freedom <= 0:
        raise ValueError("The number of bins is too small to calculate the p-value. "
                         "Please increase the number of bins or check your data.")
    p_value = 1 - chi2.cdf(hl_statistics, degree_of_freedom)

    return p_value, observed_probabilities, expected_probabilities


def one_cal_interval_cen(
        preds: np.ndarray,
        left_limits: np.ndarray,
        right_limits: np.ndarray,
        target_time: Numeric,
        num_bins: int = 10,
        binning_strategy: str = "C",
        method: str = "Turnbull"
) -> (float, list, list):
    """
    Compute the one calibration score for a given set of predictions and true event times.
    Parameters
    ----------
    preds: np.ndarray
        The predicted probabilities of experiencing the event at the time of interest.
    left_limits: np.ndarray
        The left limits of the interval-censored event times.
    right_limits: np.ndarray
        The right limits of the interval-censored event times.
    target_time: Numeric
        The time of interest.
    num_bins: int
        The number of bins to divide the predictions into.
    binning_strategy: str
        The strategy to bin the predictions. The options are: "C" (default), and "H".
        C-statistics means the predictions are divided into equal-sized bins based on the predicted probabilities.
        H-statistics means the predictions are divided into equal-increment bins from 0 to 1.
    method: str
        The method to handle censored patients. The options are: "Turnbull" (default), and "MidPoint".
        "MidPoint" method simply treats the midpoint of the interval as the event time, and
        uses the DN's method (Kaplan-Meier estimate of the survival function).
        "Turnbull" method uses the Turnbull estimator for the survival function
        to compute the average observed probabilities in each bin.
    Returns
    -------
    score: float
        The one calibration score.
    observed_probabilities: list
        The observed probabilities in each bin.
    expected_probabilities: list
        The expected probabilities in each bin.
    """
    if binning_strategy == "C":
        sorted_idx = np.argsort(-preds)
        sorted_predictions = preds[sorted_idx]
        sorted_left = left_limits[sorted_idx]
        sorted_right = right_limits[sorted_idx]

        binned_left = np.array_split(sorted_left, num_bins)
        binned_right = np.array_split(sorted_right, num_bins)
        binned_predictions = np.array_split(sorted_predictions, num_bins)
    elif binning_strategy == "H":
        # Create bins from 0 to 1 with equal increments
        bin_edges = np.linspace(0, 1, num_bins + 1)
        binned_left = []
        binned_right = []
        binned_predictions = []

        for i in range(num_bins):
            # Get the indices of predictions that fall into the current bin
            bin_mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
            binned_left.append(left_limits[bin_mask])
            binned_right.append(right_limits[bin_mask])
            binned_predictions.append(preds[bin_mask])
    else:
        error = "Please enter one of 'C','H' for binning_strategy."
        raise TypeError(error)

    hl_statistics = 0
    observed_probabilities = []
    expected_probabilities = []

    for b in range(num_bins):
        bin_size = len(binned_predictions[b])

        if bin_size == 0:
            # This is for H-statistics binning strategy,
            # If a bin has no data, skip it
            continue

        l_limits = np.array(binned_left[b])
        r_limits = np.array(binned_right[b])
        mean_prob = np.mean(binned_predictions[b])

        if method == "MidPoint":
            mid = l_limits + (r_limits - l_limits) / 2.0
            finite_mid = np.isfinite(mid)
            event_times = np.where(finite_mid, mid, left_limits)
            event_indicators = finite_mid.astype(int)

            km_model = KaplanMeier(event_times, event_indicators)
            event_probability = 1 - km_model.predict(target_time)
        elif method == "Turnbull":
            tb = TurnbullEstimator().fit(left_limits, right_limits)
            event_probability = 1 - tb.predict(target_time)
        else:
            error = "Please enter one of 'MidPoint','Turnbull' for method."
            raise TypeError(error)
        hl_statistics += (bin_size * event_probability - bin_size * mean_prob) ** 2 / (
                    bin_size * mean_prob * (1 - mean_prob))

        observed_probabilities.append(event_probability)
        expected_probabilities.append(mean_prob)

    # recalculate the number of bins as the number of bins with data
    num_bins = len(observed_probabilities)
    degree_of_freedom = num_bins - 1 if num_bins <= 15 else num_bins - 2
    if degree_of_freedom <= 0:
        raise ValueError("The number of bins is too small to calculate the p-value. "
                         "Please increase the number of bins or check your data.")
    p_value = 1 - chi2.cdf(hl_statistics, degree_of_freedom)

    return p_value, observed_probabilities, expected_probabilities


def integrated_calibration_index(
        preds: NumericArrayLike,
        event_time: NumericArrayLike,
        event_indicator: NumericArrayLike,
        target_time: Numeric,
        knots: int = 3,
        draw_figure: bool = False,
        figure_range: tuple = None,
) -> (dict, plt.figure):
    """
    Compute the Integrated Calibration Index (ICI) for a given set of predictions and true event times.
    The method is presented in [1]. The implementation is based on the R code available in Appendix A of [1].

    We choose the implementation using splines + CoxPH (instead of the hazard regression) because
    (1) the two methods can compariable performance and the difference is negligible (support by the paper)
    (2) as far as I know, there is no implementation of flexible adaptive hazard regression in Python

    [1] Austin et al., Graphical calibration curves and the integrated calibration index (ICI) for survival models.
    Stat Med. 2020

    Parameters
    ----------
    preds: NumericArrayLike
        The predicted probabilities of experiencing the event at the time of interest.
    event_time: NumericArrayLike
        The true event times.
    event_indicator: NumericArrayLike
        The indicator of whether the event is observed or not.
    target_time: Numeric
        The time of interest for calibration.
    knots: int
        The number of knots to use for the spline basis. Default is 3.
        Austin et al. (2020) [1] compared 3-5 knots and found that 3 knots is best.
    draw_figure: bool
        Whether to plot the graphical calibration curve and return the plot. Default is False.
    figure_range: tuple
        The range of the x-axis and y-axis for the plot.
        It should be a tuple of the form (x_min, x_max, y_min, y_max).
        If None, it will be set to the range of predicted survival probabilities.
        Default is None.
    Returns
    -------
    summary: dict
        A dictionary containing the integrated calibration index (ICI), E50, E90, E_max, and the information about the calibration curve.
    plot: plt.figure
        The plot of the graphical calibration curve if make_plot is True, otherwise None.
    """
    preds, event_time, event_indicator = check_and_convert(preds, event_time, event_indicator)
    # get cdfs and cumulative log-log (CLL) values
    pred_clls = np.log(-np.log(1 - preds))

    spline = dmatrix(f"bs(x, df={knots}, include_intercept=False)", {"x": pred_clls}, return_type='dataframe')
    fit_info = spline.design_info
    df = pd.concat([pd.Series(event_time, name='time'), pd.Series(event_indicator, name='event'), spline], axis=1)
    # these model-based estimates are used as the value of observed risks
    cal_fitter = CoxPHFitter().fit(df, duration_col='time', event_col='event')

    # these model-based estimates are used as the value of observed risks
    cal_pred = 1 - cal_fitter.predict_survival_function(spline, times=[target_time]).T.values.flatten()
    abs_err = np.abs(preds - cal_pred)
    ici = abs_err.mean()
    e50 = np.median(abs_err)
    e90 = np.quantile(abs_err, 0.9)
    e_max = np.max(abs_err)
    summary = {
        "ICI": ici,
        "E50": e50,
        "E90": e90,
        "E_max": e_max
    }

    grid = np.linspace(np.quantile(preds, 0.01), np.quantile(preds, 0.99), 100)
    grid_cll = np.log(-np.log(1 - grid))

    spline_grid = dmatrix(fit_info, {"x": grid_cll}, return_type='dataframe')
    cal_pred = 1 - cal_fitter.predict_survival_function(spline_grid, times=[target_time]).T.values.flatten()

    summary["curve"] = {
        "grid": grid,
        "cal_pred": cal_pred,
    }

    if draw_figure:
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(grid, cal_pred, label='Calibration Curve', color='blue')
        ax.plot(grid, grid, label='Perfect Calibration', linestyle='--', color='grey')
        ax.set_xlabel('Predicted Survival Probability')
        ax.set_ylabel('Observed Survival Probability')
        ax.set_title('Graphical Calibration Curve')
        ax.legend()
        if figure_range is not None:
            ax.set_xlim(figure_range[0], figure_range[1])
            ax.set_ylim(figure_range[2], figure_range[3])
    else:
        fig = None

    return summary, fig


if __name__ == "__main__":
    import lifelines

    # load data
    data = lifelines.datasets.load_gbsg2()
    # preprocessing
    data.rename(columns={'cens': 'event'}, inplace=True)
    data['horTh'] = data['horTh'].map({'no': 0, 'yes': 1})
    data['menostat'] = data['menostat'].map({'Pre': 0, 'Post': 1})
    data['tgrade'] = data['tgrade'].map({'I': 1, 'II': 2, 'III': 3})
    # randomly divide the data into training and validation sets
    df_train = data.sample(frac=0.7, random_state=42)  # 70% for training
    df_train = df_train.reset_index(drop=True)
    df_test = data.drop(df_train.index)  # remaining 30% for testing
    df_test = df_test.reset_index(drop=True)
    x_test = df_test.drop(columns=['time', 'event']).values

    cph = CoxPHFitter()
    cph.fit(df_train, duration_col='time', event_col='event')

    year = 1
    survs_cox = cph.predict_survival_function(x_test, times=[365 * year]).T.values.flatten()
    p, obs_probs, exp_probs = one_calibration(
        preds=1-survs_cox,
        event_time=df_test['time'].values,
        event_indicator=df_test['event'].values,
        target_time=365 * year,
        num_bins=10,
        binning_strategy="H",
        method="DN"
    )

    ici_summary, ici_fig = integrated_calibration_index(
        preds=1-survs_cox,
        event_time=df_test['time'].values,
        event_indicator=df_test['event'].values,
        target_time=365 * year,
        draw_figure=True,)
    print(ici_summary)
    if ici_fig is not None:
        ici_fig.show()
