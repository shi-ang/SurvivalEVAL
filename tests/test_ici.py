import lifelines
from lifelines import CoxPHFitter
from matplotlib import pyplot as plt

from SurvivalEVAL.Evaluations.SingleTimeCalibration import (
    integrated_calibration_index,
    one_calibration,
)

# load data
data = lifelines.datasets.load_gbsg2()
# preprocessing
data.rename(columns={"cens": "event"}, inplace=True)
data["horTh"] = data["horTh"].map({"no": 0, "yes": 1})
data["menostat"] = data["menostat"].map({"Pre": 0, "Post": 1})
data["tgrade"] = data["tgrade"].map({"I": 1, "II": 2, "III": 3})
# randomly divide the data into training and validation sets
df_train = data.sample(frac=0.7, random_state=42)  # 70% for training
df_train = df_train.reset_index(drop=True)
df_test = data.drop(df_train.index)  # remaining 30% for testing
df_test = df_test.reset_index(drop=True)
x_test = df_test.drop(columns=["time", "event"]).values

cph = CoxPHFitter()
cph.fit(df_train, duration_col="time", event_col="event")

year = 1
survs_cox = cph.predict_survival_function(x_test, times=[365 * year]).T.values.flatten()
p, hl_statistics, obs_probs, exp_probs = one_calibration(
    preds=1 - survs_cox,
    event_time=df_test["time"].values,
    event_indicator=df_test["event"].values,
    target_time=365 * year,
    num_bins=10,
    binning_strategy="H",
    method="DN",
)

ici_summary, ici_fig = integrated_calibration_index(
    preds=1 - survs_cox,
    event_time=df_test["time"].values,
    event_indicator=df_test["event"].values,
    target_time=365 * year,
    draw_figure=True,
)
print(ici_summary)
if ici_fig is not None:
    plt.show()
