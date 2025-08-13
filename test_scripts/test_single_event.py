#%% Load libraries
from lifelines import NelsonAalenFitter
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

from SurvivalEVAL.NonparametricEstimator.SingleEvent import (KaplanMeier, NelsonAalen,
                                                             CopulaGraphic, TurnbullEstimator)


#%% Test the Nelson-Aalen estimator and compare with lifelines implementation
np.random.seed(42)
n_samples = 1000
event_times = np.random.exponential(scale=10, size=n_samples)
event_indicators = np.random.binomial(n=1, p=0.7, size=n_samples)
event_indicators = (event_indicators >= 0.5).astype(int)

# create the Nelson-Aalen estimator
na_estimator = NelsonAalen(event_times, event_indicators)
# create the lifelines Nelson-Aalen fitter
na_fitter = NelsonAalenFitter()
na_fitter.fit(event_times, event_indicators)

# compare the cumulative hazard functions
times = np.linspace(0, 30, 100)
na_cumulative_hazard = na_estimator.predict(times)
lifelines_cumulative_hazard = na_fitter.cumulative_hazard_at_times(times).values

# make some predictions at random times
random_times = np.random.uniform(0, 100, size=10)
na_predictions = na_estimator.predict(random_times)
lifelines_predictions = na_fitter.predict(random_times)

mse = np.mean((na_cumulative_hazard - lifelines_cumulative_hazard) ** 2)

#%% Test the Copula Graphical estimator
times = np.array([1, 3, 5, 4, 4, 7, 8, 10, 13, 15])
events = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 1])
clayton_estimator = CopulaGraphic(event_times, event_indicators, alpha=18, type="Clayton")

gumbel_estimator = CopulaGraphic(times, events, alpha=18, type="Gumbel")

frank_estimator = CopulaGraphic(times, events, alpha=18, type="Frank")


#%% Test the Turnbull estimator
os.chdir("../..")
data = pd.read_csv("data/breast.csv")
data.right = data.right.fillna(np.inf)

# group1
data1 = data.loc[data["ther"] == 1].copy()
tb1 = TurnbullEstimator().fit(data1.left.values, data1.right.values)

# group2
data2 = data.loc[data["ther"] == 0].copy()
tb2 = TurnbullEstimator().fit(data2.left.values, data2.right.values)

# plotting
plt.figure(figsize=(7, 5))
plt.step(tb1.survival_times_, tb1.survival_probabilities_, where="post", linestyle="-",
         label="Radiotherapy (intervals)")
plt.step(tb2.survival_times_, tb2.survival_probabilities_, where="post", linestyle="-",
         label="Radio + Chemo (intervals)")
plt.xlabel("Time")
plt.ylabel("S(t)")
plt.legend()
plt.title("Turnbull Interval-Censored Survival")
plt.tight_layout()
plt.show()

# compare midpoint-based KM with Turnbull
# Midpoints:
p_mid = data["left"].to_numpy(float) + (data["right"].to_numpy(float) - data["left"].to_numpy(float)) / 2.0
finite_mid = np.isfinite(p_mid)
pm = np.where(finite_mid, p_mid, data["left"].to_numpy(float))
cens = finite_mid.astype(int)  # 1 == event, 0 == right-censored

# KM by group
km1 = KaplanMeier(pm[data["ther"] == 1], cens[data["ther"] == 1])
km0 = KaplanMeier(pm[data["ther"] == 0], cens[data["ther"] == 0])
times1, surv1 = km1.survival_times, km1.survival_probabilities
times0, surv0 = km0.survival_times, km0.survival_probabilities

plt.figure(figsize=(7, 5))
# Interval-censored (solid)
plt.step(tb1.survival_times_, tb1.survival_probabilities_, where="post", linestyle="-",
         label="Radiotherapy (intervals)")
plt.step(tb2.survival_times_, tb2.survival_probabilities_, where="post", linestyle="-",
         label="Radio + Chemo (intervals)")
# Midpoint-based KM (dashed)
if times1.size:
    plt.step(times1, surv1, where="post", linestyle="--", label="Radiotherapy (midpoints)")
if times0.size:
    plt.step(times0, surv0, where="post", linestyle="--", label="Radio + Chemo (midpoints)")

plt.xlabel("Time")
plt.ylabel("S(t)")
plt.legend()
plt.title("Interval-Censored (Turnbull) vs Midpoint KM")
plt.tight_layout()
plt.show()
