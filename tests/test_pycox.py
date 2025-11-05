import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import torch
from pycox.datasets import metabric
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from SurvivalEVAL.Evaluator import PycoxEvaluator


def _build_preprocessor():
    numeric_cols = ["x0", "x1", "x2", "x3", "x8"]
    categorical_cols = ["x4", "x5", "x6", "x7"]

    ohe = OneHotEncoder(handle_unknown="ignore")
    if hasattr(ohe, "sparse_output"):
        ohe.sparse_output = False
    else:
        ohe.sparse = False

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", ohe, categorical_cols),
        ]
    )


def _prepare_features(df: pd.DataFrame, preprocessor=None):
    if preprocessor is None:
        preprocessor = _build_preprocessor()
        features = preprocessor.fit_transform(df)
    else:
        features = preprocessor.transform(df)
    return features.astype(np.float32), preprocessor


def _make_discrete_targets(durations: np.ndarray, events: np.ndarray, cuts: np.ndarray):
    idx = np.searchsorted(cuts, durations, side="right")
    idx = np.clip(idx, 0, cuts.size - 1)
    return idx.astype(np.int64), events.astype(np.float32)


def _logistic_hazard_loss(
    logits: torch.Tensor, idx: torch.Tensor, events: torch.Tensor
) -> torch.Tensor:
    hazards = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    log_survival = torch.log(1.0 - hazards)
    log_hazard = torch.log(hazards)

    cum_log_survival = torch.cumsum(log_survival, dim=1)
    gather_idx = idx.unsqueeze(1)
    sum_including = cum_log_survival.gather(1, gather_idx).squeeze(1)

    sum_prev = torch.zeros_like(sum_including)
    idx_positive = idx > 0
    if idx_positive.any():
        previous_idx = idx[idx_positive] - 1
        sum_prev[idx_positive] = cum_log_survival[idx_positive, previous_idx]

    loss = torch.zeros_like(sum_including)
    event_mask = events.bool()
    if event_mask.any():
        loss[event_mask] = -(
            sum_prev[event_mask] + log_hazard[event_mask, idx[event_mask]]
        )
    if (~event_mask).any():
        loss[~event_mask] = -sum_including[~event_mask]

    return loss.mean()


def _train_logistic_hazard(
    x_train, idx_train, events_train, cuts, epochs=40, batch_size=128
):
    device = torch.device("cpu")
    torch.manual_seed(123)
    input_dim = x_train.shape[1]
    num_durations = cuts.size

    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, num_durations),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(idx_train),
        torch.from_numpy(events_train),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_idx, batch_events in loader:
            batch_x = batch_x.to(device)
            batch_idx = batch_idx.to(device)
            batch_events = batch_events.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = _logistic_hazard_loss(logits, batch_idx, batch_events)
            loss.backward()
            optimizer.step()

    return model


def _predict_survival(
    model: nn.Module, x: np.ndarray, cuts: np.ndarray
) -> pd.DataFrame:
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        hazards = torch.sigmoid(logits).cpu().numpy()
    survival = np.cumprod(1.0 - hazards, axis=1)
    survival = np.concatenate(
        [np.ones((survival.shape[0], 1), dtype=survival.dtype), survival], axis=1
    )
    time_grid = np.concatenate([[0.0], cuts])
    return pd.DataFrame(survival.T, index=time_grid)


@pytest.fixture(scope="module")
def trained_pycox_evaluator():
    df = metabric.read_df().sample(n=1200, random_state=42)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    x_train, preprocessor = _prepare_features(df_train)
    x_test, _ = _prepare_features(df_test, preprocessor=preprocessor)

    durations_train = df_train["duration"].to_numpy(float)
    events_train = df_train["event"].to_numpy(int)
    durations_test = df_test["duration"].to_numpy(float)
    events_test = df_test["event"].to_numpy(int)

    num_durations = 20
    cuts = np.linspace(
        durations_train.min(), durations_train.max(), num_durations + 1, endpoint=True
    )[1:]

    idx_train, events_train_disc = _make_discrete_targets(
        durations_train, events_train, cuts
    )
    _make_discrete_targets(durations_test, events_test, cuts)

    model = _train_logistic_hazard(
        x_train,
        idx_train,
        events_train_disc,
        cuts=cuts,
        epochs=40,
        batch_size=128,
    )

    surv_test_df = _predict_survival(model, x_test, cuts)

    evaluator = PycoxEvaluator(
        surv=surv_test_df,
        event_times=durations_test,
        event_indicators=events_test,
        train_event_times=durations_train,
        train_event_indicators=events_train,
        predict_time_method="RMST",
    )

    return {
        "evaluator": evaluator,
        "cuts": cuts,
        "durations_test": durations_test,
        "events_test": events_test,
    }


def test_pycox_concordance_and_brier(trained_pycox_evaluator):
    evaluator = trained_pycox_evaluator["evaluator"]
    durations_test = trained_pycox_evaluator["durations_test"]

    c_index, concordant_pairs, total_pairs = evaluator.concordance(method="Margin")
    assert 0.0 <= c_index <= 1.0
    assert concordant_pairs <= total_pairs

    brier = evaluator.brier_score(
        target_time=float(np.median(durations_test)), IPCW_weighted=True
    )
    assert np.isfinite(brier)


def test_pycox_integrated_scores(trained_pycox_evaluator):
    evaluator = trained_pycox_evaluator["evaluator"]

    ibs = evaluator.integrated_brier_score(num_points=15, IPCW_weighted=True)
    assert np.isfinite(ibs)

    mae_margin = evaluator.mae(method="Margin")
    assert mae_margin >= 0.0


def test_pycox_calibration(trained_pycox_evaluator):
    evaluator = trained_pycox_evaluator["evaluator"]
    durations_test = trained_pycox_evaluator["durations_test"]

    target_time = float(np.quantile(durations_test, 0.5))
    p_value_one, _, _ = evaluator.one_calibration(target_time=target_time, method="DN")
    assert 0.0 <= p_value_one <= 1.0

    p_value_d, _ = evaluator.d_calibration()
    assert 0.0 <= p_value_d <= 1.0
