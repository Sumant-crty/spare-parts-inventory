"""A global XGBoost model trained across all parts (Tweedie loss suits many zeros)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

LAGS = 12


def features(Y: np.ndarray, t: int, months: pd.DatetimeIndex) -> pd.DataFrame:
    """Features for forecasting month t from months < t, one row per part."""
    hist = Y[:, :t]
    obs = ~np.isnan(hist)
    h = np.nan_to_num(hist)
    cols = {f"lag_{k}": Y[:, t - k] if t - k >= 0 else np.full(len(Y), np.nan) for k in range(1, LAGS + 1)}
    f = pd.DataFrame(cols)
    for w in (3, 6, 12, 24):
        f[f"mean_{w}"] = np.nanmean(hist[:, -w:], axis=1) if t >= 1 else np.nan
        f[f"nonzero_{w}"] = (h[:, -w:] > 0).sum(axis=1)
    nz_any = (h > 0).any(axis=1)
    last_nz = np.where(nz_any, t - 1 - np.argmax((h > 0)[:, ::-1], axis=1), -1)
    f["months_since_demand"] = np.where(nz_any, t - last_nz, t + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["mean_size"] = np.where(nz_any, h.sum(axis=1) / (h > 0).sum(axis=1), 0.0)
    f["age_months"] = obs.sum(axis=1)
    f["mean_all"] = np.nanmean(hist, axis=1)
    f["month"] = months[t].month if t < len(months) else (months[-1] + pd.DateOffset(months=t - len(months) + 1)).month
    return f


def training_frame(Y: np.ndarray, months: pd.DatetimeIndex, start: int, end: int):
    X, y = [], []
    for t in range(start, end):
        target = Y[:, t]
        keep = ~np.isnan(target)
        X.append(features(Y, t, months)[keep])
        y.append(target[keep])
    return pd.concat(X, ignore_index=True), np.concatenate(y)


def fit(Y: np.ndarray, months: pd.DatetimeIndex, until: int, first: int = 12) -> xgb.XGBRegressor:
    """Train on targets in months [first, until)."""
    X, y = training_frame(Y, months, first, until)
    model = xgb.XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=1.3,
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        n_jobs=-1,
    )
    return model.fit(X, y)


def predict(model, Y: np.ndarray, t: int, months: pd.DatetimeIndex) -> np.ndarray:
    return model.predict(features(Y, t, months)).clip(min=0)
