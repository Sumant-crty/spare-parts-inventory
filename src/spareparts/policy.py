"""Serving: reorder recommendations from a part's monthly history (used by the bundle and the API).

The deployed model is trained on all history, so its errors on that history are in-sample and
too optimistic. The error variance is therefore multiplied by `inflation`, the ratio of
out-of-sample to in-sample squared error measured during evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import global_model
from .forecast import tsb
from .inventory import error_variance, order_up_to

MIN_MONTHS = 13  # the global model needs 12 lags; shorter histories fall back to TSB


def recommend_batch(
    model, Y: np.ndarray, start: pd.Timestamp, service: float, protection: int, inflation: float
) -> pd.DataFrame:
    """Recommendations for the month after the history, one row per part (all histories equal length)."""
    n, T = Y.shape
    if T >= MIN_MONTHS:
        months = pd.date_range(start, periods=T, freq="MS")
        f = np.full((n, T + 1), np.nan)
        for t in range(12, T + 1):
            f[:, t] = global_model.predict(model, Y, t, months)
        variance = error_variance(Y, f)[:, T] * inflation
        method = "XGBoost (global, Tweedie)"
    else:
        f = np.vstack([tsb(row) for row in Y])
        variance = error_variance(Y, f)[:, T]
        method = "TSB (short history)"
    rate = f[:, T]
    S = order_up_to(rate, variance, service, protection)
    return pd.DataFrame(
        {"forecast_per_month": rate, "order_up_to": S, "safety_stock": S - rate * protection, "method": method}
    )
