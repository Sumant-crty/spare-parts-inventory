"""Demand-rate forecasts for intermittent series, computed in one causal pass.

Each method takes one part's history y (oldest first) and returns f with len(y) + 1 values,
where f[t] is the forecast per month made with y[:t] only (f[0] is NaN). Smoothing constants
use common literature values rather than being tuned, to avoid overfitting short series.
"""

from __future__ import annotations

import numpy as np

ALPHA = 0.1  # smoothing for level / demand size
BETA = 0.05  # smoothing for demand probability in TSB


def zero(y: np.ndarray) -> np.ndarray:
    f = np.zeros(len(y) + 1)
    f[0] = np.nan
    return f


def naive(y: np.ndarray) -> np.ndarray:
    return np.concatenate([[np.nan], y])


def moving_average(y: np.ndarray, window: int = 12) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(y)])
    t = np.arange(1, len(y) + 1)
    lo = np.maximum(0, t - window)
    return np.concatenate([[np.nan], (c[t] - c[lo]) / (t - lo)])


def ses(y: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    f = np.full(len(y) + 1, np.nan)
    level = y[0]
    f[1] = level
    for t in range(1, len(y)):
        level += alpha * (y[t] - level)
        f[t + 1] = level
    return f


def croston(y: np.ndarray, alpha: float = ALPHA, sba: bool = False) -> np.ndarray:
    """Croston: smooth non-zero sizes and the intervals between them. sba applies (1 - alpha/2)."""
    f = np.full(len(y) + 1, np.nan)
    size = interval = None
    last = None
    for t, v in enumerate(y):
        if v > 0:
            if size is None:
                size, interval = float(v), float(t + 1)
            else:
                size += alpha * (v - size)
                interval += alpha * ((t - last) - interval)
            last = t
        rate = 0.0 if size is None else size / interval
        f[t + 1] = rate * (1 - alpha / 2) if sba else rate
    return f


def sba(y: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    return croston(y, alpha, sba=True)


def tsb(y: np.ndarray, alpha: float = ALPHA, beta: float = BETA) -> np.ndarray:
    """Teunter-Syntetos-Babai: updates the demand probability every month, so the forecast
    decays when a part stops selling (obsolescence), unlike Croston."""
    f = np.full(len(y) + 1, np.nan)
    prob, size = None, None
    for t, v in enumerate(y):
        if size is None:
            if v > 0:
                size, prob = float(v), 1.0 / (t + 1)
        else:
            if v > 0:
                prob += beta * (1 - prob)
                size += alpha * (v - size)
            else:
                prob += beta * (0 - prob)
        f[t + 1] = 0.0 if size is None else prob * size
    return f


METHODS = {
    "zero": zero,
    "naive": naive,
    "moving average (12 m)": moving_average,
    "exponential smoothing": ses,
    "Croston": croston,
    "SBA": sba,
    "TSB": tsb,
}


def forecast_matrix(Y: np.ndarray, method) -> np.ndarray:
    """parts x (months + 1) one-step-ahead forecasts."""
    return np.vstack([method(row) for row in Y])
