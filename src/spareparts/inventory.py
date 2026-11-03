"""Order-up-to levels from demand forecasts, and an inventory simulation to score them.

Policy: periodic review every month (R = 1) with lead time L months. At each review the
order-up-to level S covers demand over the protection period L + R at a target cycle service
level. Unmet demand is backordered.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, poisson

LEAD_TIME = 1
REVIEW = 1
HOLDING = 0.25 / 12  # holding cost per unit-month, as a share of unit value
BACKORDER = 1.0  # cost per unit backordered per month, as a share of unit value


def error_variance(y: np.ndarray, f: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Smoothed squared one-step error per part, available at each month (causal).

    y: parts x T demand, f: parts x (T + 1) forecasts. Returns parts x (T + 1).
    """
    n, T = y.shape
    v = np.full((n, T + 1), np.nan)
    cur = np.nanmean(y[:, :12], axis=1) + 0.25  # start at roughly the mean (Poisson-like)
    for t in range(T):
        v[:, t] = cur
        e = y[:, t] - f[:, t]
        ok = ~np.isnan(e)
        cur = np.where(ok, cur + alpha * (e**2 - cur), cur)
    v[:, T] = cur
    return v


def order_up_to(rate, variance, service: float, protection: int = LEAD_TIME + REVIEW) -> np.ndarray:
    """Quantile of protection-period demand: negative binomial if over-dispersed, else Poisson."""
    m = np.maximum(np.asarray(rate, float), 0) * protection
    v = np.maximum(np.asarray(variance, float) * protection, m)
    S = np.zeros_like(m)
    pos = m > 1e-9
    over = pos & (v > m * 1.0001)
    pois = pos & ~over
    S[pois] = poisson.ppf(service, m[pois])
    n_ = m[over] ** 2 / (v[over] - m[over])
    p_ = m[over] / v[over]
    S[over] = nbinom.ppf(service, n_, p_)
    return S


def empirical_order_up_to(y_hist: np.ndarray, service: float, window: int = 24, protection: int = LEAD_TIME + REVIEW):
    """Quantile of past protection-period demand totals (no distribution assumed)."""
    h = y_hist[:, -window:]
    sums = np.stack([h[:, i : i + protection].sum(axis=1) for i in range(h.shape[1] - protection + 1)], axis=1)
    return np.ceil(np.quantile(sums, service, axis=1, method="higher"))


def simulate(demand: np.ndarray, S: np.ndarray, lead_time: int = LEAD_TIME) -> dict:
    """Simulate an order-up-to policy. demand, S: parts x months (S known at the start of each month)."""
    n, T = demand.shape
    net = S[:, 0].copy()  # start with stock at the first order-up-to level
    pipeline = np.zeros((n, T + lead_time + 1))
    served = short = 0.0
    on_hand = np.zeros(n)
    backorders = np.zeros(n)
    orders = 0
    for t in range(T):
        net += pipeline[:, t]
        position = net + pipeline[:, t + 1 : t + lead_time + 1].sum(axis=1)
        q = np.maximum(0, S[:, t] - position)
        pipeline[:, t + lead_time] += q
        orders += int((q > 0).sum())
        d = demand[:, t]
        s = np.minimum(np.maximum(net, 0), d)
        served += s.sum()
        short += (d - s).sum()
        net -= d
        on_hand += np.maximum(net, 0)
        backorders += np.maximum(-net, 0)
    total = demand.sum()
    months = n * T
    on_hand_sum, backorder_sum = on_hand.sum(), backorders.sum()
    return {
        "fill_rate": float(served / total) if total else 1.0,
        "avg_stock_per_part": float(on_hand_sum / months),
        "avg_backorders_per_part": float(backorder_sum / months),
        "orders_per_part_month": orders / months,
        "cost_per_part_month": float((HOLDING * on_hand_sum + BACKORDER * backorder_sum) / months),
        "_per_part": {"stock": on_hand / T, "backorders": backorders / T},  # for bootstrap intervals; not reported
    }


def cost_with(result: dict, holding: float, backorder: float) -> float:
    return holding * result["avg_stock_per_part"] + backorder * result["avg_backorders_per_part"]
