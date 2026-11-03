import numpy as np

from spareparts.data import classify
from spareparts.forecast import croston, moving_average, sba, ses, tsb
from spareparts.inventory import order_up_to, simulate

Y = np.array([0, 0, 3, 0, 0, 0, 2, 0, 4, 0, 0, 0], float)


def test_forecasts_are_causal():
    """Changing future demand must not change earlier forecasts."""
    changed = Y.copy()
    changed[8:] = 99
    for method in (moving_average, ses, croston, sba, tsb):
        a, b = method(Y), method(changed)
        np.testing.assert_allclose(a[:9], b[:9], err_msg=method.__name__)


def test_croston_and_sba():
    f = croston(Y, alpha=0.5)
    # sizes 3 -> 2.5 -> 3.25; intervals 3 -> 3.5 -> 2.75
    assert np.isclose(f[-1], 3.25 / 2.75)
    assert np.isclose(sba(Y, alpha=0.5)[-1], f[-1] * 0.75)
    assert f[2] == 0  # no forecast before the first demand


def test_tsb_decays_after_demand_stops():
    y = np.concatenate([np.ones(12), np.zeros(36)])
    f = tsb(y)
    assert f[12] > f[24] > f[48] > 0
    assert croston(y)[48] == croston(y)[12]  # Croston keeps forecasting the old rate


def test_classification():
    assert classify(np.array([1, 1, 1, 1, 1, 1], float))["pattern"] == "smooth"
    assert classify(np.array([0, 0, 2, 0, 0, 2, 0, 0, 2], float))["pattern"] == "intermittent"
    assert classify(np.array([0, 0, 1, 0, 0, 9, 0, 0, 2], float))["pattern"] == "lumpy"


def test_order_up_to_rises_with_service_level():
    rate, var = np.array([0.5, 0.5]), np.array([0.5, 2.0])  # Poisson-like and over-dispersed
    lo, hi = order_up_to(rate, var, 0.8), order_up_to(rate, var, 0.99)
    assert (hi >= lo).all() and hi[1] > hi[0]  # more variable demand needs more stock
    assert order_up_to(np.array([0.0]), np.array([0.0]), 0.99)[0] == 0


def test_simulation_bookkeeping():
    demand = np.array([[1, 1, 1, 1, 1, 1]], float)
    full = simulate(demand, np.full((1, 6), 2.0))  # lead time 1: order-up-to 2 covers every month
    assert full["fill_rate"] == 1.0 and full["avg_backorders_per_part"] == 0
    starved = simulate(np.array([[0, 3, 0, 0]], float), np.zeros((1, 4)))
    assert starved["fill_rate"] == 0.0 and starved["avg_backorders_per_part"] > 0
