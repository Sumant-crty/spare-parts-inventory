"""Forecast spare-parts demand, set reorder levels, simulate inventory and compare policies.

Usage: python -m spareparts.train
"""

from __future__ import annotations

import json
import pathlib
import time

import joblib
import numpy as np
import pandas as pd

from . import __version__, global_model
from .data import classify, load_series
from .forecast import METHODS, forecast_matrix
from .inventory import (
    BACKORDER,
    HOLDING,
    LEAD_TIME,
    REVIEW,
    cost_with,
    empirical_order_up_to,
    error_variance,
    order_up_to,
    simulate,
)
from .policy import recommend_batch

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
VAL = range(27, 39)  # April 2000 - March 2001: choose service levels here
TEST = range(39, 51)  # April 2001 - March 2002: report results here
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995]
COVER = [0.5, 1, 1.5, 2, 3, 4, 6, 9, 12, 18]  # months of average demand for the rule of thumb
P = LEAD_TIME + REVIEW


def accuracy(Y: np.ndarray, F: np.ndarray, period: range) -> dict:
    """Forecast errors scaled per part, then averaged over parts (zero-demand parts excluded)."""
    t = list(period)
    y, f = Y[:, t], F[:, t]
    scale = Y[:, : t[0]].mean(axis=1)
    naive_sq = np.mean(np.diff(Y[:, : t[0]], axis=1) ** 2, axis=1)
    ok = (scale > 0) & (naive_sq > 0)
    mae = np.mean(np.abs(f - y), axis=1)[ok] / scale[ok]
    bias = np.mean(f - y, axis=1)[ok] / scale[ok]
    rmsse = np.sqrt(np.mean((f - y) ** 2, axis=1)[ok] / naive_sq[ok])
    return {"scaled_mae": float(mae.mean()), "scaled_bias": float(bias.mean()), "rmsse": float(rmsse.mean())}


def run_policy(Y, S_fn, period: range) -> dict:
    t = list(period)
    S = np.column_stack([S_fn(i) for i in t])
    return simulate(Y[:, t], S)


def choose(results: dict, holding: float, backorder: float) -> float:
    return min(results, key=lambda k: cost_with(results[k], holding, backorder))


def run() -> dict:
    t0 = time.time()
    REPORTS.mkdir(exist_ok=True)
    raw = load_series()
    complete = raw.notna().all(axis=1)
    Ydf = raw[complete]
    Y, months = Ydf.to_numpy(), Ydf.columns
    patterns = pd.Series([classify(r[: TEST.start])["pattern"] for r in Y], index=Ydf.index)
    print(f"{len(Y)} parts with complete history ({(~complete).sum()} dropped: series end early)")

    # --- forecasts (one-step-ahead, causal)
    F = {name: forecast_matrix(Y, m) for name, m in METHODS.items()}
    xg = np.full((len(Y), Y.shape[1] + 1), np.nan)
    m_val = global_model.fit(Y, months, until=VAL.start)
    for t in range(12, TEST.start):
        xg[:, t] = global_model.predict(m_val, Y, t, months)  # months before VAL.start are in-sample warm-up
    m_test = global_model.fit(Y, months, until=TEST.start)
    for t in TEST:
        xg[:, t] = global_model.predict(m_test, Y, t, months)
    F["XGBoost (global, Tweedie)"] = xg
    acc = {name: accuracy(Y, f, TEST) for name, f in F.items()}

    # --- inventory simulation for every method and service level
    V = {name: error_variance(Y, f) for name, f in F.items()}
    sims = {"val": {}, "test": {}}
    for split, period in (("val", VAL), ("test", TEST)):
        for name in F:
            if name == "zero":
                continue
            sims[split][name] = {
                s: run_policy(Y, lambda t, n=name, s=s: order_up_to(F[n][:, t], V[n][:, t], s), period) for s in LEVELS
            }
        sims[split]["empirical (past 24 months)"] = {
            s: run_policy(Y, lambda t, s=s: empirical_order_up_to(Y[:, :t], s), period) for s in LEVELS
        }
        ma = F["moving average (12 m)"]
        sims[split]["rule of thumb (k months of average demand)"] = {
            k: run_policy(Y, lambda t, k=k, ma=ma: np.ceil(k * ma[:, t]), period) for k in COVER
        }

    def chosen_results(holding, backorder):
        out = {}
        for name, res in sims["val"].items():
            param = choose(res, holding, backorder)
            r = sims["test"][name][param]
            out[name] = {
                "parameter": param,
                **{k: v for k, v in r.items() if k != "_per_part"},
                "cost": cost_with(r, holding, backorder),
            }
        return out

    main = chosen_results(HOLDING, BACKORDER)
    rule = "rule of thumb (k months of average demand)"
    best = min((n for n in main if n != rule), key=lambda n: main[n]["cost"])
    sens = []
    for b in (0.25, 1.0, 4.0):
        r = chosen_results(HOLDING, b)
        bb = min((n for n in r if n != rule), key=lambda n: r[n]["cost"])
        sens.append(
            {
                "backorder_cost": b,
                "best_method": bb,
                "best_cost": r[bb]["cost"],
                "best_fill_rate": r[bb]["fill_rate"],
                "best_stock": r[bb]["avg_stock_per_part"],
                "rule_cost": r[rule]["cost"],
                "rule_fill_rate": r[rule]["fill_rate"],
                "rule_stock": r[rule]["avg_stock_per_part"],
                "cost_saving": 1 - r[bb]["cost"] / r[rule]["cost"],
            }
        )

    # 95% intervals for the cost gap between the best method and the alternatives, resampling parts
    def part_cost(name):
        pp = sims["test"][name][main[name]["parameter"]]["_per_part"]
        return HOLDING * pp["stock"] + BACKORDER * pp["backorders"]

    rng = np.random.default_rng(0)
    boot_idx = [rng.integers(0, len(Y), len(Y)) for _ in range(1000)]
    cost_gap = {}
    for other in (rule, "TSB", "exponential smoothing", "Croston"):
        d = part_cost(best) - part_cost(other)
        rel = [d[i].mean() / part_cost(other)[i].mean() for i in boot_idx]
        cost_gap[other] = {
            "relative_cost_change": float(d.mean() / part_cost(other).mean()),
            "ci95": [float(x) for x in np.percentile(rel, [2.5, 97.5])],
        }

    # stock needed for a given fill rate on the test year (descriptive frontier)
    def stock_at(name, fill):
        res = sorted(sims["test"][name].values(), key=lambda r: r["fill_rate"])
        fr, st = [r["fill_rate"] for r in res], [r["avg_stock_per_part"] for r in res]
        return float(np.interp(fill, fr, st)) if fr[0] <= fill <= fr[-1] else None

    frontier = {
        n: {f"stock_for_{int(f * 100)}pct_fill": stock_at(n, f) for f in (0.85, 0.90, 0.95)} for n in sims["test"]
    }

    # --- by demand pattern, best method vs rule of thumb at their chosen settings
    def by_pattern(name):
        t = list(TEST)
        if name == rule:
            k = main[rule]["parameter"]
            S = np.column_stack([np.ceil(k * F["moving average (12 m)"][:, i]) for i in t])
        elif name.startswith("empirical"):
            S = np.column_stack([empirical_order_up_to(Y[:, :i], main[name]["parameter"]) for i in t])
        else:
            S = np.column_stack([order_up_to(F[name][:, i], V[name][:, i], main[name]["parameter"]) for i in t])
        out = {}
        for pat in ("intermittent", "lumpy"):
            mask = (patterns == pat).to_numpy()
            r = simulate(Y[mask][:, t], S[mask])
            out[pat] = {k: v for k, v in r.items() if k != "_per_part"} | {"parts": int(mask.sum())}
        return out

    patterns_out = {best: by_pattern(best), rule: by_pattern(rule)}

    from .plots import save_figures

    save_figures(REPORTS, sims["test"], main, acc, Y, F, best, rule)

    # --- deployable model: retrained on all history; recommendations for April 2002
    m_all = global_model.fit(Y, months, until=Y.shape[1])
    oos_months = range(VAL.start, TEST.stop)  # out-of-sample one-step forecasts exist for these months
    ins = np.column_stack([global_model.predict(m_all, Y, t, months) for t in oos_months])
    oos = xg[:, list(oos_months)]
    actual = Y[:, list(oos_months)]
    inflation = float(np.mean((actual - oos) ** 2) / np.mean((actual - ins) ** 2))
    level = main["XGBoost (global, Tweedie)"]["parameter"]
    recs = recommend_batch(m_all, Y, months[0], level, P, inflation)
    recs.index = Ydf.index
    recs["pattern"] = patterns
    recs["last_12m_demand"] = Y[:, -12:].sum(axis=1)

    results = {
        "parts": int(len(Y)),
        "dropped_incomplete": int((~complete).sum()),
        "patterns": patterns.value_counts().to_dict(),
        "periods": {
            "validation": [str(months[VAL.start].date()), str(months[VAL.stop - 1].date())],
            "test": [str(months[TEST.start].date()), str(months[TEST.stop - 1].date())],
        },
        "assumptions": {
            "lead_time_months": LEAD_TIME,
            "review_months": REVIEW,
            "holding_cost_per_unit_month": HOLDING,
            "backorder_cost_per_unit_month": BACKORDER,
        },
        "forecast_accuracy_test": acc,
        "chosen_policies_test": main,
        "best_method": best,
        "serving_variance_inflation": inflation,
        "cost_gap_vs_best_ci": cost_gap,
        "sensitivity_backorder_cost": sens,
        "frontier_test": frontier,
        "by_pattern_test": patterns_out,
        "runtime_s": round(time.time() - t0, 1),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=1, default=float))
    bundle = {
        "version": __version__,
        "model": m_all,
        "service_level": level,
        "inflation": inflation,
        "history": Ydf,
        "recommendations": recs,
        "as_of": str((months[-1] + pd.DateOffset(months=1)).date()),
        "lead_time": LEAD_TIME,
        "review": REVIEW,
        "test_summary": {"best": main[best], "rule": main[rule], "best_method": best},
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT / "models" / "policy.joblib", compress=3)
    print(f"done in {results['runtime_s']} s -> reports/metrics.json, models/policy.joblib")
    return results


if __name__ == "__main__":
    run()
