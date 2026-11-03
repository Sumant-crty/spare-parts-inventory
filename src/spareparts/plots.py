"""Report figures."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREY, GREEN, GOLD, PURPLE = "#2F5C9E", "#E8772E", "#8A8F98", "#5E8C6A", "#C99A2E", "#7B5EA7"


def _style(ax, title):
    ax.set_title(title, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def save_figures(out: pathlib.Path, sims_test, chosen, acc, Y, F, best, rule):
    out.mkdir(parents=True, exist_ok=True)

    # 1. stock vs fill-rate frontier
    show = ["XGBoost (global, Tweedie)", "TSB", "exponential smoothing", "Croston", rule]
    colors = {
        "XGBoost (global, Tweedie)": PURPLE,
        "TSB": BLUE,
        "exponential smoothing": GOLD,
        "Croston": GREEN,
        rule: ORANGE,
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for name in dict.fromkeys(show):
        if name not in sims_test:
            continue
        res = sorted(sims_test[name].values(), key=lambda r: r["fill_rate"])
        ax.plot(
            [100 * r["fill_rate"] for r in res],
            [r["avg_stock_per_part"] for r in res],
            marker="o",
            ms=3,
            color=colors[name],
            label=name,
        )
        c = chosen[name]
        ax.scatter(
            [100 * c["fill_rate"]], [c["avg_stock_per_part"]], s=90, color=colors[name], edgecolor="black", zorder=3
        )
    ax.set_xlabel("fill rate: demand served from stock (%)")
    ax.set_ylabel("average stock per part (units)")
    _style(ax, "Stock needed for each service level, test year (big dots = chosen policy)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(), fig.savefig(out / "stock_vs_fill_rate.png", dpi=150), plt.close(fig)

    # 2. forecast accuracy vs inventory outcome
    names = [n for n in acc if n in chosen or n == "zero"]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    x = np.arange(len(names))
    ax.bar(x, [acc[n]["scaled_mae"] for n in names], color=[GREY if n == "zero" else BLUE for n in names])
    ax.set_xticks(x, names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("scaled mean absolute error")
    _style(ax, "Forecast error, test year (the 'zero' forecast scores well but stocks nothing)")
    fig.tight_layout(), fig.savefig(out / "forecast_error.png", dpi=150), plt.close(fig)

    # 3. demand of a few typical parts with the best method's forecast
    rng = np.random.default_rng(1)
    idx = rng.choice(len(Y), 3, replace=False)
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 5.2), sharex=True)
    for ax, i in zip(axes, idx):
        ax.bar(range(Y.shape[1]), Y[i], color=GREY, label="monthly demand")
        ax.plot(range(Y.shape[1]), F[best][i, : Y.shape[1]], color=BLUE, lw=1.6, label=f"{best} forecast")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Intermittent demand: three example parts", fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("month (Jan 1998 = 0)")
    fig.tight_layout(), fig.savefig(out / "example_parts.png", dpi=150), plt.close(fig)
