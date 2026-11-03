"""Synthetic intermittent demand and a small model bundle, so tests run without the dataset."""

import pathlib

import joblib
import numpy as np
import pandas as pd
import pytest

from spareparts import global_model
from spareparts.policy import recommend_batch


def synthetic_demand(n: int = 60, months: int = 30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.5, (n, 1))
    occurs = rng.random((n, months)) < p
    sizes = rng.poisson(2, (n, months)) + 1
    return (occurs * sizes).astype(float)


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory) -> pathlib.Path:
    Y = synthetic_demand()
    months = pd.date_range("2020-01-01", periods=Y.shape[1], freq="MS")
    model = global_model.fit(Y, months, until=Y.shape[1])
    recs = recommend_batch(model, Y, months[0], 0.95, 2, 1.0)
    recs.index = [f"P{i}" for i in range(len(Y))]
    recs["pattern"], recs["last_12m_demand"] = "intermittent", Y[:, -12:].sum(axis=1)
    summary = {"fill_rate": 0.9, "avg_stock_per_part": 3.0}
    bundle = {
        "version": "test",
        "model": model,
        "service_level": 0.95,
        "inflation": 1.0,
        "history": pd.DataFrame(Y, index=recs.index, columns=months),
        "recommendations": recs,
        "as_of": "2022-07-01",
        "lead_time": 1,
        "review": 1,
        "test_summary": {"best": summary, "rule": summary, "best_method": "XGBoost (global, Tweedie)"},
    }
    path = tmp_path_factory.mktemp("models") / "policy.joblib"
    joblib.dump(bundle, path)
    return path


@pytest.fixture()
def client(bundle_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(bundle_path))
    import importlib

    from api import main

    main = importlib.reload(main)
    from fastapi.testclient import TestClient

    return TestClient(main.app)
