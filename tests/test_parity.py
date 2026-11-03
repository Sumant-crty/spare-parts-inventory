"""The API must reproduce the precomputed recommendations exactly. Needs the trained model."""

import pathlib

import numpy as np
import pytest

MODEL = pathlib.Path(__file__).resolve().parents[1] / "models" / "policy.joblib"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="needs the trained model")


def test_history_endpoint_matches_bundle():
    from spareparts.predictor import Predictor

    p = Predictor(MODEL)
    hist = p.b["history"]
    for part in hist.sample(20, random_state=0).index:
        r = p.reorder(hist.loc[part].tolist(), str(hist.columns[0].date()), 0, 0, 0)
        expected = p.part(part)
        assert r["order_up_to_level"] == expected["order_up_to_level"], part
        assert np.isclose(r["forecast_per_month"], expected["forecast_per_month"]), part
