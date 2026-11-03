"""Load the car-parts series and classify demand patterns (Syntetos-Boylan)."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parents[2] / "data"
TSF = DATA / "car_parts_dataset_with_missing_values.tsf"
START = pd.Timestamp("1998-01-01")
ADI_CUT, CV2_CUT = 1.32, 0.49  # Syntetos, Boylan & Croston (2005)


def load_series(path: pathlib.Path = TSF) -> pd.DataFrame:
    """Parts x months matrix of unit sales; NaN where the value is missing in the source."""
    rows = {}
    in_data = False
    for line in path.read_text().splitlines():
        if line.startswith("@data"):
            in_data = True
            continue
        if not in_data or not line.strip():
            continue
        name, _, values = line.split(":", 2)
        rows[name] = [np.nan if v == "?" else float(v) for v in values.split(",")]
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.columns = pd.date_range(START, periods=df.shape[1], freq="MS")
    df.index.name = "part"
    return df


def classify(demand: np.ndarray) -> dict:
    """Average inter-demand interval (ADI) and squared CV of non-zero demand sizes."""
    d = demand[~np.isnan(demand)]
    nz = d[d > 0]
    if len(nz) == 0:
        return {"adi": np.inf, "cv2": np.nan, "pattern": "no demand"}
    adi = len(d) / len(nz)
    cv2 = float((nz.std() / nz.mean()) ** 2) if len(nz) > 1 else 0.0
    if adi < ADI_CUT:
        pattern = "smooth" if cv2 < CV2_CUT else "erratic"
    else:
        pattern = "intermittent" if cv2 < CV2_CUT else "lumpy"
    return {"adi": float(adi), "cv2": cv2, "pattern": pattern}
