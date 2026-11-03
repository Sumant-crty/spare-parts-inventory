"""FastAPI service: spare-parts reorder advice.

Run: uvicorn api.main:app --reload   (MODEL_PATH defaults to models/policy.joblib)
"""

from __future__ import annotations

import os
import pathlib

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from spareparts.predictor import Predictor

ROOT = pathlib.Path(__file__).resolve().parents[1]
predictor = Predictor(pathlib.Path(os.getenv("MODEL_PATH", ROOT / "models" / "policy.joblib")))
app = FastAPI(
    title="Spare-parts inventory",
    version=predictor.info["version"],
    description="Demand forecasts, order-up-to levels and order quantities for intermittent spare-parts demand.",
)


class History(BaseModel):
    start_month: str = Field(pattern=r"^\d{4}-\d{2}$", description="month of the first value, YYYY-MM")
    history: list[float] = Field(min_length=3, max_length=240, description="units sold per month, oldest first")
    on_hand: float = Field(0, ge=0)
    on_order: float = Field(0, ge=0)
    backorders: float = Field(0, ge=0)
    service_level: float | None = Field(None, ge=0.5, le=0.999)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_info():
    return predictor.info


@app.get("/parts/{part_id}")
def part(
    part_id: str, on_hand: float = Query(0, ge=0), on_order: float = Query(0, ge=0), backorders: float = Query(0, ge=0)
):
    r = predictor.part(part_id, on_hand, on_order, backorders)
    if r is None:
        raise HTTPException(404, "unknown part; use POST /reorder with its demand history")
    return r


@app.post("/reorder")
def reorder(h: History):
    if any(v < 0 for v in h.history):
        raise HTTPException(422, "demand cannot be negative")
    return predictor.reorder(h.history, h.start_month + "-01", h.on_hand, h.on_order, h.backorders, h.service_level)
