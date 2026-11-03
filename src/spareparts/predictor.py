"""Reorder advice for known parts (precomputed) or any part from its monthly demand history."""

from __future__ import annotations

import pathlib

import joblib
import numpy as np
import pandas as pd

from .policy import recommend_batch


class Predictor:
    def __init__(self, bundle_path: str | pathlib.Path):
        self.b = joblib.load(bundle_path)
        self.recs = self.b["recommendations"]
        self.protection = self.b["lead_time"] + self.b["review"]

    @property
    def info(self) -> dict:
        b = self.b
        best, rule = b["test_summary"]["best"], b["test_summary"]["rule"]
        return {
            "version": b["version"],
            "recommendations_for": b["as_of"],
            "parts": len(self.recs),
            "method": b["test_summary"]["best_method"],
            "service_level": b["service_level"],
            "lead_time_months": b["lead_time"],
            "review_months": b["review"],
            "test_year": {
                "fill_rate": round(best["fill_rate"], 3),
                "avg_stock_per_part": round(best["avg_stock_per_part"], 2),
                "rule_of_thumb_fill_rate": round(rule["fill_rate"], 3),
                "rule_of_thumb_avg_stock": round(rule["avg_stock_per_part"], 2),
            },
        }

    @staticmethod
    def _advice(r: pd.Series, on_hand: float, on_order: float, backorders: float) -> dict:
        position = on_hand + on_order - backorders
        return {
            "forecast_per_month": round(float(r.forecast_per_month), 3),
            "order_up_to_level": int(r.order_up_to),
            "safety_stock": round(float(r.safety_stock), 2),
            "inventory_position": position,
            "order_quantity": int(max(0, r.order_up_to - position)),
            "method": r.method,
        }

    def part(self, part_id: str, on_hand: float = 0, on_order: float = 0, backorders: float = 0) -> dict | None:
        if part_id not in self.recs.index:
            return None
        r = self.recs.loc[part_id]
        return {
            "part": part_id,
            "for_month": self.b["as_of"],
            "pattern": r.pattern,
            **self._advice(r, on_hand, on_order, backorders),
        }

    def reorder(
        self,
        history: list[float],
        start_month: str,
        on_hand: float,
        on_order: float,
        backorders: float,
        service_level: float | None = None,
    ) -> dict:
        Y = np.asarray(history, float)[None, :]
        start = pd.Timestamp(start_month)
        service = service_level or self.b["service_level"]
        r = recommend_batch(self.b["model"], Y, start, service, self.protection, self.b["inflation"]).iloc[0]
        next_month = start + pd.DateOffset(months=len(history))
        return {
            "for_month": str(next_month.date()),
            "service_level": service,
            **self._advice(r, on_hand, on_order, backorders),
        }
