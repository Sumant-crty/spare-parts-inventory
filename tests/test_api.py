from conftest import synthetic_demand


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/model").json()["parts"] == 60


def test_known_part_and_order_quantity(client):
    r = client.get("/parts/P0", params={"on_hand": 1, "on_order": 1}).json()
    assert r["inventory_position"] == 2
    assert r["order_quantity"] == max(0, r["order_up_to_level"] - 2)
    assert client.get("/parts/NOPE").status_code == 404


def test_reorder_from_history(client):
    hist = synthetic_demand(n=1, months=24, seed=5)[0].tolist()
    r = client.post("/reorder", json={"start_month": "2021-01", "history": hist, "on_hand": 0}).json()
    assert r["for_month"] == "2023-01-01" and r["method"].startswith("XGBoost")
    assert r["order_quantity"] == r["order_up_to_level"] >= 0
    hi = client.post("/reorder", json={"start_month": "2021-01", "history": hist, "service_level": 0.99}).json()
    lo = client.post("/reorder", json={"start_month": "2021-01", "history": hist, "service_level": 0.6}).json()
    assert hi["order_up_to_level"] >= lo["order_up_to_level"]


def test_short_history_falls_back_to_tsb(client):
    r = client.post("/reorder", json={"start_month": "2021-01", "history": [0, 2, 0, 0, 1]}).json()
    assert r["method"] == "TSB (short history)"


def test_rejects_bad_input(client):
    assert client.post("/reorder", json={"start_month": "2021-01", "history": [1, -2, 0]}).status_code == 422
    assert client.post("/reorder", json={"start_month": "Jan 2021", "history": [1, 2, 0]}).status_code == 422
    assert client.post("/reorder", json={"start_month": "2021-01", "history": [1]}).status_code == 422
