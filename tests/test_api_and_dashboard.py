"""
SmartSpend AI - Phase 3 Integration Tests for API and Dashboard
Validates all FastAPI endpoints using TestClient, enforces 100% data consistency
between dashboard aggregates and raw/metrics data, verifies Pydantic validation (422),
proves zero train-serve skew, and tests fail-fast startup behavior.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pytest
import json
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app, app_state
from src.api.artifact_loader import load_all_phase1_phase2_artifacts, load_latest_artifact

DATA_PATH = "data/raw/transactions.csv"
CONFIG_PATH = "config.yaml"

@pytest.fixture(scope="module")
def client():
    """
    TestClient context manager that triggers FastAPI lifespan startup/shutdown.
    """
    with TestClient(app) as test_client:
        yield test_client

def test_1_root_serves_frontend(client: TestClient):
    """Assert Root (/) serves index.html with HTTP 200 and text/html."""
    response = client.get("/")
    assert response.status_code == 200
    assert "SmartSpend" in response.text
    assert "<!DOCTYPE html>" in response.text

def test_2_summary_data_consistency(client: TestClient):
    """Assert /api/summary values match raw CSV and Phase 1-2 metrics 100%."""
    response = client.get("/api/summary")
    assert response.status_code == 200
    summary = response.json()
    
    df = pd.read_csv(DATA_PATH)
    
    # 1. Total spend and transaction counts
    expected_total_spend = float(round(df["amount"].sum(), 2))
    assert summary["total_spend"] == expected_total_spend
    assert summary["total_transactions"] == len(df)
    
    # 2. Needs vs Wants consistency
    assert round(summary["needs_amount"] + summary["wants_amount"], 2) == expected_total_spend
    assert round(summary["needs_percentage"] + summary["wants_percentage"], 1) == 100.0
    
    # 3. Impulse counts
    expected_impulse_count = int(df["is_impulse"].sum())
    assert summary["impulse_transactions_count"] == expected_impulse_count
    
    # 4. Category breakdown sum equals total spend
    cat_sum = sum(c["total_amount"] for c in summary["category_breakdown"].values())
    assert round(cat_sum, 2) == expected_total_spend

def test_3_transactions_list_pagination_and_filters(client: TestClient):
    """Assert /api/transactions returns full schema and respects limit, skip, and filters."""
    # Default paginated fetch
    res = client.get("/api/transactions?limit=10&skip=0")
    assert res.status_code == 200
    data = res.json()
    
    assert data["limit"] == 10
    assert data["skip"] == 0
    assert len(data["items"]) == 10
    assert data["total_count"] == 1646
    
    # Schema check on item
    first_item = data["items"][0]
    required_fields = ["transaction_id", "date", "time", "merchant", "amount", "category", "is_wants", "impulse_score", "is_nudge_alert"]
    for field in required_fields:
        assert field in first_item, f"Missing field '{field}' in transaction item"

    # Category Filter
    res_food = client.get("/api/transactions?category=food&limit=100")
    assert res_food.status_code == 200
    data_food = res_food.json()
    for item in data_food["items"]:
        assert item["category"] == "food"

    # Search filter
    res_search = client.get("/api/transactions?search=grab&limit=50")
    assert res_search.status_code == 200
    data_search = res_search.json()
    assert data_search["filtered_count"] > 0
    for item in data_search["items"]:
        match = "grab" in item["merchant"].lower() or "grab" in item["memo"].lower() or "grab" in item["category"].lower()
        assert match is True

    # Nudge Only filter
    res_nudge = client.get("/api/transactions?is_nudge_only=true&limit=50")
    assert res_nudge.status_code == 200
    data_nudge = res_nudge.json()
    for item in data_nudge["items"]:
        assert item["is_nudge_alert"] is True
        assert item["impulse_score"] >= 70

def test_4_heatmap_matrix_dimensions_and_sum(client: TestClient):
    """Assert /api/heatmap returns 7x24 matrix matching total transactions."""
    res = client.get("/api/heatmap")
    assert res.status_code == 200
    heatmap = res.json()
    
    assert len(heatmap["days"]) == 7
    assert len(heatmap["hours"]) == 24
    assert len(heatmap["matrix"]) == 7
    
    total_heatmap_tx = 0
    total_heatmap_spend = 0.0
    
    for row in heatmap["matrix"]:
        assert len(row) == 24
        for cell in row:
            total_heatmap_tx += cell["transaction_count"]
            total_heatmap_spend += cell["total_amount"]
            assert cell["total_amount"] >= 0
            assert cell["transaction_count"] >= 0
            assert cell["impulse_count"] >= 0
            
    df = pd.read_csv(DATA_PATH)
    assert total_heatmap_tx == len(df)
    assert round(total_heatmap_spend, 2) == float(round(df["amount"].sum(), 2))

def test_5_metrics_endpoint_transparency(client: TestClient):
    """Assert /api/metrics provides Phase 1 and 2 metrics alongside config nudge_threshold."""
    res = client.get("/api/metrics")
    assert res.status_code == 200
    metrics_data = res.json()
    
    assert metrics_data["status"] == "success"
    assert metrics_data["nudge_threshold"] == 70
    assert "phase1_categorization" in metrics_data
    assert "phase2_needs_wants" in metrics_data
    assert "phase2_impulse_scoring" in metrics_data

def test_6_live_prediction_valid_input(client: TestClient):
    """Assert POST /api/predict performs dual inference with valid output ranges."""
    payload = {
        "merchant": "GrabFood Delivery",
        "memo": "สั่งพิซซ่ามื้อดึกหิวมากหลังเที่ยงคืน",
        "amount": 450.00,
        "date": "2025-01-26",
        "time": "23:45"
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    pred = res.json()
    
    assert pred["predicted_category"] == "food"
    assert 0.0 <= pred["category_confidence"] <= 1.0
    assert isinstance(pred["is_wants"], bool)
    assert 0 <= pred["impulse_score_v1"] <= 100
    assert isinstance(pred["is_nudge_alert"], bool)
    assert 0.0 <= pred["impulse_probability_v2"] <= 1.0
    assert pred["score_breakdown"]["late_night_score"] == 25.0
    assert pred["score_breakdown"]["payday_score"] == 25.0

def test_7_live_prediction_input_validation_422(client: TestClient):
    """Assert POST /api/predict returns HTTP 422 on invalid input formats."""
    # Negative amount
    res1 = client.post("/api/predict", json={
        "merchant": "7-Eleven",
        "amount": -50.0,
        "date": "2025-01-10",
        "time": "12:00"
    })
    assert res1.status_code == 422

    # Invalid Date format
    res2 = client.post("/api/predict", json={
        "merchant": "7-Eleven",
        "amount": 50.0,
        "date": "2025/01/10",
        "time": "12:00"
    })
    assert res2.status_code == 422

    # Invalid Time format
    res3 = client.post("/api/predict", json={
        "merchant": "7-Eleven",
        "amount": 50.0,
        "date": "2025-01-10",
        "time": "25:99"
    })
    assert res3.status_code == 422

    # Empty merchant
    res4 = client.post("/api/predict", json={
        "merchant": "",
        "amount": 50.0,
        "date": "2025-01-10",
        "time": "12:00"
    })
    assert res4.status_code == 422

def test_8_zero_train_serve_skew(client: TestClient):
    """Assert real-time API inference gives identical result to direct offline module calls."""
    from src.nlp.preprocessing import prepare_text_feature
    from src.scoring.impulse_rules import is_late_night, is_payday_window
    
    test_merchant = "Shopee Official"
    test_memo = "ซื้อหูฟังบลูทูธ"
    test_amount = 1200.0
    test_date = "2025-01-26"
    test_time = "23:15"
    
    # 1. Real-time API Path
    api_res = client.post("/api/predict", json={
        "merchant": test_merchant,
        "memo": test_memo,
        "amount": test_amount,
        "date": test_date,
        "time": test_time
    }).json()
    
    # 2. Direct Offline Module Path
    artifacts = app_state["artifacts"]
    vec = artifacts["vectorizer"].transform([prepare_text_feature(test_merchant, test_memo)])
    direct_cat = str(artifacts["category_model"].predict(vec)[0])
    
    assert api_res["predicted_category"] == direct_cat
    assert api_res["score_breakdown"]["late_night_score"] == (25.0 if is_late_night(test_time) else 0.0)
    assert api_res["score_breakdown"]["payday_score"] == (25.0 if is_payday_window(test_date) else 0.0)

def test_9_fail_fast_startup_on_missing_artifacts():
    """Assert dynamic artifact loader raises FileNotFoundError if artifact path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_all_phase1_phase2_artifacts(artifacts_dir="non_existent_artifacts_dir_xyz")
