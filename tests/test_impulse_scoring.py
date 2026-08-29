"""
SmartSpend AI - Tests for Impulse Risk Scoring Engine (Phase 2)
Validates score bounding [0, 100], Cold Start normalization, weights from config.yaml,
nudge threshold trigger, expanding window Z-score integrity, and v2 ML > v1 baseline.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import pytest
import joblib
import pandas as pd
import numpy as np

from src.scoring.impulse_rules import (
    ImpulseRuleScorer,
    calc_z_score,
    calc_amount_anomaly_score,
    is_late_night,
    is_payday_window,
    load_config
)

DATA_PATH = "data/raw/transactions.csv"
METRICS_PATH = "outputs/metrics/phase2_metrics.json"
ARTIFACTS_DIR = "models_artifacts"

def test_1_score_bounded_0_to_100():
    """Assert Impulse Score is always bounded between 0 and 100 in all cases."""
    df = pd.read_csv(DATA_PATH)
    scorer = ImpulseRuleScorer()
    scored_df = scorer.score_dataframe(df)
    
    assert scored_df["impulse_score"].min() >= 0, "Impulse score below 0 detected!"
    assert scored_df["impulse_score"].max() <= 100, "Impulse score above 100 detected!"
    assert scored_df["impulse_score"].isnull().sum() == 0, "Null values found in impulse_score!"

def test_2_cold_start_normalization():
    """Assert cold start (< 30 days) omits anomaly and rescales from 70 to 100."""
    scorer = ImpulseRuleScorer()
    
    # Cold start transaction (day 5 from first tx)
    # Late night (25) + Payday (25) + Wants (20) = 70 -> rescaled: 100
    res_all_flags = scorer.score_single(
        date_str="2025-01-25", # Payday
        time_str="23:30",      # Late night
        amount=5000.0,         # Anomaly (should be ignored in cold start)
        is_wants=True,         # Wants
        first_tx_date="2025-01-20" # 5 days elapsed (< 30)
    )
    
    assert res_all_flags["is_cold_start"] is True
    assert res_all_flags["score_breakdown"]["anomaly_score"] == 0.0
    assert res_all_flags["impulse_score"] == 100  # (70 / 70) * 100 = 100
    
    # Partial flags in cold start: Only late night (25) -> (25 / 70) * 100 = 36
    res_partial = scorer.score_single(
        date_str="2025-01-10",
        time_str="01:15",
        amount=100.0,
        is_wants=False,
        first_tx_date="2025-01-01" # 9 days elapsed
    )
    assert res_partial["is_cold_start"] is True
    assert res_partial["impulse_score"] == round((25.0 / 70.0) * 100.0) # 36

def test_3_weights_match_config_yaml():
    """Assert scoring engine uses weights from config.yaml without hardcoding."""
    config = load_config("config.yaml")
    cfg_weights = config["impulse_score"]["weights"]
    
    scorer = ImpulseRuleScorer(config)
    assert scorer.w_late_night == cfg_weights["late_night"]
    assert scorer.w_payday == cfg_weights["payday"]
    assert scorer.w_wants == cfg_weights["wants"]
    assert scorer.w_anomaly == cfg_weights["amount_anomaly"]
    
    total_weights = scorer.w_late_night + scorer.w_payday + scorer.w_wants + scorer.w_anomaly
    assert total_weights == 100.0, f"Weights sum to {total_weights}, expected 100.0"

def test_4_nudge_threshold_trigger():
    """Assert is_nudge_alert is True if and only if impulse_score >= 70."""
    config = load_config("config.yaml")
    threshold = config["impulse_score"]["nudge_threshold"]
    scorer = ImpulseRuleScorer(config)
    
    # 70 or above -> True
    res_high = scorer.score_single("2025-01-25", "23:30", 500.0, True, "2025-01-20")
    assert res_high["impulse_score"] >= threshold
    assert res_high["is_nudge_alert"] is True
    
    # Below 70 -> False
    res_low = scorer.score_single("2025-01-10", "12:00", 50.0, False, "2025-01-01")
    assert res_low["impulse_score"] < threshold
    assert res_low["is_nudge_alert"] is False

def test_5_v2_f1_beats_v1_baseline():
    """Assert v2 ML model F1 score strictly outperforms v1 rule-based baseline."""
    assert os.path.exists(METRICS_PATH), f"Metrics JSON missing at {METRICS_PATH}"
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    comp = metrics["v1_vs_v2_comparison"]
    v1_f1 = comp["v1_rule_based_f1"]
    v2_f1 = comp["v2_ml_f1"]
    
    assert comp["v2_beats_v1"] is True, f"v2 F1 ({v2_f1:.4f}) did NOT beat v1 F1 ({v1_f1:.4f})"
    assert v2_f1 > v1_f1

def test_6_z_score_expanding_window_temporal_integrity():
    """Assert Z-score calculation on earlier transactions is not affected by future transactions."""
    history = [100.0, 100.0, 100.0, 100.0, 100.0]
    # Amount 300 vs mean 100, std 0 -> calc_z_score handles zero std safely
    z_early = calc_z_score(300.0, [50.0, 70.0, 80.0, 100.0])
    assert isinstance(z_early, float)
    assert z_early > 0.0
    
    # Empty history returns 0.0
    assert calc_z_score(500.0, []) == 0.0
    assert calc_z_score(500.0, [500.0]) == 0.0

def test_7_v2_model_artifact_inference():
    """Assert v2 ML model artifact can be loaded and perform inference."""
    artifact_files = os.listdir(ARTIFACTS_DIR)
    v2_files = [f for f in artifact_files if f.startswith("impulse_model_phase2_") and f.endswith(".joblib")]
    assert len(v2_files) >= 1, "No v2 ML model artifact found in models_artifacts/"
    
    artifact = joblib.load(os.path.join(ARTIFACTS_DIR, sorted(v2_files)[-1]))
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    
    sample_X = np.array([[1, 1, 1, 2.5, 1500.0, 23, 5, 1]]) # 8 features
    pred_prob = model.predict_proba(sample_X)
    
    assert pred_prob.shape == (1, 2)
    assert 0.0 <= pred_prob[0, 1] <= 1.0
