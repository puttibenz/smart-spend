"""
SmartSpend AI - Tests for Needs vs Wants Module (Phase 2)
Validates boolean non-null outputs, default category mappings, amount spike overrides,
absence of late-night double counting, accuracy >= 0.70, and expanding window temporal integrity.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import pytest
import pandas as pd
import numpy as np

from src.needs_wants.classify_needs_wants import NeedsWantsClassifier, load_config
from src.needs_wants.evaluate_needs_wants import evaluate_needs_wants

DATA_PATH = "data/raw/transactions.csv"
METRICS_PATH = "outputs/metrics/phase2_needs_wants_eval.json"

def test_1_is_wants_boolean_and_non_null():
    """Assert is_wants is always boolean and has zero nulls."""
    df = pd.read_csv(DATA_PATH)
    classifier = NeedsWantsClassifier()
    preds = classifier.classify_dataframe(df)
    
    assert len(preds) == len(df)
    assert preds.isnull().sum() == 0, "Predicted is_wants contains null values!"
    for val in preds:
        assert isinstance(val, (bool, np.bool_)), f"Expected boolean, got {type(val)}"

def test_2_amount_spike_override_logic():
    """Assert amount > 1.5 * historical_median overrides default Need to Want."""
    classifier = NeedsWantsClassifier()
    
    # Food is default Need (False)
    # Historical amounts: [60, 70, 80] -> Median = 70. Multiplier 1.5 -> Threshold = 105
    history = [60.0, 70.0, 80.0]
    
    # Normal amount <= 105 -> Should be False (Need)
    assert classifier.classify_single(category="food", amount=85.0, category_history_amounts=history) is False
    assert classifier.classify_single(category="food", amount=105.0, category_history_amounts=history) is False
    
    # Spike amount > 105 -> Should be overridden to True (Want)
    assert classifier.classify_single(category="food", amount=120.0, category_history_amounts=history) is True
    assert classifier.classify_single(category="food", amount=500.0, category_history_amounts=history) is True

def test_3_no_late_night_double_counting():
    """Assert late_night is NOT present in classify_needs_wants.py (per FIX #6)."""
    with open("src/needs_wants/classify_needs_wants.py", "r", encoding="utf-8") as f:
        code_content = f.read().lower()
        
    assert "is_late_night" not in code_content
    assert "late_night_window" not in code_content
    assert "23:00" not in code_content

def test_4_default_category_mappings():
    """Assert default mappings match config (bills=False, shopping=True, entertainment=True)."""
    classifier = NeedsWantsClassifier()
    
    # Without history / normal amount
    assert classifier.classify_single(category="bills", amount=500.0) is False
    assert classifier.classify_single(category="shopping", amount=150.0) is True
    assert classifier.classify_single(category="entertainment", amount=200.0) is True
    assert classifier.classify_single(category="food", amount=60.0) is False
    assert classifier.classify_single(category="transport", amount=40.0) is False

def test_5_needs_wants_accuracy_meets_threshold():
    """Assert evaluation accuracy against ground truth is_wants >= 0.70."""
    metrics = evaluate_needs_wants()
    acc = metrics["value"]
    floor = metrics["accuracy_floor"]
    
    assert acc >= floor, f"Needs/Wants Accuracy ({acc:.4f}) < Floor ({floor:.2f})"
    assert metrics["meets_minimum_bar"] is True

def test_6_expanding_window_temporal_integrity():
    """Assert early transactions in timeline are NOT influenced by future transactions."""
    classifier = NeedsWantsClassifier()
    
    # Create sample timeline
    df_early = pd.DataFrame([
        {"date": "2025-01-01", "time": "12:00", "category": "food", "amount": 100.0},
        {"date": "2025-01-02", "time": "12:00", "category": "food", "amount": 100.0},
        {"date": "2025-01-03", "time": "12:00", "category": "food", "amount": 120.0}, # Normal (120 <= 150) -> False
    ])
    
    preds_early = classifier.classify_dataframe(df_early)
    assert bool(preds_early.iloc[2]) is False  # Median of [100, 100] = 100, 120 < 150 -> False
    
    # Now append future huge amounts that would distort global median if leak occurred
    df_with_future = pd.concat([
        df_early,
        pd.DataFrame([
            {"date": "2025-12-01", "time": "12:00", "category": "food", "amount": 5000.0},
            {"date": "2025-12-02", "time": "12:00", "category": "food", "amount": 5000.0},
            {"date": "2025-12-03", "time": "12:00", "category": "food", "amount": 5000.0}
        ])
    ]).reset_index(drop=True)
    
    preds_with_future = classifier.classify_dataframe(df_with_future)
    
    # The third transaction MUST still be False (expanding window does not look ahead)
    assert bool(preds_with_future.iloc[2]) is False, (
        "TEMPORAL LEAK DETECTED: Early transaction prediction changed when future transactions were appended!"
    )
