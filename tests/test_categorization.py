"""
SmartSpend AI - Tests for Phase 1: Expense Categorization
Validates CSV schema, non-null values, row consistency, text preprocessing,
model artifacts, and ensures ML models strictly outperform baseline.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import pytest
import pandas as pd
import joblib

from src.nlp.preprocessing import clean_text, tokenize_text, prepare_text_feature
from src.nlp.vectorizer import load_vectorizer
from src.models.evaluate import evaluate_models

DATA_PATH = "data/raw/transactions.csv"
METRICS_PATH = "outputs/metrics/phase1_metrics.json"
ARTIFACTS_DIR = "models_artifacts"

EXPECTED_SCHEMA = {
    "transaction_id": "object",
    "date": "object",
    "time": "object",
    "merchant": "object",
    "memo": "object",
    "amount": "float64",
    "category": "object",
    "is_wants": "bool",
    "is_impulse": "bool"
}

EXPECTED_CATEGORIES = {"food", "transport", "shopping", "bills", "entertainment", "other"}

def test_transactions_csv_exists_and_schema():
    """Assert (a) Output schema of CSV matches all 9 columns and types."""
    assert os.path.exists(DATA_PATH), f"Raw transactions file not found at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    
    assert list(df.columns) == list(EXPECTED_SCHEMA.keys()), (
        f"Columns mismatch! Expected: {list(EXPECTED_SCHEMA.keys())}, Got: {list(df.columns)}"
    )
    
    # Check data types
    for col, expected_dtype in EXPECTED_SCHEMA.items():
        assert str(df[col].dtype) == expected_dtype, (
            f"Column {col} dtype mismatch. Expected {expected_dtype}, got {df[col].dtype}"
        )
    
    # Check category values
    unique_cats = set(df["category"].unique())
    assert unique_cats.issubset(EXPECTED_CATEGORIES), (
        f"Unexpected categories found: {unique_cats - EXPECTED_CATEGORIES}"
    )

def test_no_missing_values_in_required_columns():
    """Assert (b) No missing/null values in required columns."""
    df = pd.read_csv(DATA_PATH)
    required_cols = ["transaction_id", "date", "time", "merchant", "memo", "amount", "category", "is_wants", "is_impulse"]
    for col in required_cols:
        null_count = df[col].isnull().sum()
        assert null_count == 0, f"Column '{col}' contains {null_count} null/missing values!"

def test_row_count_consistency():
    """Assert (d) Total rows generated = train_rows + test_rows."""
    raw_df = pd.read_csv(DATA_PATH)
    train_df = pd.read_csv("data/processed/train_split.csv")
    test_df = pd.read_csv("data/processed/test_split.csv")
    
    assert len(raw_df) > 0, "Raw dataset is empty!"
    assert len(raw_df) == len(train_df) + len(test_df), (
        f"Row count mismatch! Raw: {len(raw_df)}, Train+Test: {len(train_df) + len(test_df)}"
    )

def test_text_preprocessing_and_tokenization():
    """Test text cleaning and PyThaiNLP tokenization with mixed Thai/English input."""
    # Test English + Thai + special characters
    sample_text = "GrabFood! สั่งข้าวเหนียวหมูปิ้ง @ 120-THB"
    cleaned = clean_text(sample_text)
    assert "grabfood" in cleaned
    assert "@" not in cleaned
    assert "!" not in cleaned
    
    tokens = tokenize_text(sample_text)
    assert len(tokens) > 0
    assert "grabfood" in tokens or "grab" in tokens
    
    feature_text = prepare_text_feature("Starbucks Siam Paragon", "Caramel Macchiato เย็น")
    assert isinstance(feature_text, str)
    assert len(feature_text) > 0

def test_ml_macro_f1_beats_baseline():
    """Assert (c) Macro-F1 of ML strictly outperforms keyword baseline."""
    assert os.path.exists(METRICS_PATH), f"Metrics JSON not found at {METRICS_PATH}"
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    ml_f1 = metrics["value"]
    baseline_f1 = metrics["baseline_value"]
    
    assert metrics["ml_beats_baseline"] is True, "ml_beats_baseline flag is False in metrics JSON!"
    assert ml_f1 > baseline_f1, (
        f"ML Macro-F1 ({ml_f1:.4f}) did NOT beat Keyword Baseline Macro-F1 ({baseline_f1:.4f})"
    )

def test_artifacts_saved_correctly():
    """Assert fitted vectorizer and model joblib files exist and can be loaded."""
    artifact_files = os.listdir(ARTIFACTS_DIR)
    vec_files = [f for f in artifact_files if f.startswith("vectorizer_phase1_") and f.endswith(".joblib")]
    logreg_files = [f for f in artifact_files if f.startswith("logreg_phase1_") and f.endswith(".joblib")]
    lgbm_files = [f for f in artifact_files if f.startswith("lightgbm_phase1_") and f.endswith(".joblib")]
    
    assert len(vec_files) >= 1, "No vectorizer artifact found in models_artifacts/"
    assert len(logreg_files) >= 1, "No logreg artifact found in models_artifacts/"
    assert len(lgbm_files) >= 1, "No lightgbm artifact found in models_artifacts/"
    
    # Test loading vectorizer
    vec_path = os.path.join(ARTIFACTS_DIR, sorted(vec_files)[-1])
    vec = load_vectorizer(vec_path)
    transformed = vec.transform(["กาแฟ อเมซอน เย็น"])
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > 0
