"""
SmartSpend AI - Tests for Phase 1: Expense Categorization (v5 Final Protocol)
Validates schema, non-null values, exact distribution quotas, realistic Macro-F1 bounds,
unseen merchant zero-shot integrity, generalization thresholds, minimum training samples safeguard,
sub-type coverage in unseen set, and error analysis artifact exports.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import pytest
import yaml
import pandas as pd
import joblib

from src.nlp.preprocessing import clean_text, tokenize_text, prepare_text_feature
from src.nlp.vectorizer import load_vectorizer

DATA_PATH = "data/raw/transactions.csv"
TRAIN_SPLIT_PATH = "data/processed/train_split.csv"
TEST_SPLIT_PATH = "data/processed/test_split.csv"
METRICS_PATH = "outputs/metrics/phase1_metrics.json"
ERROR_CSV_PATH = "outputs/metrics/phase1_error_analysis.csv"
ARTIFACTS_DIR = "models_artifacts"
CONFIG_PATH = "config.yaml"

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

def load_test_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def test_1_transactions_csv_schema():
    """Assert (1) Output schema of CSV matches all 9 columns and types."""
    assert os.path.exists(DATA_PATH), f"Raw transactions file not found at {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    
    for col, expected_dtype in EXPECTED_SCHEMA.items():
        assert col in df.columns, f"Missing required column: {col}"
        assert str(df[col].dtype) == expected_dtype, f"Column {col} dtype mismatch: {df[col].dtype} vs {expected_dtype}"
    
    unique_cats = set(df["category"].unique())
    assert unique_cats.issubset(EXPECTED_CATEGORIES), f"Unexpected categories: {unique_cats - EXPECTED_CATEGORIES}"

def test_2_no_missing_values():
    """Assert (2) No missing/null values in required columns."""
    df = pd.read_csv(DATA_PATH)
    required_cols = list(EXPECTED_SCHEMA.keys())
    for col in required_cols:
        null_count = df[col].isnull().sum()
        assert null_count == 0, f"Column '{col}' has {null_count} null values"

def test_3_category_distribution_quota():
    """Assert (3) Exact category distribution matches target within tolerance <= ±1.0%."""
    config = load_test_config()
    target_dist = config["data_generation"]["category_distribution"]
    tolerance = config["data_generation"].get("category_distribution_tolerance", 0.01)
    
    df = pd.read_csv(DATA_PATH)
    actual_dist = df["category"].value_counts(normalize=True).to_dict()
    
    for cat, target_pct in target_dist.items():
        actual_pct = actual_dist.get(cat, 0.0)
        gap = abs(actual_pct - target_pct)
        assert gap <= tolerance, (
            f"Category '{cat}' drift exceeded! Target: {target_pct:.2%}, Actual: {actual_pct:.2%}, Gap: {gap:.4f}"
        )

def test_4_performance_bounds_and_ml_beats_baseline():
    """Assert (4) ML Macro-F1 beats Baseline AND lies strictly within [0.75, 0.95]."""
    config = load_test_config()
    floor = config["evaluation_thresholds"]["macro_f1_floor"]
    ceiling = config["evaluation_thresholds"]["macro_f1_ceiling"]
    
    assert os.path.exists(METRICS_PATH), f"Metrics JSON not found at {METRICS_PATH}"
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    ml_f1 = metrics["value"]
    baseline_f1 = metrics["baseline_value"]
    
    assert metrics["ml_beats_baseline"] is True, "ml_beats_baseline is False in metrics!"
    assert ml_f1 > baseline_f1, f"ML F1 ({ml_f1:.4f}) did not beat Baseline F1 ({baseline_f1:.4f})"
    assert floor <= ml_f1 <= ceiling, f"ML F1 ({ml_f1:.4f}) out of bounds [{floor}, {ceiling}]"

def test_5_generalization_gap_and_unseen_floor():
    """Assert (5) Unseen accuracy >= 0.60 AND Generalization Gap <= 0.20."""
    config = load_test_config()
    unseen_floor = config["evaluation_thresholds"]["unseen_accuracy_floor"]
    max_gap = config["evaluation_thresholds"]["max_generalization_gap"]
    
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    overlap = metrics["merchant_overlap_analysis"]
    unseen_acc = overlap["unseen_merchant_accuracy"]
    gen_gap = overlap["generalization_gap"]
    
    assert unseen_acc >= unseen_floor, f"Unseen Accuracy ({unseen_acc:.4f}) < Floor ({unseen_floor:.4f})"
    assert gen_gap <= max_gap, f"Generalization Gap ({gen_gap:.4f}) > Maximum ({max_gap:.4f})"

def test_6_unseen_merchant_integrity():
    """Assert (6) 100% Zero-shot guarantee: unseen merchants in test set NEVER appear in train set."""
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    test_df = pd.read_csv(TEST_SPLIT_PATH)
    
    test_unseen_df = test_df[test_df["is_unseen_merchant"] == True]
    unseen_merchants = set(test_unseen_df["merchant"].unique())
    train_merchants = set(train_df["merchant"].unique())
    
    overlap = unseen_merchants.intersection(train_merchants)
    assert len(overlap) == 0, f"Unseen merchant data leak detected! Overlap: {overlap}"

def test_7_minimum_training_samples_safeguard():
    """Assert (7) Every category in train_split.csv has >= 40 samples."""
    config = load_test_config()
    min_samples = config["evaluation_thresholds"].get("min_train_samples_per_category", 40)
    
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    cat_counts = train_df["category"].value_counts().to_dict()
    
    for cat in EXPECTED_CATEGORIES:
        count = cat_counts.get(cat, 0)
        assert count >= min_samples, f"Category '{cat}' has only {count} training samples (< {min_samples})"

def test_8_subtype_coverage_in_unseen_set():
    """Assert (8) Every (category, sub_type) group with >= 2 merchants has at least 1 merchant in unseen set."""
    from src.data_generation.generate_synthetic_transactions import MERCHANT_POOLS
    test_df = pd.read_csv(TEST_SPLIT_PATH)
    unseen_merchants = set(test_df[test_df["is_unseen_merchant"] == True]["merchant"].unique())
    
    for cat, sub_dict in MERCHANT_POOLS.items():
        for sub_type, m_list in sub_dict.items():
            if len(m_list) >= 2:
                unseen_in_pool = [m for m in m_list if m in unseen_merchants]
                assert len(unseen_in_pool) >= 1, (
                    f"Sub-type pool '{cat}::{sub_type}' with {len(m_list)} merchants has NO unseen merchant representation!"
                )

def test_9_artifact_and_error_analysis_exports():
    """Assert (9) Model joblib files and error analysis CSV exist and have valid structure."""
    assert os.path.exists(ERROR_CSV_PATH), f"Error analysis CSV missing at {ERROR_CSV_PATH}"
    err_df = pd.read_csv(ERROR_CSV_PATH)
    expected_cols = [
        "transaction_id", "merchant", "memo", "category_true",
        "category_pred_baseline", "best_ml_pred", "confidence_margin",
        "is_unseen_merchant", "sub_type", "error_attribution"
    ]
    for col in expected_cols:
        assert col in err_df.columns, f"Missing column in error CSV: {col}"
        
    artifact_files = os.listdir(ARTIFACTS_DIR)
    vec_files = [f for f in artifact_files if f.startswith("vectorizer_phase1_") and f.endswith(".joblib")]
    logreg_files = [f for f in artifact_files if f.startswith("logreg_phase1_") and f.endswith(".joblib")]
    lgbm_files = [f for f in artifact_files if f.startswith("lightgbm_phase1_") and f.endswith(".joblib")]
    
    assert len(vec_files) >= 1, "Missing vectorizer artifact"
    assert len(logreg_files) >= 1, "Missing logreg artifact"
    assert len(lgbm_files) >= 1, "Missing lgbm artifact"
