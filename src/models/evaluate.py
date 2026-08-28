"""
SmartSpend AI - Evaluation Module
Evaluates baseline and ML classifiers on the test set, generates confusion matrices,
saves metrics JSON according to schema, and enforces the ML vs Baseline fallback guardrail.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import yaml
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.models.train_classifier import KeywordBaselineClassifier, load_config

def evaluate_models(
    test_split_path: str = "data/processed/test_split.csv",
    config_path: str = "config.yaml",
    artifacts_dir: str = "models_artifacts",
    output_metrics_dir: str = "outputs/metrics"
) -> dict:
    config = load_config(config_path)
    
    if not os.path.exists(test_split_path):
        raise FileNotFoundError(f"Test split dataset not found at {test_split_path}. Run train_classifier.py first.")
    
    test_df = pd.read_csv(test_split_path)
    y_test = test_df["category"].values
    raw_texts = (test_df["merchant"].fillna("") + " " + test_df["memo"].fillna("")).values
    processed_texts = test_df["processed_text"].values
    
    # Locate saved vectorizer and models in models_artifacts
    artifact_files = os.listdir(artifacts_dir)
    vec_files = [f for f in artifact_files if f.startswith("vectorizer_phase1_") and f.endswith(".joblib")]
    logreg_files = [f for f in artifact_files if f.startswith("logreg_phase1_") and f.endswith(".joblib")]
    lgbm_files = [f for f in artifact_files if f.startswith("lightgbm_phase1_") and f.endswith(".joblib")]
    
    if not (vec_files and logreg_files and lgbm_files):
        raise FileNotFoundError("Missing artifact files in models_artifacts/. Please ensure all models and vectorizer are saved.")
    
    # Load most recent artifacts
    vec_path = os.path.join(artifacts_dir, sorted(vec_files)[-1])
    logreg_path = os.path.join(artifacts_dir, sorted(logreg_files)[-1])
    lgbm_path = os.path.join(artifacts_dir, sorted(lgbm_files)[-1])
    
    vectorizer = joblib.load(vec_path)
    logreg = joblib.load(logreg_path)
    lgbm = joblib.load(lgbm_path)
    
    # Vectorize test texts
    X_test_tfidf = vectorizer.transform(processed_texts)
    
    # 1. Baseline Predictions
    baseline_clf = KeywordBaselineClassifier(config["baseline_keywords"])
    y_pred_baseline = baseline_clf.predict(raw_texts)
    
    # 2. Logistic Regression Predictions
    y_pred_logreg = logreg.predict(X_test_tfidf)
    
    # 3. LightGBM Predictions
    y_pred_lgbm = lgbm.predict(X_test_tfidf)
    
    # Calculate Metrics
    categories = sorted(list(np.unique(y_test)))
    
    def get_metrics_dict(y_true, y_pred, model_name):
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_true, y_pred, target_names=categories, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=categories).tolist()
        return {
            "model_name": model_name,
            "accuracy": float(round(acc, 4)),
            "macro_f1": float(round(macro_f1, 4)),
            "weighted_f1": float(round(weighted_f1, 4)),
            "confusion_matrix": cm,
            "labels": categories,
            "classification_report": report
        }

    metrics_baseline = get_metrics_dict(y_test, y_pred_baseline, "Keyword Baseline")
    metrics_logreg = get_metrics_dict(y_test, y_pred_logreg, "Logistic Regression")
    metrics_lgbm = get_metrics_dict(y_test, y_pred_lgbm, "LightGBM")
    
    best_ml_model = "Logistic Regression" if metrics_logreg["macro_f1"] >= metrics_lgbm["macro_f1"] else "LightGBM"
    best_ml_metrics = metrics_logreg if best_ml_model == "Logistic Regression" else metrics_lgbm
    
    ml_beats_baseline = bool(best_ml_metrics["macro_f1"] > metrics_baseline["macro_f1"])
    
    timestamp = datetime.now().isoformat()
    
    # Structure metrics output JSON according to AGENTS.md schema requirements:
    # Must have: phase, component, metric_name, value, baseline_value, timestamp
    metrics_payload = {
        "phase": 1,
        "component": "Expense Categorization",
        "metric_name": "macro_f1",
        "value": best_ml_metrics["macro_f1"],
        "baseline_value": metrics_baseline["macro_f1"],
        "best_ml_model": best_ml_model,
        "ml_beats_baseline": ml_beats_baseline,
        "timestamp": timestamp,
        "models_evaluated": {
            "baseline": metrics_baseline,
            "logistic_regression": metrics_logreg,
            "lightgbm": metrics_lgbm
        },
        "sample_counts": {
            "test_total": len(y_test),
            "per_category": {cat: int(sum(y_test == cat)) for cat in categories}
        }
    }
    
    os.makedirs(output_metrics_dir, exist_ok=True)
    metrics_file = os.path.join(output_metrics_dir, "phase1_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("PHASE 1 EVALUATION RESULTS")
    print("="*60)
    print(f"Test Set Size: {len(y_test)} transactions")
    print(f"\n1. Keyword Baseline    -> Accuracy: {metrics_baseline['accuracy']:.4f} | Macro-F1: {metrics_baseline['macro_f1']:.4f}")
    print(f"2. Logistic Regression -> Accuracy: {metrics_logreg['accuracy']:.4f} | Macro-F1: {metrics_logreg['macro_f1']:.4f}")
    print(f"3. LightGBM            -> Accuracy: {metrics_lgbm['accuracy']:.4f} | Macro-F1: {metrics_lgbm['macro_f1']:.4f}")
    print(f"\nBest ML Model: {best_ml_model} (Macro-F1: {best_ml_metrics['macro_f1']:.4f} vs Baseline: {metrics_baseline['macro_f1']:.4f})")
    print(f"ML Beats Baseline: {'PASS [YES]' if ml_beats_baseline else 'FAIL [NO]'}")
    print(f"Metrics saved to: {metrics_file}")
    print("="*60)
    
    # Enforce Fallback Guardrail
    if not ml_beats_baseline:
        error_msg = (
            f"FALLBACK TRIGGERED: Best ML Model ({best_ml_model}) Macro-F1 ({best_ml_metrics['macro_f1']:.4f}) "
            f"did NOT beat Keyword Baseline ({metrics_baseline['macro_f1']:.4f}). "
            "Halting pipeline per AGENTS.md fallback protocol. Please review outputs/metrics/phase1_metrics.json."
        )
        print(f"\n[ERROR] {error_msg}")
        raise RuntimeError(error_msg)
        
    return metrics_payload

if __name__ == "__main__":
    evaluate_models()
