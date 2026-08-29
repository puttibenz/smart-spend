"""
SmartSpend AI - Machine Learning Impulse Model (v2 ML)
Trains and evaluates ML models (Logistic Regression & LightGBM) on behavioral features,
enforces v2_f1 > v1_f1 acceptance criteria, saves artifacts, and compares against v1 rule-based baseline.
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
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from src.scoring.impulse_rules import ImpulseRuleScorer, is_late_night, is_payday_window, load_config

def build_behavioral_features(df: pd.DataFrame, scorer: ImpulseRuleScorer) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extracts chronological expanding behavioral features for ML training.
    """
    scored_df = scorer.score_dataframe(df)
    
    late_flags = [int(is_late_night(t, scorer.late_night_window)) for t in scored_df["time"]]
    payday_flags = [int(is_payday_window(d, scorer.payday_days)) for d in scored_df["date"]]
    wants_flags = scored_df["is_wants"].astype(int).values
    z_scores_filled = scored_df["z_score"].fillna(0.0).values
    amounts = scored_df["amount"].values
    
    dt_series = pd.to_datetime(scored_df["date"] + " " + scored_df["time"])
    hours = dt_series.dt.hour.values
    days_of_week = dt_series.dt.dayofweek.values
    is_weekend = (days_of_week >= 5).astype(int)
    
    feature_matrix = pd.DataFrame({
        "late_night": late_flags,
        "payday": payday_flags,
        "is_wants": wants_flags,
        "z_score": z_scores_filled,
        "amount": amounts,
        "hour": hours,
        "day_of_week": days_of_week,
        "is_weekend": is_weekend
    })
    
    y = scored_df["is_impulse"].values.astype(int)
    return feature_matrix, y

def train_and_evaluate_v2(
    data_path: str = "data/raw/transactions.csv",
    config_path: str = "config.yaml",
    artifacts_dir: str = "models_artifacts",
    metrics_path: str = "outputs/metrics/phase2_metrics.json"
) -> dict:
    config = load_config(config_path)
    random_state = config["model_training"]["random_state"]
    test_size = config["model_training"]["test_size"]
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    df = pd.read_csv(data_path)
    scorer = ImpulseRuleScorer(config)
    
    print("Extracting behavioral features for v2 ML training...")
    X_df, y = build_behavioral_features(df, scorer)
    feature_names = list(X_df.columns)
    
    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    print(f"Train samples: {len(X_train)} (Positive: {sum(y_train)}) | Test samples: {len(X_test)} (Positive: {sum(y_test)})")
    
    # 1. Train Logistic Regression
    print("Training Logistic Regression on behavioral features...")
    logreg = LogisticRegression(
        random_state=random_state,
        class_weight="balanced",
        max_iter=1000
    )
    logreg.fit(X_train, y_train)
    
    # 2. Train LightGBM
    print("Training LightGBM Classifier on behavioral features...")
    lgbm = LGBMClassifier(
        random_state=random_state,
        class_weight="balanced",
        n_estimators=80,
        learning_rate=0.08,
        verbose=-1
    )
    lgbm.fit(X_train, y_train)
    
    # Evaluate Models on Test Set
    def eval_model(model, X, y_true, name):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        acc = float(round(accuracy_score(y_true, y_pred), 4))
        prec = float(round(precision_score(y_true, y_pred, zero_division=0), 4))
        rec = float(round(recall_score(y_true, y_pred, zero_division=0), 4))
        f1 = float(round(f1_score(y_true, y_pred, zero_division=0), 4))
        roc_auc = float(round(roc_auc_score(y_true, y_proba), 4))
        pr_auc = float(round(average_precision_score(y_true, y_proba), 4))
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        return {
            "model_name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm
        }
        
    metrics_logreg = eval_model(logreg, X_test, y_test, "Logistic Regression (v2 ML)")
    metrics_lgbm = eval_model(lgbm, X_test, y_test, "LightGBM (v2 ML)")
    
    best_v2_model_name = "Logistic Regression" if metrics_logreg["f1_score"] >= metrics_lgbm["f1_score"] else "LightGBM"
    best_v2_metrics = metrics_logreg if best_v2_model_name == "Logistic Regression" else metrics_lgbm
    best_v2_model = logreg if best_v2_model_name == "Logistic Regression" else lgbm
    
    # Save Best v2 Artifact
    os.makedirs(artifacts_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d")
    model_save_path = os.path.join(artifacts_dir, f"impulse_model_phase2_{today_str}.joblib")
    
    joblib.dump({
        "model": best_v2_model,
        "feature_names": feature_names,
        "metrics": best_v2_metrics,
        "model_type": best_v2_model_name
    }, model_save_path)
    print(f"Saved v2 model artifact to: {model_save_path}")
    
    # Load v1 metrics to compare
    v1_f1 = 0.0
    existing_payload = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            existing_payload = json.load(f)
            v1_f1 = existing_payload.get("v1_rule_based", {}).get("f1_score", 0.0)

    v2_f1 = best_v2_metrics["f1_score"]
    v2_beats_v1 = bool(v2_f1 > v1_f1)
    
    # Update metrics JSON
    existing_payload["v2_ml"] = {
        "best_model": best_v2_model_name,
        "precision": best_v2_metrics["precision"],
        "recall": best_v2_metrics["recall"],
        "f1_score": best_v2_metrics["f1_score"],
        "accuracy": best_v2_metrics["accuracy"],
        "roc_auc": best_v2_metrics["roc_auc"],
        "pr_auc": best_v2_metrics["pr_auc"],
        "models_evaluated": {
            "logistic_regression": metrics_logreg,
            "lightgbm": metrics_lgbm
        },
        "feature_names": feature_names,
        "artifact_path": model_save_path
    }
    
    existing_payload["v1_vs_v2_comparison"] = {
        "v1_rule_based_f1": v1_f1,
        "v2_ml_f1": v2_f1,
        "f1_gain": float(round(v2_f1 - v1_f1, 4)),
        "v2_beats_v1": v2_beats_v1
    }
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing_payload, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*65)
    print("PHASE 2: v2 ML IMPULSE MODEL EVALUATION RESULTS")
    print("="*65)
    print(f"v1 Rule-Based F1    : {v1_f1:.4f}")
    print(f"v2 ML Best Model    : {best_v2_model_name}")
    print(f"v2 Precision        : {best_v2_metrics['precision']:.4f}")
    print(f"v2 Recall           : {best_v2_metrics['recall']:.4f}")
    print(f"v2 F1-Score         : {v2_f1:.4f} (Gain: {v2_f1 - v1_f1:+.4f})")
    print(f"v2 ROC-AUC          : {best_v2_metrics['roc_auc']:.4f} | PR-AUC: {best_v2_metrics['pr_auc']:.4f}")
    print(f"v2 Beats v1 Baseline: {'PASS [YES]' if v2_beats_v1 else 'FAIL [NO]'}")
    print(f"Metrics updated at  : {metrics_path}")
    print("="*65)
    
    # Enforce Fallback Guardrail (Fail-Stop)
    if not v2_beats_v1:
        err_msg = (
            f"FALLBACK TRIGGERED: v2 ML Model ({best_v2_model_name}) F1 ({v2_f1:.4f}) "
            f"did NOT beat v1 Rule-Based F1 ({v1_f1:.4f}). Halting per Phase 2 Protocol."
        )
        print(f"\n[ERROR] {err_msg}")
        raise RuntimeError(err_msg)
        
    return existing_payload

if __name__ == "__main__":
    from typing import Tuple
    train_and_evaluate_v2()
