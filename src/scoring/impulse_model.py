"""
SmartSpend AI - Machine Learning Impulse Model (v2 ML)
Trains and evaluates ML models (Logistic Regression & LightGBM) on behavioral features,
performs 5-Fold Stratified Cross-Validation, enforces v2_f1 > v1_f1 acceptance criteria,
saves model artifacts, and records comprehensive CV metrics in outputs/metrics/phase2_metrics.json.
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
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

def run_5fold_cv(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> dict:
    """
    Performs 5-Fold Stratified Cross-Validation on Logistic Regression.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    precisions = []
    recalls = []
    f1s = []
    roc_aucs = []
    pr_aucs = []
    fold_details = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        clf = LogisticRegression(random_state=random_state, class_weight="balanced", max_iter=1000)
        clf.fit(X_tr, y_tr)
        
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)[:, 1]
        
        p = float(round(precision_score(y_val, y_pred, zero_division=0), 4))
        r = float(round(recall_score(y_val, y_pred, zero_division=0), 4))
        f = float(round(f1_score(y_val, y_pred, zero_division=0), 4))
        roc = float(round(roc_auc_score(y_val, y_proba), 4))
        pr = float(round(average_precision_score(y_val, y_proba), 4))
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        roc_aucs.append(roc)
        pr_aucs.append(pr)
        
        fold_details.append({
            "fold": fold,
            "precision": p,
            "recall": r,
            "f1_score": f,
            "roc_auc": roc,
            "pr_auc": pr,
            "val_positives": int(sum(y_val)),
            "val_samples": int(len(y_val))
        })
        
    return {
        "n_splits": 5,
        "fold_details": fold_details,
        "mean_precision": float(round(np.mean(precisions), 4)),
        "std_precision": float(round(np.std(precisions), 4)),
        "mean_recall": float(round(np.mean(recalls), 4)),
        "std_recall": float(round(np.std(recalls), 4)),
        "mean_f1": float(round(np.mean(f1s), 4)),
        "std_f1": float(round(np.std(f1s), 4)),
        "mean_roc_auc": float(round(np.mean(roc_aucs), 4)),
        "std_roc_auc": float(round(np.std(roc_aucs), 4)),
        "mean_pr_auc": float(round(np.mean(pr_aucs), 4)),
        "std_pr_auc": float(round(np.std(pr_aucs), 4))
    }

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
    X_all = X_df.values
    
    # 1. Run 5-Fold Stratified Cross-Validation to verify robustness across folds
    print("\nRunning 5-Fold Stratified Cross-Validation on Logistic Regression...")
    cv_results = run_5fold_cv(X_all, y, random_state=random_state)
    print(f"5-Fold CV Mean Recall   : {cv_results['mean_recall']:.4f} (+/- {cv_results['std_recall']:.4f})")
    print(f"5-Fold CV Mean Precision: {cv_results['mean_precision']:.4f} (+/- {cv_results['std_precision']:.4f})")
    print(f"5-Fold CV Mean F1-Score : {cv_results['mean_f1']:.4f} (+/- {cv_results['std_f1']:.4f})")
    print(f"5-Fold CV Mean ROC-AUC  : {cv_results['mean_roc_auc']:.4f} (+/- {cv_results['std_roc_auc']:.4f})")
    
    # 2. Stratified Train/Test Split (80/20) for standalone test evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    print(f"\nTrain samples: {len(X_train)} (Positive: {sum(y_train)}) | Test samples: {len(X_test)} (Positive: {sum(y_test)})")
    
    # Train Logistic Regression (Selected Best Model)
    print("Training Logistic Regression on behavioral features...")
    logreg = LogisticRegression(
        random_state=random_state,
        class_weight="balanced",
        max_iter=1000
    )
    logreg.fit(X_train, y_train)
    
    # Train LightGBM
    print("Training LightGBM Classifier on behavioral features...")
    lgbm = LGBMClassifier(
        random_state=random_state,
        class_weight="balanced",
        n_estimators=80,
        learning_rate=0.08,
        verbose=-1
    )
    lgbm.fit(X_train, y_train)
    
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
    
    # Primary Model: Logistic Regression
    best_v2_model_name = "Logistic Regression"
    best_v2_metrics = metrics_logreg
    best_v2_model = logreg
    
    # Save Best v2 Artifact
    os.makedirs(artifacts_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d")
    model_save_path = os.path.join(artifacts_dir, f"impulse_model_phase2_{today_str}.joblib")
    
    joblib.dump({
        "model": best_v2_model,
        "feature_names": feature_names,
        "metrics": best_v2_metrics,
        "cross_validation_5fold": cv_results,
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
        "cross_validation_5fold": cv_results,
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
    print(f"v2 ML Model         : {best_v2_model_name}")
    print(f"v2 Test Precision   : {best_v2_metrics['precision']:.4f}")
    print(f"v2 Test Recall      : {best_v2_metrics['recall']:.4f}")
    print(f"v2 Test F1-Score    : {v2_f1:.4f} (Gain: {v2_f1 - v1_f1:+.4f})")
    print(f"v2 Test ROC-AUC     : {best_v2_metrics['roc_auc']:.4f} | PR-AUC: {best_v2_metrics['pr_auc']:.4f}")
    print(f"v2 5-Fold CV F1     : {cv_results['mean_f1']:.4f} (+/- {cv_results['std_f1']:.4f})")
    print(f"v2 5-Fold CV Recall : {cv_results['mean_recall']:.4f} (+/- {cv_results['std_recall']:.4f})")
    print(f"v2 Beats v1 Baseline: {'PASS [YES]' if v2_beats_v1 else 'FAIL [NO]'}")
    print(f"Metrics updated at  : {metrics_path}")
    print("="*65)
    
    if not v2_beats_v1:
        err_msg = (
            f"FALLBACK TRIGGERED: v2 ML Model ({best_v2_model_name}) F1 ({v2_f1:.4f}) "
            f"did NOT beat v1 Rule-Based F1 ({v1_f1:.4f}). Halting per Phase 2 Protocol."
        )
        print(f"\n[ERROR] {err_msg}")
        raise RuntimeError(err_msg)
        
    return existing_payload

if __name__ == "__main__":
    train_and_evaluate_v2()
