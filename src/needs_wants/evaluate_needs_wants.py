"""
SmartSpend AI - Needs vs Wants Evaluation Module
Evaluates classify_needs_wants predictions against is_wants ground truth,
computes classification metrics, saves metrics JSON, and enforces accuracy >= 0.70.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.needs_wants.classify_needs_wants import NeedsWantsClassifier, load_config

def evaluate_needs_wants(
    data_path: str = "data/raw/transactions.csv",
    config_path: str = "config.yaml",
    output_metrics_dir: str = "outputs/metrics"
) -> dict:
    config = load_config(config_path)
    acc_floor = config.get("evaluation_thresholds_phase2", {}).get("needs_wants_accuracy_floor", 0.70)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Transactions data not found at {data_path}")
        
    df = pd.read_csv(data_path)
    if "is_wants" not in df.columns:
        raise ValueError("Column 'is_wants' (ground truth) missing in transactions dataset.")
        
    classifier = NeedsWantsClassifier(config)
    y_pred = classifier.classify_dataframe(df).values
    y_true = df["is_wants"].values.astype(bool)
    
    acc = float(round(accuracy_score(y_true, y_pred), 4))
    prec = float(round(precision_score(y_true, y_pred, zero_division=0), 4))
    rec = float(round(recall_score(y_true, y_pred, zero_division=0), 4))
    f1 = float(round(f1_score(y_true, y_pred, zero_division=0), 4))
    
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true, y_pred,
        target_names=["Needs (False)", "Wants (True)"],
        output_dict=True,
        zero_division=0
    )
    
    # Category-level accuracy breakdown
    category_acc = {}
    for cat in df["category"].unique():
        cat_mask = (df["category"] == cat)
        c_acc = float(round(accuracy_score(y_true[cat_mask], y_pred[cat_mask]), 4))
        category_acc[cat] = {
            "total_samples": int(sum(cat_mask)),
            "accuracy": c_acc,
            "true_wants_count": int(sum(y_true[cat_mask])),
            "pred_wants_count": int(sum(y_pred[cat_mask]))
        }
        
    meets_bar = bool(acc >= acc_floor)
    timestamp = datetime.now().isoformat()
    
    metrics_payload = {
        "phase": 2,
        "component": "Needs vs Wants Classifier",
        "timestamp": timestamp,
        "metric_name": "accuracy",
        "value": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "accuracy_floor": acc_floor,
        "meets_minimum_bar": meets_bar,
        "confusion_matrix": cm,
        "classification_report": report,
        "category_breakdown": category_acc,
        "sample_counts": {
            "total_transactions": len(df),
            "ground_truth_wants": int(sum(y_true)),
            "ground_truth_needs": int(sum(~y_true)),
            "predicted_wants": int(sum(y_pred)),
            "predicted_needs": int(sum(~y_pred))
        }
    }
    
    os.makedirs(output_metrics_dir, exist_ok=True)
    out_file = os.path.join(output_metrics_dir, "phase2_needs_wants_eval.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*60)
    print("NEEDS VS WANTS EVALUATION RESULTS")
    print("="*60)
    print(f"Total Transactions: {len(df)}")
    print(f"Accuracy  : {acc:.4f} (Target Floor >= {acc_floor:.2f}) -> {'PASS' if meets_bar else 'FAIL'}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"\nSaved metrics to: {out_file}")
    print("="*60)
    
    if not meets_bar:
        err_msg = f"NEEDS/WANTS ACCURACY FAILED: Accuracy {acc:.4f} < Floor {acc_floor:.4f}. Halting Phase 2."
        print(f"\n[ERROR] {err_msg}")
        raise RuntimeError(err_msg)
        
    return metrics_payload

if __name__ == "__main__":
    evaluate_needs_wants()
