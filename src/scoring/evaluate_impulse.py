"""
SmartSpend AI - Impulse Scoring Evaluation & Correlation Analysis Module
Evaluates v1 Rule-Based engine against is_impulse ground truth, computes Feature Correlation Matrix,
records performance metrics, and attaches synthetic data leakage disclaimers.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

from src.scoring.impulse_rules import ImpulseRuleScorer, load_config, is_late_night, is_payday_window

def evaluate_impulse_v1(
    data_path: str = "data/raw/transactions.csv",
    config_path: str = "config.yaml",
    output_metrics_dir: str = "outputs/metrics"
) -> dict:
    config = load_config(config_path)
    thresholds = config.get("evaluation_thresholds_phase2", {})
    p_floor = thresholds.get("v1_precision_floor", 0.50)
    r_floor = thresholds.get("v1_recall_floor", 0.50)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    df = pd.read_csv(data_path)
    if "is_impulse" not in df.columns:
        raise ValueError("Column 'is_impulse' (ground truth) missing in dataset.")

    scorer = ImpulseRuleScorer(config)
    scored_df = scorer.score_dataframe(df)
    
    y_true = scored_df["is_impulse"].values.astype(bool)
    y_pred_nudge = scored_df["is_nudge_alert"].values.astype(bool)
    y_scores_normalized = scored_df["impulse_score"].values / 100.0
    
    # 1. Classification Metrics for v1
    acc = float(round(accuracy_score(y_true, y_pred_nudge), 4))
    prec = float(round(precision_score(y_true, y_pred_nudge, zero_division=0), 4))
    rec = float(round(recall_score(y_true, y_pred_nudge, zero_division=0), 4))
    f1 = float(round(f1_score(y_true, y_pred_nudge, zero_division=0), 4))
    
    try:
        roc_auc = float(round(roc_auc_score(y_true, y_scores_normalized), 4))
        pr_auc = float(round(average_precision_score(y_true, y_scores_normalized), 4))
    except ValueError:
        roc_auc = 0.5
        pr_auc = 0.5
        
    cm = confusion_matrix(y_true, y_pred_nudge).tolist()
    
    meets_bar = bool(prec >= p_floor and rec >= r_floor)

    # 2. Feature Correlation Matrix
    late_flags = [int(is_late_night(t)) for t in scored_df["time"]]
    payday_flags = [int(is_payday_window(d)) for d in scored_df["date"]]
    wants_flags = scored_df["is_wants"].astype(int).values
    z_scores_filled = scored_df["z_score"].fillna(0.0).values
    amounts = scored_df["amount"].values
    hours = [int(t.split(":")[0]) for t in scored_df["time"]]
    
    feature_df = pd.DataFrame({
        "late_night": late_flags,
        "payday": payday_flags,
        "is_wants": wants_flags,
        "z_score": z_scores_filled,
        "amount": amounts,
        "hour": hours,
        "is_impulse": y_true.astype(int)
    })
    
    pearson_corr = feature_df.corr(method="pearson")["is_impulse"].to_dict()
    spearman_corr = feature_df.corr(method="spearman")["is_impulse"].to_dict()
    
    correlation_analysis = {}
    for feat in ["late_night", "payday", "is_wants", "z_score", "amount", "hour"]:
        correlation_analysis[feat] = {
            "pearson_r": float(round(pearson_corr.get(feat, 0.0), 4)),
            "spearman_rho": float(round(spearman_corr.get(feat, 0.0), 4))
        }

    # 3. Assemble Metrics Payload
    timestamp = datetime.now().isoformat()
    disclaimer = (
        "Synthetic Data Notice: Impulse patterns (late-night delivery & payday splurge) were "
        "injected during data generation. High correlation and precision/recall reflect adherence "
        "to synthetic behavioral assumptions and serve as a proof-of-concept baseline."
    )
    
    metrics_payload = {
        "phase": 2,
        "component": "Impulse Risk Scoring Engine",
        "timestamp": timestamp,
        "data_leakage_disclaimer": disclaimer,
        "v1_rule_based": {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "accuracy": acc,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm,
            "v1_meets_minimum_bar": meets_bar,
            "thresholds": {
                "precision_floor": p_floor,
                "recall_floor": r_floor,
                "nudge_threshold": scorer.nudge_threshold
            }
        },
        "feature_correlation_analysis": correlation_analysis,
        "sample_counts": {
            "total_transactions": len(scored_df),
            "ground_truth_impulse_true": int(sum(y_true)),
            "ground_truth_impulse_false": int(sum(~y_true)),
            "v1_nudge_alerts_count": int(sum(y_pred_nudge)),
            "cold_start_samples": int(sum(scored_df["is_cold_start"]))
        }
    }
    
    os.makedirs(output_metrics_dir, exist_ok=True)
    out_path = os.path.join(output_metrics_dir, "phase2_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
        
    print("\n" + "="*65)
    print("PHASE 2: v1 RULE-BASED IMPULSE SCORING EVALUATION")
    print("="*65)
    print(f"Total Transactions : {len(scored_df)}")
    print(f"v1 Precision       : {prec:.4f} (Floor >= {p_floor:.2f}) -> {'PASS' if prec >= p_floor else 'WARN'}")
    print(f"v1 Recall          : {rec:.4f} (Floor >= {r_floor:.2f}) -> {'PASS' if rec >= r_floor else 'WARN'}")
    print(f"v1 F1-Score        : {f1:.4f}")
    print(f"v1 ROC-AUC         : {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"v1 Meets Min Bar   : {meets_bar}")
    
    print("\nFeature Correlation with is_impulse:")
    for feat, scores in correlation_analysis.items():
        print(f"  - {feat:12s}: Pearson r={scores['pearson_r']:+.4f} | Spearman rho={scores['spearman_rho']:+.4f}")
        
    print(f"\nMetrics saved to: {out_path}")
    print("="*65)
    
    return metrics_payload

if __name__ == "__main__":
    evaluate_impulse_v1()
