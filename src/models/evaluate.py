"""
SmartSpend AI - Evaluation Module (v5 Final Protocol)
Computes comprehensive evaluation metrics: Macro-F1, Seen vs Unseen Merchant Overlap Analysis,
Sub-type breakdown, Normalized Confusion Matrix, Confidence Margins, Error Attribution,
exports phase1_error_analysis.csv, saves phase1_metrics.json, and enforces quantitative guardrails.
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
    train_split_path: str = "data/processed/train_split.csv",
    config_path: str = "config.yaml",
    artifacts_dir: str = "models_artifacts",
    output_metrics_dir: str = "outputs/metrics"
) -> dict:
    config = load_config(config_path)
    eval_thresholds = config["evaluation_thresholds"]
    
    macro_f1_floor = eval_thresholds["macro_f1_floor"]
    macro_f1_ceiling = eval_thresholds["macro_f1_ceiling"]
    unseen_acc_floor = eval_thresholds["unseen_accuracy_floor"]
    max_gen_gap = eval_thresholds["max_generalization_gap"]
    
    if not os.path.exists(test_split_path) or not os.path.exists(train_split_path):
        raise FileNotFoundError("Split datasets not found. Run train_classifier.py first.")
    
    train_df = pd.read_csv(train_split_path)
    test_df = pd.read_csv(test_split_path)
    
    y_test = test_df["category"].values
    raw_texts = (test_df["merchant"].fillna("") + " " + test_df["memo"].fillna("")).values
    processed_texts = test_df["processed_text"].values
    is_unseen = test_df["is_unseen_merchant"].values
    sub_types = test_df["sub_type"].values
    
    # Locate latest saved artifacts
    artifact_files = os.listdir(artifacts_dir)
    vec_files = sorted([f for f in artifact_files if f.startswith("vectorizer_phase1_") and f.endswith(".joblib")])
    logreg_files = sorted([f for f in artifact_files if f.startswith("logreg_phase1_") and f.endswith(".joblib")])
    lgbm_files = sorted([f for f in artifact_files if f.startswith("lightgbm_phase1_") and f.endswith(".joblib")])
    
    vectorizer = joblib.load(os.path.join(artifacts_dir, vec_files[-1]))
    logreg = joblib.load(os.path.join(artifacts_dir, logreg_files[-1]))
    lgbm = joblib.load(os.path.join(artifacts_dir, lgbm_files[-1]))
    
    X_test_tfidf = vectorizer.transform(processed_texts)
    categories = sorted(list(np.unique(y_test)))
    
    # Predictions
    baseline_clf = KeywordBaselineClassifier(config["baseline_keywords"])
    y_pred_baseline = np.array(baseline_clf.predict(raw_texts))
    
    y_pred_logreg = logreg.predict(X_test_tfidf)
    y_proba_logreg = logreg.predict_proba(X_test_tfidf)
    
    y_pred_lgbm = lgbm.predict(X_test_tfidf)
    y_proba_lgbm = lgbm.predict_proba(X_test_tfidf)

    def calc_metrics(y_true, y_pred, model_name):
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_true, y_pred, target_names=categories, output_dict=True, zero_division=0)
        raw_cm = confusion_matrix(y_true, y_pred, labels=categories)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_cm = (raw_cm.astype('float') / raw_cm.sum(axis=1)[:, np.newaxis] * 100)
            norm_cm = np.nan_to_num(norm_cm, nan=0.0).round(2).tolist()
            
        return {
            "model_name": model_name,
            "accuracy": float(round(acc, 4)),
            "macro_f1": float(round(macro_f1, 4)),
            "weighted_f1": float(round(weighted_f1, 4)),
            "raw_confusion_matrix": raw_cm.tolist(),
            "normalized_confusion_matrix_pct": norm_cm,
            "labels": categories,
            "classification_report": report
        }

    metrics_baseline = calc_metrics(y_test, y_pred_baseline, "Keyword Baseline")
    metrics_logreg = calc_metrics(y_test, y_pred_logreg, "Logistic Regression")
    metrics_lgbm = calc_metrics(y_test, y_pred_lgbm, "LightGBM")
    
    # Select Best ML Model
    best_ml_model_name = "Logistic Regression" if metrics_logreg["macro_f1"] >= metrics_lgbm["macro_f1"] else "LightGBM"
    best_ml_metrics = metrics_logreg if best_ml_model_name == "Logistic Regression" else metrics_lgbm
    best_y_pred = y_pred_logreg if best_ml_model_name == "Logistic Regression" else y_pred_lgbm
    best_y_proba = y_proba_logreg if best_ml_model_name == "Logistic Regression" else y_proba_lgbm
    
    # 1. Merchant Overlap Analysis (Seen vs Unseen)
    seen_mask = (~is_unseen)
    unseen_mask = is_unseen
    
    seen_acc = float(round(accuracy_score(y_test[seen_mask], best_y_pred[seen_mask]), 4))
    unseen_acc = float(round(accuracy_score(y_test[unseen_mask], best_y_pred[unseen_mask]), 4))
    seen_macro_f1 = float(round(f1_score(y_test[seen_mask], best_y_pred[seen_mask], average="macro", zero_division=0), 4))
    unseen_macro_f1 = float(round(f1_score(y_test[unseen_mask], best_y_pred[unseen_mask], average="macro", zero_division=0), 4))
    gen_gap = float(round(seen_acc - unseen_acc, 4))
    
    # Breakdown per category
    category_overlap_analysis = {}
    for cat in categories:
        cat_mask = (y_test == cat)
        c_seen_mask = cat_mask & seen_mask
        c_unseen_mask = cat_mask & unseen_mask
        
        c_seen_acc = float(round(accuracy_score(y_test[c_seen_mask], best_y_pred[c_seen_mask]), 4)) if sum(c_seen_mask) > 0 else None
        c_unseen_acc = float(round(accuracy_score(y_test[c_unseen_mask], best_y_pred[c_unseen_mask]), 4)) if sum(c_unseen_mask) > 0 else None
        
        category_overlap_analysis[cat] = {
            "total_test_support": int(sum(cat_mask)),
            "seen_samples": int(sum(c_seen_mask)),
            "seen_accuracy": c_seen_acc,
            "unseen_samples": int(sum(c_unseen_mask)),
            "unseen_accuracy": c_unseen_acc,
            "is_noisy_metric": bool(sum(cat_mask) < 20)
        }

    # Breakdown per sub-type
    unique_subtypes = sorted(list(np.unique(sub_types)))
    subtype_overlap_analysis = {}
    for st in unique_subtypes:
        st_mask = (sub_types == st)
        st_seen_mask = st_mask & seen_mask
        st_unseen_mask = st_mask & unseen_mask
        
        st_seen_acc = float(round(accuracy_score(y_test[st_seen_mask], best_y_pred[st_seen_mask]), 4)) if sum(st_seen_mask) > 0 else None
        st_unseen_acc = float(round(accuracy_score(y_test[st_unseen_mask], best_y_pred[st_unseen_mask]), 4)) if sum(st_unseen_mask) > 0 else None
        
        subtype_overlap_analysis[st] = {
            "total_samples": int(sum(st_mask)),
            "seen_samples": int(sum(st_seen_mask)),
            "seen_accuracy": st_seen_acc,
            "unseen_samples": int(sum(st_unseen_mask)),
            "unseen_accuracy": st_unseen_acc
        }

    # 2. Confidence Margins Calculation
    # Top-1 vs Top-2 probability
    sorted_proba = np.sort(best_y_proba, axis=1)[:, ::-1]
    top1_probs = sorted_proba[:, 0]
    top2_probs = sorted_proba[:, 1]
    confidence_margins = top1_probs - top2_probs
    
    is_correct = (best_y_pred == y_test)
    mean_margin_correct = float(round(np.mean(confidence_margins[is_correct]), 4)) if sum(is_correct) > 0 else 0.0
    mean_margin_misclassified = float(round(np.mean(confidence_margins[~is_correct]), 4)) if sum(~is_correct) > 0 else 0.0

    # 3. Error Attribution (Baseline vs ML)
    baseline_wrong = (y_pred_baseline != y_test)
    ml_wrong = (best_y_pred != y_test)
    
    shared_errors_count = int(sum(baseline_wrong & ml_wrong))
    ml_wins_count = int(sum(baseline_wrong & ~ml_wrong))
    baseline_wins_count = int(sum(~baseline_wrong & ml_wrong))
    both_correct_count = int(sum(~baseline_wrong & ~ml_wrong))
    
    # 4. Export outputs/metrics/phase1_error_analysis.csv
    error_records = []
    classes_list = list(logreg.classes_)
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        t_id = row.get("transaction_id", str(i))
        true_c = y_test[i]
        base_p = y_pred_baseline[i]
        ml_p = best_y_pred[i]
        
        # Determine Error Attribution Type
        if not baseline_wrong[i] and not ml_wrong[i]:
            err_type = "both_correct"
        elif baseline_wrong[i] and not ml_wrong[i]:
            err_type = "ml_wins_baseline_wrong"
        elif not baseline_wrong[i] and ml_wrong[i]:
            err_type = "baseline_wins_ml_wrong"
        else:
            err_type = "shared_error_both_wrong"
            
        error_records.append({
            "transaction_id": t_id,
            "merchant": row["merchant"],
            "memo": row["memo"],
            "category_true": true_c,
            "category_pred_baseline": base_p,
            "category_pred_logreg": y_pred_logreg[i],
            "category_pred_lgbm": y_pred_lgbm[i],
            "best_ml_pred": ml_p,
            "top1_prob": float(round(top1_probs[i], 4)),
            "top2_prob": float(round(top2_probs[i], 4)),
            "confidence_margin": float(round(confidence_margins[i], 4)),
            "is_unseen_merchant": bool(is_unseen[i]),
            "sub_type": sub_types[i],
            "is_ml_misclassified": bool(ml_wrong[i]),
            "error_attribution": err_type
        })
        
    error_df = pd.DataFrame(error_records)
    os.makedirs(output_metrics_dir, exist_ok=True)
    error_csv_path = os.path.join(output_metrics_dir, "phase1_error_analysis.csv")
    error_df.to_csv(error_csv_path, index=False, encoding="utf-8")

    # 5. Effective Train/Test Ratio
    total_samples = len(train_df) + len(test_df)
    effective_train_ratio = float(round(len(train_df) / total_samples, 4))
    effective_test_ratio = float(round(len(test_df) / total_samples, 4))

    # Construct Final Metrics Payload
    timestamp = datetime.now().isoformat()
    metrics_payload = {
        "phase": 1,
        "component": "Expense Categorization",
        "timestamp": timestamp,
        "effective_split": {
            "total_transactions": total_samples,
            "train_samples": len(train_df),
            "train_ratio_pct": float(round(effective_train_ratio * 100, 2)),
            "test_samples": len(test_df),
            "test_ratio_pct": float(round(effective_test_ratio * 100, 2))
        },
        "metric_name": "macro_f1",
        "value": best_ml_metrics["macro_f1"],
        "baseline_value": metrics_baseline["macro_f1"],
        "best_ml_model": best_ml_model_name,
        "ml_beats_baseline": bool(best_ml_metrics["macro_f1"] > metrics_baseline["macro_f1"]),
        "merchant_overlap_analysis": {
            "seen_merchant_accuracy": seen_acc,
            "seen_merchant_macro_f1": seen_macro_f1,
            "seen_merchant_samples": int(sum(seen_mask)),
            "unseen_merchant_accuracy": unseen_acc,
            "unseen_merchant_macro_f1": unseen_macro_f1,
            "unseen_merchant_samples": int(sum(unseen_mask)),
            "generalization_gap": gen_gap,
            "per_category_breakdown": category_overlap_analysis,
            "per_subtype_breakdown": subtype_overlap_analysis
        },
        "confidence_margin_analysis": {
            "mean_margin_correct_predictions": mean_margin_correct,
            "mean_margin_misclassified_predictions": mean_margin_misclassified
        },
        "error_attribution": {
            "both_correct": both_correct_count,
            "ml_wins_baseline_wrong": ml_wins_count,
            "baseline_wins_ml_wrong": baseline_wins_count,
            "shared_error_both_wrong": shared_errors_count
        },
        "models_evaluated": {
            "baseline": metrics_baseline,
            "logistic_regression": metrics_logreg,
            "lightgbm": metrics_lgbm
        }
    }
    
    metrics_json_path = os.path.join(output_metrics_dir, "phase1_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "="*70)
    print("PHASE 1 EVALUATION RESULTS (v5 Final Protocol)")
    print("="*70)
    print(f"Effective Split: {len(train_df)} train ({effective_train_ratio:.2%}) / {len(test_df)} test ({effective_test_ratio:.2%})")
    print(f"\n1. Keyword Baseline    -> Accuracy: {metrics_baseline['accuracy']:.4f} | Macro-F1: {metrics_baseline['macro_f1']:.4f}")
    print(f"2. Logistic Regression -> Accuracy: {metrics_logreg['accuracy']:.4f} | Macro-F1: {metrics_logreg['macro_f1']:.4f}")
    print(f"3. LightGBM            -> Accuracy: {metrics_lgbm['accuracy']:.4f} | Macro-F1: {metrics_lgbm['macro_f1']:.4f}")
    print(f"\nBest ML Model: {best_ml_model_name} (Macro-F1: {best_ml_metrics['macro_f1']:.4f})")
    
    print("\nMerchant Overlap Analysis (Seen vs Unseen Generalization):")
    print(f"  - Seen Merchant Accuracy   : {seen_acc:.4f} (Samples: {sum(seen_mask)})")
    print(f"  - Unseen Merchant Accuracy : {unseen_acc:.4f} (Samples: {sum(unseen_mask)})")
    print(f"  - Generalization Gap       : {gen_gap:.4f} (Threshold <= {max_gen_gap:.2f})")
    
    print("\nError Attribution Summary:")
    print(f"  - Both Correct          : {both_correct_count} tx ({both_correct_count/len(test_df):.2%})")
    print(f"  - ML Wins (Base Wrong)  : {ml_wins_count} tx ({ml_wins_count/len(test_df):.2%})")
    print(f"  - Base Wins (ML Wrong)  : {baseline_wins_count} tx ({baseline_wins_count/len(test_df):.2%})")
    print(f"  - Both Wrong            : {shared_errors_count} tx ({shared_errors_count/len(test_df):.2%})")
    
    print(f"\nExports:")
    print(f"  - Metrics JSON : {metrics_json_path}")
    print(f"  - Error CSV    : {error_csv_path}")
    print("="*70)

    # Enforce All Acceptance Guardrails (Fail-Stop)
    guardrail_failures = []
    
    if best_ml_metrics["macro_f1"] <= metrics_baseline["macro_f1"]:
        guardrail_failures.append(f"ML Macro-F1 ({best_ml_metrics['macro_f1']:.4f}) <= Baseline ({metrics_baseline['macro_f1']:.4f})")
        
    if best_ml_metrics["macro_f1"] < macro_f1_floor or best_ml_metrics["macro_f1"] > macro_f1_ceiling:
        guardrail_failures.append(f"ML Macro-F1 ({best_ml_metrics['macro_f1']:.4f}) out of bounds [{macro_f1_floor}, {macro_f1_ceiling}]")
        
    if unseen_acc < unseen_acc_floor:
        guardrail_failures.append(f"Unseen Merchant Accuracy ({unseen_acc:.4f}) < Floor ({unseen_acc_floor:.4f})")
        
    if gen_gap > max_gen_gap:
        guardrail_failures.append(f"Generalization Gap ({gen_gap:.4f}) > Maximum Gap ({max_gen_gap:.4f})")
        
    if guardrail_failures:
        err_msg = "ACCEPTANCE GUARDRAIL FAILED:\n" + "\n".join(f"  - {f}" for f in guardrail_failures)
        print(f"\n[ERROR] {err_msg}")
        raise RuntimeError(err_msg)
        
    print("\n[SUCCESS] ALL ACCEPTANCE CRITERIA PASSED 100%!")
    return metrics_payload

if __name__ == "__main__":
    evaluate_models()
