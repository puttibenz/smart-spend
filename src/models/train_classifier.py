"""
SmartSpend AI - Train Classifier Module (v5 Final Protocol)
Implements Custom 2-Stage Stratified Split with unseen merchant routing,
minimum training sample safeguards, effective ratio reporting, and artifact persistence.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import yaml
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from src.nlp.preprocessing import prepare_text_feature
from src.nlp.vectorizer import build_vectorizer, save_vectorizer

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class KeywordBaselineClassifier:
    """
    Keyword matching baseline classifier for expense categorization.
    Matches predefined keywords from config.yaml against transaction text,
    falling back to 'other' if no keyword matches.
    """
    def __init__(self, keyword_dict: dict, fallback_category: str = "other"):
        self.keyword_dict = {
            cat: [kw.lower() for kw in kws]
            for cat, kws in keyword_dict.items()
        }
        self.fallback_category = fallback_category

    def predict_single(self, text: str) -> str:
        text_lower = text.lower()
        match_scores = {}
        for cat, kws in self.keyword_dict.items():
            score = sum(1 for kw in kws if kw in text_lower)
            if score > 0:
                match_scores[cat] = score
        
        if match_scores:
            return max(match_scores, key=match_scores.get)
        return self.fallback_category

    def predict(self, texts: list) -> list:
        return [self.predict_single(t) for t in texts]

def train_and_save_all(
    data_path: str = "data/raw/transactions.csv",
    config_path: str = "config.yaml",
    artifacts_dir: str = "models_artifacts",
    processed_dir: str = "data/processed"
):
    config = load_config(config_path)
    random_state = config["model_training"]["random_state"]
    test_size = config["model_training"]["test_size"]
    min_train_samples = config["evaluation_thresholds"].get("min_train_samples_per_category", 40)
    
    print(f"Loading transaction dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Preprocessing text features (merchant + memo)...")
    df["processed_text"] = [
        prepare_text_feature(m, memo)
        for m, memo in zip(df["merchant"], df["memo"])
    ]

    # Custom 2-Stage Stratified Split
    print("\nExecuting Custom 2-Stage Stratified Split:")
    # 1. Force all unseen merchant transactions to test set 100%
    unseen_mask = (df["is_unseen_merchant"] == True)
    test_unseen_df = df[unseen_mask].copy()
    seen_df = df[~unseen_mask].copy()
    
    print(f"  - Total transactions: {len(df)}")
    print(f"  - Unseen merchant transactions (forced to test): {len(test_unseen_df)} ({len(test_unseen_df)/len(df):.2%})")
    print(f"  - Seen merchant transactions (split 80/20): {len(seen_df)} ({len(seen_df)/len(df):.2%})")
    
    # 2. Stratified split on seen merchant pool
    train_df, test_seen_df = train_test_split(
        seen_df,
        test_size=test_size,
        stratify=seen_df["category"],
        random_state=random_state
    )
    
    test_df = pd.concat([test_seen_df, test_unseen_df]).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    effective_train_pct = len(train_df) / len(df)
    effective_test_pct = len(test_df) / len(df)
    print(f"\nEffective Train/Test Ratio: {len(train_df)} train ({effective_train_pct:.2%}) / {len(test_df)} test ({effective_test_pct:.2%})")

    # Unseen Merchant Integrity Verification
    train_merchants = set(train_df["merchant"].unique())
    unseen_merchants = set(test_unseen_df["merchant"].unique())
    overlap_merchants = train_merchants.intersection(unseen_merchants)
    assert len(overlap_merchants) == 0, (
        f"DATA LEAK DETECTED! Unseen merchants found in training set: {overlap_merchants}"
    )
    print("  -> Unseen Merchant Integrity: PASS [100% Zero-shot Guarantee]")

    # Minimum Training Samples Safeguard Check
    print(f"\nChecking Minimum Training Samples Safeguard (Floor >= {min_train_samples} samples):")
    train_cat_counts = train_df["category"].value_counts().to_dict()
    failed_safeguards = {}
    for cat, count in train_cat_counts.items():
        status = "PASS" if count >= min_train_samples else "FAIL"
        print(f"  - {cat:15s}: {count:3d} samples [{status}]")
        if count < min_train_samples:
            failed_safeguards[cat] = count
            
    if failed_safeguards:
        err_msg = (
            f"SAFEGUARD FAILED (No-Seed-Hunting Protocol): Categories {failed_safeguards} "
            f"have fewer than {min_train_samples} training samples. Halting pipeline for User Review."
        )
        print(f"\n[ERROR] {err_msg}")
        raise RuntimeError(err_msg)

    # 3. Fit TF-IDF Vectorizer ONLY on train_df
    print("\nFitting TF-IDF Vectorizer on train_split only...")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(train_df["processed_text"])
    X_test_tfidf = vectorizer.transform(test_df["processed_text"])
    y_train = train_df["category"].values
    y_test = test_df["category"].values

    # 4. Train Classifiers
    print("Training Logistic Regression (random_state=42, balanced)...")
    logreg = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        C=1.0,
        class_weight="balanced"
    )
    logreg.fit(X_train_tfidf, y_train)

    print("Training LightGBM Classifier (random_state=42, balanced)...")
    lgbm = LGBMClassifier(
        random_state=random_state,
        n_estimators=120,
        learning_rate=0.1,
        class_weight="balanced",
        verbose=-1
    )
    lgbm.fit(X_train_tfidf, y_train)

    # 5. Save Artifacts & Processed CSVs
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d")
    
    vec_path = os.path.join(artifacts_dir, f"vectorizer_phase1_{today_str}.joblib")
    logreg_path = os.path.join(artifacts_dir, f"logreg_phase1_{today_str}.joblib")
    lgbm_path = os.path.join(artifacts_dir, f"lightgbm_phase1_{today_str}.joblib")
    
    save_vectorizer(vectorizer, vec_path)
    joblib.dump(logreg, logreg_path)
    joblib.dump(lgbm, lgbm_path)
    
    train_df.to_csv(os.path.join(processed_dir, "train_split.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test_split.csv"), index=False)
    
    print(f"\nDatasets saved to {processed_dir}/")
    print(f"Artifacts saved to {artifacts_dir}/")
    
    return {
        "train_df": train_df,
        "test_df": test_df,
        "effective_train_pct": effective_train_pct,
        "effective_test_pct": effective_test_pct
    }

if __name__ == "__main__":
    train_and_save_all()
