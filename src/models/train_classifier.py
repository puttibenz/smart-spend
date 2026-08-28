"""
SmartSpend AI - Train Classifier Module
Trains keyword matching baseline and ML classifiers (Logistic Regression & LightGBM)
for expense categorization, with strict random_state=42 and artifact serialization.
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
from sklearn.pipeline import Pipeline

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
        # Count matches per category to handle cases with multiple keywords
        match_scores = {}
        for cat, kws in self.keyword_dict.items():
            score = sum(1 for kw in kws if kw in text_lower)
            if score > 0:
                match_scores[cat] = score
        
        if match_scores:
            # Return category with highest keyword matches
            return max(match_scores, key=match_scores.get)
        return self.fallback_category

    def predict(self, texts: list) -> list:
        return [self.predict_single(t) for t in texts]

def train_and_save_all(
    data_path: str = "data/raw/transactions.csv",
    config_path: str = "config.yaml",
    artifacts_dir: str = "models_artifacts"
):
    config = load_config(config_path)
    random_state = config["model_training"]["random_state"]
    test_size = config["model_training"]["test_size"]
    
    print(f"Loading transaction dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocess text features (merchant + memo)
    print("Preprocessing text features (merchant + memo)...")
    df["processed_text"] = [
        prepare_text_feature(m, memo)
        for m, memo in zip(df["merchant"], df["memo"])
    ]
    
    X = df["processed_text"].values
    y = df["category"].values
    raw_texts = (df["merchant"].fillna("") + " " + df["memo"].fillna("")).values
    
    # Stratified Train/Test Split
    print(f"Performing Stratified Train/Test Split ({1-test_size:.0%}/{test_size:.0%}) with random_state={random_state}...")
    (
        X_train, X_test,
        y_train, y_test,
        raw_train, raw_test,
        idx_train, idx_test
    ) = train_test_split(
        X, y, raw_texts, df.index.values,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # 1. Initialize & Fit TF-IDF Vectorizer on X_train
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 2. Train Baseline Classifier
    print("Configuring Keyword Baseline Classifier...")
    baseline_clf = KeywordBaselineClassifier(config["baseline_keywords"])

    # 3. Train Logistic Regression
    print("Training Logistic Regression (random_state=42)...")
    logreg = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        C=1.0,
        class_weight="balanced"
    )
    logreg.fit(X_train_tfidf, y_train)

    # 4. Train LightGBM Classifier
    print("Training LightGBM Classifier (random_state=42)...")
    lgbm = LGBMClassifier(
        random_state=random_state,
        n_estimators=120,
        learning_rate=0.1,
        class_weight="balanced",
        verbose=-1
    )
    lgbm.fit(X_train_tfidf, y_train)

    # Save Artifacts
    os.makedirs(artifacts_dir, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d")
    
    vec_path = os.path.join(artifacts_dir, f"vectorizer_phase1_{today_str}.joblib")
    logreg_path = os.path.join(artifacts_dir, f"logreg_phase1_{today_str}.joblib")
    lgbm_path = os.path.join(artifacts_dir, f"lightgbm_phase1_{today_str}.joblib")
    
    # Also save standard symlinks/files for straightforward loading
    save_vectorizer(vectorizer, vec_path)
    joblib.dump(logreg, logreg_path)
    joblib.dump(lgbm, lgbm_path)
    
    # Save test dataset and split for evaluation
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    test_df = df.iloc[idx_test].copy()
    test_df["processed_text"] = X_test
    test_df.to_csv(os.path.join(processed_dir, "test_split.csv"), index=False)
    
    train_df = df.iloc[idx_train].copy()
    train_df["processed_text"] = X_train
    train_df.to_csv(os.path.join(processed_dir, "train_split.csv"), index=False)

    print(f"Artifacts successfully saved to {artifacts_dir}/")
    return {
        "vectorizer": vectorizer,
        "baseline_clf": baseline_clf,
        "logreg": logreg,
        "lgbm": lgbm,
        "X_train": X_train,
        "X_test": X_test,
        "X_test_tfidf": X_test_tfidf,
        "y_train": y_train,
        "y_test": y_test,
        "raw_test": raw_test,
        "date_str": today_str
    }

if __name__ == "__main__":
    train_and_save_all()
