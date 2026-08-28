"""
SmartSpend AI - Text Vectorizer Module
Manages TF-IDF vectorization and artifact serialization for Phase 1 and Phase 3 inference.
"""

import os
import joblib
from typing import Tuple, List
from sklearn.feature_extraction.text import TfidfVectorizer
from src.nlp.preprocessing import tokenize_text

def build_vectorizer(
    max_features: int = 1500,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1
) -> TfidfVectorizer:
    """
    Builds a TfidfVectorizer configured for pre-tokenized Thai/English text.
    Since text is already tokenized into whitespace-separated tokens,
    we use a simple whitespace-based tokenizer or identity analyzer.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b"
    )

def save_vectorizer(vectorizer: TfidfVectorizer, filepath: str) -> None:
    """Saves the fitted vectorizer to a joblib artifact file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(vectorizer, filepath)
    print(f"Saved vectorizer artifact to: {filepath}")

def load_vectorizer(filepath: str) -> TfidfVectorizer:
    """Loads a fitted vectorizer from a joblib artifact file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Vectorizer artifact not found at: {filepath}")
    return joblib.load(filepath)
