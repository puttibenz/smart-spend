"""
SmartSpend AI - Needs vs Wants Classifier Module
Classifies transactions into Needs (False) or Wants (True) at the transaction level,
using expanding window category medians and amount spike overrides without late-night double-counting.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class NeedsWantsClassifier:
    """
    Transaction-level Needs vs Wants Classifier.
    - Default Mapping: bills=False, shopping=True, entertainment=True, food/transport/other=False
    - Amount Override: If amount > multiplier * expanding_category_median -> is_wants = True
    - Strictly avoids double-counting late-night signals (handled in impulse_rules).
    """
    def __init__(self, config: Optional[dict] = None, config_path: str = "config.yaml"):
        if config is None:
            config = load_config(config_path)
        self.multiplier = float(config["needs_wants"]["override_amount_multiplier"])
        self.default_mapping = config["needs_wants"]["default_mapping"]

    def classify_single(
        self,
        category: str,
        amount: float,
        category_history_amounts: Optional[List[float]] = None
    ) -> bool:
        """
        Classifies a single transaction.
        category_history_amounts: List of strictly preceding amounts in the same category.
        """
        default_is_wants = bool(self.default_mapping.get(category, False))
        
        # If already default Want (e.g. shopping, entertainment), return True
        if default_is_wants:
            return True
            
        # If it's a default Need, check for Amount Spike override using historical median
        if category_history_amounts and len(category_history_amounts) >= 1:
            median_amt = float(np.median(category_history_amounts))
            if amount > (self.multiplier * median_amt):
                return True
                
        return False

    def classify_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """
        Classifies a chronological DataFrame of transactions using an expanding window.
        Assumes df is sorted by date/time or will sort chronologically internally.
        """
        work_df = df.copy()
        
        # Ensure chronological order
        has_datetime = False
        if "date" in work_df.columns and "time" in work_df.columns:
            work_df["_sort_dt"] = pd.to_datetime(work_df["date"] + " " + work_df["time"])
            work_df = work_df.sort_values("_sort_dt").reset_index(drop=True)
            has_datetime = True

        category_histories: Dict[str, List[float]] = {}
        is_wants_results: List[bool] = []

        for _, row in work_df.iterrows():
            cat = str(row["category"])
            amt = float(row["amount"])
            
            hist = category_histories.get(cat, [])
            res = self.classify_single(category=cat, amount=amt, category_history_amounts=hist)
            is_wants_results.append(res)
            
            # Append current amount to history for subsequent transactions
            if cat not in category_histories:
                category_histories[cat] = []
            category_histories[cat].append(amt)

        work_df["is_wants_pred"] = is_wants_results
        
        # If we sorted, restore original index order if needed or return mapped series
        if has_datetime and "_sort_dt" in work_df.columns:
            work_df.drop(columns=["_sort_dt"], inplace=True)
            
        return work_df["is_wants_pred"]

def main():
    config = load_config()
    data_path = "data/raw/transactions.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    classifier = NeedsWantsClassifier(config)
    preds = classifier.classify_dataframe(df)
    
    df["is_wants_pred"] = preds
    print(f"Classified {len(df)} transactions.")
    print("Predicted Needs vs Wants Breakdown:")
    print(df["is_wants_pred"].value_counts(normalize=True) * 100)

if __name__ == "__main__":
    main()
