"""
SmartSpend AI - Impulse Rules Scoring Engine (v1 Rule-Based)
Calculates Impulse Risk Score (0-100) using weighted behavioral features,
expanding window Z-score calculation, and Cold Start rescaling.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def is_late_night(time_str: str, late_night_window: List[str] = ["23:00", "02:00"]) -> bool:
    """
    Checks if time_str falls within late-night window (default 23:00 to 02:00).
    """
    if not time_str or not isinstance(time_str, str):
        return False
    try:
        parts = time_str.strip().split(":")
        hour = int(parts[0])
        return hour >= 23 or hour <= 2
    except (ValueError, IndexError):
        return False

def is_payday_window(date_str: str, payday_days: List[int] = [25, 26, 27]) -> bool:
    """
    Checks if date_str falls within payday proximity window (default 25th, 26th, 27th).
    """
    if not date_str or not isinstance(date_str, str):
        return False
    try:
        dt = datetime.strptime(date_str.strip()[:10], "%Y-%m-%d")
        return dt.day in payday_days
    except ValueError:
        return False

def calc_z_score(amount: float, history_amounts: List[float]) -> float:
    """
    Calculates Z-score for amount relative to strictly preceding historical amounts in same category.
    Returns 0.0 if fewer than 2 history samples or std is 0.
    """
    if not history_amounts or len(history_amounts) < 2:
        return 0.0
    
    mean = float(np.mean(history_amounts))
    std = float(np.std(history_amounts, ddof=1))
    
    if std <= 1e-6:
        return 0.0
        
    return float((amount - mean) / std)

def calc_amount_anomaly_score(z_score: float, weight: float = 30.0) -> float:
    """
    Step-wise tiered anomaly scoring:
    - Z >= 2.0 -> 30.0 pts
    - 1.0 <= Z < 2.0 -> 15.0 pts
    - Z < 1.0 -> 0.0 pts
    """
    if z_score >= 2.0:
        return float(weight)
    elif z_score >= 1.0:
        return float(weight * 0.5)
    return 0.0

class ImpulseRuleScorer:
    """
    v1 Rule-Based Impulse Risk Scoring Engine.
    Weights loaded strictly from config.yaml (default: late_night=25, payday=25, wants=20, anomaly=30).
    """
    def __init__(self, config: Optional[dict] = None, config_path: str = "config.yaml"):
        if config is None:
            config = load_config(config_path)
        self.config = config
        
        cfg_impulse = config["impulse_score"]
        self.weights = cfg_impulse["weights"]
        self.w_late_night = float(self.weights["late_night"])
        self.w_payday = float(self.weights["payday"])
        self.w_wants = float(self.weights["wants"])
        self.w_anomaly = float(self.weights["amount_anomaly"])
        
        self.nudge_threshold = float(cfg_impulse.get("nudge_threshold", 70))
        self.cold_start_days = int(cfg_impulse.get("cold_start_days", 30))
        self.late_night_window = cfg_impulse.get("late_night_window", ["23:00", "02:00"])
        self.payday_days = cfg_impulse.get("payday_days", [25, 26, 27])

    def score_single(
        self,
        date_str: str,
        time_str: str,
        amount: float,
        is_wants: bool,
        first_tx_date: Optional[str] = None,
        category_history_amounts: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Scores a single transaction.
        """
        late_flag = is_late_night(time_str, self.late_night_window)
        payday_flag = is_payday_window(date_str, self.payday_days)
        wants_flag = bool(is_wants)
        
        late_score = self.w_late_night if late_flag else 0.0
        payday_score = self.w_payday if payday_flag else 0.0
        wants_score = self.w_wants if wants_flag else 0.0
        
        # Cold start check: Elapsed days from first transaction
        is_cold_start = True
        days_elapsed = 0
        if first_tx_date:
            try:
                dt_curr = datetime.strptime(date_str[:10], "%Y-%m-%d")
                dt_first = datetime.strptime(first_tx_date[:10], "%Y-%m-%d")
                days_elapsed = (dt_curr - dt_first).days
                if days_elapsed >= self.cold_start_days:
                    is_cold_start = False
            except ValueError:
                is_cold_start = True

        z_score = 0.0
        anomaly_score = 0.0

        if is_cold_start:
            # Rescale sum of 3 features (max 70) to 0-100 scale
            raw_subtotal = late_score + payday_score + wants_score
            scale_base = self.w_late_night + self.w_payday + self.w_wants  # 70.0
            score = round((raw_subtotal / scale_base) * 100.0)
        else:
            z_score = calc_z_score(amount, category_history_amounts or [])
            anomaly_score = calc_amount_anomaly_score(z_score, self.w_anomaly)
            score = round(late_score + payday_score + wants_score + anomaly_score)

        # Enforce bounds [0, 100]
        final_score = int(min(100, max(0, score)))
        nudge_alert = bool(final_score >= self.nudge_threshold)

        return {
            "impulse_score": final_score,
            "is_nudge_alert": nudge_alert,
            "is_cold_start": is_cold_start,
            "days_elapsed": days_elapsed,
            "z_score": float(round(z_score, 4)),
            "flags": {
                "late_night": late_flag,
                "payday": payday_flag,
                "is_wants": wants_flag,
                "anomaly_triggered": bool(anomaly_score > 0)
            },
            "score_breakdown": {
                "late_night_score": late_score,
                "payday_score": payday_score,
                "wants_score": wants_score,
                "anomaly_score": anomaly_score
            }
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scores a DataFrame of transactions in strict chronological expanding order.
        """
        work_df = df.copy()
        if "date" in work_df.columns and "time" in work_df.columns:
            work_df["_sort_dt"] = pd.to_datetime(work_df["date"] + " " + work_df["time"])
            work_df = work_df.sort_values("_sort_dt").reset_index(drop=True)

        first_tx_date = work_df["date"].iloc[0] if len(work_df) > 0 else None
        
        category_histories: Dict[str, List[float]] = {}
        
        scores = []
        nudges = []
        cold_starts = []
        z_scores = []
        
        for _, row in work_df.iterrows():
            cat = str(row["category"])
            amt = float(row["amount"])
            d_str = str(row["date"])
            t_str = str(row["time"])
            wants_val = bool(row.get("is_wants", False))
            
            cat_hist = category_histories.get(cat, [])
            
            res = self.score_single(
                date_str=d_str,
                time_str=t_str,
                amount=amt,
                is_wants=wants_val,
                first_tx_date=first_tx_date,
                category_history_amounts=cat_hist
            )
            
            scores.append(res["impulse_score"])
            nudges.append(res["is_nudge_alert"])
            cold_starts.append(res["is_cold_start"])
            z_scores.append(res["z_score"])
            
            # Append to history for future rows
            if cat not in category_histories:
                category_histories[cat] = []
            category_histories[cat].append(amt)

        work_df["impulse_score"] = scores
        work_df["is_nudge_alert"] = nudges
        work_df["is_cold_start"] = cold_starts
        work_df["z_score"] = z_scores

        if "_sort_dt" in work_df.columns:
            work_df.drop(columns=["_sort_dt"], inplace=True)

        return work_df

def main():
    config = load_config()
    data_path = "data/raw/transactions.csv"
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return
        
    df = pd.read_csv(data_path)
    scorer = ImpulseRuleScorer(config)
    scored_df = scorer.score_dataframe(df)
    
    print("Scoring completed:")
    print(f"Total transactions: {len(scored_df)}")
    print(f"Impulse Score Mean: {scored_df['impulse_score'].mean():.2f}")
    print(f"Impulse Score Max : {scored_df['impulse_score'].max()}")
    print(f"Impulse Score Min : {scored_df['impulse_score'].min()}")
    print("\nNudge Alert Count:")
    print(scored_df["is_nudge_alert"].value_counts())
    print("\nCold Start Count:")
    print(scored_df["is_cold_start"].value_counts())

if __name__ == "__main__":
    main()
