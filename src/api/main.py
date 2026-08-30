"""
SmartSpend AI - FastAPI Backend Application
Serves RESTful APIs for transaction analytics, summary KPIs, spending heatmap,
evaluation metrics, and real-time multi-model inference without train-serve skew.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Direct imports from Phase 1 and Phase 2 modules (Zero Train-Serve Skew)
from src.nlp.preprocessing import prepare_text_feature, clean_text, tokenize_text
from src.needs_wants.classify_needs_wants import NeedsWantsClassifier, load_config
from src.scoring.impulse_rules import (
    ImpulseRuleScorer,
    is_late_night,
    is_payday_window,
    calc_z_score
)
from src.api.schemas import (
    TransactionInput,
    PredictionOutput,
    ScoreBreakdown,
    SummaryResponse,
    HeatmapResponse,
    HeatmapCell,
    CategoryStat,
    MonthlyTrendStat,
    TransactionsListResponse,
    TransactionItem
)
from src.api.artifact_loader import load_all_phase1_phase2_artifacts

CONFIG_PATH = "config.yaml"
DATA_PATH = "data/raw/transactions.csv"
FRONTEND_DIR = "src/frontend"

# Application global state
app_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup & Shutdown lifecycle manager.
    Loads config, dataset, pre-scored cache, and serialized ML artifacts with fail-fast validation.
    """
    print("[STARTUP] Initializing SmartSpend AI API...")
    
    # 1. Load Configuration
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"[FATAL] Configuration file not found at {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    app_state["config"] = config
    
    # 2. Load Raw Transactions Dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"[FATAL] Transactions dataset not found at {DATA_PATH}")
    raw_df = pd.read_csv(DATA_PATH)
    
    # Sort chronologically
    raw_df["_dt"] = pd.to_datetime(raw_df["date"] + " " + raw_df["time"])
    raw_df = raw_df.sort_values("_dt").reset_index(drop=True)
    raw_df.drop(columns=["_dt"], inplace=True)
    app_state["raw_df"] = raw_df
    
    # 3. Initialize Classifiers & Scorers
    nw_classifier = NeedsWantsClassifier(config)
    impulse_scorer = ImpulseRuleScorer(config)
    app_state["nw_classifier"] = nw_classifier
    app_state["impulse_scorer"] = impulse_scorer
    
    # 4. Precompute & Cache Scored DataFrame for fast UI serving
    print("[STARTUP] Pre-scoring full transaction history...")
    scored_df = impulse_scorer.score_dataframe(raw_df)
    app_state["scored_df"] = scored_df
    
    # 5. Load ML Model Artifacts (Fail-Fast check)
    print("[STARTUP] Loading serialized ML artifacts dynamically...")
    artifacts = load_all_phase1_phase2_artifacts()
    app_state["artifacts"] = artifacts
    
    print(f"[STARTUP] Successfully loaded: Vectorizer, {artifacts['category_model_name']} Classifier, and v2 Impulse ML Model.")
    print("[STARTUP] SmartSpend AI API is ready to serve requests.")
    
    yield
    print("[SHUTDOWN] Cleaning up SmartSpend AI API state...")
    app_state.clear()

app = FastAPI(
    title="SmartSpend AI - API",
    description="NLP Expense Categorization & Impulse Risk Scoring Engine",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for open client requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static directory if exists
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serves the interactive web dashboard SPA."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "SmartSpend AI API is running. Frontend not found at src/frontend/index.html."})

@app.get("/api/summary", response_model=SummaryResponse, tags=["Analytics"])
async def get_summary_metrics():
    """
    Returns high-level KPI metrics, Needs vs Wants distribution, Impulse statistics,
    category breakdown, and monthly trend aggregated from transaction history.
    """
    df = app_state["scored_df"]
    
    total_spend = float(round(df["amount"].sum(), 2))
    total_tx = len(df)
    
    # Needs vs Wants Breakdown
    wants_mask = df["is_wants"] == True
    wants_amt = float(round(df.loc[wants_mask, "amount"].sum(), 2))
    needs_amt = float(round(df.loc[~wants_mask, "amount"].sum(), 2))
    wants_pct = float(round((wants_amt / total_spend * 100.0) if total_spend > 0 else 0.0, 2))
    needs_pct = float(round((needs_amt / total_spend * 100.0) if total_spend > 0 else 0.0, 2))
    
    # Impulse Stats
    impulse_mask = df["is_impulse"] == True
    impulse_count = int(sum(impulse_mask))
    impulse_amt = float(round(df.loc[impulse_mask, "amount"].sum(), 2))
    impulse_pct = float(round((impulse_amt / total_spend * 100.0) if total_spend > 0 else 0.0, 2))
    
    nudge_count = int(sum(df["is_nudge_alert"] == True))
    
    # Category Breakdown
    cat_breakdown = {}
    for cat, cat_grp in df.groupby("category"):
        c_amt = float(round(cat_grp["amount"].sum(), 2))
        c_count = len(cat_grp)
        c_pct = float(round((c_amt / total_spend * 100.0) if total_spend > 0 else 0.0, 2))
        c_wants_amt = float(round(cat_grp.loc[cat_grp["is_wants"] == True, "amount"].sum(), 2))
        c_needs_amt = float(round(cat_grp.loc[cat_grp["is_wants"] == False, "amount"].sum(), 2))
        
        cat_breakdown[cat] = CategoryStat(
            total_amount=c_amt,
            transaction_count=c_count,
            percentage_of_total=c_pct,
            needs_amount=c_needs_amt,
            wants_amount=c_wants_amt
        )
        
    # Monthly Trend
    monthly_trend = []
    df_temp = df.copy()
    df_temp["month"] = df_temp["date"].str[:7]
    for month_str, m_grp in df_temp.groupby("month"):
        m_total = float(round(m_grp["amount"].sum(), 2))
        m_wants = float(round(m_grp.loc[m_grp["is_wants"] == True, "amount"].sum(), 2))
        m_needs = float(round(m_grp.loc[m_grp["is_wants"] == False, "amount"].sum(), 2))
        m_impulse = float(round(m_grp.loc[m_grp["is_impulse"] == True, "amount"].sum(), 2))
        
        monthly_trend.append(MonthlyTrendStat(
            month=month_str,
            total_amount=m_total,
            transaction_count=len(m_grp),
            needs_amount=m_needs,
            wants_amount=m_wants,
            impulse_amount=m_impulse
        ))
        
    monthly_trend.sort(key=lambda x: x.month)
    
    return SummaryResponse(
        total_spend=total_spend,
        total_transactions=total_tx,
        needs_amount=needs_amt,
        needs_percentage=needs_pct,
        wants_amount=wants_amt,
        wants_percentage=wants_pct,
        impulse_transactions_count=impulse_count,
        impulse_spending_amount=impulse_amt,
        impulse_spending_percentage=impulse_pct,
        nudge_alerts_count=nudge_count,
        category_breakdown=cat_breakdown,
        monthly_trend=monthly_trend
    )

@app.get("/api/transactions", response_model=TransactionsListResponse, tags=["Transactions"])
async def get_transactions(
    limit: int = Query(default=50, ge=1, le=500),
    skip: int = Query(default=0, ge=0),
    category: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    is_nudge_only: bool = Query(default=False)
):
    """
    Fetches paginated transaction records with category, needs/wants, impulse score, and nudge tags.
    """
    df = app_state["scored_df"]
    filtered_df = df.copy()
    
    if category and category.lower() != "all":
        filtered_df = filtered_df[filtered_df["category"].str.lower() == category.lower()]
        
    if is_nudge_only:
        filtered_df = filtered_df[filtered_df["is_nudge_alert"] == True]
        
    if search:
        s_lower = search.strip().lower()
        search_mask = (
            filtered_df["merchant"].str.lower().str.contains(s_lower, na=False) |
            filtered_df["memo"].str.lower().str.contains(s_lower, na=False) |
            filtered_df["category"].str.lower().str.contains(s_lower, na=False)
        )
        filtered_df = filtered_df[search_mask]
        
    total_count = len(df)
    filtered_count = len(filtered_df)
    
    # Sort descending by date & time for recent activity
    sorted_df = filtered_df.sort_values(by=["date", "time"], ascending=[False, False])
    page_df = sorted_df.iloc[skip : skip + limit]
    
    items = []
    for _, row in page_df.iterrows():
        items.append(TransactionItem(
            transaction_id=str(row["transaction_id"]),
            date=str(row["date"]),
            time=str(row["time"]),
            merchant=str(row["merchant"]),
            memo=str(row["memo"]) if pd.notna(row["memo"]) else "",
            amount=float(row["amount"]),
            category=str(row["category"]),
            is_wants=bool(row["is_wants"]),
            impulse_score=int(row["impulse_score"]),
            is_nudge_alert=bool(row["is_nudge_alert"]),
            is_cold_start=bool(row["is_cold_start"])
        ))
        
    return TransactionsListResponse(
        total_count=total_count,
        filtered_count=filtered_count,
        limit=limit,
        skip=skip,
        items=items
    )

@app.get("/api/heatmap", response_model=HeatmapResponse, tags=["Analytics"])
async def get_spending_heatmap():
    """
    Returns a 2D Matrix of 7 Days (Mon-Sun) x 24 Hours (0-23)
    containing total spending, transaction count, and impulse frequency.
    """
    df = app_state["scored_df"].copy()
    
    dt_series = pd.to_datetime(df["date"] + " " + df["time"])
    df["dow"] = dt_series.dt.dayofweek  # 0=Monday, 6=Sunday
    df["hour"] = dt_series.dt.hour
    
    day_names = ["จันทร์ (Mon)", "อังคาร (Tue)", "พุธ (Wed)", "พฤหัสบดี (Thu)", "ศุกร์ (Fri)", "เสาร์ (Sat)", "อาทิตย์ (Sun)"]
    hours = list(range(24))
    
    matrix = []
    max_amt = 0.0
    max_cnt = 0
    
    for dow_idx in range(7):
        row_cells = []
        d_df = df[df["dow"] == dow_idx]
        for h_idx in range(24):
            dh_df = d_df[d_df["hour"] == h_idx]
            tot_amt = float(round(dh_df["amount"].sum(), 2))
            t_count = len(dh_df)
            imp_count = int(sum(dh_df["is_impulse"] == True))
            late_flag = is_late_night(f"{h_idx:02d}:00")
            
            if tot_amt > max_amt:
                max_amt = tot_amt
            if t_count > max_cnt:
                max_cnt = t_count
                
            row_cells.append(HeatmapCell(
                day_of_week=dow_idx,
                day_name=day_names[dow_idx],
                hour=h_idx,
                total_amount=tot_amt,
                transaction_count=t_count,
                impulse_count=imp_count,
                is_late_night=late_flag
            ))
        matrix.append(row_cells)
        
    return HeatmapResponse(
        days=day_names,
        hours=hours,
        matrix=matrix,
        max_amount_cell=max_amt,
        max_count_cell=max_cnt
    )

@app.get("/api/metrics", tags=["Transparency"])
async def get_evaluation_metrics():
    """
    Returns evaluation metrics from Phase 1 and Phase 2 alongside configuration thresholds.
    """
    config = app_state.get("config", {})
    nudge_thresh = config.get("impulse_score", {}).get("nudge_threshold", 70)
    
    p1_metrics = {}
    p2_needs_wants = {}
    p2_metrics = {}
    
    if os.path.exists("outputs/metrics/phase1_metrics.json"):
        with open("outputs/metrics/phase1_metrics.json", "r", encoding="utf-8") as f:
            p1_metrics = json.load(f)
            
    if os.path.exists("outputs/metrics/phase2_needs_wants_eval.json"):
        with open("outputs/metrics/phase2_needs_wants_eval.json", "r", encoding="utf-8") as f:
            p2_needs_wants = json.load(f)
            
    if os.path.exists("outputs/metrics/phase2_metrics.json"):
        with open("outputs/metrics/phase2_metrics.json", "r", encoding="utf-8") as f:
            p2_metrics = json.load(f)
            
    return {
        "status": "success",
        "nudge_threshold": nudge_thresh,
        "phase1_categorization": p1_metrics,
        "phase2_needs_wants": p2_needs_wants,
        "phase2_impulse_scoring": p2_metrics,
        "models_loaded": {
            "category_model": app_state["artifacts"]["category_model_name"],
            "impulse_v2_model": "Logistic Regression"
        }
    }

@app.post("/api/predict", response_model=PredictionOutput, tags=["Real-time Inference"])
async def predict_single_transaction(payload: TransactionInput):
    """
    Performs real-time AI inference on a new transaction input:
    1. NLP Expense Categorization (TF-IDF + Best ML Classifier)
    2. Needs vs Wants Tagging (Expanding Category Median Override)
    3. Dual Impulse Scoring (v1 Rule-Based 0-100 & v2 ML Risk Probability)
    """
    artifacts = app_state["artifacts"]
    raw_df = app_state["raw_df"]
    nw_classifier: NeedsWantsClassifier = app_state["nw_classifier"]
    impulse_scorer: ImpulseRuleScorer = app_state["impulse_scorer"]
    
    merchant = payload.merchant.strip()
    memo = payload.memo.strip() if payload.memo else ""
    amount = float(payload.amount)
    date_str = payload.date.strip()
    time_str = payload.time.strip()
    
    # -------------------------------------------------------------
    # 1. NLP Category Prediction
    # -------------------------------------------------------------
    text_feature = prepare_text_feature(merchant, memo)
    vec = artifacts["vectorizer"].transform([text_feature])
    
    cat_model = artifacts["category_model"]
    pred_category = str(cat_model.predict(vec)[0])
    
    if hasattr(cat_model, "predict_proba"):
        probs = cat_model.predict_proba(vec)[0]
        classes = list(cat_model.classes_)
        conf = float(probs[classes.index(pred_category)])
    else:
        conf = 1.0
        
    # -------------------------------------------------------------
    # 2. Needs vs Wants Classification (Historical Context)
    # -------------------------------------------------------------
    # Retrieve all historical amounts in the predicted category
    cat_history_amounts = raw_df[raw_df["category"] == pred_category]["amount"].tolist()
    
    is_wants = nw_classifier.classify_single(
        category=pred_category,
        amount=amount,
        category_history_amounts=cat_history_amounts
    )
    
    default_wants = bool(nw_classifier.default_mapping.get(pred_category, False))
    if default_wants:
        needs_wants_reason = f"หมวดหมู่ '{pred_category}' จัดเป็น Wants โดยค่าเริ่มต้น"
    elif is_wants:
        med_val = np.median(cat_history_amounts) if cat_history_amounts else 0.0
        needs_wants_reason = f"ยอดเงิน {amount:,.2f} บาท สูงกว่า 1.5 เท่าของค่ามัธยฐาน ({med_val:,.2f} บ.)"
    else:
        needs_wants_reason = f"หมวดหมู่ '{pred_category}' จัดเป็น Needs ในระดับปกติ"
        
    # -------------------------------------------------------------
    # 3. v1 Rule-Based Impulse Scoring
    # -------------------------------------------------------------
    # Use earliest historical transaction date so live predict has full historical context
    first_tx_date = raw_df["date"].iloc[0] if len(raw_df) > 0 else date_str
    
    v1_res = impulse_scorer.score_single(
        date_str=date_str,
        time_str=time_str,
        amount=amount,
        is_wants=is_wants,
        first_tx_date=first_tx_date,
        category_history_amounts=cat_history_amounts
    )
    
    v1_score = int(v1_res["impulse_score"])
    is_nudge = bool(v1_res["is_nudge_alert"])
    breakdown = v1_res["score_breakdown"]
    
    # -------------------------------------------------------------
    # 4. v2 ML Impulse Risk Probability
    # -------------------------------------------------------------
    late_flag = int(is_late_night(time_str, impulse_scorer.late_night_window))
    payday_flag = int(is_payday_window(date_str, impulse_scorer.payday_days))
    wants_flag = int(is_wants)
    z_val = float(v1_res["z_score"])
    
    dt_obj = datetime.strptime(date_str + " " + time_str, "%Y-%m-%d %H:%M")
    hour_val = dt_obj.hour
    dow_val = dt_obj.weekday()
    weekend_flag = 1 if dow_val >= 5 else 0
    
    feature_vector = np.array([[
        late_flag,
        payday_flag,
        wants_flag,
        z_val,
        amount,
        hour_val,
        dow_val,
        weekend_flag
    ]])
    
    impulse_ml_model = artifacts["impulse_model_v2"]
    v2_proba = float(round(impulse_ml_model.predict_proba(feature_vector)[0, 1], 4))
    
    # Overall Risk Level categorization
    if v1_score >= 70 or v2_proba >= 0.70:
        risk_level = "ความเสี่ยงสูง (High Risk Nudge)"
    elif v1_score >= 40 or v2_proba >= 0.40:
        risk_level = "ความเสี่ยงปานกลาง (Moderate Risk)"
    else:
        risk_level = "ความเสี่ยงต่ำ (Low Risk)"
        
    return PredictionOutput(
        merchant=merchant,
        memo=memo,
        amount=amount,
        date=date_str,
        time=time_str,
        predicted_category=pred_category,
        category_confidence=float(round(conf, 4)),
        is_wants=is_wants,
        needs_wants_reason=needs_wants_reason,
        impulse_score_v1=v1_score,
        is_nudge_alert=is_nudge,
        score_breakdown=ScoreBreakdown(
            late_night_score=breakdown["late_night_score"],
            payday_score=breakdown["payday_score"],
            wants_score=breakdown["wants_score"],
            anomaly_score=breakdown["anomaly_score"]
        ),
        impulse_probability_v2=v2_proba,
        risk_level=risk_level
    )
