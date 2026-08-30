"""
SmartSpend AI - API Pydantic Schemas
Defines request and response schemas for REST endpoints with strict validation.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator
import re
from datetime import datetime

class TransactionInput(BaseModel):
    """
    Input schema for live transaction prediction.
    Enforces non-empty merchant, positive amount, and valid date/time strings.
    """
    merchant: str = Field(..., min_length=1, description="Merchant or shop name")
    memo: Optional[str] = Field(default="", description="Transaction memo / note")
    amount: float = Field(..., gt=0, description="Transaction amount in THB (must be > 0)")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    time: str = Field(..., description="Time in HH:MM (24-hour) format")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v.strip()):
            raise ValueError("Date must be in YYYY-MM-DD format")
        try:
            datetime.strptime(v.strip(), "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid calendar date")
        return v.strip()

    @field_validator("time")
    @classmethod
    def validate_time(cls, v: str) -> str:
        if not re.match(r"^\d{2}:\d{2}$", v.strip()):
            raise ValueError("Time must be in HH:MM format")
        try:
            datetime.strptime(v.strip(), "%H:%M")
        except ValueError:
            raise ValueError("Invalid time (must be 00:00 to 23:59)")
        return v.strip()

class ScoreBreakdown(BaseModel):
    late_night_score: float
    payday_score: float
    wants_score: float
    anomaly_score: float

class PredictionOutput(BaseModel):
    """
    Dual output response schema for real-time transaction prediction.
    """
    merchant: str
    memo: str
    amount: float
    date: str
    time: str
    predicted_category: str
    category_confidence: float
    is_wants: bool
    needs_wants_reason: str
    impulse_score_v1: int
    is_nudge_alert: bool
    score_breakdown: ScoreBreakdown
    impulse_probability_v2: float
    risk_level: str

class CategoryStat(BaseModel):
    total_amount: float
    transaction_count: int
    percentage_of_total: float
    needs_amount: float
    wants_amount: float

class MonthlyTrendStat(BaseModel):
    month: str
    total_amount: float
    transaction_count: int
    needs_amount: float
    wants_amount: float
    impulse_amount: float

class SummaryResponse(BaseModel):
    """
    Dashboard summary statistics schema.
    """
    total_spend: float
    total_transactions: int
    needs_amount: float
    needs_percentage: float
    wants_amount: float
    wants_percentage: float
    impulse_transactions_count: int
    impulse_spending_amount: float
    impulse_spending_percentage: float
    nudge_alerts_count: int
    category_breakdown: Dict[str, CategoryStat]
    monthly_trend: List[MonthlyTrendStat]

class HeatmapCell(BaseModel):
    day_of_week: int  # 0=Monday, 6=Sunday
    day_name: str
    hour: int
    total_amount: float
    transaction_count: int
    impulse_count: int
    is_late_night: bool

class HeatmapResponse(BaseModel):
    days: List[str]
    hours: List[int]
    matrix: List[List[HeatmapCell]]
    max_amount_cell: float
    max_count_cell: int

class TransactionItem(BaseModel):
    transaction_id: str
    date: str
    time: str
    merchant: str
    memo: str
    amount: float
    category: str
    is_wants: bool
    impulse_score: int
    is_nudge_alert: bool
    is_cold_start: bool

class TransactionsListResponse(BaseModel):
    total_count: int
    filtered_count: int
    limit: int
    skip: int
    items: List[TransactionItem]
