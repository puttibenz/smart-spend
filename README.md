# SmartSpend AI 🧠💸

> **ระบบจำแนกหมวดหมู่รายจ่ายอัตโนมัติด้วย NLP + ระบบประเมินความเสี่ยงพฤติกรรมใช้จ่ายตามอารมณ์ (Impulse Buying Risk Scoring) สำหรับ First Jobber พร้อม Interactive Web Dashboard & Real-Time API**

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112.2-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.1-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-32%2F32%20passed%20(100%25)-success.svg)](https://pytest.org)

---

## 📌 Executive Summary

**SmartSpend AI** เป็น Solo Portfolio Project ที่พัฒนาขึ้นเพื่อตอบโจทย์ปัญหาทางการเงินของกลุ่ม **First Jobber (คนเริ่มทำงาน)** ซึ่งมักประสบปัญหาการซื้อของตามอารมณ์ (Impulse Buying) ในช่วงดึกหรือช่วงหลังเงินเดือนออก โดยระบบประกอบด้วย 3 ส่วนหลัก:

1. **Expense Categorization (NLP):** จำแนกหมวดหมู่รายจ่ายภาษาไทย/อังกฤษ 6 หมวดหมู่อัตโนมัติ (`food`, `transport`, `shopping`, `bills`, `entertainment`, `other`) ด้วย TF-IDF + Machine Learning
2. **Impulse Risk Scoring & Needs/Wants Engine:**
   - **Needs vs Wants Tagging:** จำแนกระดับ Transaction ด้วย Expanding Category Median (Amount > 1.5x Median) ปราศจาก Feature Double-Counting
   - **Dual Impulse Scoring:** ประเมินความเสี่ยงด้วย **v1 Rule-Based Score (0–100)** พร้อม Cold Start Rescaling (30 วันแรก) และ **v2 Machine Learning Model (Logistic Regression)** ที่ผ่านการพิสูจน์ด้วย **5-Fold Stratified Cross-Validation**
3. **Interactive Dashboard & REST API:** พัฒนาด้วย FastAPI + Responsive SPA (Tailwind CSS + Chart.js) แสดงผล KPIs, Needs vs Wants, Spending Heatmap 7×24 และ Live AI Simulator Form สำหรับทดสอบทำนายสด

---

## 📊 Key Evaluation & Benchmark Results

### 1. Phase 1 — Expense Categorization Performance
ทดสอบบน Test Set ที่มี Realistic Noise (Ambiguous Memos, Typos) และ **20% Unseen Merchants (Zero-Shot Generalization)**:

| Model | Accuracy | Macro-F1 | Weighted-F1 | Status vs Baseline |
|---|---|---|---|---|
| **Keyword Baseline** | 77.95% | **0.7319** | 78.42% | *Baseline Reference* |
| **Logistic Regression (Best Model)** | **94.56%** | **0.9396** | **94.57%** | **PASS (+20.77% Macro-F1 Gain)** |
| **LightGBM** | 88.07% | 0.8909 | 88.24% | **PASS** |

* **Seen Merchant Accuracy:** 98.79% (244/247 tx)
* **Unseen Merchant Accuracy:** 92.05% (382/415 tx) *(Floor ≥ 60.00%)*
* **Generalization Gap:** 6.74% *(Threshold ≤ 20.00%)*

---

### 2. Phase 2 — Needs/Wants & Impulse Scoring Performance
* **Needs vs Wants Accuracy เทียบ Ground Truth:** **83.48%** (Precision: 78.41%, Recall: 92.34%, F1: 84.80%)
* **Impulse Risk Scoring Comparison:**

| Model / Engine | Test Precision | Test Recall | Test F1-Score | ROC-AUC | 5-Fold CV Mean F1 | Status |
|---|---|---|---|---|---|---|
| **v1 Rule-Based (Baseline)** | **82.54%** | 29.38% | **0.4333** | 0.9708 | - | *Baseline* |
| **v2 ML (Logistic Regression)** | 67.31% | **100.00%** | **0.8046** | **0.9832** | **0.7963 ± 0.0236** | **PASS (+37.13% Gain)** |

---

### 3. Phase 3 — Dashboard & API Verification
* **Data Consistency:** ตัวเลขสถิติบน Dashboard ตรงกับชุดข้อมูลจริง 1,646 รายการ 100%
* **Zero Train-Serve Skew:** การทำนายสดผ่าน `POST /api/predict` เรียกใช้โมดูล Preprocessing และ Feature Extraction เดียวกันกับการเทรนโดยตรง
* **Automated Test Coverage:** Unit & Integration Tests **ผ่านครบ 32/32 Tests (100%)**

---

## 🏗️ Project Architecture & Folder Structure

```
smartspend-ai/
├── config.yaml                   # Centralized configuration & hyper-parameters
├── requirements.txt               # Pinned library dependencies (Python 3.11+)
├── data/
│   ├── raw/                      # Synthetic transaction dataset (transactions.csv)
│   └── processed/                # Preprocessed train/test splits (Custom 2-Stage Split)
├── src/
│   ├── data_generation/          # Decoupled pools synthetic data generator
│   │   └── generate_synthetic_transactions.py
│   ├── nlp/                      # NLP cleaning, tokenization (PyThaiNLP), TF-IDF vectorizer
│   │   ├── preprocessing.py
│   │   └── vectorizer.py
│   ├── models/                   # Categorization training and evaluation
│   │   ├── train_classifier.py
│   │   └── evaluate.py
│   ├── needs_wants/              # Needs vs Wants classifier with expanding median override
│   │   ├── classify_needs_wants.py
│   │   └── evaluate_needs_wants.py
│   ├── scoring/                  # v1 Rule-based impulse engine & v2 ML classifier
│   │   ├── impulse_rules.py
│   │   ├── evaluate_impulse.py
│   │   └── impulse_model.py
│   ├── api/                      # FastAPI Backend & Dynamic Artifact Loader
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── artifact_loader.py
│   └── frontend/                 # Interactive SPA Dashboard (Tailwind CSS + Chart.js)
│       ├── index.html
│       └── dashboard.js
├── tests/                        # 32 Automated Unit & Integration Tests
│   ├── test_categorization.py
│   ├── test_needs_wants.py
│   ├── test_impulse_scoring.py
│   └── test_api_and_dashboard.py
├── models_artifacts/              # Serialized joblib pipelines (Vectorizer, LogReg, v2 ML)
├── outputs/
│   ├── metrics/                  # JSON metrics & error analysis CSV
│   └── phase_summaries/          # Phase 1, Phase 2, and Phase 3 review summaries
└── README.md
```

---

## 🚀 Quickstart & Setup Guide

### 1. Clone Repository & Setup Environment

```powershell
# Clone repository
git clone https://github.com/puttibenz/smart-spend.git
cd smart-spend

# Create Virtual Environment (Python 3.11+)
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (32 Tests)

```powershell
.\.venv\Scripts\pytest.exe tests/ -v
```

### 3. Launch Interactive Web Dashboard & API

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

เปิดเว็บเบราว์เซอร์ไปที่: **`http://127.0.0.1:8000`** เพื่อใช้งาน Dashboard

---

## 📡 REST API Documentation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | เสิร์ฟ Interactive Dashboard SPA |
| `GET` | `/api/summary` | สรุปภาพรวม KPIs, Needs vs Wants, Impulse stats, Monthly Trend |
| `GET` | `/api/transactions` | รายการธุรกรรมทั้งหมด (รองรับ `limit`, `skip`, `category`, `search`, `is_nudge_only`) |
| `GET` | `/api/heatmap` | ข้อมูลตาราง 2 มิติ (Day of Week 7 วัน $\times$ Hour 24 ชม.) |
| `GET` | `/api/metrics` | ข้อมูล Performance Metrics จาก Phase 1 & 2 และ Config Thresholds |
| `POST` | `/api/predict` | ทำนายหมวดหมู่ NLP + Needs/Wants + Dual Impulse Risk Score แบบ Real-Time |

---

## ⚠️ Known Limitations & Methodological Rigor

1. **Synthetic Data Disclaimer:** ข้อมูลทั้งหมดเป็น Synthetic Data ที่จำลองพฤติกรรม First Jobber ค่าประสิทธิภาพโมเดลที่สูงสะท้อนถึงการจับสัญญาณที่ฝังไว้ได้อย่างถูกต้อง แต่ในสถานการณ์จริง พฤติกรรมมนุษย์มีปัจจัยทางอารมณ์และบริบทที่ซับซ้อนกว่า
2. **Cold Start Rescaling:** ในช่วง 30 วันแรก การคำนวณ Z-score จะถูกข้าม และ Rescale คะแนนจาก 3 ปัจจัยที่เหลือ (70 คะแนน) เป็น 100 คะแนนเต็ม เพื่อให้ระบบยังทำงานได้แม้ไม่มีประวัติในอดีต

---

## 👨‍💻 Author & Credits

* **Developer:** Puttibenz (SmartSpend AI Portfolio)
* **Tech Stack:** Python 3.11, FastAPI, Scikit-Learn, LightGBM, PyThaiNLP, Chart.js, Tailwind CSS
