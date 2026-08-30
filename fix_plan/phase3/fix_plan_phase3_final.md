# Implementation Plan — Phase 3 Refinement: Dashboard & API (Final Protocol)

แก้ปัญหาเชิงสถาปัตยกรรมของการนำโมเดลจาก Phase 1-2 มาให้บริการ Real-time ผ่าน API — ป้องกัน **Train-Serve Skew**, นิยาม **Historical Context สำหรับ Live Predict**, ทำ **Dynamic Artifact Loading**, เพิ่ม **Input Validation**, และกำหนด **Default Model Selection** ให้ชัดเจน

> เอกสารนี้แก้ไขจาก `Phase3_plan.md` ฉบับเดิม โดยคง Frontend/Backend scope ทั้งหมดไว้เหมือนเดิม แต่เพิ่มรายละเอียดเชิงเทคนิค 5 จุดที่จำเป็นก่อนเริ่มโค้ดจริง เพื่อป้องกัน Live Predictor ให้ผลลัพธ์ผิดแบบไม่มี error เตือน

---

## User Review Required — สรุปข้อตกลงทั้งหมด (คงเดิม + แก้ไข)

> [!IMPORTANT]
> **คงเดิมจาก Phase3_plan.md:**
> 1. Frontend: SPA ด้วย HTML5 + Tailwind + Chart.js, 4 KPI Cards, Doughnut/Bar Chart, Heatmap, Live AI Predictor, Transaction Explorer
> 2. Backend: FastAPI serve static frontend ที่ `/`, endpoints `GET /api/summary`, `GET /api/transactions`, `GET /api/heatmap`, `GET /api/metrics`, `POST /api/predict`
> 3. Integration test เทียบตัวเลข Dashboard กับ CSV/Metrics JSON แบบ 100%
>
> **แก้ไขใหม่ (จุดที่พบในรอบตรวจ):**
>
> 4. **[FIX — สำคัญที่สุด] ป้องกัน Train-Serve Skew:**
>    - `src/api/main.py` **ห้ามเขียน feature extraction logic ซ้ำ** — ต้อง `import` และเรียกใช้ฟังก์ชันเดิมโดยตรงจาก:
>      - `src/nlp/preprocessing.py` (text cleaning สำหรับ categorization)
>      - `src/nlp/vectorizer.py` (TF-IDF transform)
>      - `src/needs_wants/classify_needs_wants.py` (Needs/Wants override logic)
>      - `src/scoring/impulse_rules.py` (`ImpulseRuleScorer`, `is_late_night`, `is_payday_window`)
>      - `src/scoring/impulse_model.py` (`build_behavioral_features` — ใช้ฟังก์ชันเดียวกับตอนเทรน v2 ML เป๊ะ)
>    - เหตุผล: ถ้าเขียน logic คำนวณ feature ใหม่แยกใน API แม้ตั้งใจให้เหมือนเดิม มีความเสี่ยงสูงที่จะต่างกันเล็กน้อย (timezone, การปัดเศษ, ลำดับ column) ทำให้โมเดล predict ผิดเพี้ยนแบบไม่มี error เตือน
>
> 5. **[NEW] นิยาม Historical Context สำหรับ Live Predict ให้ชัดเจน:**
>    - Transaction ใหม่จาก `POST /api/predict` ให้ถือว่า **"เกิดหลังรายการสุดท้ายใน dataset เสมอ"**
>    - ใช้ **ทุก transaction ใน `data/raw/transactions.csv`** เป็นประวัติทั้งหมดสำหรับคำนวณ Category Median (Needs/Wants) และ Z-score (Impulse Amount Anomaly) ของ transaction ใหม่นั้น
>    - **ห้ามตีความเป็น Cold Start เสมอ** (ไม่ใช่ว่า live predict ไม่มีประวัติเลย) เพราะจะทำให้ Amount Anomaly ไม่ทำงานเลยในโหมด demo
>
> 6. **[NEW] Dynamic Model Artifact Loading:**
>    - ห้าม hardcode ชื่อไฟล์ `.joblib` ที่มี `{date}` ต่อท้าย (เช่น `logreg_phase1_20260829.joblib`)
>    - ใช้ `glob` หาไฟล์ล่าสุดใน `models_artifacts/` ตาม pattern แต่ละประเภท (`vectorizer_phase1_*.joblib`, `{model}_phase1_*.joblib`, `impulse_model_phase2_*.joblib`) โดยเลือกไฟล์ที่มี timestamp ใหม่สุดถ้ามีหลายไฟล์
>    - **Fail-fast ตอน startup:** ถ้าหา artifact ไม่เจอ ให้ FastAPI app raise error ชัดเจนตอน startup (ไม่ใช่ crash แบบกำกวมตอนเรียก endpoint ครั้งแรก)
>
> 7. **[NEW] Pydantic Schema สำหรับ `POST /api/predict`:**
>    - สร้าง `TransactionInput` model ด้วย Pydantic กำหนด:
>      - `merchant: str` (ห้ามว่าง)
>      - `memo: str` (อนุญาตว่างได้ ตาม noise pattern ambiguous memo ที่เจอใน Phase 1)
>      - `amount: float` (ต้อง `> 0`)
>      - `date: str` (validate format `YYYY-MM-DD`)
>      - `time: str` (validate format `HH:MM`, 24h)
>    - ถ้า validate ไม่ผ่าน FastAPI ต้องคืน HTTP 422 พร้อมข้อความชัดเจน (Pydantic default behavior)
>
> 8. **[NEW] Default Model Selection สำหรับ Category Prediction:**
>    - `POST /api/predict` ต้องดึงชื่อโมเดลที่ดีที่สุดจาก `outputs/metrics/phase1_metrics.json` field `best_ml_model` มาใช้เป็นตัวทำนาย category จริง
>    - **ห้าม hardcode** ชื่อโมเดล (เช่น "Logistic Regression") ตรงๆ ในโค้ด API
>
> 9. **[CLARIFY] Nudge Threshold ต้องดึงจาก Config ผ่าน API ไม่ hardcode ใน Frontend:**
>    - `dashboard.js` ห้าม hardcode ค่า `nudge_threshold = 70`
>    - ให้ `/api/metrics` หรือ endpoint ที่เกี่ยวข้อง ส่งค่านี้กลับมาจาก `config.yaml` เพื่อให้ frontend อ่านมาใช้แสดงผล (ตาม Guardrail #5 เดิมใน AGENTS.md)

---

## Proposed Changes

### Git Branch
- สร้างและสลับไปที่ branch: **`phase-3-dashboard`**
- **ห้าม merge เข้า `main` เอง** — รอ user อนุมัติที่ STOP-FOR-REVIEW เสมอ

---

### Module 1: FastAPI Backend (`src/api/`)

#### [NEW] `src/api/__init__.py`
- Package initialization

#### [NEW] `src/api/schemas.py` (ไฟล์ใหม่ — เพิ่มจาก plan เดิม)
- `TransactionInput` (Pydantic model) ตาม NEW #7 ด้านบน
- `PredictionOutput` (Pydantic model): `predicted_category: str`, `category_confidence: float`, `is_wants: bool`, `impulse_score_v1: float`, `is_nudge_alert: bool`, `impulse_probability_v2: float`

#### [NEW] `src/api/artifact_loader.py` (ไฟล์ใหม่ — เพิ่มจาก plan เดิม)
- ฟังก์ชัน `load_latest_artifact(pattern: str) -> Any` ใช้ `glob` + sort by filename/mtime หาไฟล์ล่าสุด
- ฟังก์ชัน `load_all_phase1_phase2_artifacts()` โหลด vectorizer, best categorization model (อ้างอิงจาก `phase1_metrics.json`), v2 impulse model — คืน error ชัดเจนถ้าไฟล์ไม่ครบ

#### [NEW] `src/api/main.py`
- Startup event: เรียก `load_all_phase1_phase2_artifacts()` — **fail-fast** ถ้าไฟล์ไม่ครบ (ตาม NEW #6)
- โหลด `config.yaml` เก็บไว้ใน app state สำหรับส่งค่า `nudge_threshold` ผ่าน API (ตาม CLARIFY #9)
- Import ฟังก์ชันจาก Phase 1-2 โดยตรง **ห้ามเขียน feature logic ซ้ำ** (ตาม FIX #4):
  ```python
  from src.nlp.preprocessing import clean_text, tokenize
  from src.nlp.vectorizer import transform_text
  from src.needs_wants.classify_needs_wants import classify_transaction
  from src.scoring.impulse_rules import ImpulseRuleScorer, is_late_night, is_payday_window
  from src.scoring.impulse_model import build_behavioral_features
  ```
- Endpoints:
  - `GET /`: เสิร์ฟ `src/frontend/index.html`
  - `GET /api/summary`: สรุปยอดเงิน, Needs vs Wants, Impulse stats, monthly breakdown
  - `GET /api/transactions`: รายการธุรกรรมพร้อม filter (limit, skip, category, search)
  - `GET /api/heatmap`: Matrix 7×24 (Day × Hour)
  - `GET /api/metrics`: รวม metrics จาก Phase 1-2 **plus `nudge_threshold` จาก config**
  - `POST /api/predict`:
    - รับ `TransactionInput`, validate ผ่าน Pydantic อัตโนมัติ
    - โหลดข้อมูลทั้งหมดจาก `data/raw/transactions.csv` เป็น historical context (ตาม NEW #5 — **ไม่ใช่ cold start เสมอ**)
    - เรียกฟังก์ชันเดิมจาก Phase 1-2 คำนวณ category, is_wants, impulse_score_v1, impulse_probability_v2 ตามลำดับ
    - คืนค่า `PredictionOutput`

---

### Module 2: Interactive Frontend (`src/frontend/`)

#### [NEW] `src/frontend/index.html`
- Header, 4 KPI Cards, Chart Section (Doughnut + Bar), Heatmap Section, Live AI Prediction Form, Transaction Explorer Table
- (คงเดิมตาม Phase3_plan.md ไม่มีการแก้ไขในส่วนนี้)

#### [NEW] `src/frontend/dashboard.js`
- Fetch จาก `/api/summary`, `/api/transactions`, `/api/heatmap`, `/api/metrics`
- Render Chart.js
- Interactive Filters, Pagination, Live Form Submit (`POST /api/predict`)
- **อ่านค่า `nudge_threshold` จาก response ของ `/api/metrics`** แทนการ hardcode (ตาม CLARIFY #9)

---

### Module 3: Automated Tests & DoD Verification

#### [NEW] `tests/test_api_and_dashboard.py`
1. Root endpoint เสิร์ฟ frontend สำเร็จ (HTTP 200)
2. `GET /api/summary` ตรงกับข้อมูลจริงจาก CSV และ Metrics JSON 100%
3. `GET /api/transactions` ครบทุกฟิลด์ (`category`, `is_wants`, `impulse_score`, `is_nudge_alert`)
4. `GET /api/heatmap` คืนค่า Matrix 7×24 ครบ
5. `GET /api/metrics` คืนค่า metrics Phase 1-2 **plus `nudge_threshold`**
6. `POST /api/predict` ทำนายรายการสดได้ถูกต้อง คืนค่าครบทุกมิติ
7. **[NEW]** `POST /api/predict` กับ input ที่ไม่ valid (amount ติดลบ, date format ผิด) ต้องคืน HTTP 422
8. **[NEW]** Test เปรียบเทียบ: transaction เดียวกันที่ผ่าน `POST /api/predict` (real-time path) กับที่ผ่าน `impulse_model.py` (batch/training path) ต้องได้ผลลัพธ์ตรงกัน — ยืนยันว่าไม่มี train-serve skew จริง
9. **[NEW]** Test startup: ลบไฟล์ artifact ชั่วคราวแล้วยืนยันว่า FastAPI app แจ้ง error ชัดเจนตอน startup ไม่ใช่ crash แบบกำกวม

#### [NEW] `outputs/phase_summaries/phase3_summary.md`
- สรุปผลการพัฒนา, รายละเอียด Endpoints, หลักฐานความถูกต้องของตัวเลข
- **[NEW]** ระบุผลการตรวจ train-serve consistency (test #8) อย่างชัดเจน

---

## Verification Plan

### Automated Tests
```powershell
.\.venv\Scripts\pytest.exe tests/test_api_and_dashboard.py -v
```

### Manual Verification
```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
- เปิด `http://127.0.0.1:8000` ตรวจ UI, กราฟ, Heatmap, Live Predictor
- **[NEW]** ทดสอบกรอกฟอร์ม Live Predict ด้วย transaction ที่มีอยู่จริงใน `transactions.csv` (คัดลอก merchant/memo/amount/date/time มาโดยตรง) แล้วเทียบผลลัพธ์กับค่าที่มีอยู่แล้วในข้อมูล ว่า category/is_wants/impulse_score ตรงกันหรือใกล้เคียงสมเหตุสมผล
- **[NEW]** ทดสอบกรอกฟอร์มด้วย input ผิดรูปแบบ (amount ติดลบ) ยืนยันว่า error message อ่านเข้าใจได้ ไม่ใช่ stack trace ดิบ
- ตรวจว่าไม่มีชื่อไฟล์ `.joblib` ที่มี date hardcode อยู่ในซอร์สโค้ด API (`grep -r "2026" src/api/`)

---

## Status: ✅ Approved — พร้อมให้เริ่มโค้ดจริงได้ทันที
