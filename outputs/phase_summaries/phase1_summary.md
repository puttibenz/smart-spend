# Phase 1 Summary: Expense Categorization

**Project:** SmartSpend AI  
**Phase:** 1 — Expense Categorization (NLP Core)  
**Branch:** `phase-1-categorization`  
**Status:** Completed & Ready for Review  

---

## 1. Tasks Completed (Definition of Done)

- [x] **`config.yaml`:** กำหนดค่า Parameters ทั้งหมด (Data Gen, Random Seed = 42, Keyword Baseline Dictionary, Placeholders สำหรับ Phase 2)
- [x] **`generate_synthetic_transactions.py`:** สร้าง Synthetic Dataset 12 เดือน จำนวน 1,554 transactions ตรงตาม 9 Columns Schema และ Category Distribution
- [x] **`preprocessing.py` & `vectorizer.py`:** ฟังก์ชัน Clean Text, PyThaiNLP (`newmm`) Tokenization, และ TF-IDF Vectorizer
- [x] **`train_classifier.py`:** เทรน Keyword Baseline, Logistic Regression, และ LightGBM พร้อมบันทึก Artifacts ลง `models_artifacts/`
- [x] **`evaluate.py`:** คำนวณ Accuracy, Macro-F1, Confusion Matrix, Classification Report และบันทึก `outputs/metrics/phase1_metrics.json`
- [x] **`tests/test_categorization.py`:** ทดสอบ Automated Unit & Integration Tests ผ่าน 100% (6/6 tests passed)

---

## 2. Evaluation Results (Baseline vs ML)

**Test Set Size:** 311 transactions (Stratified 20% split, `random_state=42`)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Result vs Baseline |
|---|---|---|---|---|
| **Keyword Baseline** | 91.32% | **0.8626** | 92.43% | *Baseline Reference* |
| **Logistic Regression** | **100.00%** | **1.0000** | **100.00%** | **PASS [BEATS BASELINE]** |
| **LightGBM** | **100.00%** | **1.0000** | **100.00%** | **PASS [BEATS BASELINE]** |

> **สรุปผล:** ML Models ทั้งสองตัวชนะ Keyword Baseline อย่างมีนัยสำคัญ (+13.74% Macro-F1) ผ่านเกณฑ์ Definition of Done และบันทึกลง `outputs/metrics/phase1_metrics.json` เรียบร้อยแล้ว

---

## 3. Artifacts Generated & Saved

1. **Dataset:**
   - `data/raw/transactions.csv` (1,554 rows, 9 columns)
   - `data/processed/train_split.csv` (1,243 rows)
   - `data/processed/test_split.csv` (311 rows)
2. **Model Artifacts (`models_artifacts/`):**
   - `vectorizer_phase1_20260828.joblib` (TF-IDF fitted vectorizer)
   - `logreg_phase1_20260828.joblib` (Logistic Regression classifier)
   - `lightgbm_phase1_20260828.joblib` (LightGBM classifier)
3. **Metrics Output:**
   - `outputs/metrics/phase1_metrics.json`

---

## 4. Git Commits on `phase-1-categorization`

| Commit Hash | Type(Scope) & Message |
|---|---|
| `c2c97bf` | `chore(phase1): add config.yaml with parameters and baseline keywords` |
| `8931270` | `feat(phase1): implement synthetic transaction generator with fixed seed` |
| `a66bdfb` | `feat(phase1): implement text preprocessing and vectorizer module` |
| `de4847c` | `feat(phase1): train baseline and ML classifiers and evaluate on test set` |
| `0b8234e` | `test(phase1): add unit and integration tests for expense categorization DoD` |

---

## 5. Notes & Observations

- **Data Generalization Note:** บน synthetic data ที่มีแพทเทิร์นชัดเจน โมเดล ML สามารถแยกแยะ feature ได้อย่างสมบูรณ์แบบ (Macro-F1 1.0000) ขณะที่ Keyword Baseline มีข้อจำกัดเมื่อเจอคำที่อยู่นอกพจนานุกรม
- **Phase 3 Readiness:** ได้บันทึก TF-IDF Vectorizer คู่กับโมเดลเรียบร้อยแล้ว เพื่อให้พร้อมนำไปประกอบเป็น API / Inference pipeline ใน Phase 3

---

## 6. Open Decisions for Phase 2 (Impulse Risk Scoring)

ก่อนเริ่ม **Phase 2** มีประเด็นที่ต้องยืนยันกับ User ดังนี้:
1. **Open Decision #2 (Needs vs Wants Override Logic):**
   - กำหนดเกณฑ์ Transaction-level override อย่างไร? (เช่น ถ้ายอดใช้จ่ายเกินกี่เท่าของ Median ในหมวดหมู่นั้น หรือเป็น Late-night จึงจะ override เป็น `is_wants = True`)
2. **Open Decision #3 (Impulse Score Nudge Threshold):**
   - เกณฑ์คะแนนความเสี่ยงสำหรับเตือนแจ้งเตือนบน Dashboard (ปัจจุบันตั้ง default ไว้ที่ 70 ใน config.yaml)
3. **Open Decision #4 (Correlation Analysis Timing):**
   - ตรวจสอบ Correlation ระหว่าง Features กับ `is_impulse` หลังรัน v1 ทันที หรือรวมทำพร้อม v2 ML?
