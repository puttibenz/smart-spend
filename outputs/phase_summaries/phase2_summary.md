# Phase 2 Summary: Impulse Risk Scoring Engine (v2 Final Protocol)

**Project:** SmartSpend AI  
**Phase:** 2 — Impulse Risk Scoring & Needs/Wants Classification  
**Branch:** `phase-2-impulse-scoring`  
**Status:** ✅ Completed, Evaluated & Ready for Review  
**Protocol Version:** v2 Final Protocol (Late-night Double Counting Fix, Expanding Window Z-Score, Needs/Wants Ground Truth Eval, 5-Fold Stratified Cross-Validation)

---

## 1. Tasks Completed (Definition of Done)

- [x] **`config.yaml`:** ปรับค่าพารามิเตอร์ของ Phase 2 ครบถ้วน (`override_amount_multiplier: 1.5`, `default_mapping`, `weights: 25/25/20/30`, `nudge_threshold: 70`, `cold_start_days: 30`)
- [x] **`classify_needs_wants.py`:** จำแนก Needs vs Wants ระดับ Transaction ด้วย Expanding Category Median (Amount > 1.5x Median) พร้อม **ตัดเงื่อนไข Late-Night ออกทั้งหมด** เพื่อป้องกัน Feature Double-Counting
- [x] **`evaluate_needs_wants.py`:** ประเมินความแม่นยำเทียบ Ground Truth `is_wants` (ได้ Accuracy 83.48% ผ่านเกณฑ์ Floor ≥ 70%) บันทึกลง `outputs/metrics/phase2_needs_wants_eval.json`
- [x] **`impulse_rules.py`:** พัฒนา v1 Rule-Based Engine คำนวณคะแนน Impulse Score (0–100) รองรับ Cold Start (30 วันแรกคิดจาก 3 ฟีเจอร์ ปรับสเกลเต็ม 100) และคำนวณ Z-score แบบ Expanding Window
- [x] **`evaluate_impulse.py`:** ประเมินผล v1 เทียบ Ground Truth `is_impulse`, วิเคราะห์ Feature Correlation Matrix, และแนบ Data Leakage Disclaimer
- [x] **`impulse_model.py`:** เทรน v2 ML Classifier (Logistic Regression) บน Behavioral Features พร้อมทำ **5-Fold Stratified Cross-Validation** ยืนยันความเสถียรข้าม Fold โดย **v2 ทำคะแนน F1-Score ชนะ v1 ได้อย่างขาดลอย (+37.13%)**
- [x] **`tests/test_needs_wants.py` & `tests/test_impulse_scoring.py`:** Automated Tests ผ่าน 100% ครบทั้ง 13/13 Tests

---

## 2. Needs vs Wants Evaluation Results

ประเมินความแม่นยำของ Rule-based (`amount > 1.5 * expanding_median`) เทียบกับ Ground Truth `is_wants` (1,646 รายการ):

| Metric | Measured Value | Target Threshold | Status |
|---|---|---|---|
| **Accuracy** | **83.48%** | ≥ 70.00% (`needs_wants_accuracy_floor`) | **PASS** |
| **Precision** | **78.41%** | - | *Good Precision* |
| **Recall** | **92.34%** | - | *High Coverage* |
| **F1-Score** | **84.80%** | - | *Balanced* |

---

## 3. Impulse Scoring: v1 Rule-Based vs v2 ML Comparison

ประเมินความสามารถในการตรวจจับการใช้จ่ายตามอารมณ์เทียบกับ Ground Truth `is_impulse`:

| Model / Engine | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Status vs Criteria |
|---|---|---|---|---|---|
| **v1 Rule-Based (Baseline)** | **82.54%** | 29.38% | **0.4333** | 0.9708 | *Baseline Reference* |
| **v2 ML (Logistic Regression)** | 67.31% | **100.00%** | **0.8046** | **0.9832** | **PASS [F1 Gain +37.13%]** |
| **v2 ML (LightGBM)** | 71.74% | 94.29% | 0.8148 | 0.9818 | *Alternative ML* |

---

## 4. 5-Fold Stratified Cross-Validation (ความเสถียรของโมเดล)

เพื่อพิสูจน์ว่า Recall 100% ไม่ใช่ความบังเอิญจาก Test Split ชุดเล็ก ได้ทำ **5-Fold Stratified Cross-Validation** บนชุดข้อมูลทั้งหมด 1,646 รายการ (Positive = 177 รายการ):

| Fold | Val Samples (Positive) | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Fold 1** | 330 tx (36 positive) | 69.23% | **100.00%** | 0.8182 | 0.9737 |
| **Fold 2** | 329 tx (35 positive) | 64.81% | **100.00%** | 0.7865 | 0.9785 |
| **Fold 3** | 329 tx (35 positive) | 70.00% | **100.00%** | 0.8235 | 0.9754 |
| **Fold 4** | 329 tx (35 positive) | 66.04% | **100.00%** | 0.7955 | 0.9832 |
| **Fold 5** | 329 tx (36 positive) | 61.02% | **100.00%** | 0.7579 | 0.9707 |
| **Mean ± Std** | **329.2 ± 0.4** | **66.22% ± 3.24%** | **100.00% ± 0.00%** | **0.7963 ± 0.0236** | **0.9763 ± 0.0043** |

> **บทวิเคราะห์ความเสถียร (Cross-Validation Insights):**
> 1. **Recall มีค่า 100% ทุก Fold โดยไม่มีความแปรปรวน ($\pm 0.00\%$):** ยืนยันว่าโมเดล Logistic Regression (`class_weight='balanced'`) สามารถแยกขอบเขต Positive ของ Synthetic Behavioral Signals (`late_night`, `payday`, `is_wants`) ได้อย่างสมบูรณ์แบบข้ามทุก Fold ไม่ใช่ผลจากโชค
> 2. **ความแปรปรวนเกิดขึ้นเฉพาะที่ Precision ($66.22\% \pm 3.24\%$):** เกิดจากการยอมแลก False Positive เล็กน้อยในแต่ละ Fold เพื่อการันตีไม่ให้เคส Impulse หลุดรอด

---

## 5. Feature Correlation Matrix with `is_impulse`

| Feature Name | Description | Pearson $r$ | Spearman $\rho$ | Insights |
|---|---|---|---|---|
| `late_night` | รายการเกิดช่วง 23:00–02:00 | **+0.5451** | +0.5451 | **สัญญาณความเสี่ยงสูงสุด** (สั่งอาหาร/ช้อปปิ้งรอบดึก) |
| `payday` | รายการเกิดช่วงวันที่ 25, 26, 27 | **+0.4449** | +0.4449 | สัญญาณความเสี่ยงสูงมาก (ช่วงเงินเดือนออก) |
| `is_wants` | รายการจัดเป็น Wants | **+0.3475** | +0.3475 | สัมพันธ์ปานกลาง |
| `z_score` | ยอดเงินกระโดดสูงกว่าปกติ | **+0.2799** | +0.2735 | สัมพันธ์เชิงบวก |
| `amount` | จำนวนเงินดิบ | +0.1549 | +0.2176 | สัมพันธ์เชิงบวกเล็กน้อย |
| `hour` | ชั่วโมงที่เกิดรายการ | -0.2916 | -0.2125 | สัมพันธ์เชิงลบ (ชั่วโมงดึก 0, 1, 2 สัมพันธ์กับ impulse) |

---

## 6. Artifacts Generated & Saved

1. **Model Artifacts (`models_artifacts/`):**
   - `impulse_model_phase2_20260829.joblib` (Trained Logistic Regression Pipeline พร้อม 5-Fold CV metrics)
2. **Evaluation Metrics JSON (`outputs/metrics/`):**
   - `phase2_needs_wants_eval.json` (Needs vs Wants Accuracy & Classification Report)
   - `phase2_metrics.json` (v1 vs v2 Comparison, 5-Fold CV Summary, Correlation Matrix, Data Leakage Disclaimer)

---

## 7. Known Limitations & Methodology Notes

- **Synthetic Data Leakage Disclaimer:** เนื่องจากพฤติกรรม Impulse ถูกออกแบบผ่านเงื่อนไข Late-night และ Payday ในขั้นตอน Data Generation ค่า Correlation และ Recall ที่สูงของโมเดลจึงสะท้อนถึงการจับสัญญาณที่ฝังไว้ได้อย่างถูกต้อง แต่ในการใช้งานจริงกับมนุษย์ ปัจจัยทางอารมณ์มีความซับซ้อนกว่านี้
- **Cold Start Scaling:** ในช่วง 30 วันแรก การ Rescale คะแนนจากฐาน 70 ไปเต็ม 100 อาจทำให้ธุรกรรมที่มี Late-night หรือ Payday ได้รับคะแนนสูงกว่าช่วงปกติเล็กน้อย ซึ่งเป็น Trade-off ที่ยอมรับได้เพื่อไม่ให้ระบบหยุดทำงานเมื่อไม่มีประวัติ Z-score

---

## 8. Git Commits on `phase-2-impulse-scoring`

| Commit Hash | Conventional Commit Message |
|---|---|
| `beac0e5` | `chore(phase2): update config.yaml with Phase 2 parameters and thresholds` |
| `c0c7fce` | `feat(phase2): implement Needs vs Wants classification with expanding median and evaluation` |
| `7aae155` | `feat(phase2): implement rule-based impulse score engine with cold start normalization` |
| `69512bd` | `feat(phase2): implement v1 impulse evaluation and feature correlation analysis` |
| `7b6e985` | `feat(phase2): train and evaluate v2 ML impulse model against v1 baseline` |
| `e9174fe` | `test(phase2): add unit and integration tests for Needs/Wants and Impulse Scoring DoD` |
| `38ddf31` | `feat(phase2): add 5-fold cross validation for Logistic Regression and update evaluation` |

---

## 9. Next Step: Phase 3 (Dashboard & API)

เมื่อได้รับการอนุมัติ Merge เข้า `main` จาก User แล้ว เราพร้อมเปิด branch `phase-3-dashboard` เพื่อสร้าง FastAPI Backend และ Web Visualization ต่อไปครับ
