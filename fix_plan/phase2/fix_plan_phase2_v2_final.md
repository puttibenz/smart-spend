# Implementation Plan — Phase 2 Refinement: Impulse Risk Scoring Engine (Final Protocol v2)

แก้ปัญหา **Feature Double-Counting** (late-night ถูกนับคะแนนซ้ำผ่านทั้ง `wants` และ `late_night` feature), เพิ่ม **Quantitative Acceptance Criteria** ที่ขาดหายไป, และเพิ่มการ **Evaluate Needs/Wants Classifier เทียบ Ground Truth** ที่มีอยู่แล้วในข้อมูล — พร้อมยืนยันเรื่อง Temporal Leakage ของ Z-score

> เอกสารนี้แก้ไขจาก `implementation_plan_phase2.md` ฉบับเดิม โดยคง Open Decision ทั้งหมดที่ตกลงกันแล้วไว้ (override logic, weight, cold start, nudge threshold) แต่แก้ 3 จุดสำคัญที่ตรวจพบก่อนอนุมัติให้เริ่มโค้ด

---

## User Review Required — สรุปข้อตกลงทั้งหมด (คงเดิม + แก้ไข)

> [!IMPORTANT]
> **คงเดิมจากการสัมภาษณ์ /grill-me:**
> 1. Default Category Mapping: `bills` = Needs; `shopping`, `entertainment` = Wants; `food`, `transport`, `other` = default Needs
> 2. Amount Anomaly (Z-score) Scoring: `Z ≥ 2.0 → 30 คะแนน`, `1.0 ≤ Z < 2.0 → 15 คะแนน`, `Z < 1.0 → 0 คะแนน`
> 3. Cold Start Period 30 วันแรก: ข้าม Amount Anomaly, คิดคะแนนเต็ม 70, rescale เป็น `round((raw_score/70.0)*100.0)`
> 4. Nudge Alert Threshold: `score >= 70`
> 5. Feature Correlation Analysis: คำนวณทันทีหลัง v1 เสร็จ ก่อนเริ่ม v2
>
> **แก้ไขใหม่ (จุดที่พบในรอบตรวจ):**
>
> 6. **[FIX — สำคัญที่สุด] แก้ Feature Double-Counting ของ Late-night:**
>    - **เอาเงื่อนไข "late-night" ออกจาก Needs/Wants Override Logic ทั้งหมด**
>    - Override เป็น `is_wants = True` เหลือแค่เงื่อนไขเดียว: **ยอดเงิน > 1.5 เท่าของ Category Median**
>    - เหตุผล: `late_night` มี weight 25 ของตัวเองอยู่แล้วใน Impulse Score ถ้าปล่อยให้ late-night ไป trigger `is_wants=True` ด้วย จะทำให้ transaction ดึกได้คะแนนซ้ำจากสัญญาณเดียวกัน (25 + 20 = 45 คะแนนจากสัญญาณเดียว) ทำให้ nudge alert ไวเกินจริงเฉพาะกลุ่ม transaction ดึก
>
> 7. **[NEW] Quantitative Acceptance Criteria สำหรับ v1 Rule-Based:**
>    - `v1_precision >= 0.50` (เทียบ ground truth `is_impulse`)
>    - `v1_recall >= 0.50`
>    - เกณฑ์นี้ตั้งไว้ไม่สูงเกินไปเพราะรู้อยู่แล้วว่า v1 มี data leakage บางส่วน (weight เป็น assumption ไม่ใช่ tuned จริง) — เป้าหมายคือแค่ยืนยันว่า v1 "ใช้งานได้จริงในระดับหนึ่ง" ไม่ใช่สุ่มทาย
>    - ถ้าไม่ผ่าน → ไม่ raise error ทันที (เพราะ v1 เป็นแค่ baseline ตั้งต้น) แต่ต้องบันทึกไว้ใน `phase2_summary.md` อย่างชัดเจนและ flag ให้ user ตัดสินใจว่าจะปรับ weight หรือไปต่อ v2 เลย
>
> 8. **[NEW] Quantitative Acceptance Criteria สำหรับ v2 ML (Fallback Pattern เดียวกับ Phase 1):**
>    - `v2_f1 > v1_f1` (ML ต้องชนะ v1 Rule-Based) — ถ้าไม่ผ่าน ห้าม agent แก้ feature เพิ่มเองเรื่อยๆ ให้บันทึกผลที่แพ้ลง metrics พร้อมสมมติฐาน แล้ว **[STOP-FOR-REVIEW]** รอ user ตัดสินใจ (เหมือน Phase 1 Fallback Protocol ใน `AGENTS.md`)
>
> 9. **[NEW] Needs/Wants Classifier ต้อง Evaluate เทียบ Ground Truth ที่มีอยู่แล้ว:**
>    - CSV จาก Phase 1 มีคอลัมน์ `is_wants` เป็น ground truth อยู่แล้ว (generator กำหนดไว้ตอนสร้าง synthetic data)
>    - ต้องคำนวณ Accuracy/Precision/Recall ของ `classify_needs_wants.py` (rule 1.5x median) เทียบกับคอลัมน์นี้ บันทึกผลลง `outputs/metrics/phase2_needs_wants_eval.json`
>    - Threshold ขั้นต่ำ: `needs_wants_accuracy >= 0.70` — ถ้าต่ำกว่านี้ ให้หยุดและรายงาน ไม่ใช่เดินหน้าต่อไป Impulse Score ทั้งที่ input (`is_wants`) ยังไม่แม่นพอ
>
> 10. **[CLARIFY] Z-score / Category Median ต้องคำนวณแบบ Expanding Window เท่านั้น (ป้องกัน Temporal Leakage):**
>     - ห้ามคำนวณ median/mean ของ category จากข้อมูล**ทั้งชุด** (รวม transaction ในอนาคต) เพราะจะเป็นการ "รู้อนาคต" ตอนประเมิน transaction ปัจจุบัน
>     - ต้องคำนวณจาก transaction ที่เกิด**ก่อนหน้าตามเวลาเท่านั้น** (expanding window เรียงตาม `date` + `time`)
>     - นี่คือหลักการเดียวกับ Cold Start ที่ตกลงไว้แล้ว (30 วันแรกไม่มีประวัติพอ) — ต้องระบุให้ agent เขียนโค้ดสอดคล้องกัน ไม่ใช่แค่ cold start แต่ทุก transaction หลังจากนั้นก็ต้องใช้ historical data ณ จุดเวลานั้นเท่านั้น ไม่ใช่สถิติจากข้อมูลทั้งหมด

---

## Proposed Changes

### Git Branch
- สร้างและสลับไปที่ branch: **`phase-2-impulse-scoring`**
- **ห้าม merge เข้า `main` เอง** — ต้องรอ user อนุมัติที่ STOP-FOR-REVIEW เสมอ (ตาม Guardrail #7 ใน AGENTS.md)

---

### Configuration Updates

#### [MODIFY] `config.yaml`
```yaml
needs_wants:
  override_amount_multiplier: 1.5
  # late_night ถูกเอาออกจาก override logic แล้ว — ดู FIX #6
  default_mapping:
    bills: false          # false = Need
    shopping: true         # true = Want
    entertainment: true
    food: false
    transport: false
    other: false

impulse_score:
  weights:
    late_night: 25
    payday: 25
    wants: 20
    amount_anomaly: 30
  nudge_threshold: 70
  cold_start_days: 30
  late_night_window:
    - "23:00"
    - "02:00"
  payday_days: [25, 26, 27]

evaluation_thresholds_phase2:
  needs_wants_accuracy_floor: 0.70
  v1_precision_floor: 0.50
  v1_recall_floor: 0.50
  # v2 ต้อง f1 > v1_f1 (ไม่ต้องตั้ง floor ตายตัว — เทียบสัมพัทธ์กับ v1 เสมอ)
```

---

### Module 1: Needs vs Wants Classification

#### [NEW] `src/needs_wants/classify_needs_wants.py`
- คำนวณ Category Median แบบ **Expanding Window** (เฉพาะ transaction ที่เกิดก่อนหน้าตามเวลา — ดู CLARIFY #10)
- Transaction-level mapping ตาม default category
- Override เป็น `is_wants = True` เมื่อ **amount > 1.5x Category Median เท่านั้น** (เอา late-night ออกแล้วตาม FIX #6)
- คืนค่า boolean `is_wants` ที่ไม่มี null

#### [NEW] `src/needs_wants/evaluate_needs_wants.py` (ไฟล์ใหม่ — เพิ่มจาก plan เดิม)
- เทียบผลลัพธ์จาก `classify_needs_wants.py` กับคอลัมน์ `is_wants` ground truth ในข้อมูล Phase 1
- คำนวณ Accuracy, Precision, Recall, Confusion Matrix
- บันทึกลง `outputs/metrics/phase2_needs_wants_eval.json`
- **Fail-stop:** ถ้า `accuracy < needs_wants_accuracy_floor (0.70)` → `raise RuntimeError` พร้อมตัวเลขจริง ไม่เดินหน้าต่อ Impulse Score

#### [NEW] `tests/test_needs_wants.py`
1. `is_wants` เป็น boolean เสมอ ไม่มี null
2. Override logic ทำงานถูกต้องเฉพาะเคส Amount Spike (**ไม่มี late-night test case แล้ว** ตาม FIX #6)
3. Default mapping ของแต่ละ Category ถูกต้อง
4. **[NEW]** Accuracy เทียบ ground truth ≥ 0.70 ตาม threshold ใน config
5. **[NEW]** ยืนยันว่า Category Median คำนวณแบบ expanding window จริง (test case: transaction ต้นๆ ของ timeline ต้องไม่ได้รับผลจาก transaction ท้ายๆ)

---

### Module 2: Impulse Scoring Engine (v1 Rule-Based)

#### [NEW] `src/scoring/impulse_rules.py`
- ดึงค่าน้ำหนัก (25/25/20/30) จาก `config.yaml` ห้าม hardcode
- `is_late_night(time_str)`: ตรวจสอบช่วง 23:00–02:00
- `is_payday_window(date_str)`: ตรวจสอบวันที่ 25, 26, 27
- `calc_z_score(amount, history)`: คำนวณ Z-score จาก **ประวัติแบบ expanding window เท่านั้น** (ตาม CLARIFY #10 — ใช้ historical data เดียวกับที่ `classify_needs_wants.py` ใช้)
- `calc_impulse_score(tx, history)`: คำนวณคะแนน 0–100 พร้อม Cold Start handling
- `is_nudge_alert(score)`: ตรวจสอบ score ≥ nudge_threshold (70)

---

### Module 3: Evaluation & Correlation Analysis

#### [NEW] `src/scoring/evaluate_impulse.py`
- คำนวณ Precision, Recall, F1, PR-AUC, ROC-AUC ของ v1 เทียบ `is_impulse` ground truth
- **[NEW] Fail-stop เชิงเตือน (ไม่ raise error แต่ flag ชัดเจน):** ถ้า `precision < 0.50` หรือ `recall < 0.50` → บันทึก `"v1_meets_minimum_bar": false` ใน metrics JSON พร้อมเหตุผล แล้วเขียนไว้เด่นชัดใน `phase2_summary.md` ให้ user เห็นทันที
- คำนวณ Feature Correlation Matrix (`late_night`, `payday`, `is_wants`, `z_score` vs `is_impulse`)
- แนบ Data Leakage Disclaimer ตามเดิม

---

### Module 4: Machine Learning Impulse Model (v2 ML)

#### [NEW] `src/scoring/impulse_model.py`
- สกัดฟีเจอร์: `[late_night_flag, payday_flag, is_wants, z_score, amount, hour]`
- เทรน `LogisticRegression` และ `LGBMClassifier` (Stratified split, `random_state=42`)
- **[NEW] Fail-stop:** ถ้า `v2_f1 <= v1_f1` → `raise RuntimeError` พร้อมตัวเลขจริง (Fallback Pattern เดียวกับ Phase 1 — ห้าม agent เพิ่ม feature เองเรื่อยๆ โดยไม่หยุด)
- บันทึก Artifacts ลง `models_artifacts/{model_name}_phase2_{date}.joblib`

---

### Module 5: Automated Tests & Review

#### [NEW] `tests/test_impulse_scoring.py`
1. Impulse Score อยู่ในช่วง `[0, 100]` เสมอ
2. Cold Start Case (< 30 วัน) ไม่ error, normalize ถูกต้อง
3. ค่าน้ำหนักตรงกับ `config.yaml` ไม่ hardcode ซ้ำ
4. Nudge flag ทำงานถูกต้องเมื่อ score ≥ 70
5. v2 ML Model inference คืนค่า probability ได้ถูกต้อง
6. **[NEW]** `v2_f1 > v1_f1` ตาม Acceptance Criteria #8
7. **[NEW]** Z-score ของ transaction แรกๆ ในประวัติต้องไม่ได้รับอิทธิพลจาก transaction ที่เกิดทีหลัง (expanding window integrity check)

#### [NEW] `outputs/phase_summaries/phase2_summary.md`
- สรุปผล Needs/Wants Accuracy เทียบ ground truth
- สรุปผล v1 vs v2 (Precision/Recall/F1) พร้อมระบุชัดว่า v1 ผ่านเกณฑ์ขั้นต่ำหรือไม่
- ตาราง Correlation Analysis
- Known Limitation: Cold Start rescaling ทำให้คะแนนช่วง 30 วันแรก "ไวกว่า" หลังจากนั้นสำหรับพฤติกรรมเดียวกัน (ระบุไว้ตามที่ตกลง)
- Commit Hashes

---

## Verification Plan

### Automated Tests
```powershell
.\.venv\Scripts\pytest.exe tests/test_needs_wants.py -v
.\.venv\Scripts\pytest.exe tests/test_impulse_scoring.py -v
```

### Manual Verification
- ตรวจ `outputs/metrics/phase2_needs_wants_eval.json` — accuracy เทียบ ground truth ≥ 0.70
- ตรวจ `outputs/metrics/phase2_metrics.json` — มี correlation matrix, v1 vs v2 comparison, `v1_meets_minimum_bar` flag
- ตรวจว่า `is_wants` ใน `classify_needs_wants.py` **ไม่มี** เงื่อนไข late-night หลงเหลืออยู่ (grep หา `late_night` ในไฟล์นี้ต้องไม่เจอ)
- ตรวจว่า Z-score / Category Median คำนวณแบบ expanding window จริง (สุ่มเช็ค transaction ต้นๆ ของ timeline ว่าค่า median ที่ใช้คำนวณไม่รวมข้อมูลอนาคต)
- ตรวจโมเดลใน `models_artifacts/`
- ตรวจ `phase2_summary.md` ครบถ้วน

---

## Status: ✅ Approved — พร้อมให้เริ่มโค้ดจริงได้ทันที
