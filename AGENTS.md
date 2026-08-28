# AGENTS.md — SmartSpend AI

> ไฟล์นี้คือ context ที่ AI coding agent (เช่น Antigravity, Claude Code, Cursor) ควรโหลดอัตโนมัติทุกครั้งที่เริ่มทำงานในโปรเจกต์นี้ เป็นเวอร์ชันสรุปจาก **Agent Implementation Guide** ใน `SmartSpend_AI_PRD_v2.md` — ถ้าต้องการ context เชิงธุรกิจ/product เต็มรูปแบบ (Problem Statement, Data Strategy, Success Metrics ฯลฯ) ให้อ่านไฟล์ PRD หลักประกอบด้วย ไฟล์นี้โฟกัสเฉพาะสิ่งที่ agent ต้อง "ทำตาม" ตอนเขียนโค้ด

---

## Project Summary

SmartSpend AI — ระบบจำแนกหมวดหมู่รายจ่ายอัตโนมัติด้วย NLP + ให้คะแนนความเสี่ยง Impulse Buying สำหรับ First Jobber เป็น solo portfolio project ใช้ synthetic data ทั้งหมด แบ่งงานเป็น 3 Phase หลัก (Categorization → Impulse Scoring → Dashboard) ดูรายละเอียดธุรกิจเต็มใน `SmartSpend_AI_PRD_v2.md`

---

## Folder Structure

```
smartspend-ai/
├── config.yaml                   # threshold/weight ทั้งหมดที่ยังเป็น assumption
├── requirements.txt               # pinned version ทุก library (ห้ามใช้ >=)
├── data/
│   ├── raw/                      # synthetic CSV ที่ generate ได้
│   └── processed/                # ข้อมูลหลัง preprocessing
├── src/
│   ├── data_generation/
│   │   └── generate_synthetic_transactions.py
│   ├── nlp/
│   │   ├── preprocessing.py
│   │   └── vectorizer.py
│   ├── models/
│   │   ├── train_classifier.py
│   │   └── evaluate.py
│   ├── scoring/
│   │   ├── impulse_rules.py      # v1 rule-based — คำนวณ score เท่านั้น
│   │   ├── evaluate_impulse.py   # แยกจาก impulse_rules.py — Precision/Recall
│   │   └── impulse_model.py      # v2 ML (Phase 2 เท่านั้น)
│   ├── needs_wants/
│   │   └── classify_needs_wants.py
│   ├── api/                      # Phase 3 เท่านั้น
│   │   └── main.py
│   └── frontend/                 # Phase 3 เท่านั้น
│       ├── index.html
│       └── dashboard.js
├── notebooks/                    # EDA/experiment เท่านั้น ไม่ใช่ production code
├── tests/
│   ├── test_categorization.py
│   ├── test_impulse_scoring.py
│   └── test_needs_wants.py
├── models_artifacts/              # .pkl / .joblib — ตั้งชื่อ {model_name}_{phase}_{date}.joblib
├── outputs/
│   ├── metrics/                  # JSON evaluation ต่อ phase
│   └── phase_summaries/          # summary ก่อนหยุดรอ user รีวิว
└── README.md
```

**Python version:** 3.11+ (ระบุเป็น comment บนสุดของ `requirements.txt`)
**`requirements.txt`:** pin version ชัดเจนเสมอ (`scikit-learn==1.5.1` ไม่ใช่ `scikit-learn>=1.5`)
**`config.yaml`** ต้องมีอย่างน้อย: `impulse_score.weights` (25/25/20/30), `impulse_score.nudge_threshold` (70), `impulse_score.late_night_window` (23:00-02:00), `needs_wants.override_amount_multiplier`, `data_generation.random_seed` (42)

---

## Data Schema

**`data/raw/transactions.csv`:**

| Column | Type | หมายเหตุ |
|---|---|---|
| `transaction_id` | string (UUID) | unique ต่อแถว |
| `date` | date (`YYYY-MM-DD`) | |
| `time` | string (`HH:MM`, 24h) | |
| `merchant` | string | |
| `memo` | string | ไทย/อังกฤษปน |
| `amount` | float | ทศนิยม 2 ตำแหน่ง |
| `category` | string (enum) | `food`, `transport`, `shopping`, `bills`, `entertainment`, `other` |
| `is_wants` | bool | ground truth ระดับ transaction |
| `is_impulse` | bool | ground truth สำหรับ evaluate Impulse Score |

**`outputs/metrics/phase{N}_metrics.json`** ต้องมี key: `phase`, `component`, `metric_name`, `value`, `baseline_value`, `timestamp`

**Category distribution (ตายตัว ไม่สุ่มเอง):** `food` 35%, `transport` 20%, `shopping` 20%, `bills` 10%, `entertainment` 10%, `other` 5%
**Random seed:** ต้อง fix (เช่น 42) เพื่อ reproducibility ข้าม run

---

## Task Checklist (Definition of Done)

**Phase 1 — Expense Categorization**
- [ ] `generate_synthetic_transactions.py` → CSV 6-12 เดือน ตรง schema, fixed seed, สัดส่วน category ตรงตามที่กำหนด
- [ ] `preprocessing.py` → cleaned text พร้อม unit test
- [ ] `train_classifier.py` → เทรน **baseline (keyword matching)** และ **ML (LogReg + LightGBM)** แยกกัน, save โมเดลลง `models_artifacts/`
- [ ] `evaluate.py` → Macro-F1 + confusion matrix ทั้งคู่ — **ML ต้องชนะ baseline ถึงผ่าน** (ถ้าแพ้ → หยุดรอ user ตาม fallback ด้านล่าง)
- [ ] `pytest tests/test_categorization.py` ผ่านทั้งหมด — ต้อง assert: (a) schema ตรงทุก column (b) ไม่มี missing value ใน column บังคับ (c) Macro-F1 ML > baseline (d) จำนวนแถว output = จำนวนที่ generate
- **[STOP-FOR-REVIEW]**

**Phase 2 — Impulse Risk Scoring**
- [ ] `classify_needs_wants.py` → mapping ระดับ **transaction** ไม่ใช่ category (ต้องยืนยัน logic กับ user ก่อน — ดู Open Decision #2)
- [ ] `pytest tests/test_needs_wants.py` ผ่าน — assert: (a) `is_wants` เป็น boolean เสมอ ไม่มี null (b) override logic ทำงานถูกต้องตามเกณฑ์ที่ตกลง
- [ ] `impulse_rules.py` → weight ตาม `config.yaml` **ห้ามแก้ตัวเลขน้ำหนัก (25/25/20/30) เอง**
- [ ] `evaluate_impulse.py` → Precision/Recall เทียบ `is_impulse` + disclaimer เรื่อง data leakage แนบในผลลัพธ์เสมอ
- [ ] `impulse_model.py` (v2 ML) — ทำเฉพาะหลัง v1 มีผลลัพธ์และ user รีวิวแล้ว
- [ ] `pytest tests/test_impulse_scoring.py` ผ่าน — assert: (a) score อยู่ 0-100 เสมอ (b) cold-start case ไม่ error (c) weight ตรงกับ `config.yaml`
- **[STOP-FOR-REVIEW]**

**Phase 3 — Dashboard**
- [ ] `api/main.py` → endpoint: transaction list พร้อม category/is_wants/impulse_score, สรุป Needs vs Wants, ข้อมูล heatmap
- [ ] `frontend/index.html` + `dashboard.js` → Needs vs Wants breakdown, Spending Heatmap, สรุปแนวโน้มรายสัปดาห์/รายเดือน
- [ ] เทียบตัวเลขในกราฟกับ metrics JSON ว่าตรงกัน
- **[STOP-FOR-REVIEW]**

---

## Fallback: ถ้า ML แพ้ Baseline

ห้ามเพิ่ม feature engineering ต่อเองเรื่อยๆ โดยไม่หยุด — บันทึกผลที่แพ้ลง metrics JSON ตามปกติ พร้อมสมมติฐานว่าทำไมแพ้ (เช่น data leakage) แล้วหยุดรอ user ตัดสินใจว่าจะ (ก) ใช้ v1/baseline ต่อ หรือ (ข) ให้ agent ลอง feature เพิ่มแบบมีขอบเขตชัดเจน — **ไม่ใช่ agent ตัดสินใจเอง**

---

## Stop-for-Review Protocol

เมื่อถึง **[STOP-FOR-REVIEW]** ทำตามลำดับนี้เสมอ:

1. สร้างไฟล์ `outputs/phase_summaries/phase{N}_summary.md` — มี: task ที่เสร็จ, ผลลัพธ์ metrics เทียบ baseline, ปัญหา/assumption ที่พบ, คำถามที่ค้าง
2. Commit งานทั้งหมดของ Phase ให้ครบ แล้วใส่ commit hash ล่าสุดใน summary — **ห้าม merge เข้า `main`** จนกว่า user จะตอบ
3. พิมพ์สรุปสั้นๆ ในแชท พร้อมลิงก์ summary file และคำถามที่ค้าง (ถ้ามี) ตรงๆ
4. ถ้า user ไม่ตอบและเป็น autonomous session ยาว — **ห้าม auto-continue เอง** ให้จบ session ไว้ที่จุดนี้

---

## Git Workflow

**Branch:**

| Branch | ใช้ทำอะไร |
|---|---|
| `main` | เฉพาะโค้ดที่ user รีวิว+อนุมัติแล้วเท่านั้น |
| `phase-1-categorization` | งาน Phase 1 |
| `phase-2-impulse-scoring` | งาน Phase 2 |
| `phase-3-dashboard` | งาน Phase 3 |

Agent ทำงานใน branch ของ Phase ปัจจุบันเสมอ ห้าม merge เข้า `main` เอง

**ความถี่ commit:** commit ทุกครั้งที่ checkbox ใน Task Checklist เสร็จ 1 อัน (ไม่รวมทั้ง Phase ในคอมมิตเดียว ไม่ปนหลาย component เข้าด้วยกัน)

**Commit message — Conventional Commits:** `<type>(<scope>): <description>`

| Type | ใช้เมื่อ |
|---|---|
| `feat` | เพิ่มฟีเจอร์ใหม่ |
| `test` | เพิ่ม/แก้ test |
| `fix` | แก้บั๊ก |
| `docs` | แก้ summary/README/PRD |
| `chore` | infra เช่น เพิ่ม dependency |

ตัวอย่าง:
```
feat(phase1): implement synthetic transaction generator with fixed seed
test(phase1): add schema validation tests for generated CSV
feat(phase2): implement rule-based impulse score in impulse_rules.py
fix(phase2): correct cold-start normalization when baseline missing
docs(phase1): add phase1_summary.md for review
```

`scope` = `phase1`/`phase2`/`phase3` หรือชื่อ component (`scoring`, `data-gen`)

---

## Guardrails — ห้ามทำเองโดยไม่ถาม

1. **ห้ามข้าม Phase** — ต้องผ่าน Definition of Done + user รีวิวก่อนเสมอ
2. **ห้ามแก้ weight ใน Impulse Score (25/25/20/30)** โดยไม่รายงาน user — เป็น assumption ที่ยังไม่ validate ถ้าจะปรับต้องเสนอเหตุผล (เช่น correlation) แล้วถามก่อน
3. **ห้ามเพิ่ม dependency นอกเหนือ Tech Stack** โดยไม่ถามก่อน
4. **ห้ามเขียน Phase 4** (OCR, LINE Notify, Monte Carlo) จนกว่า user จะสั่งชัดเจน
5. **ห้าม hardcode threshold** ที่ยังเป็น assumption — ต้องดึงจาก `config.yaml` เท่านั้น
6. **ห้ามใช้ library version แบบ open-ended** (`>=`) ใน `requirements.txt`
7. **ห้าม merge branch เข้า `main` เอง** — เสนอได้ที่ STOP-FOR-REVIEW แต่รอ user ยืนยันเสมอ
8. **ห้าม force push บน `main`** และห้าม squash/rewrite history ที่ user รีวิวผ่านแล้ว

---

## Open Decisions — ต้องถาม User ก่อน (ห้ามเดาเอง)

| # | ประเด็น |
|---|---|
| 1 | จำนวนเดือนของ synthetic data และจำนวน transaction/วันโดยเฉลี่ย |
| 2 | Logic ที่แน่นอนของ transaction-level Needs/Wants override (เกณฑ์ amount) |
| 3 | Threshold ของ Impulse Score สำหรับ nudge alert (ปัจจุบันตั้งคร่าวๆ ที่ 70) |
| 4 | จังหวะรัน correlation analysis เพื่อปรับ weight (หลัง v1 ทันที หรือรวบไปพร้อม v2) |

---

## Reference

รายละเอียดเชิงธุรกิจ/product ทั้งหมด (Problem Statement, Data Strategy เหตุผล, Success Metrics, Known Limitations) อยู่ใน **`SmartSpend_AI_PRD_v2.md`** — ให้อ่านไฟล์นั้นประกอบก่อนเริ่ม Phase ใหม่ทุกครั้ง โดยเฉพาะช่วงที่ยังไม่คุ้นกับ context ของโปรเจกต์
