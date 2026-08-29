# SmartSpend AI: First-Jobber Automated Expense Categorization & Impulse Buying Predictor

**Version:** 2.5 (Revised — added Git Commit & Branching Strategy)
**Status:** Personal / Portfolio Project — Solo Build
**Type:** Data Science / Applied NLP Project

---

## 📌 Project Overview

**SmartSpend AI** คือระบบจัดการการเงินส่วนบุคคลที่ออกแบบมาสำหรับกลุ่ม **First Jobber** โดยเฉพาะ ใช้ **NLP** จำแนกหมวดหมู่รายจ่ายอัตโนมัติ และใช้ **Behavioral Feature Engineering** ตรวจจับพฤติกรรมการใช้จ่ายตามอารมณ์ (Impulse Buying) เพื่อสร้าง Financial Awareness ให้ผู้ใช้

โปรเจกต์นี้ทำเป็น **Solo Data Science Project** เพื่อฝึกฝนและสร้าง Portfolio ด้าน NLP + Behavioral Analytics — **ไม่ผูกกับผลิตภัณฑ์ธนาคารใดๆ** ใช้ข้อมูลจำลอง (synthetic data) ทั้งหมด

---

## 🎯 Problem Statement & Target User

**Target User:** First Jobber อายุ 22-28 ปี เพิ่งเริ่มมีรายได้ประจำ ยังไม่มีวินัยทางการเงินที่แน่นอน

**Pain Points:**

| # | Pain Point | รายละเอียด |
|---|---|---|
| 1 | Tracking Friction | ผู้ใช้ละเลยการจดบันทึกเพราะต้องพิมพ์เองทีละรายการ ไม่มีการแยกหมวดหมู่อัตโนมัติ |
| 2 | The Latte Factor | รายจ่ายยิบย่อยสะสมโดยไม่รู้ตัว แยกไม่ออกระหว่าง Needs (จำเป็น) กับ Wants (ต้องการ) |
| 3 | Impulse Buying & Payday Splurge | ใช้จ่ายตามอารมณ์ โดยเฉพาะช่วงกลางคืนและสัปดาห์แรกหลังเงินเดือนออก |

---

## 🗂️ Data Strategy (เพิ่มใหม่ — จุดที่ PRD เดิมขาด)

เนื่องจากไม่มีข้อมูลธุรกรรมจริง และข้อมูลการเงินส่วนบุคคลมีความ sensitive จึงใช้แนวทาง **Synthetic Data Generation** แทนการหา public dataset หรือใช้ข้อมูลจริง

### ทำไมต้อง Synthetic ไม่ใช่ Public Dataset (เช่น Kaggle)
- Dataset สาธารณะส่วนใหญ่เป็น transaction ทั่วไป **ไม่มี label "impulse buying"** ให้ ทำให้ประเมินผลโมเดลส่วนนี้ไม่ได้เลย
- ข้อความ memo/ร้านค้าใน dataset ต่างประเทศไม่สะท้อนบริบทไทย (ร้านสะดวกซื้อ, แอปส่งอาหาร, ธุรกรรมพร้อมเพย์)
- Synthetic data ให้คุณควบคุม **ground truth** ได้เต็มที่ — รู้ชัดว่า record ไหนคือ impulse จริง จึงวัด precision/recall ได้แม่นยำ

### วิธีสร้าง Synthetic Data
ใช้ Python (`Faker`, `numpy`, `pandas`) จำลอง transaction log ของ user สมมติ 1 คน เป็นระยะเวลา 6–12 เดือน โดยฝัง pattern ที่สมจริงไว้ล่วงหน้า:

- **Fixed pattern:** เงินเดือนเข้าทุกวันที่ 25, ค่าเช่า/ค่าเน็ต/ประกันตัดทุกเดือนวันที่คงที่
- **Routine pattern:** ค่าอาหารเที่ยง, ค่าเดินทาง (BTS/Grab) เกิดสม่ำเสมอวันธรรมดา
- **Injected impulse pattern (สำหรับสร้าง label):** ช้อปปิ้งออนไลน์ความถี่/มูลค่าพุ่งขึ้นในช่วง 23:00–02:00 และ 3 วันแรกหลังเงินเดือนออก — **ติด flag `is_impulse=True` ไว้ตอน generate** เพื่อใช้เป็น ground truth ทดสอบโมเดลภายหลัง
- **Noise:** สุ่ม merchant name, จำนวนเงินแบบมี variance, บาง transaction ผิดปกติแบบสุ่ม (ไม่ใช่ impulse) เพื่อไม่ให้โมเดล overfit กับ pattern ที่ตั้งไว้ตรงเกินไป
- **Category distribution (สัดส่วนเป้าหมาย):** กำหนดตายตัวไว้ล่วงหน้า ไม่ปล่อยให้ agent สุ่มเอง — `food` 35%, `transport` 20%, `shopping` 20%, `bills` 10%, `entertainment` 10%, `other` 5% (ปรับได้ แต่ต้องระบุในโค้ดเป็นค่าคงที่ ไม่ใช่สุ่มแบบ uniform)
- **Random Seed:** ต้อง fix `random_seed = 42` (หรือค่าคงที่ใดๆ) ไว้ในสคริปต์ เพื่อให้รันซ้ำได้ผลลัพธ์เดิมทุกครั้ง — จำเป็นต่อการเทียบ metrics ข้าม run ตาม schema ด้านล่าง

**Deliverable ของขั้นตอนนี้:** สคริปต์ `generate_synthetic_transactions.py` ที่ output เป็น CSV — schema เต็มระบุไว้ใน **Agent Implementation Guide → Data Schema** (มี `transaction_id`, `date`, `time`, `merchant`, `memo`, `amount`, `category`, `is_wants`, `is_impulse` รวม 9 column) ให้ยึด schema ตรงนั้นเป็นจุดอ้างอิงเดียว

---

## 💡 Key Features & Roadmap (จัดเป็น Phase — เดิมรวมเป็นก้อนเดียวจนกว้างเกินทำคนเดียว)

**สรุป Effort Estimate (ทำแบบ part-time, solo):**

| Phase | เนื้อหา | ประมาณเวลา |
|---|---|---|
| 1 | Synthetic data generation + Expense Categorization | 1.5–2 สัปดาห์ |
| 2 | Impulse Risk Scoring (v1 rule-based + v2 ML) | 1–1.5 สัปดาห์ |
| 3 | Dashboard & Visualization | 1 สัปดาห์ |
| 4 | Stretch goals | ไม่กำหนด (ทำถ้ามีเวลาเหลือ) |

### Phase 1 — MVP: Expense Categorization (NLP Core)
เป้าหมาย: ทำ pipeline จำแนกหมวดหมู่รายจ่ายให้ทำงานได้จริงก่อน

- **Text Preprocessing:** ทำความสะอาด memo/merchant name, ตัดคำไทย-อังกฤษด้วย `PyThaiNLP`
- **Vectorization:** TF-IDF (baseline) → ลอง Sentence Embeddings เทียบผลภายหลัง
- **Multi-Class Classification:** ทำนายหมวดหมู่ (อาหาร, คมนาคม, ช้อปปิ้ง, บิลรายเดือน ฯลฯ) ด้วย Logistic Regression และ LightGBM เปรียบเทียบกัน
- ⚠️ **Needs vs Wants Tagging ไม่อยู่ใน Phase 1** — ย้ายไปทำใน **Phase 2** เพราะ logic ที่ถูกต้องต้อง sync กับ Impulse Risk Score (ดูรายละเอียดใน Phase 2 ด้านล่าง) Phase 1 ทำแค่ category classification เท่านั้น
- **Evaluation:** Accuracy, Macro-F1 (เพราะ class ไม่สมดุลกัน เช่น "บิลรายเดือน" มีน้อยกว่า "อาหาร" มาก), Confusion Matrix

### Phase 2 — Impulse Risk Scoring Engine
เป้าหมาย: ให้คะแนนความเสี่ยงการใช้จ่ายตามอารมณ์แบบมี logic ชัดเจน

**ขั้นแรกของ Phase นี้ — Needs vs Wants Tagging (ระดับ transaction):**
ห้าม map ทั้ง category เป็น Needs/Wants แบบตายตัว (เช่น "อาหาร" = Need เสมอ) เพราะจะขัดกับ Impulse Score ที่ต้องใช้ "Category = Wants" เป็น feature หนึ่งในการให้คะแนน — ถ้า "อาหาร" ถูก fix เป็น Need ตลอด การสั่ง delivery แพงตอนตี 1 จะไม่ถูกนับเป็น Wants เลย ทั้งที่ควรเป็น impulse ชัดเจน จึงต้องทำ mapping ที่ **ระดับ transaction ไม่ใช่ระดับ category**: ใช้ category เป็น default ก่อน แล้ว override เป็น Wants เมื่อเข้าเงื่อนไขพิเศษ (เช่น amount สูงกว่า median ของ category นั้นอย่างมีนัยสำคัญ, หรือเกิดช่วง late-night)

**v1 — Rule-Based Weighted Score (เริ่มจากง่ายก่อน):**

คำนวณ `Impulse Risk Score (0–100)` จาก weighted sum ของ feature ต่อไปนี้:

| Feature | เงื่อนไข | น้ำหนัก |
|---|---|---|
| Late-night flag | เวลาธุรกรรมอยู่ 23:00–02:00 | 25 |
| Payday-proximity flag | อยู่ในช่วง 3 วันแรกหลังเงินเดือนออก | 25 |
| Category = Wants | หมวดที่จัดเป็น Wants (ระดับ transaction ตาม logic ด้านบน) | 20 |
| Amount anomaly | ยอดเงิน z-score สูงกว่าค่าเฉลี่ยของ user เอง (ไม่ใช่ threshold ตายตัว เพราะแต่ละคนใช้จ่ายไม่เท่ากัน) | 30 |

> ⚠️ **น้ำหนักเหล่านี้เป็น assumption เริ่มต้น ยังไม่ผ่านการ validate ด้วยข้อมูลใดๆ** (ตัวเลข 25/25/20/30 กำหนดขึ้นเองแบบ heuristic ให้รวมกันได้ 100) แผนคือหลังสร้าง synthetic data แล้วให้ปรับน้ำหนักใหม่โดยดู correlation ของแต่ละ feature กับ `is_impulse` (ground truth) เช่น feature ไหน correlation สูงกว่าควรได้น้ำหนักมากกว่า แทนที่จะคงตัวเลขที่เดาไว้ตอนแรก

> ⚠️ **Cold Start Problem:** feature "Amount anomaly" ต้องเทียบกับ baseline เฉลี่ยของ user เอง ซึ่งต้องมีประวัติธุรกรรมสะสมอย่างน้อย ~30 วันก่อนถึงจะคำนวณ z-score ได้อย่างมีความหมาย ผู้ใช้ใหม่ที่ยังไม่มี baseline นี้ ระบบจะข้าม feature นี้ไปก่อน (ให้น้ำหนัก 0 ชั่วคราว) และ normalize คะแนนรวมจาก 3 feature ที่เหลือแทน จนกว่าจะมี baseline เพียงพอ

**v2 — Upgrade เป็น ML (เมื่อมี label เพียงพอจาก synthetic data):**
เทรน Logistic Regression หรือ Isolation Forest บน feature เดียวกัน เทียบผลกับ v1 rule-based ว่าปรับปรุง precision/recall ได้จริงหรือไม่ — เป็นจุดที่โชว์ทักษะ "รู้จักเทียบ baseline" ได้ดีใน portfolio

> ⚠️ **ถ้า ML แพ้ Baseline:** ห้าม agent แก้ปัญหาด้วยการเพิ่ม feature engineering ต่อเองเรื่อยๆ โดยไม่หยุด — ให้บันทึกผลลัพธ์ที่แพ้ลง metrics JSON ตามปกติ พร้อมสมมติฐานว่าทำไมถึงแพ้ (เช่น "data leakage ทำให้ v1 ได้เปรียบเพราะ v1 ใช้ feature ตรงกับ pattern ที่ inject ไว้") แล้วหยุดรอ user ตัดสินใจว่าจะ (ก) ใช้ v1 rule-based ต่อไปเป็นทางออกสุดท้าย หรือ (ข) ให้ agent ลอง feature เพิ่มแบบมีขอบเขตชัดเจน — **ไม่ใช่ agent ตัดสินใจเอง**

**Proactive Nudge:** ธุรกรรมที่ score > threshold (เช่น 70) แสดง tag แจ้งเตือนใน dashboard

### Phase 3 — Dashboard & Visualization
เป้าหมาย: แสดงผลลัพธ์จาก Phase 1-2 ในรูปแบบที่ user ดูแล้วเข้าใจพฤติกรรมตัวเอง

- Needs vs Wants breakdown (pie/donut chart)
- Spending Behavior Heatmap (วัน x ชั่วโมง)
- สรุปแนวโน้มรายสัปดาห์/รายเดือน

> **Definition of Done ของ Phase 3 อยู่ใน Agent Implementation Guide → Task Checklist** (เดิม section นี้ไม่มี DoD ระบุไว้ ทำให้ agent ไม่รู้ว่า "เสร็จ" คืออะไร — แก้แล้ว ดูรายละเอียดด้านล่าง)

### Phase 4 (Stretch, ไม่ใช่เป้าหมายหลักของรอบแรก)
- [ ] OCR อ่านสลิปอัตโนมัติ
- [ ] Personal Goal Recommendation ด้วย Monte Carlo Simulation
- [ ] แจ้งเตือนผ่าน LINE Notify / Telegram

> **หมายเหตุ:** Phase 4 ใส่ไว้เป็น future roadmap เท่านั้น ไม่ต้องทำในรอบแรกเพื่อไม่ให้ scope บวมจนไม่จบ

---

## 🛠️ Tech Stack

**Data Generation & Science**
- Python 3.11+, `Faker` (synthetic data), Pandas, NumPy, Scikit-learn
- NLP: PyThaiNLP, TF-IDF Vectorizer
- Models: Logistic Regression, LightGBM (classification), Isolation Forest (anomaly, Phase 2 v2)

**Backend & API** *(เริ่มทำหลัง Phase 1-2 เสร็จ เพื่อไม่ให้เสียเวลา build infra ก่อนมี model ที่ใช้งานได้)*
- FastAPI, Pydantic, SQLite/SQLAlchemy

**Frontend & Visualization**
- HTML5, Tailwind CSS, Chart.js / Plotly

---

## 📏 Success Metrics (ปรับปรุงตาม v5 Protocol)

| Component | Metric | เป้าหมาย |
|---|---|---|
| Expense Categorization — Baseline | Macro-F1 ของ keyword/rule-based matching ล้วน (ไม่ใช้ ML) | ใช้เป็นเส้นฐานเทียบว่า ML ให้คุณค่าเพิ่มจริงหรือไม่ (~0.70-0.75) |
| Expense Categorization — ML | Macro-F1 (บน Test Set ที่มี Realistic Noise) | ต้องสูงกว่า Baseline อย่างมีนัยสำคัญ และอยู่ในช่วงที่สมจริง `[0.75, 0.95]` |
| Generalization — Unseen Merchant | Accuracy ของกลุ่มร้านค้าใหม่ที่ไม่เคยเห็นใน Train Set | `≥ 0.60` (Floor) และ Generalization Gap ระหว่าง Seen vs Unseen `≤ 0.20` |
| Impulse Risk Score (v1 rule-based) | Precision/Recall เทียบ ground truth `is_impulse` | ใช้เป็น baseline อ้างอิงสำหรับ v2 |
| Impulse Risk Score (v2 ML) | ปรับปรุงจาก v1 ได้อย่างมีนัยสำคัญ | Recall เพิ่มขึ้นโดย Precision ไม่ลดมาก |

---

## ⚠️ Known Limitations
- **Data Leakage & Mitigation ด้วย Unseen Partitioning:** ใน Synthetic Data ทั่วไป โมเดลอาจจดจำ Template ได้ 100% จึงได้ออกแบบ **Custom 2-Stage Stratified Split** กักร้านค้ากลุ่ม Unseen 20% เข้า Test Set ทั้งหมดเพื่อประเมิน Zero-shot Generalization อย่างแท้จริง อย่างไรก็ดี พฤติกรรมจริงของมนุษย์มีความซับซ้อนและมีคำสแลง/บริบทที่หลากหลายกว่าที่จำลอง
- **Impulse Pattern Leakage:** เพราะ impulse pattern (late-night + payday window) ถูก inject ไว้เองตอน generate data และ v1 Impulse Score ก็ใช้ feature ชุดเดียวกันนี้ตรงๆ ผลลัพธ์ precision/recall ที่ได้จึงมีแนวโน้มสูงเพราะโมเดลเรียนรู้จากกฎที่ตั้งไว้ ต้องตระหนักถึงข้อจำกัดนี้เสมอ
- **"Impulse Buying" เชิงจิตวิทยา:** การนิยามด้วย rule/feature เป็นการประมาณเชิงพฤติกรรม ไม่ใช่การวัดอารมณ์หรือแรงจูงใจจริง
- **Dataset ขนาด 1 User Profile:** ในอนาคตสามารถขยายเป็นหลาย Persona (สายประหยัด, สายช้อปปิ้ง, ฟรีแลนซ์) เพื่อทดสอบ Robustness

---

## 🤖 Agent Implementation Guide

> Section นี้เขียนสำหรับ **AI Coding Agent (เช่น Antigravity)** ที่จะรับ PRD นี้ไปเขียนโค้ดต่อโดยตรง เป้าหมายคือให้ agent ทำงานได้โดยไม่ต้องเดา และรู้ว่าจุดไหนต้องหยุดถาม user ก่อน

### Folder Structure

```
smartspend-ai/
├── config.yaml                   # threshold/weight ทั้งหมดที่ยังเป็น assumption (ดู Guardrail #5)
├── requirements.txt               # pinned version ทุก library
├── data/
│   ├── raw/                      # synthetic CSV ที่ generate ได้
│   └── processed/                # ข้อมูลหลัง preprocessing
├── src/
│   ├── data_generation/
│   │   └── generate_synthetic_transactions.py
│   ├── nlp/
│   │   ├── preprocessing.py      # text cleaning, PyThaiNLP tokenization
│   │   └── vectorizer.py         # TF-IDF / embeddings
│   ├── models/
│   │   ├── train_classifier.py   # Expense categorization (LogReg / LightGBM)
│   │   └── evaluate.py           # metrics: accuracy, macro-F1, confusion matrix
│   ├── scoring/
│   │   ├── impulse_rules.py      # v1 rule-based weighted score (คำนวณ score เท่านั้น)
│   │   ├── evaluate_impulse.py   # แยกจาก impulse_rules.py: Precision/Recall เทียบ is_impulse ground truth, save metrics JSON
│   │   └── impulse_model.py      # v2 ML model (Phase 2 เท่านั้น)
│   ├── needs_wants/
│   │   └── classify_needs_wants.py   # transaction-level mapping (ดู Open Decision #2)
│   ├── api/                      # Phase 3 เท่านั้น — FastAPI endpoints
│   │   └── main.py
│   └── frontend/                 # Phase 3 เท่านั้น — HTML/Tailwind/Chart.js
│       ├── index.html
│       └── dashboard.js
├── notebooks/                    # EDA และ experiment เท่านั้น ไม่ใช่ production code
├── tests/
│   ├── test_categorization.py
│   ├── test_impulse_scoring.py
│   └── test_needs_wants.py
├── models_artifacts/              # โมเดลที่เทรนแล้ว (.pkl / .joblib) — save ที่นี่เท่านั้น, ตั้งชื่อไฟล์แบบ {model_name}_{phase}_{date}.joblib
├── outputs/
│   ├── metrics/                  # JSON ผลลัพธ์ evaluation ต่อ phase
│   └── phase_summaries/          # ไฟล์สรุปที่ agent สร้างก่อนหยุดรอ user รีวิว (ดู "Stop-for-Review Protocol")
└── README.md
```

**`requirements.txt` ต้อง pin version ชัดเจน** ไม่ใช้ `>=` แบบเปิดกว้าง (เช่น `scikit-learn==1.5.1` ไม่ใช่ `scikit-learn>=1.5`) เพื่อไม่ให้ผลลัพธ์เปลี่ยนข้าม session เพราะ library update — ระบุ Python version ที่ใช้ไว้บนสุดของไฟล์เป็น comment ด้วย (แนะนำ Python 3.11+)

**`config.yaml` ต้องรวม** อย่างน้อย: `impulse_score.weights` (25/25/20/30), `impulse_score.nudge_threshold` (70), `impulse_score.late_night_window` (23:00-02:00), `needs_wants.override_amount_multiplier`, `data_generation.random_seed` (42) — ค่าทั้งหมดนี้คือ "assumption ที่ยังไม่ validate" ตาม Guardrail #2 และ #5 ห้ามกระจาย hardcode ไว้หลายไฟล์

### Data Schema (exact column spec)

**Synthetic transaction CSV** (`data/raw/transactions.csv`):

| Column | Type | หมายเหตุ |
|---|---|---|
| `transaction_id` | string (UUID) | unique ต่อแถว |
| `date` | date (`YYYY-MM-DD`) | |
| `time` | string (`HH:MM`, 24h) | ใช้คำนวณ late-night flag |
| `merchant` | string | ชื่อร้าน/บริการ |
| `memo` | string | ข้อความ free-text ที่ user กรอกเอง (ไทย/อังกฤษปน) |
| `amount` | float | บาท, ทศนิยม 2 ตำแหน่ง |
| `category` | string (enum) | ground truth: `food`, `transport`, `shopping`, `bills`, `entertainment`, `other` |
| `is_wants` | bool | ground truth ระดับ transaction (ไม่ใช่ category-level) |
| `is_impulse` | bool | ground truth สำหรับ evaluate Impulse Score |

**Evaluation output** (`outputs/metrics/phase{N}_metrics.json`): ต้องมี key `phase`, `component`, `metric_name`, `value`, `baseline_value`, `timestamp` เป็นอย่างน้อย เพื่อให้เทียบผลข้าม run ได้

### Task Checklist ต่อ Phase (Definition of Done)

**Phase 1 — Expense Categorization**
- [ ] `generate_synthetic_transactions.py` รันแล้วได้ CSV 6-12 เดือน ตรง schema ด้านบนทุก column, ใช้ fixed random seed, สัดส่วน category ตรงตามที่กำหนดใน Data Strategy
- [ ] `preprocessing.py` รับ raw memo/merchant → คืน cleaned text (มี unit test เทียบ input/output ตัวอย่าง)
- [ ] `train_classifier.py` เทรน **baseline (keyword/rule matching)** และ **ML model (LogReg + LightGBM)** แยกกัน บันทึกทั้งคู่ลง metrics JSON, save โมเดลลง `models_artifacts/`
- [ ] `evaluate.py` output Macro-F1, confusion matrix ของทั้ง baseline และ ML — ML ต้องชนะ baseline ถึงจะถือว่า Phase 1 ผ่าน (ถ้าแพ้ ดู fallback ใน Phase 2 section — ใช้หลักการเดียวกัน: หยุดรอ user)
- [ ] `pytest tests/test_categorization.py` ผ่านทั้งหมด — **ต้อง assert อย่างน้อย:** (a) output schema ของ CSV ตรงทุก column/type ตามสเปค (b) ไม่มี missing value ใน column บังคับ (c) Macro-F1 ของ ML > Macro-F1 ของ baseline (d) จำนวนแถวใน output ตรงกับที่ generate (ไม่มี row หายระหว่าง pipeline)
- **[STOP-FOR-REVIEW]** ดู "Stop-for-Review Protocol" ด้านล่าง ก่อนเริ่ม Phase 2

**Phase 2 — Impulse Risk Scoring**
- [ ] `classify_needs_wants.py` ทำ mapping ระดับ transaction ตาม Open Decision #2 (ต้องยืนยัน logic กับ user ก่อนเขียนโค้ดจริง)
- [ ] `pytest tests/test_needs_wants.py` ผ่านทั้งหมด — **ต้อง assert อย่างน้อย:** (a) ทุก transaction ได้ค่า `is_wants` เป็น boolean เสมอ ไม่มี null (b) override logic ทำงานตามเกณฑ์ที่ตกลงกับ user ใน Open Decision #2 จริง (ทดสอบด้วย test case ที่ควร override และไม่ควร override)
- [ ] `impulse_rules.py` implement weight ตามตารางใน Phase 2 section, **ดึงค่าจาก `config.yaml`** — **ห้ามแก้ตัวเลขน้ำหนักเอง** (ดู guardrail ด้านล่าง)
- [ ] `evaluate_impulse.py` คำนวณ Precision/Recall ของ v1 เทียบ `is_impulse` ground truth บันทึกลง metrics JSON พร้อม disclaimer เรื่อง data leakage แนบในผลลัพธ์
- [ ] `impulse_model.py` (v2 ML) ทำเฉพาะหลัง v1 มีผลลัพธ์และ user รีวิวแล้วเท่านั้น
- [ ] `pytest tests/test_impulse_scoring.py` ผ่านทั้งหมด — **ต้อง assert อย่างน้อย:** (a) score อยู่ในช่วง 0-100 เสมอ (b) cold-start case (user ไม่มี baseline) ไม่ error และ normalize คะแนนถูกต้องตามที่ระบุใน Cold Start Problem (c) weight ที่ใช้จริงตรงกับค่าใน `config.yaml` ไม่ hardcode ซ้ำในโค้ด
- **[STOP-FOR-REVIEW]** ดู "Stop-for-Review Protocol" ด้านล่าง ก่อนเริ่ม Phase 3

**Phase 3 — Dashboard**
- [ ] `api/main.py` เปิด endpoint อย่างน้อย: ดึงรายการ transaction พร้อม category/is_wants/impulse_score, ดึงสรุป Needs vs Wants, ดึงข้อมูล heatmap
- [ ] `frontend/index.html` + `dashboard.js` แสดง 3 อย่างตามที่ระบุใน Phase 3 section: Needs vs Wants breakdown, Spending Behavior Heatmap, สรุปแนวโน้มรายสัปดาห์/รายเดือน
- [ ] ทดสอบด้วยตาว่า dashboard โหลดข้อมูลจริงจาก Phase 1-2 ได้ถูกต้อง (เทียบตัวเลขในกราฟกับ metrics JSON)
- **[STOP-FOR-REVIEW]** สร้าง summary ตามปกติ แม้จะเป็น phase สุดท้ายของรอบแรกก็ตาม

### Stop-for-Review Protocol (ทำให้ "หยุดรอ user" เป็น action จับต้องได้)

เมื่อถึงจุด **[STOP-FOR-REVIEW]** agent ต้องทำตามลำดับนี้เสมอ ไม่ตีความเอง:

1. สร้างไฟล์ `outputs/phase_summaries/phase{N}_summary.md` ที่มีอย่างน้อย: รายการ task ที่ทำเสร็จ, ผลลัพธ์ metrics พร้อมเทียบ baseline, ปัญหาหรือ assumption ที่พบระหว่างทาง, และคำถามที่ค้างอยู่ (ถ้ามี)
2. Commit งานทั้งหมดของ Phase นั้นให้ครบ (ตาม Git Commit & Branching Strategy ด้านล่าง) แล้วใส่ commit hash ล่าสุดไว้ใน summary file — **ห้าม merge เข้า `main`** จนกว่าจะได้รับคำตอบจาก user ในแชท
3. พิมพ์ข้อความสรุปสั้นๆ ในแชทบอก user ว่า Phase นี้เสร็จแล้ว พร้อมลิงก์ไปที่ summary file และถามคำถามที่ค้าง (ถ้ามี) ตรงๆ ไม่ implicit
4. ถ้า user ไม่ตอบและ agent ทำงานแบบ autonomous session ยาว — **ห้าม auto-continue ไป Phase ถัดไปเอง** ให้ agent สิ้นสุด session รอบนั้นไว้ที่จุดนี้แทน

### Git Commit & Branching Strategy

**Branch Strategy — แยก branch ต่อ Phase**

| Branch | ใช้ทำอะไร |
|---|---|
| `main` | เสถียร มีเฉพาะโค้ดที่ user รีวิวและอนุมัติแล้วที่ STOP-FOR-REVIEW เท่านั้น |
| `phase-1-categorization` | งานทั้งหมดของ Phase 1 |
| `phase-2-impulse-scoring` | งานทั้งหมดของ Phase 2 |
| `phase-3-dashboard` | งานทั้งหมดของ Phase 3 |

Agent ทำงานอยู่ใน branch ของ Phase ปัจจุบันเสมอ **ห้าม merge เข้า `main` เอง** — เสนอ merge ได้ที่ขั้นตอน STOP-FOR-REVIEW แต่ต้องรอ user ยืนยันในแชทก่อนถึงจะ merge จริง (ดู Guardrail #7 ด้านล่าง)

**ความถี่ในการ Commit — commit ต่อ sub-task ที่เสร็จ**

Commit ทุกครั้งที่ checkbox ใน Task Checklist ของ Phase นั้นถูกติ๊กเสร็จ 1 อัน (ไม่ใช่ commit รวมทั้ง Phase ทีเดียว และไม่ใช่ commit ทุกบรรทัดที่แก้) เพื่อให้:
- ย้อนดู history ได้ว่า sub-task ไหนเสร็จเมื่อไหร่ ถ้าต้อง debug หรือ revert เฉพาะจุด
- ไม่ปนงานหลาย component เข้าด้วยกันในคอมมิตเดียว (เช่น ห้ามรวม ML code กับ frontend code ไว้คอมมิตเดียวกัน แม้จะอยู่คนละ branch อยู่แล้วก็ตาม)

**รูปแบบ Commit Message — Conventional Commits**

ใช้ฟอร์แมต `<type>(<scope>): <description>` เพราะ scan history ได้ง่าย และ scope ผูกกับ Phase/component ในโครงสร้างไฟล์ที่มีอยู่แล้วพอดี

| Type | ใช้เมื่อ |
|---|---|
| `feat` | เพิ่มฟีเจอร์ใหม่ (เช่น เขียนไฟล์ script ใหม่ตาม checklist) |
| `test` | เพิ่มหรือแก้ test |
| `fix` | แก้บั๊ก |
| `docs` | แก้ไฟล์ summary, README, PRD |
| `chore` | งานเชิง infra เช่น เพิ่ม dependency ใน requirements.txt |

**ตัวอย่าง:**
```
feat(phase1): implement synthetic transaction generator with fixed seed
test(phase1): add schema validation tests for generated CSV
feat(phase2): implement rule-based impulse score in impulse_rules.py
fix(phase2): correct cold-start normalization when baseline missing
docs(phase1): add phase1_summary.md for review
```

`scope` ให้ใช้ `phase1` / `phase2` / `phase3` เป็นหลัก หรือใช้ชื่อ component (`scoring`, `data-gen`) ถ้า commit นั้นชัดเจนว่าเกี่ยวกับ component เดียวและอาจถูกอ้างอิงข้าม phase ในอนาคต

**เชื่อมกับ Stop-for-Review Protocol:** ก่อนสร้างไฟล์ `phase{N}_summary.md` ต้อง commit งานทั้งหมดของ Phase นั้นให้ครบก่อน แล้วใส่ commit hash ล่าสุดไว้ใน summary file ด้วย เพื่อให้ user เช็คได้ตรงจุดว่า diff ที่จะรีวิวคือช่วงไหน

### Guardrails — สิ่งที่ Agent ห้ามทำเอง

1. **ห้ามข้าม Phase** — ห้ามเริ่ม Phase ถัดไปก่อนที่ Phase ก่อนหน้าจะผ่าน Definition of Done และ user รีวิวแล้ว
2. **ห้ามแก้ค่า weight ใน Impulse Score (25/25/20/30)** โดยไม่รายงาน user — ค่านี้เป็น assumption ที่ยังไม่ validate ตาม **Phase 2 section (⚠️ น้ำหนักเหล่านี้เป็น assumption เริ่มต้น)** และ **Open Decision #4** ถ้า agent พบว่าควรปรับ ให้เสนอค่าใหม่พร้อมเหตุผล (เช่น correlation ที่คำนวณได้) แล้วถามก่อน ไม่ implement เงียบๆ
3. **ห้ามเพิ่ม dependency นอกเหนือ Tech Stack** ที่ระบุไว้โดยไม่ถามก่อน
4. **ห้ามเขียน Phase 4 (OCR, LINE Notify, Monte Carlo)** จนกว่า user จะสั่งชัดเจน — อยู่นอก scope รอบแรก
5. **ห้าม hardcode threshold ที่ยังเป็น assumption** (เช่น "score > 70" สำหรับ nudge alert) เป็นค่าตายตัวในหลายที่ — ต้องดึงจาก `config.yaml` เท่านั้น (ดูรายการ key ที่ต้องมีใน Folder Structure section) เพื่อปรับทีเดียวได้
6. **ห้ามใช้ library version แบบ open-ended** (`>=`) ใน `requirements.txt` — ต้อง pin version ชัดเจนเสมอ เพื่อ reproducibility ข้าม session
7. **ห้าม merge branch ของ Phase เข้า `main` เอง** — เสนอ merge ได้ที่ STOP-FOR-REVIEW แต่ต้องรอ user ยืนยันในแชทก่อนเสมอ ไม่ merge ล่วงหน้าแม้จะมั่นใจว่างานเสร็จสมบูรณ์แล้ว
8. **ห้าม force push บน `main`** ในทุกกรณี และห้าม squash/rewrite history ของ commit ที่ user เคยรีวิวผ่านไปแล้ว

### Open Decisions — ต้องถาม User ก่อนเดินหน้า (ห้าม Agent เดาเอง)

| # | ประเด็น | ทำไมต้องถาม |
|---|---|---|
| 1 | จำนวนเดือนของ synthetic data (6 หรือ 12 เดือน) และจำนวน transaction/วันโดยเฉลี่ย | กระทบขนาด dataset และเวลาเทรน |
| 2 | Logic ที่แน่นอนของ transaction-level Needs/Wants override (เช่น amount สูงกว่า median เท่าไหร่ถึง override) | PRD ระบุแนวคิดไว้แต่ยังไม่มีตัวเลขเกณฑ์ที่แน่นอน |
| 3 | Threshold ของ Impulse Score สำหรับ nudge alert (ตอนนี้ตั้งไว้คร่าวๆ ที่ 70) | เป็นค่า assumption ที่ยังไม่ validate |
| 4 | จะรัน correlation analysis เพื่อปรับ weight ตอนไหน (หลัง Phase 2 v1 เสร็จทันที หรือรวบไปทำพร้อม v2) | กระทบลำดับงานใน Phase 2 |

---

## 🚀 Future Roadmap
- [ ] ขยาย synthetic data เป็นหลาย user persona (สายประหยัด, สายใช้จ่ายเกินตัว ฯลฯ) เพื่อทดสอบ generalization
- [ ] OCR อ่านสลิปอัตโนมัติ
- [ ] Personal Goal Recommendation ด้วย Monte Carlo Simulation
- [ ] เชื่อมต่อ LINE Notify / Telegram
