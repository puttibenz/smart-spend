# Implementation Plan — Phase 1 Refinement (Final Protocol v5 — Approved)

แก้ปัญหา **Data Leakage / 1.0000 Macro-F1** และ **Category Distribution Drift** ด้วยกลไกสร้างข้อมูลแบบ Strict Quota + Decoupled Templates + Multi-level Noise, **Category & Sub-type Stratified Unseen Split**, พร้อมระบบ Safeguard ที่มี **No-Seed-Hunting Protocol** ป้องกันการ p-hack ผลลัพธ์

> เอกสารนี้คือฉบับรวมสุดท้าย (v5) ที่รวมทุกจุดที่ผ่านการรีวิวจนอนุมัติแล้วจาก fix_plan 1-4 บวกกับ 2 จุดสุดท้ายที่พบในรอบตรวจ v4 — **อนุมัติให้เริ่มเขียนโค้ดได้ทันทีตาม spec นี้**

---

## User Review Required — สรุปทุกข้อตกลง (รวมทุกรอบ)

> [!IMPORTANT]
> **1. Custom 2-Stage Partitioning Split:**
> - ทุก Transaction จากร้านค้ากลุ่ม `unseen_merchants` → บังคับเข้า Test Set 100% (ห้ามหลุดเข้า Train แม้แต่แถวเดียว)
> - Transaction จาก `seen_merchants` ที่เหลือ → Stratified Split 80/20 ตามปกติ (`random_state=42`)
> - Test Set สุดท้าย = unseen ทั้งหมด + seen ส่วน 20%
>
> **2. Category & Sub-type Stratified Unseen Selection (จุดใหม่จากรอบตรวจ v4):**
> - สุ่มเลือก 20% ของร้านค้าเป็น `unseen_merchants` **แบบ stratified 2 ชั้น**:
>   - ชั้นที่ 1: แยกตาม Category (food/transport/shopping/bills/entertainment/other)
>   - ชั้นที่ 2: แยกตาม Sub-type ภายใน Category นั้น (`routine`, `wants`, `impulse_delivery`/`impulse_online`)
> - เป้าหมาย: `unseen_merchants` ต้องมีตัวแทนจากทุก sub-type ที่มีอยู่จริงในแต่ละ category ไม่ใช่กระจุกอยู่แค่ `routine` เพราะ sub-type ต่างกัน "โทนภาษา" ต่างกันมาก (ร้านทั่วไป vs ร้านหรู vs ร้านที่มักเกี่ยวกับ impulse buying) — ถ้า unseen กระจุกอยู่ sub-type เดียว `unseen_accuracy` จะไม่ได้วัด generalization ที่ครอบคลุมจริง โดยเฉพาะ sub-type ที่เชื่อมกับ Impulse Score ใน Phase 2
>
> **3. Transparent Effective Train/Test Ratio Reporting:**
> - รายงานสัดส่วนจริง (เช่น 68:32) ใน `phase1_metrics.json` และ `phase1_summary.md` เพราะสัดส่วนจะไม่ใช่ 80/20 เป๊ะอีกต่อไป
>
> **4. Minimum Training Samples Safeguard:**
> - ทุก Category ใน `train_split.csv` ต้องมี ≥ `min_train_samples_per_category` (40) samples
>
> **5. No-Seed-Hunting Protocol (จุดใหม่จากรอบตรวจ v4 — สำคัญมาก):**
> - **ห้าม agent เปลี่ยน `random_seed` ไปเรื่อยๆ เพื่อหาค่าที่ทำให้ test ผ่าน** ไม่ว่าจะเป็น safeguard ข้อ 4 (min samples) หรือเกณฑ์ใดๆ ก็ตาม
> - `random_seed: 42` ต้อง **fix ค่าเดียวตลอดทั้งโปรเจกต์** ตาม Guardrail เดิมใน AGENTS.md
> - ถ้ารันแล้ว safeguard ใดไม่ผ่าน (เช่น category `other` ได้แค่ 34/40 samples) → **agent ต้องหยุดทันที บันทึกตัวเลขจริงที่ได้ลง `outputs/phase_summaries/phase1_summary.md`** พร้อมระบุสาเหตุที่เป็นไปได้ (เช่น "unseen merchant ของหมวด other มี transaction volume สูงกว่าเฉลี่ยโดยบังเอิญ") แล้ว **[STOP-FOR-REVIEW]** รอ user ตัดสินใจว่าจะ (ก) ปรับ `unseen_merchant_pct` ลงเฉพาะหมวดที่มีปัญหา หรือ (ข) เพิ่มจำนวน transaction เป้าหมายรวม หรือ (ค) ลดเกณฑ์ `min_train_samples_per_category` — **ไม่ใช่ agent ตัดสินใจเปลี่ยน seed หรือ threshold เองเงียบๆ**

---

## Acceptance Criteria & Quantitative Thresholds (ฉบับสุดท้าย)

1. **Macro-F1 Acceptance Bounds:**
   - `0.75 <= Macro-F1 <= 0.95`
   - `ml_macro_f1 > baseline_macro_f1`
2. **Seen vs Unseen Merchant Quantitative Criteria:**
   - `unseen_merchant_accuracy >= 0.60`
   - `(seen_merchant_accuracy - unseen_merchant_accuracy) <= 0.20`
3. **Category Distribution Quota Tolerance:**
   - คลาดเคลื่อนไม่เกิน `±1.0%` จากเป้าหมาย (35/20/20/10/10/5)
4. **Data Partition Integrity & Minimum Samples:**
   - `unseen_merchants` ใน `train_split.csv` ต้องเป็น 0 เด็ดขาด
   - ทุก Category ใน `train_split.csv` ต้องมีอย่างน้อย 40 samples
5. **Sub-type Coverage in Unseen Set (ใหม่):**
   - ทุก sub-type ที่มีอยู่จริงในแต่ละ category (routine/wants/impulse_*) ต้องมีร้านค้าอย่างน้อย 1 ร้านอยู่ใน `unseen_merchants` ของ category นั้น (ยกเว้น sub-type ที่มีร้านค้าน้อยกว่า 2 ร้านอยู่แล้วในต้นฉบับ — กรณีนี้ให้ agent ระบุไว้ใน summary ว่าข้ามเพราะ pool มีจำกัด)

---

## Proposed Changes

### Git Branch
- ดำเนินการต่อบน branch: **`phase-1-categorization`**

---

### Configuration & Base Setup

#### [MODIFY] `config.yaml`
```yaml
data_generation:
  category_distribution_tolerance: 0.01  # ±1.0%
  noise:
    ambiguous_memo_pct: 0.15
    typo_pct: 0.10
    unseen_merchant_pct: 0.20

evaluation_thresholds:
  macro_f1_floor: 0.75
  macro_f1_ceiling: 0.95
  unseen_accuracy_floor: 0.60
  max_generalization_gap: 0.20
  min_train_samples_per_category: 40
```

---

### Data Generation Component

#### [MODIFY] `src/data_generation/generate_synthetic_transactions.py`
- **Strict Quota Allocation:** คำนวณจำนวนแถวแน่นอนตามสัดส่วน (35/20/20/10/10/5) จากจำนวนเป้าหมาย (~1,550 แถว)
- **Decoupled Pools (Expanded):** ขยายคลังร้านค้า (10+ ร้านต่อหมวด รวม 70+ ร้าน) และ Memos (20+ ข้อความต่อหมวด รวม 120+ ข้อความ)
- **Category & Sub-type Stratified Unseen Selection (`random_seed: 42` เท่านั้น — ห้ามเปลี่ยน):**
  1. Group ร้านค้าตาม (category, sub_type) ก่อน
  2. ในแต่ละ group สุ่มเลือก 20% เป็น `unseen_merchants` (ปัดขึ้นอย่างน้อย 1 ร้านถ้า group มีร้านค้า ≥ 2 ร้าน)
  3. ถ้า group ไหนมีร้านค้าน้อยกว่า 2 ร้าน → ข้าม ไม่บังคับมี unseen ในกลุ่มนั้น แต่ต้อง log ไว้ว่าข้ามเพราะอะไร
  4. ติด flag `is_unseen_merchant: bool` และ `sub_type: str` ให้ทุก transaction
- **Deterministic Noise Injection (`random_seed: 42` เท่านั้น):**
  - Ambiguous Memos (`"โอน"`, `"จ่ายเงิน"`, `"พร้อมเพย์"`, `"xxx"`, `""`) ตาม `ambiguous_memo_pct`
  - Typo / Slang ใน Memo ตาม `typo_pct`
- **ห้าม retry ด้วย seed อื่นถ้าผลลัพธ์ไม่เข้าเกณฑ์ — ดู No-Seed-Hunting Protocol**

---

### NLP & Training Pipeline

#### [MODIFY] `src/models/train_classifier.py`
- **Custom 2-Stage Stratified Split:**
  1. Transaction ที่ `is_unseen_merchant == True` → เข้า `test_df` ทั้งหมด 100%
  2. Transaction ที่เหลือ (`is_unseen_merchant == False`) → `train_test_split(stratify=y, test_size=0.2, random_state=42)`
  3. รวม Test Set, บันทึก `data/processed/train_split.csv` และ `test_split.csv` (พร้อมคอลัมน์ `is_unseen_merchant`, `sub_type`)
- **Safeguard Check (Fail-Stop, ไม่ retry):**
  - ตรวจว่าทุก Category ใน `train_split.csv` มี sample ≥ 40
  - ถ้าไม่ผ่าน → `raise RuntimeError` พร้อมตัวเลขจริงของทุก category ที่ไม่ผ่าน ไม่ auto-fix ด้วยการเปลี่ยน seed/threshold เอง
- คำนวณและบันทึก **Effective Train/Test Ratio** จริง
- Fit TF-IDF Vectorizer บน `train_split` เท่านั้น (ห้ามแตะ test_split)
- Train & Save Models (`LogisticRegression(random_state=42)`, `LGBMClassifier(random_state=42)`) ลง `models_artifacts/`

---

### Comprehensive Evaluation & Error Analysis

#### [MODIFY] `src/models/evaluate.py`
- ดึงเกณฑ์จาก `config.yaml` (`evaluation_thresholds`) ทั้งหมด ไม่ hardcode
- คำนวณ:
  - **Effective Train/Test Split Ratio**
  - **`merchant_overlap_analysis`** — แยก seen vs unseen, breakdown ต่อ category **และต่อ sub-type**
  - **Normalized Confusion Matrix** (% ต่อ row)
  - **Confidence Margins** (Top-1 vs Top-2 Probability)
  - **Error Attribution** (Baseline vs ML — ผิดร่วมกัน / ผิดต่างกัน)
  - **Low Support Flag** (`is_noisy_metric: True` สำหรับหมวดที่ test support < 20)
- Export `outputs/metrics/phase1_error_analysis.csv`
- บันทึก `outputs/metrics/phase1_metrics.json`
- **Enforce Fallback Guardrail (Fail-Stop เดียวกันทุกเกณฑ์):** `raise RuntimeError` หากเกณฑ์ข้อใดข้อหนึ่งไม่ผ่าน (ml vs baseline, floor/ceiling, unseen floor, generalization gap) — ไม่ silent-pass, ไม่ auto-adjust เกณฑ์เอง

---

### Automated Tests

#### [MODIFY] `tests/test_categorization.py`
1. **Schema Check:** CSV ตรง 9 คอลัมน์และ data types ถูกต้อง
2. **Non-Null Check:** ไม่มี missing values ใน required columns
3. **Category Distribution Quota Check:** คลาดเคลื่อนไม่เกิน ±1.0%
4. **Performance Bounds:** `ml_macro_f1 > baseline_macro_f1` และ `0.75 <= ml_macro_f1 <= 0.95`
5. **Generalization Gap & Floor:** `unseen_accuracy >= 0.60` และ `generalization_gap <= 0.20`
6. **Unseen Integrity Check:** `len(set(unseen_merchants).intersection(train_merchants)) == 0`
7. **Minimum Training Samples Safeguard:** ทุกหมวดใน train set มี ≥ 40 samples
8. **Sub-type Coverage Check (ใหม่):** ทุก (category, sub_type) ที่มีร้านค้า ≥ 2 ร้าน ต้องมีอย่างน้อย 1 ร้านอยู่ใน unseen set
9. **Artifact & Error Export Check:** มีไฟล์โมเดลใน `models_artifacts/` และ `outputs/metrics/phase1_error_analysis.csv`

---

### Documentation Sync

#### [MODIFY] `outputs/phase_summaries/phase1_summary.md`
- สรุปผลรอบใหม่: ตาราง Seen vs Unseen (แยกต่อ sub-type ด้วย), Effective Train/Test Ratio, Insights จาก Error Analysis, Commit Hashes
- ถ้ามี safeguard ใดไม่ผ่านและต้องหยุด → บันทึกตัวเลขจริงและเหตุผลไว้ในนี้ตาม No-Seed-Hunting Protocol

#### [MODIFY] `SmartSpend_AI_prd.md`
- อัปเดต Known Limitations และ Success Metrics ให้สะท้อน Category & Sub-type Stratified Split, Multi-level Noise, Seen vs Unseen Evaluation Strategy

---

## Verification Plan

### Automated Tests
```powershell
.\.venv\Scripts\python.exe src/data_generation/generate_synthetic_transactions.py
.\.venv\Scripts\python.exe src/models/train_classifier.py
.\.venv\Scripts\python.exe src/models/evaluate.py
.\.venv\Scripts\pytest.exe tests/test_categorization.py -v
```

### Manual Verification
- ตรวจ `outputs/metrics/phase1_metrics.json` — มี `merchant_overlap_analysis` (ต่อ category และ sub-type) และ Effective Ratio
- ตรวจ `outputs/metrics/phase1_error_analysis.csv`
- ตรวจ `phase1_summary.md` และ `SmartSpend_AI_prd.md` อัปเดตครบ
- **ถ้ามีการหยุดกลางทางเพราะ safeguard ไม่ผ่าน:** ตรวจว่า agent รายงานตัวเลขจริงตรงไปตรงมา ไม่ได้ silent-retry ด้วย seed อื่น (เช็คจาก git log ว่า `random_seed` ใน `config.yaml` ไม่ถูกแก้ระหว่างทาง)

---

## Status: ✅ Approved — พร้อมให้เริ่มโค้ดจริงได้ทันที
