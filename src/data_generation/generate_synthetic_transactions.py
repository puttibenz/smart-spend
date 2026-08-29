"""
SmartSpend AI - Synthetic Transaction Generator (v5 Final Protocol)
Generates realistic transactions with decoupled pools, strict category quota allocation,
2-level stratified unseen merchant selection, deterministic multi-level noise, and fixed seed=42.
"""

import os
import uuid
import yaml
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Decoupled Merchant Pools per Category and Sub-type
MERCHANT_POOLS = {
    "food": {
        "routine": [
            "ร้านข้าวแกงป้าพร", "ก๋วยเตี๋ยวเรืออยุธยา รสเด็ด", "ข้าวมันไก่ตอนประตูน้ำ",
            "7-Eleven สาขาอโศก", "ร้านอาหารตามสั่งลุงหนวด", "Amazon Cafe PTT",
            "ชายสี่บะหมี่เกี๊ยว", "ร้านส้มตำเจ๊ไก่ รัชดา", "โจ๊กสามย่าน สาขา 2",
            "ร้านข้าวต้มรอบดึก กวงเฮง", "CJ Express ซอย 5", "ร้านข้าวราดแกงเมืองตรัง"
        ],
        "wants": [
            "Starbucks Reserve", "Sukishi Korean Buffet", "MK Restaurants",
            "After You Dessert Cafe", "Sushi Masa Thonglor", "Haidilao Hot Pot CentralWorld",
            "Momoparadise Central Rama 9", "KFC Drive Thru", "McDonald's EmQuartier",
            "Greyhound Cafe", "Bar B Q Plaza"
        ],
        "impulse_delivery": [
            "GrabFood - Bonchon Chicken", "Lineman - ส้มตำแซ่บพรีเมียม",
            "ShopeeFood - McDonald's 24hrs", "GrabFood - ชานมไข่มุก KOI The",
            "Lineman - ยำกุ้งสดแซ่บลืมผัว", "Foodpanda - Pizza Hut Delivery"
        ]
    },
    "transport": {
        "routine": [
            "BTS Skytrain อโศก", "MRT Bangkok พระราม 9", "ปตท. PTT Station สาขา 1",
            "ปตท. PTT Station สาขา 2", "BCP บางจาก พระโขนง", "Shell Station สุขุมวิท",
            "Caltex เอกมัย", "วินมอเตอร์ไซค์ ซอยสุขุมวิท 21", "วินมอเตอร์ไซค์ หน้า MRT",
            "ทางด่วน กทพ. Easy Pass", "M-Flow ทล.9 กรมทางหลวง", "ที่จอดรถ อาคารเสริมมิตร"
        ],
        "wants": [
            "GrabCar Premium Bangkok", "Bolt Taxi Express", "AirAsia Flight Booking",
            "Nok Air Ticket", "Drive Car Rental Thailand", "Line Man Taxi VIP"
        ]
    },
    "shopping": {
        "routine": [
            "Watsons สาขา เซ็นทรัล", "Boots Pharmacy พระราม 9", "Big C Supercenter รัชดา",
            "Lotus's สุขุมวิท 50", "Tops Supermarket All Seasons", "Gourmet Market Siam Paragon",
            "Daiso Japan 60 บาท"
        ],
        "wants": [
            "Uniqlo Central Rama 9", "ZARA Siam Paragon", "H&M EmQuartier",
            "Eveandboy Mega Bangna", "Sephora Thailand", "IKEA Bangna",
            "Muji Central Chidlom", "B2S Think Space", "Gentlewoman Store"
        ],
        "impulse_online": [
            "TikTok Shop Thailand", "Shopee Flash Sale Mall", "Lazada 9.9 Official Store",
            "Apple Store Online TH", "Pop Mart Official Store", "Central Online Shopping"
        ]
    },
    "bills": {
        "routine": [
            "การไฟฟ้านครหลวง MEA", "การประปานครหลวง MWA", "AIS Fibre Broadband",
            "TrueMove H Postpaid", "3BB Broadband Member", "NT Broadband Thailand",
            "ค่าเช่าคอนโด ลุมพินีทาวเวอร์", "นิติบุคคลอาคารชุด ดิ ไอริส",
            "FWD Insurance Thailand", "AIA Life Insurance Post", "บัตรเครดิต KTC Visa",
            "บัตรเครดิต SCB CardX", "ประกันภัยรถยนต์ วิริยะประกันภัย"
        ]
    },
    "entertainment": {
        "wants": [
            "Netflix Thailand", "Spotify Premium TH", "YouTube Premium Monthly",
            "Disney+ Hotstar TH", "Steam Games Store", "PlayStation Network TH",
            "Nintendo eShop", "Major Cineplex รัชโยธิน", "SF Cinema World CentralWorld",
            "The Beer Bar Sukhumvit 11", "Glow Nightclub Thonglor", "Ticketmelon Concerts",
            "Board Game Cafe Ari", "Karaoke City Praditmanutham"
        ]
    },
    "other": {
        "routine": [
            "ธนาคารกสิกรไทย K-Mobile", "ธนาคารไทยพาณิชย์ SCB Easy", "ธนาคารกรุงไทย Krungthai Next",
            "ATM KBank หน้าคอนโด", "วัดสระเกศราชวรมหาวิหาร", "ค่าธรรมเนียมถอนเงินข้ามเขต",
            "ไปรษณีย์ไทย Kerry Express", "ตู้เติมเงิน บุญเติม"
        ],
        "wants": [
            "โอนเงินบริจาคมูลนิธิสุนัขจรจัด", "โอนร่วมบุญผ้าป่าวัดป่า",
            "PromptPay โอนเงินส่วนตัว", "ทำบุญช่วยเหลือผู้ประสบภัย"
        ]
    }
}

# Decoupled Memo Pools per Category and Sub-type
MEMO_POOLS = {
    "food": {
        "routine": [
            "ข้าวราดแกง 2 อย่างไข่ต้ม", "เส้นเล็กน้ำตกหมูพิเศษ", "ข้าวมันไก่เนื้อน่องไม่หนัง",
            "ข้าวกล่องเซเว่นพร้อมน้ำดื่ม", "กะเพราหมูกรอบไข่ดาวไม่สุก", "Iced Black Coffee หวานน้อย",
            "บะหมี่เกี๊ยวหมูแดงแห้ง", "ส้มตำไทยคอหมูย่าง", "โจ๊กหมูใส่ไข่เยี่ยวม้า",
            "ข้าวต้มปลาอินทรีย์รอบค่ำ", "แซนด์วิชทูน่ามื้อเช้า", "เกาเหลาเลือดหมูข้าวเปล่า",
            "มื้อเที่ยงกับทีม", "อาหารตามสั่งจานด่วน", "กาแฟคั่วเข้มแก้วเช้า"
        ],
        "wants": [
            "Iced Caramel Macchiato Venti", "บุฟเฟต์ปิ้งย่างเกาหลีชุดใหญ่", "สุกี้ชุดครอบครัวพร้อมเป็ดย่าง",
            "Shibuya Honey Toast และชาร้อน", "เซ็ตซูชิพรีเมียมแซลมอน 5 คำ", "ชาบูหม้อไฟหมาล่าพรีเมียม",
            "สุกี้ชาบูเนื้อวากิวไม่อั้น", "ไก่ทอด KFC บักเก็ตพิเศษ", "เบอร์เกอร์เนื้อทรัฟเฟิล",
            "ดินเนอร์ร้านอาหารอิตาเลียน", "ปิ้งย่างบาร์บีคิวฉลองปิดโปรเจกต์"
        ],
        "impulse_delivery": [
            "ไก่ทอดบอนชอนชุดปาร์ตี้ สั่งดึก", "ยำแซลมอนกุ้งสด สั่งตอนหิวรอบดึก",
            "เบอร์เกอร์ชุดใหญ่เฟรนช์ฟรายส์ดึก", "ชานมไข่มุก golden bubble x2",
            "ยำมาม่าหมูยอไข่แดงเค็มรอบดึก", "พิซซ่าถาดกลางหนานุ่มส่งรอบเที่ยงคืน"
        ]
    },
    "transport": {
        "routine": [
            "เติมเงินบัตร BTS รายเดือน", "แตะบัตร MRT สถานีพระราม 9", "เติมน้ำมันมอเตอร์ไซค์เต็มถัง",
            "เติมน้ำมัน E20 รถยนต์ส่วนตัว", "เติมน้ำมัน Gasohol 95", "เติมน้ำมันดีเซลเต็มถัง",
            "ค่าวินไปหน้าปากซอยสุขุมวิท", "ค่าวินมอไซค์ไปต่อรถไฟฟ้า", "Easy Pass เติมเงินทางด่วน",
            "M-Flow ชำระค่าผ่านทางด่วน", "ค่าจอดรถรายชั่วโมงอาคาร", "ค่ารถตู้ไปทำงานอนุสาวรีย์"
        ],
        "wants": [
            "เรียกรถ GrabCar ไปปาร์ตี้", "นั่ง Taxi Express ช่วงฝนตก", "จองตั๋วเครื่องบินไปเที่ยวภูเก็ต",
            "ตั๋วบินเชียงใหม่สุดสัปดาห์", "เช่ารถขับเที่ยวกับเพื่อน", "Grab VIP เดินทางไปสนามบิน"
        ]
    },
    "shopping": {
        "routine": [
            "ซื้อยาสีฟัน แชมพู และสบู่เหลว", "ซื้อยาแก้แพ้และวิตามินซีรวม", "ซื้อของใช้จำเป็นในห้องน้ำ",
            "น้ำยาซักผ้าและปรับผ้านุ่ม", "ของสดวัตถุดิบทำกับข้าว", "ผลไม้นำเข้าและนมจืด",
            "กล่องเก็บของและอุปกรณ์อเนกประสงค์"
        ],
        "wants": [
            "ซื้อเสื้อกล้ามและกางเกงสแล็ค", "เสื้อเบลเซอร์ทำงานคอลใหม่", "เสื้อผ้าแฟชั่นเกาหลีแมตช์ชุด",
            "สกินแคร์เซรั่มบำรุงผิวหน้า", "ลิปสติกและพาเลตต์อายแชโดว์", "โคมไฟมินิมอลแต่งโต๊ะทำงาน",
            "เครื่องเขียนสมุดโน้ตสไตล์ญี่ปุ่น", "หนังสือแปลและนวนิยายขายดี", "กระเป๋าผ้าแบรนด์ยอดฮิต"
        ],
        "impulse_online": [
            "สั่งของเล่น Art Toy กล่องสุ่มไลฟ์ดึก", "Flash Sale 00:00 หูฟังบลูทูธ",
            "เครื่องดูดฝุ่นไร้สายกดตอนเที่ยงคืน", "Apple Watch สายสปอร์ตแฟชั่น",
            "กล่องสุ่มคอลเลกชันใหม่ล่าสุด", "ช้อปปิ้งออนไลน์ช่วงลดราคาเที่ยงคืน"
        ]
    },
    "bills": {
        "routine": [
            "ชำระค่าไฟฟ้าประจำเดือน MEA", "ชำระค่าน้ำประปานครหลวง", "ค่าบริการอินเทอร์เน็ตบ้านไฟเบอร์",
            "ค่าแพ็กเกจมือถือรายเดือน 5G", "ค่าเน็ตบ้านรายเดือน 3BB", "ค่าโทรศัพท์และบรอดแบนด์",
            "โอนค่าเช่าคอนโดประจำเดือน", "ชำระค่าส่วนกลางคอนโดมิเนียม", "เบี้ยประกันชีวิตและอุบัติเหตุ",
            "เบี้ยประกันสุขภาพรายเดือน", "ชำระยอดบัตรเครดิตยอดเต็ม", "ชำระบิลบัตรเครดิตรายเดือน",
            "ต่อประกันภัยรถยนต์ชั้น 1"
        ]
    },
    "entertainment": {
        "wants": [
            "Netflix Premium รายเดือน 4K", "Spotify Individual Subscription", "YouTube Premium Family",
            "Disney+ Hotstar รายปี", "ซื้อเกมลดราคาบน Steam Store", "PlayStation Plus Member",
            "เกมดิจิทัล Nintendo Switch", "ตั๋วหนัง IMAX 3D พร้อมป๊อปคอร์น", "ตั๋วชมภาพยนตร์รอบค่ำ",
            "ดื่มคราฟต์เบียร์สังสรรค์คืนวันศุกร์", "ปาร์ตี้วันเกิดกับเพื่อนร่วมงาน", "ตั๋วคอนเสิร์ตอินดี้มิวสิก",
            "เล่นบอร์ดเกมกับกลุ่มเพื่อน", "ร้องคาราโอเกะปาร์ตี้วันหยุด"
        ]
    },
    "other": {
        "routine": [
            "โอนเงินคืนเพื่อนค่าอาหารกลางวัน", "โอนเงินคืนค่าแท็กซี่หารกัน", "โอนเงินส่วนตัวเข้าบัญชีออมทรัพย์",
            "ถอนเงินสดใช้จ่ายเบ็ดเตล็ด", "ทำบุญค่าน้ำค่าไฟวัดประจำเดือน", "ค่าธรรมเนียมธุรกรรมธนาคาร",
            "ส่งพัสดุเอกสารด่วน", "เติมเงินมือถือระบบเติมเงิน"
        ],
        "wants": [
            "ร่วมบริจาคอาหารสุนัขแมวจรจัด", "ร่วมบุญสร้างโรงพยาบาลสงฆ์",
            "โอนเงินซื้อของขวัญวันเกิดเพื่อน", "บริจาคช่วยมูลนิธิการกุศล"
        ]
    }
}

TYPO_DICT = {
    "bts": "บีทีเอส",
    "mrt": "เอ็มอาร์ที",
    "shopee": "ช้อปปี้",
    "lazada": "ลาซาด้า",
    "netflix": "เน็ตฟลิก",
    "starbucks": "ตาบัค",
    "cafe": "คาเฟ่",
    "coffee": "คอฟฟี่",
    "ข้าว": "คาว",
    "ไฟ": "ฟัย",
    "อาหาร": "อาหาน",
    "สุกี้": "สุกี้้้",
    "grab": "แกร็บ",
    "lineman": "ไลน์แมน",
}

AMBIGUOUS_MEMOS = [
    "โอน", "จ่ายเงิน", "พร้อมเพย์", "xxx", "คืนเพื่อน", "จ่ายผ่านแอป",
    "ยอดตัดบัญชี", "ชำระเงิน", "โอนเงิน", "เคลียร์ยอด", "-", "ไม่ระบุ"
]

def select_stratified_unseen_merchants(rng: random.Random, unseen_pct: float) -> tuple[set, dict]:
    """
    Selects 20% of merchants per (category, sub_type) group as unseen_merchants.
    Returns:
      - unseen_merchants_set: Set of merchant names
      - unseen_breakdown: Dictionary mapping (category, sub_type) -> list of unseen merchants
    """
    unseen_merchants_set = set()
    unseen_breakdown = {}

    for cat, sub_dict in MERCHANT_POOLS.items():
        for sub_type, m_list in sub_dict.items():
            key = f"{cat}::{sub_type}"
            if len(m_list) >= 2:
                k = max(1, int(math.ceil(len(m_list) * unseen_pct)))
                # Deterministic sample using provided rng
                chosen = rng.sample(m_list, k)
                unseen_breakdown[key] = chosen
                for m in chosen:
                    unseen_merchants_set.add(m)
            else:
                unseen_breakdown[key] = []
                print(f"[Notice] Sub-type '{key}' has fewer than 2 merchants. Skipping unseen selection for this pool.")

    return unseen_merchants_set, unseen_breakdown

def apply_noise(memo: str, rng: random.Random, ambiguous_pct: float, typo_pct: float) -> str:
    """Deterministically applies ambiguous memo substitution and typo variations."""
    r_val = rng.random()
    if r_val < ambiguous_pct:
        return rng.choice(AMBIGUOUS_MEMOS)
    elif r_val < (ambiguous_pct + typo_pct):
        # Apply typo / slang variation
        words = memo.split(" ")
        modified_words = []
        for w in words:
            w_lower = w.lower()
            if w_lower in TYPO_DICT and rng.random() < 0.7:
                modified_words.append(TYPO_DICT[w_lower])
            else:
                modified_words.append(w)
        return " ".join(modified_words)
    return memo

def generate_transactions(config: dict) -> pd.DataFrame:
    seed = config["data_generation"]["random_seed"]
    rng_py = random.Random(seed)
    rng_np = np.random.RandomState(seed)

    start_date_str = config["data_generation"].get("start_date", "2025-01-01")
    months = config["data_generation"].get("months", 12)
    avg_daily = config["data_generation"].get("avg_daily_transactions", 4.5)
    cat_dist = config["data_generation"]["category_distribution"]
    tolerance = config["data_generation"].get("category_distribution_tolerance", 0.01)
    
    noise_cfg = config["data_generation"]["noise"]
    ambiguous_memo_pct = noise_cfg.get("ambiguous_memo_pct", 0.15)
    typo_pct = noise_cfg.get("typo_pct", 0.10)
    unseen_merchant_pct = noise_cfg.get("unseen_merchant_pct", 0.20)

    # 1. 2-Level Stratified Unseen Merchant Selection
    unseen_merchants_set, unseen_breakdown = select_stratified_unseen_merchants(rng_py, unseen_merchant_pct)

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    total_days = int(months * 30.5) # ~366 days
    end_date = start_date + timedelta(days=total_days)

    total_target_transactions = int(round(total_days * avg_daily)) # ~1,647 transactions
    
    # 2. Strict Quota Allocation per Category
    categories = list(cat_dist.keys())
    cat_target_counts = {c: int(round(total_target_transactions * cat_dist[c])) for c in categories}
    diff = total_target_transactions - sum(cat_target_counts.values())
    cat_target_counts["food"] += diff

    # Create category queue to strictly meet quota
    cat_quota_remaining = cat_target_counts.copy()

    records = []
    
    # Pre-generate dates and distribution
    current_date = start_date
    day_list = []
    while current_date < end_date:
        day_list.append(current_date)
        current_date += timedelta(days=1)

    for current_date in day_list:
        day_of_month = current_date.day
        is_payday_window = (day_of_month in [25, 26, 27])
        is_weekend = (current_date.weekday() >= 5)

        # Determine number of transactions for this day
        # Ensure we distribute remaining quota across remaining days
        remaining_days = max(1, (end_date - current_date).days)
        total_remaining_tx = sum(cat_quota_remaining.values())
        daily_target = max(1, int(round(total_remaining_tx / remaining_days)))
        
        # Add slight variance
        daily_n = max(1, int(rng_np.normal(daily_target, 1.0)))
        daily_n = min(daily_n, total_remaining_tx)

        # 1. Recurring Bills on fixed days if bills quota available
        if day_of_month in [1, 2, 25, 28] and cat_quota_remaining["bills"] > 0:
            if rng_py.random() < 0.75:
                cat = "bills"
                sub_type = "routine"
                merchant = rng_py.choice(MERCHANT_POOLS[cat][sub_type])
                raw_memo = rng_py.choice(MEMO_POOLS[cat][sub_type])
                memo = apply_noise(raw_memo, rng_py, ambiguous_memo_pct, typo_pct)
                amt = round(float(rng_py.uniform(350, 4500)), 2)
                
                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "time": f"{rng_py.randint(9, 18):02d}:{rng_py.randint(0, 59):02d}",
                    "merchant": merchant,
                    "memo": memo,
                    "amount": amt,
                    "category": cat,
                    "is_wants": False,
                    "is_impulse": False,
                    "sub_type": sub_type,
                    "is_unseen_merchant": (merchant in unseen_merchants_set)
                })
                cat_quota_remaining["bills"] -= 1

        # 2. Allocate daily transactions from remaining categories
        available_cats = [c for c, count in cat_quota_remaining.items() if count > 0]
        if not available_cats:
            continue

        for _ in range(daily_n):
            available_cats = [c for c, count in cat_quota_remaining.items() if count > 0]
            if not available_cats:
                break
            
            # Probability proportional to remaining counts
            weights = [cat_quota_remaining[c] for c in available_cats]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            chosen_cat = rng_np.choice(available_cats, p=probs)
            
            # Sub-type selection
            sub_dict = MERCHANT_POOLS[chosen_cat]
            sub_types = list(sub_dict.keys())
            
            # Determine time & impulse
            late_prob = 0.25 if (is_payday_window or is_weekend) else 0.08
            is_late_night = (rng_py.random() < late_prob)
            
            if is_late_night:
                late_hour = rng_py.choice([23, 0, 1, 2])
                tx_time = f"{late_hour:02d}:{rng_py.randint(0, 59):02d}"
            else:
                normal_hour = rng_py.randint(7, 22)
                tx_time = f"{normal_hour:02d}:{rng_py.randint(0, 59):02d}"

            is_impulse = False
            is_wants = False
            
            if is_late_night and "impulse_delivery" in sub_types and rng_py.random() < 0.65:
                chosen_sub = "impulse_delivery"
                is_impulse = True
                is_wants = True
            elif is_late_night and "impulse_online" in sub_types and rng_py.random() < 0.70:
                chosen_sub = "impulse_online"
                is_impulse = True
                is_wants = True
            elif is_payday_window and "wants" in sub_types and rng_py.random() < 0.50:
                chosen_sub = "wants"
                is_impulse = True
                is_wants = True
            elif "wants" in sub_types and rng_py.random() < 0.40:
                chosen_sub = "wants"
                is_impulse = False
                is_wants = True
            else:
                chosen_sub = "routine" if "routine" in sub_types else sub_types[0]
                is_impulse = False
                is_wants = (chosen_sub == "wants")

            merchant = rng_py.choice(MERCHANT_POOLS[chosen_cat][chosen_sub])
            raw_memo = rng_py.choice(MEMO_POOLS[chosen_cat][chosen_sub])
            memo = apply_noise(raw_memo, rng_py, ambiguous_memo_pct, typo_pct)
            
            # Amount simulation
            if chosen_cat == "food":
                base_amt = rng_py.uniform(50, 250) if chosen_sub == "routine" else rng_py.uniform(250, 1200)
            elif chosen_cat == "transport":
                base_amt = rng_py.uniform(25, 120) if chosen_sub == "routine" else rng_py.uniform(200, 2500)
            elif chosen_cat == "shopping":
                base_amt = rng_py.uniform(100, 500) if chosen_sub == "routine" else rng_py.uniform(400, 3000)
            elif chosen_cat == "bills":
                base_amt = rng_py.uniform(300, 3500)
            elif chosen_cat == "entertainment":
                base_amt = rng_py.uniform(150, 1800)
            else: # other
                base_amt = rng_py.uniform(40, 500)

            if is_impulse and rng_py.random() < 0.25:
                base_amt *= rng_py.uniform(1.2, 1.6)

            amt = round(float(base_amt), 2)
            
            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "time": tx_time,
                "merchant": merchant,
                "memo": memo,
                "amount": amt,
                "category": chosen_cat,
                "is_wants": is_wants,
                "is_impulse": is_impulse,
                "sub_type": chosen_sub,
                "is_unseen_merchant": (merchant in unseen_merchants_set)
            })
            cat_quota_remaining[chosen_cat] -= 1

    df = pd.DataFrame(records)
    
    # Sort chronologically
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df.drop(columns=["datetime"], inplace=True)

    # Insert UUID transaction_id
    df.insert(0, "transaction_id", [str(uuid.uuid4()) for _ in range(len(df))])

    # Category Distribution Check
    actual_dist = df["category"].value_counts(normalize=True).to_dict()
    for cat, target_pct in cat_dist.items():
        actual_pct = actual_dist.get(cat, 0.0)
        gap = abs(actual_pct - target_pct)
        assert gap <= tolerance, (
            f"Category '{cat}' distribution drift exceeded tolerance! Target: {target_pct:.2%}, Actual: {actual_pct:.2%}, Gap: {gap:.4f}"
        )

    return df

def main():
    config = load_config()
    print("Generating synthetic transaction dataset (v5 protocol with fixed seed=42)...")
    df = generate_transactions(config)

    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "transactions.csv")
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Successfully generated {len(df)} transactions.")
    print(f"Saved to: {output_file}")
    
    print("\n1. Category Breakdown vs Target:")
    cat_counts = df["category"].value_counts()
    for cat, count in cat_counts.items():
        pct = count / len(df)
        target = config["data_generation"]["category_distribution"][cat]
        print(f"  - {cat:15s}: {pct:.2%} ({count} tx) [Target: {target:.2%}]")
        
    print("\n2. Unseen Merchant Flag Breakdown:")
    unseen_counts = df["is_unseen_merchant"].value_counts()
    for u, count in unseen_counts.items():
        pct = count / len(df)
        print(f"  - is_unseen_merchant={u}: {pct:.2%} ({count} tx)")

    print("\n3. Sub-type Breakdown:")
    sub_counts = df["sub_type"].value_counts()
    for st, count in sub_counts.items():
        pct = count / len(df)
        print(f"  - {st:20s}: {pct:.2%} ({count} tx)")

if __name__ == "__main__":
    main()
