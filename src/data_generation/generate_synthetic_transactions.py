"""
SmartSpend AI - Synthetic Transaction Generator
Generates realistic personal financial transactions with embedded behavioral patterns.
"""

import os
import uuid
import random
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Merchant and Memo Templates per Category
MERCHANT_MEMO_POOLS = {
    "food": {
        "routine": [
            ("ร้านข้าวแกงป้าพร", "ข้าวราดแกง 2 อย่าง", 50, 80, False),
            ("ก๋วยเตี๋ยวเรืออยุธยา", "เส้นเล็กน้ำตกหมูพิเศษ", 50, 70, False),
            ("ข้าวมันไก่ตอนประตูน้ำ", "ข้าวมันไก่เนื้อน่อง", 60, 80, False),
            ("7-Eleven", "ข้าวกล่องเซเว่น + น้ำเปล่า", 45, 95, False),
            ("ร้านอาหารตามสั่งลุงหนวด", "กะเพราหมูกรอบไข่ดาว", 65, 85, False),
            ("Amazon Cafe", "Iced Black Coffee", 60, 75, False),
            ("ชายสี่บะหมี่เกี๊ยว", "บะหมี่เกี๊ยวหมูแดง", 55, 70, False),
        ],
        "wants": [
            ("Starbucks", "Iced Caramel Macchiato Venti", 165, 220, True),
            ("Sukishi Buffet", "บุฟเฟต์ปิ้งย่างเกาหลีชุดใหญ่", 599, 899, True),
            ("MK Restaurants", "สุกี้ชุดครอบครัว + เป็ดย่าง", 650, 1200, True),
            ("After You Dessert Cafe", "Shibuya Honey Toast", 245, 380, True),
            ("Sushi Masa", "เซ็ตซูชิพรีเมียมแซลมอน", 450, 950, True),
            ("Haidilao Hot Pot", "ชาบูหม้อไฟหมาล่า", 800, 1600, True),
        ],
        "impulse_delivery": [
            ("GrabFood - Bonchon", "ไก่ทอดบอนชอนชุดปาร์ตี้ สั่งดึก", 380, 750, True),
            ("Lineman - ส้มตำยำแซ่บ", "ยำแซลมอนกุ้งสด สั่งตอนหิวรอบดึก", 280, 520, True),
            ("ShopeeFood - McDonald's", "เบอร์เกอร์ชุดใหญ่เฟรนช์ฟรายส์", 240, 460, True),
            ("GrabFood - ชานมไข่มุก KOI Thé", "ชานมไข่มุก golden bubble x2 แก้ว", 180, 320, True),
        ]
    },
    "transport": {
        "routine": [
            ("BTS Skytrain", "เติมเงินบัตร BTS เที่ยวเดินทาง", 300, 500, False),
            ("MRT Bangkok", "แตะบัตร MRT ไปทำงาน", 35, 45, False),
            ("ปตท. PTT Station", "เติมน้ำมันรถมอเตอร์ไซค์", 120, 180, False),
            ("ปตท. PTT Station", "เติมน้ำมัน E20 รถเก๋ง", 800, 1200, False),
            ("วินมอเตอร์ไซค์ ซอย 23", "ค่าวินไปหน้าปากซอย", 20, 40, False),
            ("ทางด่วน กทพ.", "Easy Pass เติมเงินทางด่วน", 500, 1000, False),
        ],
        "wants": [
            ("GrabCar Premium", "เรียกรถ Grab ไปปาร์ตี้", 250, 450, True),
            ("Bolt Taxi", "นั่ง Taxi กลับบ้านช่วงเร่งด่วน", 180, 320, True),
            ("AirAsia", "ตั๋วเครื่องบินไปเที่ยวเชียงใหม่", 1800, 3500, True),
        ]
    },
    "shopping": {
        "routine": [
            ("Watsons", "ซื้อยาสีฟัน แชมพู สบู่", 180, 350, False),
            ("Boots", "ซื้อยาแก้แพ้และวิตามินซี", 150, 300, False),
            ("Uniqlo", "ซื้อเสื้อกล้ามและถุงเท้าทำงาน", 290, 590, False),
            ("Big C Supercenter", "ของใช้จำเป็นในห้องน้ำ", 350, 700, False),
        ],
        "wants": [
            ("Shopee Mall", "เคสมือถือ + สติกเกอร์ตกแต่ง", 150, 350, True),
            ("Lazada Official", "เสื้อผ้าแฟชั่นเกาหลีชุดใหม่", 490, 1200, True),
            ("Uniqlo", "เสื้อกันหนาวคอลเลกชันใหม่", 990, 1990, True),
            ("Eveandboy", "สกินแคร์และเครื่องสำอางแบรนด์ดัง", 850, 2200, True),
            ("ZARA", "กางเกงสแล็คและเบลเซอร์", 1490, 2990, True),
            ("IKEA Bangna", "โคมไฟแต่งห้องและของตกแต่ง", 450, 1500, True),
        ],
        "impulse_online": [
            ("TikTok Shop", "สั่งของเล่น Art Toy กล่องสุ่มตอนไลฟ์ดึก", 390, 1200, True),
            ("Shopee Flash Sale", "Flash Sale 00:00 หูฟังบลูทูธไร้สาย", 690, 1590, True),
            ("Lazada 9.9 / Mid-Month", "เครื่องดูดฝุ่นไร้สายกดตอนเที่ยงคืน", 1290, 2990, True),
            ("Apple Store Online", "Apple Watch สายแฟชั่นใหม่", 1800, 3500, True),
        ]
    },
    "bills": {
        "routine": [
            ("การไฟฟ้านครหลวง MEA", "ชำระค่าไฟฟ้าประจำเดือน", 800, 1600, False),
            ("การประปานครหลวง MWA", "ชำระค่าน้ำประปา", 120, 250, False),
            ("AIS Fibre / 3BB", "ค่าบริการอินเทอร์เน็ตบ้าน", 599, 799, False),
            ("TrueMove H Postpaid", "ค่าแพ็กเกจมือถือรายเดือน", 499, 699, False),
            ("ค่าเช่าคอนโด / หอพัก", "โอนค่าเช่าห้องประจำเดือน", 6500, 9500, False),
            ("FWD Insurance", "เบี้ยประกันชีวิตและสุขภาพรายเดือน", 1500, 2500, False),
            ("นิติบุคคลอาคารชุด", "ชำระค่าส่วนกลางคอนโด", 1200, 1800, False),
        ],
        "wants": []
    },
    "entertainment": {
        "routine": [],
        "wants": [
            ("Netflix Thailand", "Netflix Premium รายเดือน", 419, 419, True),
            ("Spotify Premium", "Spotify Individual Subscription", 139, 139, True),
            ("YouTube Premium", "YouTube Premium รายเดือน", 179, 179, True),
            ("Major Cineplex", "ตั๋วหนัง IMAX + ป๊อปคอร์นเซ็ต", 350, 650, True),
            ("SF Cinema", "ตั๋วชมภาพยนตร์รอบค่ำ", 240, 400, True),
            ("Steam Games", "ซื้อเกมลดราคาบน Steam Store", 299, 1299, True),
            ("The Beer Bar Sukhumvit", "ดื่มคราฟต์เบียร์สังสรรค์กับเพื่อน", 600, 1400, True),
            ("Glow Nightclub", "ปาร์ตี้ศุกร์หรรษากับเพื่อนร่วมงาน", 900, 2200, True),
            ("Ticketmelon", "ตั๋วคอนเสิร์ตอินดี้เฟสติวัล", 1200, 2500, True),
        ]
    },
    "other": {
        "routine": [
            ("ธนาคารกสิกรไทย", "โอนเงินคืนเพื่อนค่าอาหารกองกลาง", 100, 300, False),
            ("วัดสระเกศราชวรมหาวิหาร", "ทำบุญค่าน้ำค่าไฟวัด", 50, 100, False),
            ("ATM KBank", "ถอนเงินสดใช้จ่ายเบ็ดเตล็ด", 500, 1000, False),
            ("ธนาคารกรุงไทย", "ชำระค่าธรรมเนียมราชการ", 40, 100, False),
        ],
        "wants": [
            ("โอนเงินบริจาคมูลนิธิ", "ร่วมบุญช่วยเหลือน้องหมาน้องแมว", 100, 300, True),
        ]
    }
}

def generate_transactions(config: dict) -> pd.DataFrame:
    seed = config["data_generation"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)

    start_date_str = config["data_generation"].get("start_date", "2025-01-01")
    months = config["data_generation"].get("months", 12)
    avg_daily = config["data_generation"].get("avg_daily_transactions", 4.5)
    cat_dist = config["data_generation"]["category_distribution"]

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    # Approximate 12 months = 365 days
    total_days = int(months * 30.5)
    end_date = start_date + timedelta(days=total_days)

    total_target_transactions = int(total_days * avg_daily)
    
    # Pre-calculate counts per category strictly according to distribution
    categories = list(cat_dist.keys())
    dist_weights = [cat_dist[c] for c in categories]
    cat_counts = {c: int(round(total_target_transactions * cat_dist[c])) for c in categories}
    
    # Adjust count difference due to rounding
    diff = total_target_transactions - sum(cat_counts.values())
    cat_counts["food"] += diff

    records = []

    # Iterate day by day
    current_date = start_date
    day_index = 0

    while current_date < end_date:
        day_of_month = current_date.day
        is_payday_window = (day_of_month in [25, 26, 27])
        is_weekend = (current_date.weekday() >= 5)

        # 1. Monthly recurring bills on fixed days
        if day_of_month in [1, 2, 25, 28]:
            if random.random() < 0.7:
                bill_item = random.choice(MERCHANT_MEMO_POOLS["bills"]["routine"])
                amt = round(float(random.uniform(bill_item[2], bill_item[3])), 2)
                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "time": f"{random.randint(9, 18):02d}:{random.randint(0, 59):02d}",
                    "merchant": bill_item[0],
                    "memo": bill_item[1],
                    "amount": amt,
                    "category": "bills",
                    "is_wants": bill_item[4],
                    "is_impulse": False,
                })

        # 2. Daily routine & discretionary transactions
        daily_n = max(1, int(np.random.normal(avg_daily, 1.2)))
        for _ in range(daily_n):
            # Choose category based on distribution
            chosen_cat = np.random.choice(categories, p=dist_weights)
            
            # Determine time
            # Late-night injection logic: 23:00 - 02:00
            # Higher chance of late night on payday window or weekends
            late_prob = 0.22 if (is_payday_window or is_weekend) else 0.08
            is_late_night = (random.random() < late_prob)
            
            if is_late_night:
                late_hour = random.choice([23, 0, 1, 2])
                tx_time = f"{late_hour:02d}:{random.randint(0, 59):02d}"
            else:
                normal_hour = random.randint(7, 22)
                tx_time = f"{normal_hour:02d}:{random.randint(0, 59):02d}"

            # Impulse decision:
            # An expense is impulse if:
            # - Category in [food, shopping, entertainment] AND
            # - (Late night OR Payday window with wants item)
            pool = MERCHANT_MEMO_POOLS[chosen_cat]
            is_impulse = False
            
            if is_late_night and "impulse_delivery" in pool and random.random() < 0.65:
                item = random.choice(pool["impulse_delivery"])
                is_impulse = True
            elif is_late_night and "impulse_online" in pool and random.random() < 0.70:
                item = random.choice(pool["impulse_online"])
                is_impulse = True
            elif is_payday_window and pool.get("wants") and random.random() < 0.55:
                item = random.choice(pool["wants"])
                is_impulse = True
            elif pool.get("wants") and random.random() < 0.35:
                item = random.choice(pool["wants"])
                is_impulse = False
            else:
                routine_list = pool.get("routine") or pool.get("wants")
                item = random.choice(routine_list)
                is_impulse = False

            # Add variance to amount
            base_amt = random.uniform(item[2], item[3])
            # If impulse, occasional surge
            if is_impulse and random.random() < 0.2:
                base_amt *= random.uniform(1.2, 1.5)

            amt = round(float(base_amt), 2)
            
            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "time": tx_time,
                "merchant": item[0],
                "memo": item[1],
                "amount": amt,
                "category": chosen_cat,
                "is_wants": bool(item[4]),
                "is_impulse": bool(is_impulse),
            })

        current_date += timedelta(days=1)
        day_index += 1

    df = pd.DataFrame(records)
    
    # Sort chronologically
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df.drop(columns=["datetime"], inplace=True)

    # Insert unique UUID transaction_id as first column
    df.insert(0, "transaction_id", [str(uuid.uuid4()) for _ in range(len(df))])

    return df

def main():
    config = load_config()
    print("Generating synthetic transaction dataset...")
    df = generate_transactions(config)

    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "transactions.csv")
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Successfully generated {len(df)} transactions.")
    print(f"Saved to: {output_file}")
    print("\nCategory Breakdown:")
    cat_counts = df["category"].value_counts(normalize=True) * 100
    for cat, pct in cat_counts.items():
        print(f"  - {cat:15s}: {pct:.2f}% ({df['category'].value_counts()[cat]} transactions)")
    
    print("\nImpulse Flag Breakdown:")
    imp_counts = df["is_impulse"].value_counts(normalize=True) * 100
    for imp, pct in imp_counts.items():
        print(f"  - is_impulse={imp}: {pct:.2f}% ({df['is_impulse'].value_counts()[imp]} transactions)")

if __name__ == "__main__":
    main()
