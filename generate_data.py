"""
Generate a realistic synthetic sales dataset for an Indian D2C e-commerce brand.

The data intentionally contains real-world quality issues (duplicates, missing
values, inconsistent text, invalid quantities, outliers) so the cleaning step
has something meaningful to do.

Usage:
    python src/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ORDERS = 12_000
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "sales_raw.csv"

rng = np.random.default_rng(SEED)

# --- Reference data ---------------------------------------------------------
REGIONS = {
    "North": ["Delhi", "Uttar Pradesh", "Punjab", "Haryana", "Rajasthan"],
    "West": ["Maharashtra", "Gujarat", "Goa", "Madhya Pradesh"],
    "South": ["Karnataka", "Tamil Nadu", "Kerala", "Telangana", "Andhra Pradesh"],
    "East": ["West Bengal", "Odisha", "Bihar", "Assam"],
}
REGION_WEIGHTS = {"North": 0.30, "West": 0.32, "South": 0.26, "East": 0.12}

PRODUCTS = {
    # category: [(product, base_price_inr), ...]
    "Men's Socks": [("Classic Crew Socks", 349), ("Merino Wool Dress Socks", 799),
                    ("Ankle Sports Socks (3-Pack)", 499), ("Bamboo Formal Socks", 599)],
    "Women's Socks": [("No-Show Liner Socks", 299), ("Printed Crew Socks", 399),
                      ("Cashmere Blend Socks", 999), ("Compression Socks", 699)],
    "Kids' Socks": [("Cartoon Ankle Socks (5-Pack)", 449), ("School Uniform Socks", 249),
                    ("Anti-Slip Toddler Socks", 299)],
    "Gift Sets": [("Luxury Gift Box (4 Pairs)", 1999), ("Festive Combo Pack", 1499)],
    "Accessories": [("Sock Organizer", 399), ("Travel Laundry Pouch", 249)],
}
CATEGORY_WEIGHTS = [0.36, 0.30, 0.14, 0.12, 0.08]

CHANNELS = ["Website", "Mobile App", "Marketplace", "Instagram Shop"]
CHANNEL_WEIGHTS = [0.38, 0.27, 0.25, 0.10]
PAYMENTS = ["UPI", "Credit Card", "Debit Card", "COD", "Net Banking", "Wallet"]
PAYMENT_WEIGHTS = [0.45, 0.17, 0.12, 0.16, 0.05, 0.05]


def seasonal_dates(n: int) -> pd.Series:
    """Order dates across 2024-2025 with festive-season and sale peaks."""
    days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    weights = np.ones(len(days))
    month = days.month
    weights[(month == 10) | (month == 11)] *= 2.2    # Diwali / festive season
    weights[month == 12] *= 1.5                        # year-end & winter
    weights[month == 1] *= 1.3                         # winter / Republic Day sale
    weights[(month >= 4) & (month <= 6)] *= 0.8        # summer slowdown
    weights[days.dayofweek >= 5] *= 1.25               # weekends
    weights *= np.where(days.year == 2025, 1.28, 1.0)  # year-on-year growth
    weights /= weights.sum()
    return pd.Series(rng.choice(days, size=n, p=weights))


def main() -> None:
    n = N_ORDERS
    n_customers = 4_200
    # A minority of customers place many orders (realistic repeat behaviour)
    cust_weights = rng.pareto(1.6, n_customers) + 1
    cust_weights /= cust_weights.sum()
    customer_ids = rng.choice([f"CUST{i:05d}" for i in range(1, n_customers + 1)],
                              size=n, p=cust_weights)

    regions = rng.choice(list(REGION_WEIGHTS), size=n, p=list(REGION_WEIGHTS.values()))
    states = [rng.choice(REGIONS[r]) for r in regions]

    categories = rng.choice(list(PRODUCTS), size=n, p=CATEGORY_WEIGHTS)
    products, prices = [], []
    for c in categories:
        name, base = PRODUCTS[c][rng.integers(len(PRODUCTS[c]))]
        products.append(name)
        prices.append(base)

    dates = seasonal_dates(n)
    festive = dates.dt.month.isin([10, 11]).to_numpy()
    quantity = rng.choice([1, 2, 3, 4, 5, 6], size=n, p=[0.48, 0.26, 0.12, 0.07, 0.04, 0.03])
    discount = np.where(festive,
                        rng.choice([0.10, 0.15, 0.20, 0.25, 0.30], size=n),
                        rng.choice([0.0, 0.0, 0.05, 0.10, 0.15], size=n))

    df = pd.DataFrame({
        "order_id": [f"ORD{100000 + i}" for i in range(n)],
        "order_date": dates.dt.strftime("%Y-%m-%d"),
        "customer_id": customer_ids,
        "region": regions,
        "state": states,
        "channel": rng.choice(CHANNELS, size=n, p=CHANNEL_WEIGHTS),
        "payment_method": rng.choice(PAYMENTS, size=n, p=PAYMENT_WEIGHTS),
        "category": categories,
        "product_name": products,
        "unit_price": prices,
        "quantity": quantity,
        "discount": discount,
    })
    df["sales"] = (df["unit_price"] * df["quantity"] * (1 - df["discount"])).round(2)
    cogs_ratio = rng.uniform(0.42, 0.58, size=n)
    df["profit"] = (df["sales"] - df["unit_price"] * df["quantity"] * cogs_ratio).round(2)
    df["order_status"] = rng.choice(["Delivered", "Cancelled", "Returned"],
                                    size=n, p=[0.91, 0.06, 0.03])

    # --- Inject realistic data-quality issues --------------------------------
    df = df.astype({"region": "object", "payment_method": "object", "channel": "object"})
    idx = rng.choice(n, 180, replace=False)
    df.loc[idx[:90], "region"] = np.nan                       # missing region
    df.loc[idx[90:], "payment_method"] = np.nan               # missing payment method
    idx = rng.choice(n, 400, replace=False)
    df.loc[idx, "channel"] = df.loc[idx, "channel"].str.lower() + " "  # messy text
    idx = rng.choice(n, 25, replace=False)
    df.loc[idx, "quantity"] = 0                                # invalid quantity
    idx = rng.choice(n, 12, replace=False)
    df.loc[idx, "sales"] = df.loc[idx, "sales"] * 40           # data-entry outliers
    dupes = df.sample(150, random_state=SEED)                  # duplicate rows
    df = pd.concat([df, dupes]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
