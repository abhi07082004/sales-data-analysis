"""
Clean and preprocess the raw sales data.

Steps
-----
1. Remove exact duplicate rows
2. Standardise text columns (trim whitespace, consistent casing)
3. Fill missing region from state; fill missing payment method as "Unknown"
4. Drop rows with invalid quantity (<= 0)
5. Recalculate sales from price x quantity x (1 - discount) to fix entry errors
6. Add date features (year, month, quarter, weekday) and margin
7. Keep only completed (Delivered) orders for revenue analysis

Usage:
    python src/clean_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sales_raw.csv"
OUT = ROOT / "data" / "processed" / "sales_clean.csv"

STATE_TO_REGION = {
    "Delhi": "North", "Uttar Pradesh": "North", "Punjab": "North", "Haryana": "North",
    "Rajasthan": "North", "Maharashtra": "West", "Gujarat": "West", "Goa": "West",
    "Madhya Pradesh": "West", "Karnataka": "South", "Tamil Nadu": "South",
    "Kerala": "South", "Telangana": "South", "Andhra Pradesh": "South",
    "West Bengal": "East", "Odisha": "East", "Bihar": "East", "Assam": "East",
}


def clean(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    log = {"raw_rows": len(df)}

    # 1. Duplicates
    df = df.drop_duplicates().copy()
    log["duplicates_removed"] = log["raw_rows"] - len(df)

    # 2. Text standardisation
    for col in ["channel", "payment_method", "region", "state", "category", "product_name"]:
        df[col] = df[col].astype("string").str.strip()
    df["channel"] = df["channel"].str.title()

    # 3. Missing values
    log["missing_region_filled"] = int(df["region"].isna().sum())
    df["region"] = df["region"].fillna(df["state"].map(STATE_TO_REGION))
    log["missing_payment_filled"] = int(df["payment_method"].isna().sum())
    df["payment_method"] = df["payment_method"].fillna("Unknown")

    # 4. Invalid quantities
    invalid_qty = df["quantity"] <= 0
    log["invalid_quantity_removed"] = int(invalid_qty.sum())
    df = df[~invalid_qty]

    # 5. Fix sales outliers caused by entry errors
    expected = (df["unit_price"] * df["quantity"] * (1 - df["discount"])).round(2)
    bad_sales = ~np.isclose(df["sales"], expected, atol=1)
    log["sales_values_corrected"] = int(bad_sales.sum())
    df.loc[bad_sales, "profit"] = df.loc[bad_sales, "profit"] - (
        df.loc[bad_sales, "sales"] - expected[bad_sales])
    df["sales"] = expected

    # 6. Feature engineering
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.to_period("M").astype(str)
    df["quarter"] = df["order_date"].dt.to_period("Q").astype(str)
    df["weekday"] = df["order_date"].dt.day_name()
    df["profit_margin"] = (df["profit"] / df["sales"]).round(4)

    # 7. Completed orders only
    log["cancelled_or_returned"] = int((df["order_status"] != "Delivered").sum())
    df = df[df["order_status"] == "Delivered"].reset_index(drop=True)
    log["clean_rows"] = len(df)

    if verbose:
        print("Cleaning summary")
        for k, v in log.items():
            print(f"  {k:<28}{v:>8,}")
    return df, log


def main() -> None:
    df, _ = clean(pd.read_csv(RAW))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved clean data to {OUT}")


if __name__ == "__main__":
    main()
