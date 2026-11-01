"""
Exploratory sales analysis: KPIs, trends, and dashboard-style charts.

Outputs
-------
- reports/figures/*.png      individual charts
- reports/sales_dashboard.png one-page dashboard
- reports/kpi_summary.csv    headline KPIs

Usage:
    python src/analyze_sales.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "sales_clean.csv"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"

sns.set_theme(style="whitegrid", palette="crest")
INR = mtick.FuncFormatter(lambda x, _: f"₹{x/1e5:.1f}L")  # values in lakhs


# --- KPI helpers --------------------------------------------------------------
def compute_kpis(df: pd.DataFrame) -> dict:
    by_year = df.groupby("year")["sales"].sum()
    yoy = (by_year.iloc[-1] / by_year.iloc[0] - 1) if len(by_year) > 1 else float("nan")
    orders_per_customer = df.groupby("customer_id")["order_id"].nunique()
    return {
        "total_revenue": round(df["sales"].sum(), 2),
        "total_profit": round(df["profit"].sum(), 2),
        "profit_margin": round(df["profit"].sum() / df["sales"].sum(), 4),
        "orders": df["order_id"].nunique(),
        "customers": df["customer_id"].nunique(),
        "avg_order_value": round(df["sales"].sum() / df["order_id"].nunique(), 2),
        "yoy_revenue_growth": round(yoy, 4),
        "repeat_customer_rate": round((orders_per_customer > 1).mean(), 4),
        "festive_revenue_share": round(
            df.loc[pd.to_datetime(df["order_date"]).dt.month.isin([10, 11]), "sales"].sum()
            / df["sales"].sum(), 4),
    }


def revenue_share(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.groupby(col).agg(revenue=("sales", "sum"), profit=("profit", "sum"),
                              orders=("order_id", "nunique"))
    out["share"] = out["revenue"] / out["revenue"].sum()
    out["margin"] = out["profit"] / out["revenue"]
    return out.sort_values("revenue", ascending=False)


# --- Charts -------------------------------------------------------------------
def plot_monthly_trend(df, ax):
    m = df.groupby("month")["sales"].sum()
    ax.plot(m.index, m.values, marker="o", linewidth=2, color="#1F6F78")
    ax.set_title("Monthly Revenue Trend")
    ax.yaxis.set_major_formatter(INR)
    ax.set_xticks(range(0, len(m), 3))
    ax.set_xticklabels(m.index[::3], rotation=45, ha="right")
    ax.set_xlabel("")


def plot_category(df, ax):
    c = revenue_share(df, "category")
    sns.barplot(x=c["revenue"], y=c.index, ax=ax, hue=c.index, legend=False)
    ax.set_title("Revenue by Category")
    ax.xaxis.set_major_formatter(INR)
    ax.set_xlabel(""); ax.set_ylabel("")


def plot_region(df, ax):
    r = revenue_share(df, "region")
    ax.pie(r["revenue"], labels=r.index, autopct="%1.0f%%", startangle=90,
           colors=sns.color_palette("crest", len(r)), wedgeprops={"width": 0.45})
    ax.set_title("Revenue Share by Region")


def plot_channel(df, ax):
    c = revenue_share(df, "channel")
    sns.barplot(x=c.index, y=c["margin"], ax=ax, hue=c.index, legend=False)
    ax.set_title("Profit Margin by Sales Channel")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel(""); ax.set_ylabel("")


def plot_weekday_heatmap(df, ax):
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df = df.assign(month_num=pd.to_datetime(df["order_date"]).dt.month)
    pivot = df.pivot_table(index="weekday", columns="month_num", values="sales",
                           aggfunc="sum").reindex(order) / 1e3
    sns.heatmap(pivot, ax=ax, cmap="crest", cbar_kws={"label": "₹ thousand"})
    ax.set_title("Revenue Heatmap: Weekday x Month")
    ax.set_xlabel("Month"); ax.set_ylabel("")


def plot_payment(df, ax):
    p = df["payment_method"].value_counts(normalize=True)
    sns.barplot(x=p.values, y=p.index, ax=ax, hue=p.index, legend=False, palette="crest")
    ax.set_title("Orders by Payment Method")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel(""); ax.set_ylabel("")


CHARTS = {
    "01_monthly_revenue_trend": plot_monthly_trend,
    "02_revenue_by_category": plot_category,
    "03_revenue_by_region": plot_region,
    "04_margin_by_channel": plot_channel,
    "05_weekday_month_heatmap": plot_weekday_heatmap,
    "06_payment_methods": plot_payment,
}


def save_individual_charts(df):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in CHARTS.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        fn(df, ax)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{name}.png", dpi=130)
        plt.close(fig)


def save_dashboard(df, kpis):
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle("Sales Performance Dashboard (2024-2025)", fontsize=20, weight="bold", y=0.995)
    kpi_text = (f"Revenue: ₹{kpis['total_revenue']/1e5:,.1f}L   |   "
                f"Profit Margin: {kpis['profit_margin']:.1%}   |   "
                f"Orders: {kpis['orders']:,}   |   AOV: ₹{kpis['avg_order_value']:,.0f}   |   "
                f"YoY Growth: {kpis['yoy_revenue_growth']:.1%}   |   "
                f"Repeat Customers: {kpis['repeat_customer_rate']:.1%}")
    fig.text(0.5, 0.955, kpi_text, ha="center", fontsize=12.5,
             bbox=dict(boxstyle="round", facecolor="#E8F1F2", edgecolor="#1F6F78"))
    gs = fig.add_gridspec(3, 2, top=0.92, hspace=0.45, wspace=0.25)
    for (name, fn), pos in zip(CHARTS.items(), [gs[0, :], gs[1, 0], gs[1, 1],
                                                 gs[2, 0], None, gs[2, 1]]):
        if pos is None:
            continue
        fn(df, fig.add_subplot(pos))
    fig.savefig(REPORT_DIR / "sales_dashboard.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(CLEAN)
    kpis = compute_kpis(df)
    pd.Series(kpis).to_csv(REPORT_DIR / "kpi_summary.csv", header=["value"])
    print("Headline KPIs")
    for k, v in kpis.items():
        print(f"  {k:<24}{v}")
    print("\nCategory performance\n", revenue_share(df, "category").round(3))
    print("\nChannel performance\n", revenue_share(df, "channel").round(3))
    save_individual_charts(df)
    save_dashboard(df, kpis)
    print(f"\nCharts saved to {FIG_DIR} and dashboard to {REPORT_DIR / 'sales_dashboard.png'}")


if __name__ == "__main__":
    main()
