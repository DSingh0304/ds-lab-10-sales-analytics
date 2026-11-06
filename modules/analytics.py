"""
Analytics & KPI Module
Provides reusable statistical summary functions and dynamic KPI calculations.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate the 5 core dynamically updating KPIs:
    1. Total Sales
    2. Total Profit
    3. Total Orders (unique Order IDs)
    4. Total Quantity Sold
    5. Profit Margin (%)
    Plus secondary executive metrics.

    Parameters:
        df: Input cleaned or filtered DataFrame.

    Returns:
        Dictionary containing metric values, formatted strings, and status flags.
    """
    if df is None or df.empty:
        return {
            "total_sales": 0.0,
            "total_profit": 0.0,
            "total_orders": 0,
            "total_quantity": 0,
            "profit_margin_pct": 0.0,
            "avg_order_value": 0.0,
            "avg_discount_pct": 0.0,
            "sales_formatted": "$0.00",
            "profit_formatted": "$0.00",
            "orders_formatted": "0",
            "quantity_formatted": "0",
            "margin_formatted": "0.00%",
        }

    # Core 5 KPIs
    total_sales = float(df["Sales"].sum()) if "Sales" in df.columns else 0.0
    total_profit = float(df["Profit"].sum()) if "Profit" in df.columns else 0.0

    if "Order ID" in df.columns:
        total_orders = int(df["Order ID"].nunique())
    else:
        total_orders = len(df)

    total_quantity = int(df["Quantity"].sum()) if "Quantity" in df.columns else 0

    if total_sales != 0:
        profit_margin_pct = (total_profit / total_sales) * 100.0
    else:
        profit_margin_pct = 0.0

    # Secondary metrics
    avg_order_value = total_sales / max(total_orders, 1)
    avg_discount_pct = float(df["Discount"].mean() * 100.0) if "Discount" in df.columns else 0.0

    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "total_orders": total_orders,
        "total_quantity": total_quantity,
        "profit_margin_pct": profit_margin_pct,
        "avg_order_value": avg_order_value,
        "avg_discount_pct": avg_discount_pct,
        # Formatted presentation strings
        "sales_formatted": f"${total_sales:,.2f}",
        "profit_formatted": f"${total_profit:,.2f}",
        "orders_formatted": f"{total_orders:,}",
        "quantity_formatted": f"{total_quantity:,}",
        "margin_formatted": f"{profit_margin_pct:.2f}%",
        "aov_formatted": f"${avg_order_value:,.2f}",
        "discount_formatted": f"{avg_discount_pct:.1f}%",
    }


def calculate_numeric_summary(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculate comprehensive descriptive statistics for numeric attributes:
    Mean, Std, Min, 25%, Median, 75%, Max, IQR, Skewness, Kurtosis.

    Parameters:
        df: Input DataFrame.
        columns: Optional list of numeric columns.

    Returns:
        DataFrame containing formatted summary statistics.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        return pd.DataFrame()

    stats_list = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue

        q25 = float(s.quantile(0.25))
        q50 = float(s.median())
        q75 = float(s.quantile(0.75))
        iqr = q75 - q25
        mean_val = float(s.mean())
        std_val = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        skew_val = float(s.skew()) if len(s) > 2 else 0.0
        kurt_val = float(s.kurtosis()) if len(s) > 3 else 0.0

        stats_list.append(
            {
                "Feature": col,
                "Count": int(len(s)),
                "Mean": round(mean_val, 2),
                "Std Dev": round(std_val, 2),
                "Min": round(float(s.min()), 2),
                "25% (Q1)": round(q25, 2),
                "Median (Q2)": round(q50, 2),
                "75% (Q3)": round(q75, 2),
                "Max": round(float(s.max()), 2),
                "IQR": round(iqr, 2),
                "Skewness": round(skew_val, 3),
                "Kurtosis": round(kurt_val, 3),
            }
        )

    return pd.DataFrame(stats_list)


def calculate_categorical_summary(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculate summary statistics for categorical attributes:
    Count, Unique Values, Top Mode, Top Mode Count, and Frequency Share (%).

    Parameters:
        df: Input DataFrame.
        columns: Optional list of categorical columns.

    Returns:
        DataFrame containing categorical summary.
    """
    if columns is None:
        columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        # Filter out date strings if any
        columns = [c for c in columns if not c.lower().endswith("date")]

    if not columns:
        return pd.DataFrame()

    cat_list = []
    total_len = len(df)
    for col in columns:
        s = df[col].dropna()
        num_valid = len(s)
        num_unique = int(s.nunique())
        if num_valid > 0:
            mode_series = s.mode()
            top_val = mode_series.iloc[0] if not mode_series.empty else "N/A"
            top_count = int((s == top_val).sum())
            top_pct = round((top_count / max(num_valid, 1)) * 100.0, 2)
        else:
            top_val = "N/A"
            top_count = 0
            top_pct = 0.0

        cat_list.append(
            {
                "Categorical Feature": col,
                "Valid Count": num_valid,
                "Unique Classes": num_unique,
                "Most Common (Mode)": str(top_val),
                "Mode Frequency": top_count,
                "Mode Share (%)": top_pct,
                "Missing Count": int(total_len - num_valid),
            }
        )

    return pd.DataFrame(cat_list)


def calculate_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate aggregated sales, profit, orders, and margin grouped by Category and Sub-Category.
    """
    if "Category" not in df.columns or "Sales" not in df.columns or "Profit" not in df.columns:
        return pd.DataFrame()

    group_cols = ["Category"]
    if "Sub-Category" in df.columns:
        group_cols.append("Sub-Category")

    agg_df = (
        df.groupby(group_cols)
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Total_Quantity=("Quantity", "sum") if "Quantity" in df.columns else ("Sales", "count"),
            Unique_Orders=("Order ID", "nunique") if "Order ID" in df.columns else ("Sales", "count"),
        )
        .reset_index()
    )

    agg_df["Profit_Margin_%"] = (agg_df["Total_Profit"] / agg_df["Total_Sales"].replace(0, np.nan)) * 100.0
    agg_df["Profit_Margin_%"] = agg_df["Profit_Margin_%"].fillna(0.0).round(2)
    agg_df["Total_Sales"] = agg_df["Total_Sales"].round(2)
    agg_df["Total_Profit"] = agg_df["Total_Profit"].round(2)

    return agg_df.sort_values(by="Total_Sales", ascending=False)


def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix across available numerical variables.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr(method="pearson").round(3)


def filter_dataset(
    df: pd.DataFrame,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    categories: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    segments: Optional[List[str]] = None,
    ship_modes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply interactive multi-criteria filtering to dataset.
    """
    filtered = df.copy()

    # Filter 1: Date Range
    if "Order Date" in filtered.columns and pd.api.types.is_datetime64_any_dtype(filtered["Order Date"]):
        if start_date is not None:
            filtered = filtered[filtered["Order Date"].dt.date >= pd.to_datetime(start_date).date()]
        if end_date is not None:
            filtered = filtered[filtered["Order Date"].dt.date <= pd.to_datetime(end_date).date()]

    # Filter 2: Category
    if categories and "Category" in filtered.columns:
        filtered = filtered[filtered["Category"].isin(categories)]

    # Filter 3: Region
    if regions and "Region" in filtered.columns:
        filtered = filtered[filtered["Region"].isin(regions)]

    # Filter 4: Customer Segment
    if segments and "Segment" in filtered.columns:
        filtered = filtered[filtered["Segment"].isin(segments)]

    # Filter 5: Ship Mode
    if ship_modes and "Ship Mode" in filtered.columns:
        filtered = filtered[filtered["Ship Mode"].isin(ship_modes)]

    return filtered
