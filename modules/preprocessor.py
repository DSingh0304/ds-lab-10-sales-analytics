"""
Data Preprocessing & Cleaning Module
Provides modular, reusable functions for cleaning dirty datasets:
- Deduplication
- Missing value imputation
- Inconsistent data standardisation (categories, text, word-numbers, invalid values)
- Robust multi-format date parsing and temporal feature engineering
- Outlier detection and treatment
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Synonym & inconsistency dictionaries
CATEGORY_MAP = {
    "technology": "Technology",
    "technology ": "Technology",
    " technology ": "Technology",
    "tech": "Technology",
    "gadgets": "Technology",
    "office supply": "Office Supplies",
    "office supplies": "Office Supplies",
    "furniture": "Furniture",
}

REGION_MAP = {
    "south": "South",
    "south ": "South",
    "northeast": "East",
    "west": "West",
    "central": "Central",
    "east": "East",
}

SEGMENT_MAP = {
    "consumer": "Consumer",
    "home office": "Home Office",
    "corporate": "Corporate",
    "unknown segment": "Consumer",
}

SHIP_MODE_MAP = {
    "sameday": "Same Day",
    "standard class": "Standard Class",
    "express ultra": "First Class",
    "first class": "First Class",
    "second class": "Second Class",
}

WORD_TO_NUMBER = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "ten percent": 0.10,
    "10%": 0.10,
    "20%": 0.20,
    "many": np.nan,
    "unknown": np.nan,
    "not available": np.nan,
    "n/a": np.nan,
    "?": np.nan,
    "inf": np.nan,
    "-inf": np.nan,
}


def clean_text_field(val: Any) -> Any:
    """Strip whitespace and collapse multiple internal spaces."""
    if pd.isna(val):
        return val
    s = str(val).strip()
    return re.sub(r"\s+", " ", s)


def clean_inconsistent_categories(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Standardize categorical values, casing, synonyms, and whitespace."""
    df_clean = df.copy()
    logs = []

    # Strip whitespace across all string / object columns
    str_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for c in str_cols:
        df_clean[c] = df_clean[c].apply(clean_text_field)

    # Standardize Category
    if "Category" in df_clean.columns:
        before_unique = df_clean["Category"].dropna().unique().tolist()

        def _map_cat(val: Any) -> Any:
            if pd.isna(val) or val is None or str(val).strip().lower() in ["nan", "none", ""]:
                return np.nan
            s = str(val).strip()
            return CATEGORY_MAP.get(s.lower(), s.title())

        df_clean["Category"] = df_clean["Category"].apply(_map_cat)
        after_unique = df_clean["Category"].dropna().unique().tolist()
        logs.append(f"Standardized 'Category' values: {before_unique} -> {after_unique}")

    # Standardize Region
    if "Region" in df_clean.columns:

        def _map_reg(val: Any) -> Any:
            if pd.isna(val) or val is None or str(val).strip().lower() in ["nan", "none", ""]:
                return np.nan
            s = str(val).strip()
            return REGION_MAP.get(s.lower(), s.title())

        df_clean["Region"] = df_clean["Region"].apply(_map_reg)
        logs.append("Normalized 'Region' casing and mappings (e.g., 'SOUTH'/'west' -> 'South'/'West').")

    # Standardize Segment
    if "Segment" in df_clean.columns:

        def _map_seg(val: Any) -> Any:
            if pd.isna(val) or val is None or str(val).strip().lower() in ["nan", "none", ""]:
                return np.nan
            s = str(val).strip()
            return SEGMENT_MAP.get(s.lower(), s.title())

        df_clean["Segment"] = df_clean["Segment"].apply(_map_seg)
        logs.append("Normalized 'Segment' labels and handled 'Unknown Segment'.")

    # Standardize Ship Mode
    if "Ship Mode" in df_clean.columns:

        def _map_ship(val: Any) -> Any:
            if pd.isna(val) or val is None or str(val).strip().lower() in ["nan", "none", ""]:
                return np.nan
            s = str(val).strip()
            return SHIP_MODE_MAP.get(s.lower(), s.title())

        df_clean["Ship Mode"] = df_clean["Ship Mode"].apply(_map_ship)
        logs.append("Normalized 'Ship Mode' values (e.g. 'SameDay' -> 'Same Day', 'Express Ultra' -> 'First Class').")

    return df_clean, logs


def clean_numeric_columns(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    impute_strategy: str = "median",
    handle_outliers: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Coerce non-numeric values (words like 'five', 'ten percent', '?', 'inf'),
    handle invalid negatives/percentages, and impute missing numeric values.
    """
    df_clean = df.copy()
    logs = []

    if numeric_cols is None:
        numeric_cols = [c for c in ["Sales", "Quantity", "Discount", "Profit"] if c in df_clean.columns]

    for col in numeric_cols:
        # Step 1: Replace known word strings
        def _map_num_word(val: Any) -> Any:
            if pd.isna(val) or val is None or str(val).strip().lower() in ["nan", "none", ""]:
                return np.nan
            s = str(val).strip()
            return WORD_TO_NUMBER.get(s.lower(), s)

        cleaned_series = df_clean[col].apply(_map_num_word)

        # Step 2: Coerce to float
        numeric_vals = pd.to_numeric(cleaned_series, errors="coerce")

        # Step 3: Handle domain-specific anomalies
        if col == "Quantity":
            # Quantities cannot be negative
            neg_mask = numeric_vals < 0
            if neg_mask.sum() > 0:
                numeric_vals = numeric_vals.abs()
                logs.append(f"Fixed {int(neg_mask.sum())} negative quantity record(s) to absolute positive value.")

            # Quantity outlier (e.g., 10000)
            if handle_outliers:
                extreme_q = numeric_vals > 50
                if extreme_q.sum() > 0:
                    med_q = numeric_vals[~extreme_q].median()
                    numeric_vals.loc[extreme_q] = med_q
                    logs.append(f"Capped {int(extreme_q.sum())} extreme Quantity outlier(s) (>50) to median ({med_q:.0f}).")

        elif col == "Discount":
            # Discounts > 1.0 (e.g., 1.75 -> 0.175 or cap)
            high_disc = numeric_vals > 1.0
            if high_disc.sum() > 0:
                numeric_vals.loc[high_disc] = numeric_vals.loc[high_disc] / 10.0
                logs.append(f"Scaled {int(high_disc.sum())} erroneous discount value(s) > 100% to valid decimal.")

        elif col == "Sales":
            # Fix negative sales anomaly
            neg_sales = numeric_vals < 0
            if neg_sales.sum() > 0:
                numeric_vals = numeric_vals.abs()
                logs.append(f"Corrected {int(neg_sales.sum())} negative Sales value(s) to positive.")

            # Extreme sales outlier (e.g. 9,999,999.99 for dining chair)
            if handle_outliers:
                extreme_sales = numeric_vals > 50000
                if extreme_sales.sum() > 0:
                    normal_sales = numeric_vals[~extreme_sales]
                    imputed_sales_val = round(normal_sales.median(), 2)
                    numeric_vals.loc[extreme_sales] = imputed_sales_val
                    logs.append(
                        f"Replaced {int(extreme_sales.sum())} extreme Sales outlier(s) (> $50,000) with category median (${imputed_sales_val:.2f})."
                    )

        # Step 4: Missing value imputation
        nan_count = int(numeric_vals.isna().sum())
        if nan_count > 0:
            if impute_strategy == "median":
                fill_val = round(float(numeric_vals.median()), 2)
            elif impute_strategy == "mean":
                fill_val = round(float(numeric_vals.mean()), 2)
            else:
                fill_val = 0.0

            numeric_vals = numeric_vals.fillna(fill_val)
            logs.append(f"Imputed {nan_count} missing value(s) in '{col}' using {impute_strategy} ({fill_val}).")

        df_clean[col] = numeric_vals

    return df_clean, logs


def clean_date_columns(
    df: pd.DataFrame,
    date_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse mixed-format dates (e.g., '2025-09-07' vs '26/12/2024'), drop or impute
    unparseable values (like 'not-a-date'), and engineer temporal features.
    """
    df_clean = df.copy()
    logs = []

    if date_cols is None:
        date_cols = [c for c in ["Order Date", "Ship Date"] if c in df_clean.columns]

    for col in date_cols:
        parsed_dates = pd.to_datetime(df_clean[col], format="mixed", errors="coerce")
        nat_count = int(parsed_dates.isna().sum())
        if nat_count > 0:
            logs.append(f"Detected {nat_count} invalid/missing date entries in '{col}' (converted to NaT).")
        df_clean[col] = parsed_dates

    # If Order Date is NaT, drop row if critical
    if "Order Date" in df_clean.columns:
        invalid_orders = df_clean["Order Date"].isna()
        if invalid_orders.sum() > 0:
            num_dropped = int(invalid_orders.sum())
            df_clean = df_clean[~invalid_orders].copy()
            logs.append(f"Dropped {num_dropped} record(s) with invalid 'Order Date' (e.g., 'not-a-date').")

        # Temporal Feature Engineering
        df_clean["Order Year"] = df_clean["Order Date"].dt.year
        df_clean["Order Month"] = df_clean["Order Date"].dt.month
        df_clean["Order Month Name"] = df_clean["Order Date"].dt.strftime("%b")
        df_clean["Order Year-Month"] = df_clean["Order Date"].dt.strftime("%Y-%m")
        df_clean["Order Day of Week"] = df_clean["Order Date"].dt.day_name()
        df_clean["Order Quarter"] = df_clean["Order Date"].dt.to_period("Q").astype(str)
        logs.append("Engineered temporal features: Year, Month, Year-Month, Day of Week, Quarter.")

    # Calculate shipping duration in days
    if "Ship Date" in df_clean.columns and "Order Date" in df_clean.columns:
        # If Ship Date is NaT, impute with Order Date + 4 days (average shipping lag)
        ship_nat = df_clean["Ship Date"].isna()
        if ship_nat.sum() > 0:
            df_clean.loc[ship_nat, "Ship Date"] = df_clean.loc[ship_nat, "Order Date"] + pd.Timedelta(days=4)
            logs.append(f"Imputed {int(ship_nat.sum())} missing 'Ship Date' entries with Order Date + 4 days shipping buffer.")

        df_clean["Shipping Days"] = (df_clean["Ship Date"] - df_clean["Order Date"]).dt.days
        # If shipping days < 0 due to data error, take absolute value or default to 1
        df_clean["Shipping Days"] = df_clean["Shipping Days"].apply(lambda d: max(int(d), 0) if pd.notna(d) else 3)
        logs.append("Calculated 'Shipping Days' delivery metric.")

    return df_clean, logs


def handle_categorical_missing(
    df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
    strategy: str = "mode",
) -> Tuple[pd.DataFrame, List[str]]:
    """Impute missing categorical entries with mode or 'Unknown'."""
    df_clean = df.copy()
    logs = []

    if cat_cols is None:
        cat_cols = [
            c for c in [
                "Customer ID",
                "Segment",
                "Country",
                "City",
                "State",
                "Region",
                "Product ID",
                "Category",
                "Sub-Category",
                "Product Name",
                "Ship Mode",
            ]
            if c in df_clean.columns
        ]

    for col in cat_cols:
        null_count = int(df_clean[col].isna().sum())
        if null_count > 0:
            if strategy == "mode" and not df_clean[col].mode().empty:
                fill_val = df_clean[col].mode().iloc[0]
            else:
                fill_val = "Unknown"
            df_clean[col] = df_clean[col].fillna(fill_val)
            logs.append(f"Imputed {null_count} missing value(s) in '{col}' with '{fill_val}'.")

    return df_clean, logs


def preprocess_dataset(
    df: pd.DataFrame,
    impute_strategy: str = "median",
    cat_strategy: str = "mode",
    handle_outliers: bool = True,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Full end-to-end cleaning and preprocessing pipeline.

    Parameters:
        df: Input raw DataFrame.
        impute_strategy: Strategy for numeric imputation ('median', 'mean', 'zero').
        cat_strategy: Strategy for categorical imputation ('mode', 'unknown').
        handle_outliers: Whether to cap/correct extreme anomalies in Sales & Quantity.

    Returns:
        Tuple of (Cleaned DataFrame, Audit Logs List, Metrics Dictionary).
    """
    audit_logs: List[str] = []
    initial_rows, initial_cols = df.shape
    df_clean = df.copy()

    # Step 1: Drop rows where all elements are NaN
    before_all_nan = len(df_clean)
    df_clean = df_clean.dropna(how="all")
    dropped_all_nan = before_all_nan - len(df_clean)
    if dropped_all_nan > 0:
        audit_logs.append(f"Dropped {dropped_all_nan} completely empty row(s).")

    # Step 2: Drop rows missing primary identifier 'Order ID' if column exists
    if "Order ID" in df_clean.columns:
        missing_order_id = df_clean["Order ID"].isna().sum()
        if missing_order_id > 0:
            df_clean = df_clean.dropna(subset=["Order ID"])
            audit_logs.append(f"Dropped {missing_order_id} record(s) lacking mandatory 'Order ID'.")

    # Step 3: Deduplication
    dups_count = int(df_clean.duplicated().sum())
    if dups_count > 0:
        df_clean = df_clean.drop_duplicates(keep="first")
        audit_logs.append(f"Identified and removed {dups_count} exact duplicate record(s).")
    else:
        audit_logs.append("No duplicate records found.")

    # Step 4: Clean text and categorical values
    df_clean, cat_logs = clean_inconsistent_categories(df_clean)
    audit_logs.extend(cat_logs)

    # Step 5: Clean date formats and engineer temporal features
    df_clean, date_logs = clean_date_columns(df_clean)
    audit_logs.extend(date_logs)

    # Step 6: Clean numeric columns (word-numbers, anomalies, nulls)
    df_clean, num_logs = clean_numeric_columns(
        df_clean,
        impute_strategy=impute_strategy,
        handle_outliers=handle_outliers,
    )
    audit_logs.extend(num_logs)

    # Step 7: Impute remaining categorical nulls
    df_clean, cat_null_logs = handle_categorical_missing(df_clean, strategy=cat_strategy)
    audit_logs.extend(cat_null_logs)

    # Reset index
    df_clean = df_clean.reset_index(drop=True)

    final_rows, final_cols = df_clean.shape
    metrics = {
        "initial_rows": initial_rows,
        "initial_cols": initial_cols,
        "final_rows": final_rows,
        "final_cols": final_cols,
        "rows_removed": initial_rows - final_rows,
        "duplicates_removed": dups_count,
        "remaining_nulls": int(df_clean.isnull().sum().sum()),
    }

    return df_clean, audit_logs, metrics
