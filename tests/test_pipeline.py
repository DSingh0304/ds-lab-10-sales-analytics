"""
Automated Test Suite for DS Lab 10 Application
Tests multi-format ingestion, validation, preprocessing, KPIs, and visualization pipelines.
"""

import io
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from modules.analytics import (
    calculate_categorical_summary,
    calculate_category_breakdown,
    calculate_correlation_matrix,
    calculate_kpis,
    calculate_numeric_summary,
    filter_dataset,
)
from modules.data_loader import load_data, validate_raw_data
from modules.preprocessor import (
    clean_date_columns,
    clean_inconsistent_categories,
    clean_numeric_columns,
    preprocess_dataset,
)
from modules.visualizations import (
    plot_category_performance,
    plot_regional_performance,
    plot_sales_profit_trend,
    plot_sales_vs_profit_scatter,
    plot_segment_donut,
    plot_shipping_duration_box,
    plot_sub_category_breakdown,
)


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Creates a synthetic dirty dataframe mirroring experiment10_raw_dataset.csv."""
    data = {
        "Order ID": ["ORD-1", "ORD-2", "ORD-3", "ORD-1", np.nan, "ORD-4"],
        "Order Date": ["2024-01-15", "16/02/2024", "2024-03-20", "2024-01-15", "not-a-date", "2024-04-10"],
        "Ship Date": ["2024-01-18", "2024-02-22", "2024-03-25", "2024-01-18", np.nan, "2024-04-15"],
        "Ship Mode": ["Second Class", "SameDay", "Standard class", "Second Class", np.nan, "First Class"],
        "Customer ID": ["C-1", "C-2", "C-3", "C-1", np.nan, "C-4"],
        "Segment": ["Consumer", "consumer", "Home Office", "Consumer", np.nan, "Corporate"],
        "Country": ["United States"] * 6,
        "City": ["Boston", "Miami", "Chicago", "Boston", np.nan, "New York"],
        "State": ["MA", "FL", "IL", "MA", np.nan, "NY"],
        "Region": ["East", "SOUTH", "west", "East", np.nan, "East"],
        "Product ID": ["P-1", "P-2", "P-3", "P-1", np.nan, "P-4"],
        "Category": ["Technology", " technology ", "Office supply", "Technology", np.nan, "Furniture"],
        "Sub-Category": ["Phones", "Accessories", "Binders", "Phones", np.nan, "Chairs"],
        "Product Name": ["Phone A", "Mouse B", "Binder C", "Phone A", np.nan, "Chair D"],
        "Sales": ["1000", "250.5", "unknown", "1000", np.nan, "-50"],
        "Quantity": ["2", "five", "4", "2", "many", "10000"],
        "Discount": ["0.1", "ten percent", "0.2", "0.1", np.nan, "1.75"],
        "Profit": ["150", "?", "45.0", "150", np.nan, "inf"],
    }
    return pd.DataFrame(data)


def test_load_csv_and_validation(tmp_path: Path, sample_raw_data: pd.DataFrame):
    """Test loading and validating a CSV file."""
    csv_file = tmp_path / "test.csv"
    sample_raw_data.to_csv(csv_file, index=False)

    df, err = load_data(str(csv_file))
    assert err is None
    assert df is not None
    assert df.shape == sample_raw_data.shape

    val = validate_raw_data(df)
    assert val["num_rows"] == 6
    assert val["num_cols"] == 18
    assert val["duplicate_count"] == 1  # Row 3 is duplicate of row 0
    assert "Sales" in val["columns"]
    assert val["is_valid"] is True


def test_load_excel(tmp_path: Path, sample_raw_data: pd.DataFrame):
    """Test loading an Excel (.xlsx) file."""
    excel_file = tmp_path / "test.xlsx"
    sample_raw_data.to_excel(excel_file, index=False)

    df, err = load_data(str(excel_file))
    assert err is None
    assert df is not None
    assert len(df) == 6


def test_load_json(tmp_path: Path, sample_raw_data: pd.DataFrame):
    """Test loading a JSON (.json) file."""
    json_file = tmp_path / "test.json"
    sample_raw_data.to_json(json_file, orient="records")

    df, err = load_data(str(json_file))
    assert err is None
    assert df is not None
    assert len(df) == 6


def test_preprocessing_pipeline(sample_raw_data: pd.DataFrame):
    """Test end-to-end preprocessing, cleaning, and feature engineering."""
    df_clean, logs, metrics = preprocess_dataset(sample_raw_data, handle_outliers=True)

    # Corrupt row with missing Order ID & not-a-date dropped
    # Duplicate row dropped
    assert len(df_clean) == 4
    assert metrics["duplicates_removed"] == 1
    assert metrics["remaining_nulls"] == 0

    # Categorical normalization
    assert set(df_clean["Category"].unique()) == {"Technology", "Office Supplies", "Furniture"}
    assert "South" in df_clean["Region"].unique()
    assert "West" in df_clean["Region"].unique()
    assert "Consumer" in df_clean["Segment"].unique()

    # Numeric conversions and outlier handling
    assert pd.api.types.is_numeric_dtype(df_clean["Sales"])
    assert pd.api.types.is_numeric_dtype(df_clean["Quantity"])
    assert pd.api.types.is_numeric_dtype(df_clean["Discount"])
    assert pd.api.types.is_numeric_dtype(df_clean["Profit"])

    # Quantity anomaly (10000) should be capped
    assert df_clean["Quantity"].max() <= 50

    # Discount > 1 should be scaled
    assert df_clean["Discount"].max() <= 1.0

    # Date parsing and temporal features
    assert "Order Year" in df_clean.columns
    assert "Order Year-Month" in df_clean.columns
    assert "Shipping Days" in df_clean.columns
    assert (df_clean["Shipping Days"] >= 0).all()


def test_kpis_calculation(sample_raw_data: pd.DataFrame):
    """Test the 5 dynamically updating KPIs."""
    df_clean, _, _ = preprocess_dataset(sample_raw_data)
    kpis = calculate_kpis(df_clean)

    assert "total_sales" in kpis
    assert "total_profit" in kpis
    assert "total_orders" in kpis
    assert "total_quantity" in kpis
    assert "profit_margin_pct" in kpis

    assert kpis["total_sales"] > 0
    assert kpis["total_orders"] == df_clean["Order ID"].nunique()
    assert kpis["total_quantity"] == int(df_clean["Quantity"].sum())

    expected_margin = (kpis["total_profit"] / kpis["total_sales"]) * 100.0
    assert abs(kpis["profit_margin_pct"] - expected_margin) < 1e-4


def test_statistical_summaries(sample_raw_data: pd.DataFrame):
    """Test numeric and categorical descriptive summary functions."""
    df_clean, _, _ = preprocess_dataset(sample_raw_data)

    num_summary = calculate_numeric_summary(df_clean)
    assert not num_summary.empty
    assert "Mean" in num_summary.columns
    assert "Median (Q2)" in num_summary.columns
    assert "IQR" in num_summary.columns
    assert "Skewness" in num_summary.columns

    cat_summary = calculate_categorical_summary(df_clean)
    assert not cat_summary.empty
    assert "Categorical Feature" in cat_summary.columns
    assert "Most Common (Mode)" in cat_summary.columns

    corr_matrix = calculate_correlation_matrix(df_clean)
    assert not corr_matrix.empty
    assert corr_matrix.shape[0] == corr_matrix.shape[1]

    breakdown = calculate_category_breakdown(df_clean)
    assert not breakdown.empty
    assert "Total_Sales" in breakdown.columns
    assert "Profit_Margin_%" in breakdown.columns


def test_filtering_and_dynamic_kpis(sample_raw_data: pd.DataFrame):
    """Test interactive filtering and dynamic updating of KPIs."""
    df_clean, _, _ = preprocess_dataset(sample_raw_data)

    # Filter by category
    tech_filtered = filter_dataset(df_clean, categories=["Technology"])
    assert len(tech_filtered) == 2
    tech_kpis = calculate_kpis(tech_filtered)
    assert tech_kpis["total_orders"] == 2

    # Filter by region
    east_filtered = filter_dataset(df_clean, regions=["East"])
    assert (east_filtered["Region"] == "East").all()


def test_visualizations(sample_raw_data: pd.DataFrame):
    """Test that all 7 Plotly interactive figures render without error."""
    df_clean, _, _ = preprocess_dataset(sample_raw_data)

    f1 = plot_sales_profit_trend(df_clean)
    assert f1 is not None

    f2 = plot_category_performance(df_clean)
    assert f2 is not None

    f3 = plot_sub_category_breakdown(df_clean)
    assert f3 is not None

    f4 = plot_regional_performance(df_clean)
    assert f4 is not None

    f5 = plot_sales_vs_profit_scatter(df_clean)
    assert f5 is not None

    f6 = plot_segment_donut(df_clean)
    assert f6 is not None

    f7 = plot_shipping_duration_box(df_clean)
    assert f7 is not None

