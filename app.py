"""
Data Science Lab 10: Multi-Format Sales Analytics & Data Cleaning Studio
An end-to-end interactive Streamlit application supporting:
- Multi-format data upload & structural validation (CSV, Excel, JSON)
- Automated & configurable preprocessing (nulls, duplicates, anomalies, dates)
- 5 Dynamic KPIs and interactive multi-criteria filtering
- Interactive Plotly visualizations and comprehensive statistical summaries
- Export cleaned datasets in CSV, Excel, and JSON formats
"""

import io
from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st

from modules.analytics import (
    calculate_categorical_summary,
    calculate_category_breakdown,
    calculate_correlation_matrix,
    calculate_kpis,
    calculate_numeric_summary,
    filter_dataset,
)
from modules.data_loader import load_data, validate_raw_data
from modules.preprocessor import preprocess_dataset
from modules.visualizations import (
    plot_category_performance,
    plot_regional_performance,
    plot_sales_profit_trend,
    plot_sales_vs_profit_scatter,
    plot_segment_donut,
    plot_shipping_duration_box,
    plot_sub_category_breakdown,
)

# Page configuration
st.set_page_config(
    page_title="Sales Analytics Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    :root {
        --ink: #1c1b19; --ink-soft: #5d5b55; --ink-faint: #8a877f;
        --line: #e4e1da; --paper: #faf9f6; --accent: #2a78d6;
        --good: #1f7a4d; --bad: #c8402f;
    }
    html, body, [class*="css"], .stApp { font-family: 'Inter', system-ui, sans-serif; }
    .block-container { padding-top: 2.2rem; max-width: 1280px; }
    h1, h2, h3, h4 { letter-spacing: -0.01em; color: var(--ink); font-weight: 600; }
    h3 { font-size: 1.15rem !important; }
    h4 { font-size: 1rem !important; margin-top: 1.4rem !important; }
    hr { border-color: var(--line) !important; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Page header */
    .page-head { border-bottom: 1px solid var(--line); padding-bottom: 14px; margin-bottom: 22px; }
    .page-head h1 { font-size: 1.6rem; margin: 0 0 6px 0; padding: 0; }
    .page-head .meta { color: var(--ink-soft); font-size: 0.88rem; }
    .page-head .meta b { color: var(--ink); font-weight: 500; }

    /* KPI strip: one ruled band instead of floating cards */
    .kpi-strip { display: grid; grid-template-columns: repeat(5, 1fr);
        border: 1px solid var(--line); border-radius: 8px; background: #fff; margin-bottom: 26px; }
    .kpi { padding: 16px 20px; border-right: 1px solid var(--line); min-width: 0; }
    .kpi:last-child { border-right: none; }
    .kpi-label { font-size: 0.78rem; color: var(--ink-soft); margin-bottom: 6px; }
    .kpi-value { font-size: 1.65rem; font-weight: 600; color: var(--ink);
        line-height: 1.15; font-variant-numeric: tabular-nums; }
    .kpi-sub { font-size: 0.78rem; color: var(--ink-faint); margin-top: 4px; }
    .kpi-sub.neg { color: var(--bad); }
    @media (max-width: 900px) {
        .kpi-strip { grid-template-columns: repeat(2, 1fr); }
        .kpi { border-bottom: 1px solid var(--line); }
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { padding: 10px 0; font-weight: 500; color: var(--ink-soft); background: none; }
    .stTabs [aria-selected="true"] { color: var(--ink); }

    /* Streamlit metrics, sidebar */
    [data-testid="stMetric"] { background: #fff; border: 1px solid var(--line);
        border-radius: 8px; padding: 12px 16px; }
    [data-testid="stMetricLabel"] { color: var(--ink-soft); }
    [data-testid="stSidebar"] { border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h3 { font-size: 0.78rem !important; text-transform: uppercase;
        letter-spacing: 0.08em; color: var(--ink-faint); font-weight: 600; margin-top: 0.4rem; }
    .brand { font-size: 1.05rem; font-weight: 600; color: var(--ink); margin: 0; }
    .brand-sub { font-size: 0.82rem; color: var(--ink-soft); margin: 2px 0 8px 0; }
    .audit-step { display: flex; gap: 14px; padding: 9px 0; border-bottom: 1px solid var(--line);
        font-size: 0.92rem; color: var(--ink); }
    .audit-step span.n { color: var(--ink-faint); font-variant-numeric: tabular-nums; min-width: 1.6rem; }
    .footer { color: var(--ink-faint); font-size: 0.8rem; padding: 10px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_default_dataset_path() -> Optional[Path]:
    """Locate the default experiment10 dataset if present."""
    default_files = [
        Path("experiment10_raw_dataset.csv"),
        Path(__file__).parent / "experiment10_raw_dataset.csv",
    ]
    for p in default_files:
        if p.exists():
            return p
    return None


# Sidebar Navigation & Upload Controls
with st.sidebar:
    st.markdown(
        '<p class="brand">DS Lab 10</p>'
        '<p class="brand-sub">Sales analytics and data cleaning</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.subheader("Data")
    uploaded_file = st.file_uploader(
        "Upload dataset (CSV, Excel, JSON)",
        type=["csv", "xlsx", "xls", "json"],
        help="Upload sales dataset in .csv, .xlsx, or .json format to validate and clean.",
    )

    use_sample = False
    default_path = get_default_dataset_path()

    if uploaded_file is None:
        if default_path:
            st.info("No file uploaded. Using the built-in lab dataset.")
            use_sample = True
        else:
            st.warning("Please upload a CSV, Excel, or JSON file to begin.")

    st.markdown("---")
    st.subheader("Cleaning")
    impute_method = st.selectbox(
        "Numeric Imputation Strategy",
        options=["median", "mean", "zero"],
        index=0,
        help="Method used to impute missing numeric values in Sales, Quantity, Discount, and Profit.",
    )
    handle_outliers = st.checkbox(
        "Correct domain anomalies & extreme outliers",
        value=True,
        help="Cap extreme quantity outliers, fix negative sales/quantities, and scale invalid discounts > 100%.",
    )


# Ingest Data
raw_df: Optional[pd.DataFrame] = None
data_source_name = ""

if uploaded_file is not None:
    raw_df, err = load_data(uploaded_file)
    data_source_name = uploaded_file.name
    if err:
        st.error(f"Failed to load file: {err}")
elif use_sample and default_path:
    raw_df, err = load_data(str(default_path))
    data_source_name = default_path.name
    if err:
        st.error(f"Failed to load sample dataset: {err}")

# Stop if no data available
if raw_df is None or raw_df.empty:
    st.info("Upload a data file in the sidebar to begin.")
    st.stop()

# Perform Raw Data Structural Validation
raw_validation = validate_raw_data(raw_df)

# Run Data Preprocessing Pipeline
cleaned_df, cleaning_logs, cleaning_metrics = preprocess_dataset(
    raw_df,
    impute_strategy=impute_method,
    cat_strategy="mode",
    handle_outliers=handle_outliers,
)

# Sidebar Interactive Filters
with st.sidebar:
    st.markdown("---")
    st.subheader("Filters")

    # Filter 1: Date Range
    start_date = None
    end_date = None
    if "Order Date" in cleaned_df.columns and pd.api.types.is_datetime64_any_dtype(cleaned_df["Order Date"]):
        min_date = cleaned_df["Order Date"].min().date()
        max_date = cleaned_df["Order Date"].max().date()

        date_selection = st.date_input(
            "Order Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filter analytics and KPIs within a specific timeline.",
        )
        if isinstance(date_selection, (tuple, list)):
            if len(date_selection) >= 1:
                start_date = date_selection[0]
            if len(date_selection) >= 2:
                end_date = date_selection[1]
        else:
            start_date = date_selection

    # Filter 2: Category
    selected_categories = []
    if "Category" in cleaned_df.columns:
        cat_options = sorted(cleaned_df["Category"].dropna().unique().tolist())
        select_all_cats = st.checkbox("Select All Categories", value=True)
        if select_all_cats:
            selected_categories = cat_options
        else:
            selected_categories = st.multiselect(
                "Product Category",
                options=cat_options,
                default=cat_options[:2] if len(cat_options) >= 2 else cat_options,
            )

    # Filter 3: Region
    selected_regions = []
    if "Region" in cleaned_df.columns:
        reg_options = sorted(cleaned_df["Region"].dropna().unique().tolist())
        select_all_regs = st.checkbox("Select All Regions", value=True)
        if select_all_regs:
            selected_regions = reg_options
        else:
            selected_regions = st.multiselect(
                "Geographic Region",
                options=reg_options,
                default=reg_options,
            )

    # Filter 4: Customer Segment
    selected_segments = []
    if "Segment" in cleaned_df.columns:
        seg_options = sorted(cleaned_df["Segment"].dropna().unique().tolist())
        selected_segments = st.multiselect(
            "Customer Segment",
            options=seg_options,
            default=seg_options,
        )

    # Filter 5: Shipping Mode
    selected_ship_modes = []
    if "Ship Mode" in cleaned_df.columns:
        ship_options = sorted(cleaned_df["Ship Mode"].dropna().unique().tolist())
        selected_ship_modes = st.multiselect(
            "Shipping Mode",
            options=ship_options,
            default=ship_options,
        )

# Apply Filters
filtered_df = filter_dataset(
    cleaned_df,
    start_date=start_date,
    end_date=end_date,
    categories=selected_categories if selected_categories else None,
    regions=selected_regions if selected_regions else None,
    segments=selected_segments if selected_segments else None,
    ship_modes=selected_ship_modes if selected_ship_modes else None,
)

# Calculate Dynamically Updating KPIs
kpis = calculate_kpis(filtered_df)

# Header Section
st.markdown(
    f"""
    <div class="page-head">
        <h1>Sales Analytics &amp; Data Cleaning Studio</h1>
        <div class="meta">Source <b>{data_source_name}</b> &nbsp;/&nbsp;
        Showing <b>{len(filtered_df):,}</b> of <b>{len(cleaned_df):,}</b> records</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPI strip
profit_neg = " neg" if kpis["total_profit"] < 0 else ""
margin_neg = " neg" if kpis["profit_margin_pct"] < 0 else ""
st.markdown(
    f"""
    <div class="kpi-strip">
        <div class="kpi"><div class="kpi-label">Total sales</div>
            <div class="kpi-value">{kpis['sales_formatted']}</div>
            <div class="kpi-sub">Avg order {kpis['aov_formatted']}</div></div>
        <div class="kpi"><div class="kpi-label">Total profit</div>
            <div class="kpi-value">{kpis['profit_formatted']}</div>
            <div class="kpi-sub{profit_neg}">{"Net loss" if profit_neg else "Net profit"}</div></div>
        <div class="kpi"><div class="kpi-label">Orders</div>
            <div class="kpi-value">{kpis['orders_formatted']}</div>
            <div class="kpi-sub">Unique order IDs</div></div>
        <div class="kpi"><div class="kpi-label">Quantity</div>
            <div class="kpi-value">{kpis['quantity_formatted']}</div>
            <div class="kpi-sub">Units sold</div></div>
        <div class="kpi"><div class="kpi-label">Profit margin</div>
            <div class="kpi-value">{kpis['margin_formatted']}</div>
            <div class="kpi-sub{margin_neg}">Avg discount {kpis['discount_formatted']}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Application Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Visualizations & Dashboard",
        "Data Upload & Structural Validation",
        "Preprocessing & Cleaning Pipeline",
        "Statistical Summaries & Distributions",
        "Export Cleaned Dataset",
    ]
)

# ==========================================
# TAB 1: INTERACTIVE VISUALIZATIONS
# ==========================================
with tab1:
    if filtered_df.empty:
        st.warning("No records match the active filter criteria. Adjust your sidebar filters to see visualizations.")
    else:
        # Visual Controls
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            st.markdown("### Performance")
        with col_ctrl2:
            time_gran = st.selectbox("Timeline Aggregation", options=["Month", "Quarter", "Week"], index=0)

        # Row 1: Time Series Trend
        fig_trend = plot_sales_profit_trend(filtered_df, freq=time_gran)
        st.plotly_chart(fig_trend, width="stretch")

        # Row 2: Category Performance & Regional Analysis
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_cat = plot_category_performance(filtered_df)
            st.plotly_chart(fig_cat, width="stretch")
        with col_chart2:
            fig_reg = plot_regional_performance(filtered_df)
            st.plotly_chart(fig_reg, width="stretch")

        # Row 3: Sub-category breakdown & Customer Segment Share
        col_chart3, col_chart4 = st.columns(2)
        with col_chart3:
            fig_sub = plot_sub_category_breakdown(filtered_df)
            st.plotly_chart(fig_sub, width="stretch")
        with col_chart4:
            fig_seg = plot_segment_donut(filtered_df)
            st.plotly_chart(fig_seg, width="stretch")

        # Row 4: Transaction Scatter Matrix & Shipping Days Box Plot
        col_chart5, col_chart6 = st.columns([3, 2])
        with col_chart5:
            fig_scatter = plot_sales_vs_profit_scatter(filtered_df)
            st.plotly_chart(fig_scatter, width="stretch")
        with col_chart6:
            fig_box = plot_shipping_duration_box(filtered_df)
            st.plotly_chart(fig_box, width="stretch")


# ==========================================
# TAB 2: DATA UPLOAD & STRUCTURAL VALIDATION
# ==========================================
with tab2:
    st.markdown("### Raw Dataset Structural Validation & Quality Report")
    st.caption("Verifies uploaded dimensions, schema types, null values, and duplicate records.")

    # High-level summary pills
    val_cols = st.columns(4)
    with val_cols[0]:
        st.metric(
            "Dimensions (Rows × Columns)",
            f"{raw_validation['num_rows']} × {raw_validation['num_cols']}",
        )
    with val_cols[1]:
        st.metric(
            "Total Missing Cells",
            f"{raw_validation['total_missing_cells']} ({raw_validation['missing_cell_percentage']}%)",
        )
    with val_cols[2]:
        st.metric(
            "Duplicate Records",
            f"{raw_validation['duplicate_count']}",
        )
    with val_cols[3]:
        st.metric(
            "Total Features",
            f"{len(raw_validation['columns'])} Columns",
        )

    st.markdown("#### 1. Column-by-Column Schema & Completeness Audit")
    st.dataframe(
        raw_validation["validation_table"],
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### 2. Duplicate Records Inspection")
    if raw_validation["duplicate_count"] > 0:
        st.warning(
            f"Found {raw_validation['duplicate_count']} duplicate records in the raw dataset. "
            "These are automatically deduplicated during preprocessing."
        )
        st.dataframe(raw_validation["duplicate_rows"], width="stretch")
    else:
        st.success("No duplicate records detected in the raw dataset.")

    st.markdown("#### 3. Raw Data Preview (First 10 Rows)")
    st.dataframe(raw_df.head(10), width="stretch")


# ==========================================
# TAB 3: PREPROCESSING & CLEANING PIPELINE
# ==========================================
with tab3:
    st.markdown("### Data Preprocessing & Cleaning Pipeline")
    st.caption("Detailed audit trail of automated transformations, imputation, and anomaly resolution.")

    # Cleaning Summary Metric Cards
    clean_cols = st.columns(4)
    with clean_cols[0]:
        st.metric("Raw Rows", f"{cleaning_metrics['initial_rows']}")
    with clean_cols[1]:
        st.metric("Duplicates Dropped", f"{cleaning_metrics['duplicates_removed']}")
    with clean_cols[2]:
        st.metric("Cleaned Rows", f"{cleaning_metrics['final_rows']}")
    with clean_cols[3]:
        st.metric("Post-Cleaning Nulls", f"{cleaning_metrics['remaining_nulls']}")

    st.markdown("#### Execution Audit Trail")
    st.markdown(
        "".join(
            f'<div class="audit-step"><span class="n">{i:02d}</span><span>{e}</span></div>'
            for i, e in enumerate(cleaning_logs, 1)
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### Before vs. After Cleaning Inspection")
    col_before, col_after = st.columns(2)
    with col_before:
        st.subheader("Raw Data Sample")
        st.dataframe(raw_df.head(8), width="stretch")
    with col_after:
        st.subheader("Cleaned & Engineered Sample")
        st.dataframe(cleaned_df.head(8), width="stretch")


# ==========================================
# TAB 4: STATISTICAL SUMMARIES
# ==========================================
with tab4:
    st.markdown("### Descriptive Statistical Summaries")
    st.caption("Computed using modular Python analytical functions.")

    st.markdown("#### 1. Numerical Variables Descriptive Statistics")
    numeric_stats = calculate_numeric_summary(filtered_df)
    if not numeric_stats.empty:
        st.dataframe(numeric_stats, width="stretch", hide_index=True)
    else:
        st.info("No numerical features available for summary.")

    st.markdown("#### 2. Categorical Variables Frequency & Dominance")
    cat_stats = calculate_categorical_summary(filtered_df)
    if not cat_stats.empty:
        st.dataframe(cat_stats, width="stretch", hide_index=True)

    st.markdown("#### 3. Pearson Correlation Matrix")
    corr_matrix = calculate_correlation_matrix(filtered_df)
    if not corr_matrix.empty:
        st.dataframe(corr_matrix, width="stretch")

    st.markdown("#### 4. Category & Sub-Category Performance Aggregations")
    cat_breakdown = calculate_category_breakdown(filtered_df)
    if not cat_breakdown.empty:
        st.dataframe(cat_breakdown, width="stretch", hide_index=True)


# ==========================================
# TAB 5: EXPORT CLEANED DATASET
# ==========================================
with tab5:
    st.markdown("### Export Cleaned Dataset")
    st.caption("Download the validated, cleaned, and feature-engineered dataset in standard formats.")

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    # Export 1: CSV
    with col_exp1:
        st.markdown("#### CSV Format")
        csv_buffer = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned CSV",
            data=csv_buffer,
            file_name="cleaned_sales_dataset.csv",
            mime="text/csv",
            width="stretch",
        )

    # Export 2: Excel (.xlsx)
    with col_exp2:
        st.markdown("#### Excel Format (.xlsx)")
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            filtered_df.to_excel(writer, index=False, sheet_name="Cleaned_Sales_Data")
        excel_buffer.seek(0)
        st.download_button(
            label="Download Cleaned Excel",
            data=excel_buffer.getvalue(),
            file_name="cleaned_sales_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

    # Export 3: JSON (.json)
    with col_exp3:
        st.markdown("#### JSON Format (.json)")
        json_buffer = filtered_df.to_json(orient="records", date_format="iso", indent=2)
        st.download_button(
            label="Download Cleaned JSON",
            data=json_buffer,
            file_name="cleaned_sales_dataset.json",
            mime="application/json",
            width="stretch",
        )

    st.markdown("---")
    st.markdown("#### Cleaned Dataset Schema & Full Data Table")
    st.dataframe(filtered_df, width="stretch")


# Footer
st.markdown("---")
st.markdown(
    '<div class="footer">Data Science Lab 10. Built with Streamlit, pandas and Plotly.</div>',
    unsafe_allow_html=True,
)
