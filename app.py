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
    page_title="Sales Analytics & Data Cleaning Studio | DS Lab 10",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Responsive Cards, Badges, Tables)
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #10b981;
        font-weight: 600;
        margin-top: 4px;
    }
    .metric-sub-negative {
        color: #ef4444;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 6px;
    }
    .badge-blue { background-color: #dbeafe; color: #1e40af; }
    .badge-green { background-color: #d1fae5; color: #065f46; }
    .badge-amber { background-color: #fef3c7; color: #92400e; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        border-radius: 8px;
    }
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
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3090/3090108.png",
        width=56,
    )
    st.title("DS Lab 10 Studio")
    st.caption("Multi-Format Data Validation, Preprocessing & Analytics")
    st.markdown("---")

    st.subheader("📁 1. Data Ingestion")
    uploaded_file = st.file_uploader(
        "Upload dataset (CSV, Excel, JSON)",
        type=["csv", "xlsx", "xls", "json"],
        help="Upload sales dataset in .csv, .xlsx, or .json format to validate and clean.",
    )

    use_sample = False
    default_path = get_default_dataset_path()

    if uploaded_file is None:
        if default_path:
            st.info("💡 No file uploaded yet. Using built-in lab benchmark dataset.")
            use_sample = True
        else:
            st.warning("Please upload a CSV, Excel, or JSON file to begin.")

    st.markdown("---")
    st.subheader("⚙️ 2. Cleaning Settings")
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
    st.info("👋 Welcome! Please upload a data file using the sidebar to begin analysis.")
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
    st.subheader("🎯 3. Interactive Filters")

    # Filter 1: Date Range
    start_date = None
    end_date = None
    if "Order Date" in cleaned_df.columns and pd.api.types.is_datetime64_any_dtype(cleaned_df["Order Date"]):
        min_date = cleaned_df["Order Date"].min().date()
        max_date = cleaned_df["Order Date"].max().date()

        date_selection = st.date_input(
            "📅 Order Date Range",
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
                "🏷️ Product Category",
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
                "🗺️ Geographic Region",
                options=reg_options,
                default=reg_options,
            )

    # Filter 4: Customer Segment
    selected_segments = []
    if "Segment" in cleaned_df.columns:
        seg_options = sorted(cleaned_df["Segment"].dropna().unique().tolist())
        selected_segments = st.multiselect(
            "👥 Customer Segment",
            options=seg_options,
            default=seg_options,
        )

    # Filter 5: Shipping Mode
    selected_ship_modes = []
    if "Ship Mode" in cleaned_df.columns:
        ship_options = sorted(cleaned_df["Ship Mode"].dropna().unique().tolist())
        selected_ship_modes = st.multiselect(
            "🚚 Shipping Mode",
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
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 22px 28px; border-radius: 12px; color: white; margin-bottom: 24px;">
        <h1 style="color: white; margin: 0; font-size: 1.9rem; font-weight: 800;">
            📊 Sales Analytics & Data Cleaning Studio
        </h1>
        <p style="margin: 6px 0 0 0; color: #e0e7ff; font-size: 0.95rem;">
            Data Science Lab 10 • Ingestion Source: <b>{data_source_name}</b> • Filtered Records: <b>{len(filtered_df):,} / {len(cleaned_df):,}</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 5 Core Dynamically Updating KPI Cards
kpi_cols = st.columns(5)

with kpi_cols[0]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">💰 Total Sales</div>
            <div class="metric-value">{kpis['sales_formatted']}</div>
            <div class="metric-sub">Avg Order: {kpis['aov_formatted']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[1]:
    profit_class = "metric-sub" if kpis["total_profit"] >= 0 else "metric-sub metric-sub-negative"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">📈 Total Profit</div>
            <div class="metric-value">{kpis['profit_formatted']}</div>
            <div class="{profit_class}">Net Profitability</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[2]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">📦 Total Orders</div>
            <div class="metric-value">{kpis['orders_formatted']}</div>
            <div class="metric-sub">Unique Order IDs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[3]:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">🏷️ Total Quantity</div>
            <div class="metric-value">{kpis['quantity_formatted']}</div>
            <div class="metric-sub">Units Sold</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[4]:
    margin_class = "metric-sub" if kpis["profit_margin_pct"] >= 0 else "metric-sub metric-sub-negative"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">📊 Profit Margin</div>
            <div class="metric-value">{kpis['margin_formatted']}</div>
            <div class="{margin_class}">Avg Disc: {kpis['discount_formatted']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main Application Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Visualizations & Dashboard",
        "🔍 Data Upload & Structural Validation",
        "🧹 Preprocessing & Cleaning Pipeline",
        "📈 Statistical Summaries & Distributions",
        "💾 Export Cleaned Dataset",
    ]
)

# ==========================================
# TAB 1: INTERACTIVE VISUALIZATIONS
# ==========================================
with tab1:
    if filtered_df.empty:
        st.warning("⚠️ No records match the active filter criteria. Adjust your sidebar filters to see visualizations.")
    else:
        # Visual Controls
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            st.markdown("### 📈 Interactive Visual Performance Analytics")
        with col_ctrl2:
            time_gran = st.selectbox("Timeline Aggregation", options=["Month", "Quarter", "Week"], index=0)

        # Row 1: Time Series Trend
        fig_trend = plot_sales_profit_trend(filtered_df, freq=time_gran)
        st.plotly_chart(fig_trend, use_container_width=True)

        # Row 2: Category Performance & Regional Analysis
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_cat = plot_category_performance(filtered_df)
            st.plotly_chart(fig_cat, use_container_width=True)
        with col_chart2:
            fig_reg = plot_regional_performance(filtered_df)
            st.plotly_chart(fig_reg, use_container_width=True)

        # Row 3: Sub-category breakdown & Customer Segment Share
        col_chart3, col_chart4 = st.columns(2)
        with col_chart3:
            fig_sub = plot_sub_category_breakdown(filtered_df)
            st.plotly_chart(fig_sub, use_container_width=True)
        with col_chart4:
            fig_seg = plot_segment_donut(filtered_df)
            st.plotly_chart(fig_seg, use_container_width=True)

        # Row 4: Transaction Scatter Matrix & Shipping Days Box Plot
        col_chart5, col_chart6 = st.columns([3, 2])
        with col_chart5:
            fig_scatter = plot_sales_vs_profit_scatter(filtered_df)
            st.plotly_chart(fig_scatter, use_container_width=True)
        with col_chart6:
            fig_box = plot_shipping_duration_box(filtered_df)
            st.plotly_chart(fig_box, use_container_width=True)


# ==========================================
# TAB 2: DATA UPLOAD & STRUCTURAL VALIDATION
# ==========================================
with tab2:
    st.markdown("### 🔍 Raw Dataset Structural Validation & Quality Report")
    st.caption("Verifies uploaded dimensions, schema types, null values, and duplicate records.")

    # High-level summary pills
    val_cols = st.columns(4)
    with val_cols[0]:
        st.metric(
            "📐 Dimensions (Rows × Columns)",
            f"{raw_validation['num_rows']} × {raw_validation['num_cols']}",
        )
    with val_cols[1]:
        st.metric(
            "🗂️ Total Missing Cells",
            f"{raw_validation['total_missing_cells']} ({raw_validation['missing_cell_percentage']}%)",
        )
    with val_cols[2]:
        st.metric(
            "📑 Duplicate Records",
            f"{raw_validation['duplicate_count']}",
        )
    with val_cols[3]:
        st.metric(
            "📋 Total Features",
            f"{len(raw_validation['columns'])} Columns",
        )

    st.markdown("#### 1. Column-by-Column Schema & Completeness Audit")
    st.dataframe(
        raw_validation["validation_table"],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### 2. Duplicate Records Inspection")
    if raw_validation["duplicate_count"] > 0:
        st.warning(
            f"Found {raw_validation['duplicate_count']} duplicate records in the raw dataset. "
            "These are automatically deduplicated during preprocessing."
        )
        st.dataframe(raw_validation["duplicate_rows"], use_container_width=True)
    else:
        st.success("✅ No duplicate records detected in the raw dataset.")

    st.markdown("#### 3. Raw Data Preview (First 10 Rows)")
    st.dataframe(raw_df.head(10), use_container_width=True)


# ==========================================
# TAB 3: PREPROCESSING & CLEANING PIPELINE
# ==========================================
with tab3:
    st.markdown("### 🧹 Data Preprocessing & Cleaning Pipeline")
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

    st.markdown("#### 📝 Execution Audit Trail")
    for idx, log_entry in enumerate(cleaning_logs, 1):
        st.markdown(f"**Step {idx}:** {log_entry}")

    st.markdown("---")
    st.markdown("#### 🔄 Before vs. After Cleaning Inspection")
    col_before, col_after = st.columns(2)
    with col_before:
        st.subheader("Raw Data Sample")
        st.dataframe(raw_df.head(8), use_container_width=True)
    with col_after:
        st.subheader("Cleaned & Engineered Sample")
        st.dataframe(cleaned_df.head(8), use_container_width=True)


# ==========================================
# TAB 4: STATISTICAL SUMMARIES
# ==========================================
with tab4:
    st.markdown("### 📈 Descriptive Statistical Summaries")
    st.caption("Computed using modular Python analytical functions.")

    st.markdown("#### 1. Numerical Variables Descriptive Statistics")
    numeric_stats = calculate_numeric_summary(filtered_df)
    if not numeric_stats.empty:
        st.dataframe(numeric_stats, use_container_width=True, hide_index=True)
    else:
        st.info("No numerical features available for summary.")

    st.markdown("#### 2. Categorical Variables Frequency & Dominance")
    cat_stats = calculate_categorical_summary(filtered_df)
    if not cat_stats.empty:
        st.dataframe(cat_stats, use_container_width=True, hide_index=True)

    st.markdown("#### 3. Pearson Correlation Matrix")
    corr_matrix = calculate_correlation_matrix(filtered_df)
    if not corr_matrix.empty:
        st.dataframe(corr_matrix, use_container_width=True)

    st.markdown("#### 4. Category & Sub-Category Performance Aggregations")
    cat_breakdown = calculate_category_breakdown(filtered_df)
    if not cat_breakdown.empty:
        st.dataframe(cat_breakdown, use_container_width=True, hide_index=True)


# ==========================================
# TAB 5: EXPORT CLEANED DATASET
# ==========================================
with tab5:
    st.markdown("### 💾 Export Cleaned Dataset")
    st.caption("Download the validated, cleaned, and feature-engineered dataset in standard formats.")

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    # Export 1: CSV
    with col_exp1:
        st.markdown("#### 📄 CSV Format")
        csv_buffer = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv_buffer,
            file_name="cleaned_sales_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Export 2: Excel (.xlsx)
    with col_exp2:
        st.markdown("#### 📊 Excel Format (.xlsx)")
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            filtered_df.to_excel(writer, index=False, sheet_name="Cleaned_Sales_Data")
        excel_buffer.seek(0)
        st.download_button(
            label="⬇️ Download Cleaned Excel",
            data=excel_buffer.getvalue(),
            file_name="cleaned_sales_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # Export 3: JSON (.json)
    with col_exp3:
        st.markdown("#### 🗄️ JSON Format (.json)")
        json_buffer = filtered_df.to_json(orient="records", date_format="iso", indent=2)
        st.download_button(
            label="⬇️ Download Cleaned JSON",
            data=json_buffer,
            file_name="cleaned_sales_dataset.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### 📋 Cleaned Dataset Schema & Full Data Table")
    st.dataframe(filtered_df, use_container_width=True)


# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 12px 0;">
        Data Science Lab 10 • Built with Streamlit, Pandas, and Plotly • Fully Modular Architecture
    </div>
    """,
    unsafe_allow_html=True,
)
