# 📊 Sales Analytics & Data Cleaning Studio (DS Lab 10)

An end-to-end interactive data science application built with **Streamlit**, **Pandas**, and **Plotly**. This platform unifies multi-format data ingestion, automated structural validation, advanced data preprocessing, dynamically updating executive KPIs, interactive multi-criteria filtering, and analytical visualizations into a single cohesive dashboard.

---

## 🌟 Key Application Features

### 1. Data Upload & Validation
- **Multi-Format Support**: Seamlessly ingest datasets in **CSV (`.csv`)**, **Excel (`.xlsx`, `.xls`)**, and **JSON (`.json`)** formats.
- **Built-in Benchmark Fallback**: Automatically loads the lab benchmark dataset (`experiment10_raw_dataset.csv`) on startup so users can immediately interact with the dashboard.
- **Structural Validation Report**:
  - Dataset dimensions (rows × columns)
  - Complete list of column names and detected data types
  - Detailed missing value audit (counts and percentages per feature)
  - Duplicate record detection and full preview of duplicate rows

### 2. Data Preprocessing & Cleaning Pipeline
- **Duplicate Handling**: Identifies and removes duplicate records while logging total dropped instances.
- **Missing Value Imputation**: Configurable numeric imputation (Median, Mean, Zero) and categorical mode/unknown imputation.
- **Inconsistent Data Standardisation**:
  - Normalizes whitespace, casing, and synonym variations across categories (e.g., `' technology '` / `'Office supply'` $\rightarrow$ `'Technology'` / `'Office Supplies'`).
  - Standardizes geographic regions (`'SOUTH'` / `'west'` $\rightarrow$ `'South'` / `'West'`).
  - Converts textual number words and symbols (`'five'` $\rightarrow$ 5.0, `'ten percent'` $\rightarrow$ 0.10, `'inf'`, `'?'`, `'unknown'` $\rightarrow$ NaN).
- **Domain Anomaly Correction**:
  - Corrects negative sales and quantities to valid absolute values.
  - Fixes erroneous discounts $> 100\%$ (e.g., $1.75 \rightarrow 0.175$).
  - Caps extreme quantity outliers ($> 50$) and sales outliers ($> \$50,000$).
- **Date Format Handling & Temporal Engineering**:
  - Parses mixed datetime formats (e.g., `YYYY-MM-DD` and `DD/MM/YYYY`) using `pd.to_datetime(format='mixed')`.
  - Drops invalid corrupt date tokens (e.g., `'not-a-date'`).
  - Engineers temporal features: `Order Year`, `Order Month`, `Order Year-Month`, `Order Day of Week`, `Order Quarter`, and delivery duration `Shipping Days` (`Ship Date - Order Date`).

### 3. Dynamic KPIs & Interactive Multi-Criteria Filters
- **5 Dynamically Updating KPIs**:
  1. 💰 **Total Sales**: Formatted currency sum of all filtered transactions.
  2. 📈 **Total Profit**: Net profitability across filtered records.
  3. 📦 **Total Orders**: Count of unique `Order ID`s.
  4. 🏷️ **Total Quantity Sold**: Sum of all units sold.
  5. 📊 **Profit Margin (%)**: $\frac{\text{Total Profit}}{\text{Total Sales}} \times 100$.
- **Secondary Executive Metrics**: Average Order Value (AOV) and Average Discount (%).
- **Interactive Filters**:
  - 📅 **Date Range Filter**: Dynamic start/end date picker constrained by dataset min/max dates.
  - 🏷️ **Category Filter**: Multi-select dropdown with "Select All" toggle.
  - 🗺️ **Region Filter**: Multi-select dropdown for geographic regions (East, West, South, Central).
  - 👥 **Customer Segment Filter**: Filter by Consumer, Corporate, and Home Office.
  - 🚚 **Shipping Mode Filter**: Filter by Standard Class, First Class, Second Class, and Same Day.

### 4. Interactive Visualizations
Created using **Plotly** with responsive layouts, tooltips, and themes:
1. **Sales & Profit Trend over Time**: Dual-axis combination chart (Monthly / Quarterly / Weekly).
2. **Category Performance**: Grouped horizontal bar chart comparing Sales vs Profit.
3. **Sub-Category Breakdown**: Stacked bar chart showing sales contributions with category tags.
4. **Regional Revenue & Profit Margin**: Bar chart overlaying profit margin % onto total sales.
5. **Transaction Matrix (Sales vs Profit)**: Scatter bubble chart sized by Quantity, colored by Category, with a Breakeven ($0 Profit) reference line.
6. **Customer Segment Donut**: Donut chart displaying revenue share across customer segments.
7. **Shipping Duration Distribution**: Box plot of delivery times across shipping modes.

### 5. Statistical Summaries & Cleaned Data Export
- Comprehensive numerical descriptive statistics table (Count, Mean, Std Dev, Min, Q1, Median, Q3, Max, IQR, Skewness, Kurtosis).
- Categorical frequency distributions and dominance share (%).
- Pearson Correlation Matrix across all numerical indicators.
- **Export Formats**: One-click download of cleaned data as **CSV**, **Excel (`.xlsx`)**, or **JSON**.

---

## 🏗️ Project Architecture

```
DS Lab 10/
├── app.py                      # Main Streamlit Dashboard Application
├── requirements.txt            # Production dependencies
├── README.md                   # Documentation and deployment manual
├── experiment10_raw_dataset.csv# Lab benchmark raw dirty dataset
├── sample_dataset.xlsx         # Excel sample file
├── sample_dataset.json         # JSON sample file
├── modules/
│   ├── __init__.py
│   ├── data_loader.py          # Ingestion (CSV, XLSX, JSON) & Structural Validation
│   ├── preprocessor.py         # Cleaning, Imputation, Text Normalization & Dates
│   ├── analytics.py            # Reusable statistical summary & KPI calculations
│   └── visualizations.py       # Plotly interactive chart generators
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py        # Automated Pytest suite
└── .streamlit/
    └── config.toml             # Streamlit server and theme configurations
```

---

## 🚀 Local Installation & Execution

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DSingh0304/ds-lab-10-sales-analytics.git
   cd ds-lab-10-sales-analytics
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run unit tests**:
   ```bash
   pytest -v
   ```

4. **Launch the application**:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

This application is ready for 1-click deployment on [Streamlit Community Cloud](https://share.streamlit.io):
1. Repository: `DSingh0304/ds-lab-10-sales-analytics`
2. Branch: `main`
3. Main file path: `app.py`

Direct 1-Click Deploy Link:
[Deploy to Streamlit Cloud](https://share.streamlit.io/deploy?repository=DSingh0304/ds-lab-10-sales-analytics&branch=main&mainModule=app.py)

---

## 🧪 Automated Testing Results

All 8 modular test suites pass cleanly:
- `test_load_csv_and_validation` ✅
- `test_load_excel` ✅
- `test_load_json` ✅
- `test_preprocessing_pipeline` ✅
- `test_kpis_calculation` ✅
- `test_statistical_summaries` ✅
- `test_filtering_and_dynamic_kpis` ✅
- `test_visualizations` ✅

