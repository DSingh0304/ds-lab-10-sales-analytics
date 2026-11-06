"""
Data Loader & Validation Module
Handles multi-format ingestion (CSV, XLSX, JSON) and comprehensive structural validation.
"""

import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd


SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls", ".json"]


def load_data(
    source: Union[str, Path, io.BytesIO, Any],
    file_type: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load data from a file path, BytesIO, or Streamlit UploadedFile object.
    Supports CSV, Excel (.xlsx, .xls), and JSON (.json).

    Parameters:
        source: File path, Path object, or file-like buffer (UploadedFile / BytesIO).
        file_type: Optional explicit file type override ('csv', 'xlsx', 'json').

    Returns:
        Tuple of (DataFrame, error_message). If successful, error_message is None.
    """
    try:
        # Determine file type
        ext = ""
        if file_type:
            ext = f".{file_type.lower().lstrip('.')}"
        elif hasattr(source, "name") and source.name:
            ext = Path(source.name).suffix.lower()
        elif isinstance(source, (str, Path)):
            ext = Path(source).suffix.lower()

        if not ext and hasattr(source, "type"):
            # Guess from MIME type if available
            mime = source.type
            if "csv" in mime:
                ext = ".csv"
            elif "excel" in mime or "spreadsheet" in mime:
                ext = ".xlsx"
            elif "json" in mime:
                ext = ".json"

        if ext not in SUPPORTED_EXTENSIONS:
            return None, f"Unsupported file extension '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"

        # Reset pointer if it's a file-like buffer
        if hasattr(source, "seek"):
            source.seek(0)

        # Ingest according to extension
        if ext == ".csv":
            try:
                df = pd.read_csv(source)
            except UnicodeDecodeError:
                if hasattr(source, "seek"):
                    source.seek(0)
                df = pd.read_csv(source, encoding="latin1")
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(source)
        elif ext == ".json":
            df = pd.read_json(source)
        else:
            return None, f"Unsupported format: {ext}"

        if df is None or df.empty:
            return None, "Uploaded file contains an empty dataset (0 rows)."

        return df, None

    except Exception as e:
        return None, f"Error parsing data file: {str(e)}"


def validate_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform structural and data quality validation on the uploaded dataset.

    Parameters:
        df: Input raw DataFrame.

    Returns:
        Dictionary containing dimensions, column names, data types, missing value
        counts/percentages, and duplicate record details.
    """
    num_rows, num_cols = df.shape
    columns = list(df.columns)
    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

    missing_series = df.isnull().sum()
    missing_pct_series = (missing_series / max(num_rows, 1)) * 100.0

    total_cells = num_rows * num_cols
    total_missing_cells = int(missing_series.sum())
    missing_cell_percentage = (total_missing_cells / max(total_cells, 1)) * 100.0

    duplicate_mask = df.duplicated(keep=False)
    duplicate_count = int(df.duplicated(keep="first").sum())
    duplicate_rows = df[duplicate_mask].copy()

    # Create a consolidated validation summary dataframe
    validation_table = pd.DataFrame(
        {
            "Column Name": columns,
            "Data Type": [dtypes_dict[c] for c in columns],
            "Non-Null Count": [num_rows - int(missing_series[c]) for c in columns],
            "Missing Count": [int(missing_series[c]) for c in columns],
            "Missing (%)": [round(missing_pct_series[c], 2) for c in columns],
            "Unique Values": [int(df[c].nunique(dropna=True)) for c in columns],
            "Sample Value": [str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else "N/A" for c in columns],
        }
    )

    return {
        "dimensions": (num_rows, num_cols),
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": columns,
        "dtypes": dtypes_dict,
        "missing_counts": missing_series.to_dict(),
        "missing_percentages": missing_pct_series.round(2).to_dict(),
        "total_missing_cells": total_missing_cells,
        "missing_cell_percentage": round(missing_cell_percentage, 2),
        "duplicate_count": duplicate_count,
        "duplicate_rows": duplicate_rows,
        "validation_table": validation_table,
        "is_valid": num_rows > 0 and num_cols > 0,
    }
