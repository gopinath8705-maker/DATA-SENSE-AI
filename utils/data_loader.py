"""
utils/data_loader.py
Handles file upload and dataset loading for CSV and Excel files.
"""

import pandas as pd
import streamlit as st
from io import BytesIO
from typing import Tuple, Optional


def load_dataset(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load a dataset from an uploaded file object.

    Supports:
        - CSV (.csv)
        - Excel (.xlsx, .xls)

    Returns:
        (DataFrame, None)  on success
        (None, error_msg)  on failure
    """
    if uploaded_file is None:
        return None, "No file provided."

    filename = uploaded_file.name.lower()

    try:
        file_bytes = BytesIO(uploaded_file.read())

        if filename.endswith(".csv"):
            # Try common encodings
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    file_bytes.seek(0)
                    df = pd.read_csv(file_bytes, encoding=encoding)
                    return df, None
                except UnicodeDecodeError:
                    continue
            return None, "Could not decode CSV file. Try saving as UTF-8."

        elif filename.endswith((".xlsx", ".xls")):
            file_bytes.seek(0)
            df = pd.read_excel(file_bytes)
            return df, None

        else:
            return None, f"Unsupported file format: '{filename}'. Please upload a CSV or Excel file."

    except pd.errors.EmptyDataError:
        return None, "The file is empty or has no data."
    except pd.errors.ParserError as e:
        return None, f"Failed to parse file: {e}"
    except Exception as e:
        return None, f"Unexpected error loading file: {e}"
