"""Tests for data loading and preprocessing."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path when running tests
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load_data import load_csv
from src.data.preprocessing import clean_and_prepare
from src.config import RAW_DIR, TARGET_COLUMN, DROP_COLUMNS


def test_clean_and_prepare_keeps_target_last():
    """Clean moves target to last column."""
    df = pd.DataFrame({
        "A": [1, 2], "B": [3, 4], "Energy Consumption": [10, 20]
    })
    out = clean_and_prepare(df, drop_columns=[], target_column="Energy Consumption")
    assert list(out.columns)[-1] == "Energy Consumption"
    assert "A" in out.columns and "B" in out.columns


def test_clean_drops_unnamed():
    """Clean drops Unnamed: 0 if present."""
    df = pd.DataFrame({
        "Unnamed: 0": [0, 1], "x": [1, 2], "Energy Consumption": [5, 10]
    })
    out = clean_and_prepare(df)
    assert "Unnamed: 0" not in out.columns


def test_load_csv_missing_file():
    """load_csv raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError):
        load_csv(Path("/nonexistent/file.csv"))
