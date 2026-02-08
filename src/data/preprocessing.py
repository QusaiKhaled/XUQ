"""Cleaning, NaN handling, and feature pipeline."""

import pandas as pd

from ..config import DROP_COLUMNS, TARGET_COLUMN


def clean_and_prepare(df: pd.DataFrame, drop_columns=None, target_column=None) -> pd.DataFrame:
    """
    Remove redundant columns, move target to end, and optionally drop rows with NaNs.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.
    drop_columns : list, optional
        Columns to drop (default from config: e.g. 'Unnamed: 0').
    target_column : str, optional
        Target column name (default from config: 'Energy Consumption').

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with target as last column.
    """
    drop_columns = drop_columns or DROP_COLUMNS
    target_column = target_column or TARGET_COLUMN

    df_clean = df.drop(columns=[c for c in drop_columns if c in df.columns], errors="ignore")
    if target_column in df_clean.columns:
        target_series = df_clean.pop(target_column)
        df_clean[target_column] = target_series
    # Drop rows with any NaN if desired (dataset had no NaNs in original notebook)
    # df_clean = df_clean.dropna()
    return df_clean
