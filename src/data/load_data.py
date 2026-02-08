"""CSV importers and simple validators."""

import pandas as pd
from pathlib import Path

from ..config import RAW_DIR, DEFAULT_CSV_NAME


def load_csv(path=None) -> pd.DataFrame:
    """
    Load the Melbourne wastewater treatment dataset from CSV.

    Parameters
    ----------
    path : str or Path, optional
        Full path to CSV. If None, uses config RAW_DIR and DEFAULT_CSV_NAME.

    Returns
    -------
    pd.DataFrame
        Raw dataframe.
    """
    if path is None:
        path = RAW_DIR / DEFAULT_CSV_NAME
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    return df
