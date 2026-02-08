"""Mutual information-based feature selection and utilities."""

import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from ..config import TARGET_COLUMN, MI_RANDOM_STATE, MIN_MI_THRESHOLD


def compute_mi_scores(X: pd.DataFrame, y: pd.Series, random_state=None) -> pd.Series:
    """
    Compute mutual information scores between each feature and target.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (no target).
    y : pd.Series
        Target vector.
    random_state : int, optional
        Random state for MI (default from config).

    Returns
    -------
    pd.Series
        MI scores indexed by feature name, sorted ascending.
    """
    random_state = random_state if random_state is not None else MI_RANDOM_STATE
    mi_scores = mutual_info_regression(X, y, random_state=random_state)
    return pd.Series(mi_scores, index=X.columns).sort_values(ascending=True)


def select_features_mi(
    df: pd.DataFrame,
    target=None,
    min_mi=None,
) -> pd.DataFrame:
    """
    Keep only features with mutual information above threshold; target always kept.

    Parameters
    ----------
    df : pd.DataFrame
        Clean dataframe including target column.
    target : str, optional
        Target column name (default from config).
    min_mi : float, optional
        Minimum MI to keep (default from config, 0 = drop only zero-MI).

    Returns
    -------
    pd.DataFrame
        Subset of columns (selected features + target).
    """
    target = target or TARGET_COLUMN
    min_mi = min_mi if min_mi is not None else MIN_MI_THRESHOLD

    X = df.drop(columns=[target])
    y = df[target]
    mi_series = compute_mi_scores(X, y)
    features_to_keep = mi_series[mi_series > min_mi].index.tolist()
    return df[features_to_keep + [target]].copy()
