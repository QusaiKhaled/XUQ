"""R², MSE, and related regression helpers."""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_regression_metrics(y_true, y_pred):
    """
    Compute MSE, RMSE, MAE, MAPE, R².

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like

    Returns
    -------
    dict
        Keys: mse, rmse, mae, mape, r2.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}
