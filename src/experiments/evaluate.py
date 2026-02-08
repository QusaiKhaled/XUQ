"""Evaluation and metrics aggregation."""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ..utils.metrics import compute_regression_metrics


def evaluate_model(model, X_test, y_test, predict_fn=None):
    """
    Generic evaluator: model must have predict(X) or pass predict_fn(model, X).
    Returns dict with mse, rmse, mae, mape, r2.
    """
    if predict_fn is not None:
        y_pred = predict_fn(model, X_test)
    else:
        y_pred = model.predict(X_test)
    return compute_regression_metrics(y_test, y_pred)
