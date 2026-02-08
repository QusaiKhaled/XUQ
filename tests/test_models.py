"""Tests for IT2-ANFIS and baseline models."""

import pytest
import numpy as np
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.it2_anfis import IT2_TSK_ANFIS
from src.models.anfis_zero import ANFISZero
from src.models.anfis_first import ANFISFirst


@pytest.fixture
def small_data():
    np.random.seed(42)
    X = np.random.randn(80, 3)
    y = 2 * X[:, 0] + 0.5 * X[:, 1] + np.random.randn(80) * 0.1
    return X, y


def test_it2_predict_shape(small_data):
    X, y = small_data
    model = IT2_TSK_ANFIS(n_rules=3, X=X[:50], verbose=False)
    pred = model.predict(X[50:])
    assert pred.shape == (30,)


def test_it2_predict_interval_shape(small_data):
    X, y = small_data
    model = IT2_TSK_ANFIS(n_rules=3, X=X[:50], verbose=False)
    low, high = model.predict_interval(X[50:])
    assert low.shape == high.shape == (30,)


def test_anfis_zero_predict(small_data):
    X, y = small_data
    X_tr, X_te = X[:50], X[50:]
    y_tr, y_te = y[:50], y[50:]
    model = ANFISZero(n_clusters=3, X=X_tr)
    model.fit(X_tr, y_tr, X_te, y_te, epochs=5, patience=3, verbose=False)
    pred = model.predict(X_te)
    assert pred.shape == (30,)


def test_anfis_first_predict(small_data):
    X, y = small_data
    X_tr, X_te = X[:50], X[50:]
    y_tr, y_te = y[:50], y[50:]
    model = ANFISFirst(n_clusters=3, X=X_tr)
    model.fit(X_tr, y_tr, X_te, y_te, epochs=5, patience=3, verbose=False)
    pred = model.predict(X_te)
    assert pred.shape == (30,)
