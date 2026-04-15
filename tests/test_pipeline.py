# tests/test_pipeline.py
import pytest
import numpy as np
import joblib
from pathlib import Path


def test_processed_data_exists():
    assert Path("models/processed_data.pkl").exists()


def test_preprocessor_exists():
    assert Path("models/preprocessor.pkl").exists()


def test_xgboost_model_exists():
    assert Path("models/xgboost.pkl").exists()


def test_logistic_regression_model_exists():
    assert Path("models/logistic_regression.pkl").exists()


def test_processed_data_shapes():
    X_train, X_test, y_train, y_test = joblib.load("models/processed_data.pkl")
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]


def test_target_is_binary():
    _, _, y_train, y_test = joblib.load("models/processed_data.pkl")
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


def test_train_test_ratio():
    X_train, X_test, y_train, y_test = joblib.load("models/processed_data.pkl")
    total = X_train.shape[0] + X_test.shape[0]
    test_ratio = X_test.shape[0] / total
    assert 0.18 <= test_ratio <= 0.22


def test_class_imbalance_preserved():
    _, _, y_train, y_test = joblib.load("models/processed_data.pkl")
    train_default_rate = y_train.mean()
    test_default_rate = y_test.mean()
    assert abs(train_default_rate - test_default_rate) < 0.01


def test_xgboost_predicts():
    X_train, X_test, y_train, y_test = joblib.load("models/processed_data.pkl")
    model = joblib.load("models/xgboost.pkl")
    proba = model.predict_proba(X_test[:10])
    assert proba.shape == (10, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_preprocessor_transform():
    X_train, X_test, y_train, y_test = joblib.load("models/processed_data.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    assert hasattr(preprocessor, "transform")
    assert hasattr(preprocessor, "fit_transform")