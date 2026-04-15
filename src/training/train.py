# src/training/train.py
import numpy as np
import joblib
import logging
import time
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from src.config import CONFIG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")


def load_processed_data() -> tuple:
    """Load the preprocessed train/test arrays from disk."""
    data = joblib.load("models/processed_data.pkl")
    X_train, X_test, y_train, y_test = data
    logger.info(f"Loaded processed data — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute class weight ratio for XGBoost to handle imbalance."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    ratio = n_neg / n_pos
    logger.info(f"Class imbalance ratio (scale_pos_weight): {ratio:.2f}")
    return ratio


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a logistic regression baseline model."""
    logger.info("Training Logistic Regression baseline...")
    start = time.time()

    model = LogisticRegression(
        max_iter=1000,
        random_state=CONFIG["model"]["random_state"],
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info(f"Logistic Regression trained in {elapsed:.1f}s")
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Train an XGBoost classifier."""
    logger.info("Training XGBoost classifier...")
    start = time.time()

    scale_pos_weight = compute_scale_pos_weight(y_train)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=CONFIG["model"]["random_state"],
        eval_metric="auc",
        early_stopping_rounds=30,
        n_jobs=-1,
        verbosity=0
    )

    eval_set = [(X_train, y_train)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    elapsed = time.time() - start
    logger.info(f"XGBoost trained in {elapsed:.1f}s | Best iteration: {model.best_iteration}")
    return model


def save_model(model, name: str):
    """Save a trained model to the models directory."""
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.pkl"
    joblib.dump(model, path)
    logger.info(f"Saved {name} to {path}")


def run_training() -> dict:
    """Train both models and return them with their train AUC scores."""
    X_train, X_test, y_train, y_test = load_processed_data()

    lr_model = train_logistic_regression(X_train, y_train)
    lr_train_auc = roc_auc_score(y_train, lr_model.predict_proba(X_train)[:, 1])
    logger.info(f"LR Train AUC: {lr_train_auc:.4f}")
    save_model(lr_model, "logistic_regression")

    xgb_model = train_xgboost(X_train, y_train)
    xgb_train_auc = roc_auc_score(y_train, xgb_model.predict_proba(X_train)[:, 1])
    logger.info(f"XGBoost Train AUC: {xgb_train_auc:.4f}")
    save_model(xgb_model, "xgboost")

    return {
        "logistic_regression": {"model": lr_model, "train_auc": lr_train_auc},
        "xgboost": {"model": xgb_model, "train_auc": xgb_train_auc},
        "X_test": X_test,
        "y_test": y_test
    }


if __name__ == "__main__":
    results = run_training()
    print(f"\n{'='*50}")
    print(f"Logistic Regression Train AUC: {results['logistic_regression']['train_auc']:.4f}")
    print(f"XGBoost Train AUC:             {results['xgboost']['train_auc']:.4f}")
    print(f"{'='*50}\n")