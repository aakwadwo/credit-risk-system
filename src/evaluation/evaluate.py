# src/evaluation/evaluate.py
import numpy as np
import joblib
import logging
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")


def compute_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute the Kolmogorov-Smirnov statistic.
    Standard metric in credit risk scoring — measures separation
    between default and non-default score distributions.
    """
    scores_default = y_proba[y_true == 1]
    scores_non_default = y_proba[y_true == 0]
    ks_stat, _ = ks_2samp(scores_default, scores_non_default)
    return ks_stat


def compute_gini(auc: float) -> float:
    """Gini coefficient — standard credit risk metric. Gini = 2*AUC - 1."""
    return 2 * auc - 1


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """Compute full evaluation metrics for a trained model."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    ks = compute_ks_statistic(y_test, y_proba)
    gini = compute_gini(auc)

    report = classification_report(y_test, y_pred, target_names=["No Default", "Default"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "model_name": model_name,
        "auc_roc": round(auc, 4),
        "gini": round(gini, 4),
        "ks_statistic": round(ks, 4),
        "avg_precision": round(avg_precision, 4),
        "precision_default": round(report["Default"]["precision"], 4),
        "recall_default": round(report["Default"]["recall"], 4),
        "f1_default": round(report["Default"]["f1-score"], 4),
        "confusion_matrix": cm.tolist()
    }

    logger.info(f"\n{model_name} Evaluation:")
    logger.info(f"  AUC-ROC:      {metrics['auc_roc']}")
    logger.info(f"  Gini:         {metrics['gini']}")
    logger.info(f"  KS Statistic: {metrics['ks_statistic']}")
    logger.info(f"  Avg Precision:{metrics['avg_precision']}")
    logger.info(f"  Recall(Default): {metrics['recall_default']}")

    return metrics


def run_evaluation() -> dict:
    """Load both models and evaluate them on the test set."""
    X_train, X_test, y_train, y_test = joblib.load("models/processed_data.pkl")

    lr_model = joblib.load("models/logistic_regression.pkl")
    xgb_model = joblib.load("models/xgboost.pkl")

    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    return {"logistic_regression": lr_metrics, "xgboost": xgb_metrics}


if __name__ == "__main__":
    results = run_evaluation()

    print(f"\n{'='*60}")
    print(f"{'Metric':<25} {'Logistic Regression':>20} {'XGBoost':>10}")
    print(f"{'-'*60}")
    metrics_to_show = ["auc_roc", "gini", "ks_statistic", "avg_precision", "recall_default", "f1_default"]
    for metric in metrics_to_show:
        lr_val = results["logistic_regression"][metric]
        xgb_val = results["xgboost"][metric]
        print(f"{metric:<25} {lr_val:>20} {xgb_val:>10}")
    print(f"{'='*60}\n")