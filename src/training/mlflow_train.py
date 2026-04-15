# src/training/mlflow_train.py
import numpy as np
import joblib
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from scipy.stats import ks_2samp
from src.config import CONFIG
from src.evaluation.evaluate import compute_ks_statistic, compute_gini, evaluate_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")


def setup_mlflow():
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
    mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])
    logger.info(f"MLflow tracking uri: {CONFIG['mlflow']['tracking_uri']}")
    logger.info(f"MLflow experiment: {CONFIG['mlflow']['experiment_name']}")


def log_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and log Logistic Regression with MLflow."""
    params = {
        "model_type": "LogisticRegression",
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": CONFIG["model"]["random_state"]
    }

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        mlflow.log_params(params)
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])

        model = LogisticRegression(
            max_iter=params["max_iter"],
            random_state=params["random_state"],
            class_weight=params["class_weight"],
            solver=params["solver"],
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, "Logistic Regression")

        mlflow.log_metric("auc_roc", metrics["auc_roc"])
        mlflow.log_metric("gini", metrics["gini"])
        mlflow.log_metric("ks_statistic", metrics["ks_statistic"])
        mlflow.log_metric("avg_precision", metrics["avg_precision"])
        mlflow.log_metric("recall_default", metrics["recall_default"])
        mlflow.log_metric("f1_default", metrics["f1_default"])
        mlflow.log_metric("precision_default", metrics["precision_default"])

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="credit_risk_lr"
        )

        joblib.dump(model, "models/logistic_regression.pkl")
        mlflow.log_artifact("models/logistic_regression.pkl")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"LR run logged — Run ID: {run_id}")
        return metrics, run_id


def log_xgboost(X_train, X_test, y_train, y_test):
    """Train and log XGBoost with MLflow."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    params = {
        "model_type": "XGBoost",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": round(scale_pos_weight, 4),
        "random_state": CONFIG["model"]["random_state"]
    }

    with mlflow.start_run(run_name="xgboost_challenger"):
        mlflow.log_params(params)
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])

        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            scale_pos_weight=scale_pos_weight,
            random_state=params["random_state"],
            eval_metric="auc",
            early_stopping_rounds=30,
            n_jobs=-1,
            verbosity=0
        )

        eval_set = [(X_train, y_train)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        mlflow.log_param("best_iteration", model.best_iteration)

        metrics = evaluate_model(model, X_test, y_test, "XGBoost")

        mlflow.log_metric("auc_roc", metrics["auc_roc"])
        mlflow.log_metric("gini", metrics["gini"])
        mlflow.log_metric("ks_statistic", metrics["ks_statistic"])
        mlflow.log_metric("avg_precision", metrics["avg_precision"])
        mlflow.log_metric("recall_default", metrics["recall_default"])
        mlflow.log_metric("f1_default", metrics["f1_default"])
        mlflow.log_metric("precision_default", metrics["precision_default"])

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="credit_risk_xgb"
        )

        joblib.dump(model, "models/xgboost.pkl")
        mlflow.log_artifact("models/xgboost.pkl")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"XGBoost run logged — Run ID: {run_id}")
        return metrics, run_id


def run_mlflow_training():
    """Run both experiments and log everything to MLflow."""
    setup_mlflow()

    X_train, X_test, y_train, y_test = joblib.load("models/processed_data.pkl")

    logger.info("Starting MLflow experiment logging...")

    lr_metrics, lr_run_id = log_logistic_regression(X_train, X_test, y_train, y_test)
    xgb_metrics, xgb_run_id = log_xgboost(X_train, X_test, y_train, y_test)

    print(f"\n{'='*60}")
    print(f"MLflow Experiment: {CONFIG['mlflow']['experiment_name']}")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'LR':>15} {'XGBoost':>15}")
    print(f"{'-'*60}")
    for metric in ["auc_roc", "gini", "ks_statistic", "avg_precision", "recall_default"]:
        print(f"{metric:<25} {lr_metrics[metric]:>15} {xgb_metrics[metric]:>15}")
    print(f"{'-'*60}")
    print(f"LR Run ID:      {lr_run_id}")
    print(f"XGBoost Run ID: {xgb_run_id}")
    print(f"{'='*60}")
    print(f"\nView UI: mlflow ui --backend-store-uri mlruns")


if __name__ == "__main__":
    run_mlflow_training()