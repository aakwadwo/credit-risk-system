# src/monitoring/monitor.py
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI).
    PSI < 0.1  : No significant change
    PSI 0.1-0.2: Moderate change, monitor closely
    PSI > 0.2  : Significant change, investigate
    """
    expected_pct, bin_edges = np.histogram(expected, bins=bins)
    actual_pct, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_pct / len(expected)
    actual_pct = actual_pct / len(actual)

    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def detect_score_drift(reference_proba: np.ndarray, current_proba: np.ndarray) -> dict:
    """
    Detect drift in model output scores between reference and current window.
    Uses PSI and KS test.
    """
    psi = compute_psi(reference_proba, current_proba)
    ks_stat, ks_pvalue = ks_2samp(reference_proba, current_proba)

    psi_status = "stable" if psi < 0.1 else "warning" if psi < 0.2 else "alert"
    ks_status = "stable" if ks_pvalue > 0.05 else "alert"

    result = {
        "psi": psi,
        "psi_status": psi_status,
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_pvalue), 4),
        "ks_status": ks_status,
        "overall_status": "alert" if "alert" in [psi_status, ks_status] else
                          "warning" if psi_status == "warning" else "stable"
    }

    logger.info(f"Drift detection — PSI: {psi} ({psi_status}) | KS: {ks_stat:.4f} ({ks_status})")
    return result


def simulate_drift_report():
    """
    Simulate a drift detection report using train data as reference
    and a perturbed sample as the current window.
    This demonstrates the monitoring capability for the exam.
    """
    X_train, X_test, y_train, y_test = joblib.load("models/processed_data.pkl")
    model = joblib.load("models/xgboost.pkl")

    reference_proba = model.predict_proba(X_train[:5000])[:, 1]
    current_proba_stable = model.predict_proba(X_test[:1000])[:, 1]
    current_proba_drifted = current_proba_stable * 1.8
    current_proba_drifted = np.clip(current_proba_drifted, 0, 1)

    print(f"\n{'='*60}")
    print(f"DRIFT MONITORING REPORT")
    print(f"{'='*60}")

    print(f"\n--- Scenario 1: Stable deployment (no drift) ---")
    stable_result = detect_score_drift(reference_proba, current_proba_stable)
    for k, v in stable_result.items():
        print(f"  {k}: {v}")

    print(f"\n--- Scenario 2: Drifted deployment (alert) ---")
    drifted_result = detect_score_drift(reference_proba, current_proba_drifted)
    for k, v in drifted_result.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*60}")
    print(f"PSI Thresholds: <0.1 stable | 0.1-0.2 warning | >0.2 alert")
    print(f"KS p-value: >0.05 stable | <=0.05 alert")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    simulate_drift_report()