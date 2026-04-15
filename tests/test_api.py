# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.model_loader import model_loader

@pytest.fixture(autouse=True, scope="session")
def load_model():
    """Load model once before all API tests."""
    model_loader.load()

client = TestClient(app)

VALID_PAYLOAD = {
    "CODE_GENDER": "M",
    "DAYS_BIRTH": -12000,
    "CNT_CHILDREN": 1,
    "CNT_FAM_MEMBERS": 3.0,
    "AMT_INCOME_TOTAL": 135000.0,
    "AMT_CREDIT": 450000.0,
    "AMT_ANNUITY": 22500.0,
    "AMT_GOODS_PRICE": 400000.0,
    "DAYS_EMPLOYED": -2000.0,
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "NAME_CONTRACT_TYPE": "Cash loans",
    "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
    "REGION_RATING_CLIENT": 2,
    "DAYS_LAST_PHONE_CHANGE": -300.0
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_predict_valid_input():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "probability_of_default" in data
    assert "risk_class" in data
    assert "risk_score" in data
    assert "model_version" in data
    assert 0.0 <= data["probability_of_default"] <= 1.0
    assert data["risk_class"] in ["low", "medium", "high"]
    assert 300 <= data["risk_score"] <= 850


def test_predict_invalid_gender():
    payload = VALID_PAYLOAD.copy()
    payload["CODE_GENDER"] = "X"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_days_birth():
    payload = VALID_PAYLOAD.copy()
    payload["DAYS_BIRTH"] = 100
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_region_rating():
    payload = VALID_PAYLOAD.copy()
    payload["REGION_RATING_CLIENT"] = 5
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_required_field():
    payload = VALID_PAYLOAD.copy()
    del payload["AMT_INCOME_TOTAL"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_low_risk_applicant():
    payload = VALID_PAYLOAD.copy()
    payload["AMT_INCOME_TOTAL"] = 500000.0
    payload["AMT_CREDIT"] = 100000.0
    payload["DAYS_EMPLOYED"] = -5000.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["probability_of_default"] < 0.6


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data