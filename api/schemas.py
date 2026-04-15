# api/schemas.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class LoanApplication(BaseModel):
    """Input schema for a single loan application prediction request."""

    # Personal info
    CODE_GENDER: str = Field(..., description="Gender: M or F")
    DAYS_BIRTH: int = Field(..., description="Age in days (negative integer)")
    CNT_CHILDREN: int = Field(..., description="Number of children")
    CNT_FAM_MEMBERS: float = Field(..., description="Number of family members")

    # Financial info
    AMT_INCOME_TOTAL: float = Field(..., description="Total annual income")
    AMT_CREDIT: float = Field(..., description="Loan credit amount")
    AMT_ANNUITY: float = Field(..., description="Loan annuity amount")
    AMT_GOODS_PRICE: Optional[float] = Field(None, description="Price of goods for loan")

    # Employment info
    DAYS_EMPLOYED: Optional[float] = Field(None, description="Days employed (negative) or None if unemployed")
    NAME_INCOME_TYPE: str = Field(..., description="Income type e.g. Working, Pensioner")
    NAME_EDUCATION_TYPE: str = Field(..., description="Education level")
    NAME_FAMILY_STATUS: str = Field(..., description="Family status")
    NAME_HOUSING_TYPE: str = Field(..., description="Housing type")

    # Loan info
    NAME_CONTRACT_TYPE: str = Field(..., description="Cash loans or Revolving loans")
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = Field(None, description="Credit bureau enquiries last year")
    REGION_RATING_CLIENT: int = Field(..., description="Region rating 1-3")
    DAYS_LAST_PHONE_CHANGE: Optional[float] = Field(None, description="Days since phone change")

    @field_validator("CODE_GENDER")
    @classmethod
    def gender_must_be_valid(cls, v):
        if v not in ["M", "F"]:
            raise ValueError("CODE_GENDER must be M or F")
        return v

    @field_validator("DAYS_BIRTH")
    @classmethod
    def birth_must_be_negative(cls, v):
        if v >= 0:
            raise ValueError("DAYS_BIRTH must be negative (days before today)")
        return v

    @field_validator("REGION_RATING_CLIENT")
    @classmethod
    def region_rating_valid(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError("REGION_RATING_CLIENT must be 1, 2, or 3")
        return v

    model_config = {"protected_namespaces": ()}


class PredictionResponse(BaseModel):
    """Output schema for prediction results."""
    probability_of_default: float = Field(..., description="Probability of loan default (0-1)")
    risk_class: str = Field(..., description="Risk classification: low, medium, high")
    risk_score: int = Field(..., description="Risk score 300-850 (higher is safer)")
    model_version: str = Field(..., description="Model used for prediction")

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    mlflow_version: str