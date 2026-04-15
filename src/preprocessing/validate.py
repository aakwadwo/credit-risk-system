# src/preprocessing/validate.py
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import logging

logger = logging.getLogger(__name__)


def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the raw application_train dataframe.
    Checks types, value ranges, and required columns.
    """
    schema = DataFrameSchema(
        columns={
            "TARGET": Column(
                int,
                checks=Check.isin([0, 1]),
                nullable=False
            ),
            "SK_ID_CURR": Column(
                int,
                checks=Check.greater_than(0),
                nullable=False
            ),
            "AMT_INCOME_TOTAL": Column(
                float,
                checks=Check.greater_than(0),
                nullable=True
            ),
            "AMT_CREDIT": Column(
                float,
                checks=Check.greater_than(0),
                nullable=True
            ),
            "AMT_ANNUITY": Column(
                float,
                checks=Check.greater_than(0),
                nullable=True
            ),
            "DAYS_BIRTH": Column(
                int,
                checks=Check.less_than(0),
                nullable=True
            ),
            "DAYS_EMPLOYED": Column(
                int,
                nullable=True
            ),
        },
        checks=Check(
            lambda df: df.shape[0] > 0,
            error="Dataframe is empty"
        ),
        coerce=True
    )

    try:
        validated_df = schema.validate(df, lazy=True)
        logger.info("Data validation passed successfully")
        return validated_df
    except pa.errors.SchemaErrors as e:
        logger.warning(f"Validation found {len(e.failure_cases)} issue(s) — proceeding with caution")
        logger.warning(f"\n{e.failure_cases[['column', 'check', 'count']].drop_duplicates()}")
        return df


if __name__ == "__main__":
    from src.ingestion.ingest import load_raw_data
    from src.config import CONFIG

    df = load_raw_data(CONFIG["data"]["main_file"])
    validated = validate_raw_data(df)
    print(f"Validation complete. Shape: {validated.shape}")