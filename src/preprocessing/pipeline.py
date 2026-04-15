# src/preprocessing/pipeline.py
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.config import CONFIG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")


KNOWN_ANOMALIES = {
    "DAYS_EMPLOYED": 365243
}


def fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Replace known data entry anomalies with NaN."""
    df = df.copy()
    for col, bad_value in KNOWN_ANOMALIES.items():
        if col in df.columns:
            n = (df[col] == bad_value).sum()
            if n > 0:
                logger.info(f"Replacing {n:,} anomalous values in {col} with NaN")
                df[col] = df[col].replace(bad_value, np.nan)
    return df


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """Drop columns where more than threshold% of values are missing."""
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    logger.info(f"Dropping {len(cols_to_drop)} columns with >{threshold*100:.0f}% missing: {cols_to_drop[:5]}...")
    return df.drop(columns=cols_to_drop)


def split_features_target(df: pd.DataFrame) -> tuple:
    """Separate features from target variable."""
    target_col = CONFIG["preprocessing"]["target_column"]
    drop_cols = CONFIG["preprocessing"]["drop_columns"]

    y = df[target_col]
    X = df.drop(columns=[target_col] + [c for c in drop_cols if c in df.columns])
    logger.info(f"Features: {X.shape[1]} columns | Target: {target_col}")
    return X, y


def get_feature_types(X: pd.DataFrame) -> tuple:
    """Return lists of numeric and categorical column names."""
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    logger.info(f"Numeric features: {len(numeric_cols)} | Categorical features: {len(categorical_cols)}")
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Build the sklearn ColumnTransformer preprocessing pipeline."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor


def run_pipeline(save: bool = True) -> dict:
    """Full preprocessing pipeline: load → clean → split → fit → save."""
    from src.ingestion.ingest import load_raw_data
    from src.preprocessing.validate import validate_raw_data

    df = load_raw_data(CONFIG["data"]["main_file"])
    df = validate_raw_data(df)
    df = fix_anomalies(df)
    df = drop_high_missing(df, threshold=0.6)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["preprocessing"]["test_size"],
        random_state=CONFIG["preprocessing"]["random_state"],
        stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

    numeric_cols, categorical_cols = get_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    logger.info(f"Processed train shape: {X_train_processed.shape}")

    if save:
        processed_dir = Path(CONFIG["data"]["processed_dir"])
        processed_dir.mkdir(parents=True, exist_ok=True)
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(preprocessor, "models/preprocessor.pkl")
        joblib.dump((X_train_processed, X_test_processed, y_train.values, y_test.values),
                    "models/processed_data.pkl")
        logger.info("Saved preprocessor and processed data to models/")

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "preprocessor": preprocessor,
        "feature_names": numeric_cols + categorical_cols
    }


if __name__ == "__main__":
    result = run_pipeline(save=True)
    print(f"\nPipeline complete.")
    print(f"X_train shape: {result['X_train'].shape}")
    print(f"X_test shape:  {result['X_test'].shape}")
    print(f"y_train distribution: {pd.Series(result['y_train']).value_counts(normalize=True).round(3).to_dict()}")