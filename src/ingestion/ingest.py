# src/ingestion/ingest.py
import pandas as pd
import logging
from pathlib import Path
from src.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filename: str) -> pd.DataFrame:
    """Load a raw CSV file from the configured raw data directory."""
    raw_dir = Path(CONFIG["data"]["raw_dir"])
    filepath = raw_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info(f"Loading {filepath} ...")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")
    return df


def get_basic_summary(df: pd.DataFrame) -> dict:
    """Return a basic structural summary of the dataframe."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "missing_total": int(missing.sum()),
        "columns_with_missing": int((missing > 0).sum()),
        "top_missing": missing_pct[missing_pct > 0].sort_values(ascending=False).head(10).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "target_distribution": df["TARGET"].value_counts(normalize=True).round(4).to_dict() if "TARGET" in df.columns else {}
    }
    return summary


if __name__ == "__main__":
    df = load_raw_data(CONFIG["data"]["main_file"])
    summary = get_basic_summary(df)

    print(f"\n{'='*50}")
    print(f"Dataset shape: {summary['n_rows']:,} rows × {summary['n_cols']} columns")
    print(f"Total missing values: {summary['missing_total']:,}")
    print(f"Columns with missing data: {summary['columns_with_missing']}")
    print(f"\nTarget distribution:")
    for k, v in summary["target_distribution"].items():
        label = "Default (1)" if k == 1 else "No default (0)"
        print(f"  {label}: {v*100:.1f}%")
    print(f"\nTop 10 columns by missing %:")
    for col, pct in summary["top_missing"].items():
        print(f"  {col}: {pct}%")
    print(f"{'='*50}\n")