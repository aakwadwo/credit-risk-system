# api/model_loader.py
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton model loader — loads model and preprocessor once
    at startup, reuses them for every request.
    """
    _instance = None
    _model = None
    _preprocessor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self):
        """Load model and preprocessor from disk."""
        model_path = Path("models/xgboost.pkl")
        preprocessor_path = Path("models/preprocessor.pkl")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")

        self._model = joblib.load(model_path)
        self._preprocessor = joblib.load(preprocessor_path)
        logger.info("Model and preprocessor loaded successfully")

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def preprocessor(self):
        if self._preprocessor is None:
            raise RuntimeError("Preprocessor not loaded. Call load() first.")
        return self._preprocessor

    @property
    def is_loaded(self):
        return self._model is not None and self._preprocessor is not None


model_loader = ModelLoader()