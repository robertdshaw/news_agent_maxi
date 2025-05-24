import pickle
import logging
from pathlib import Path


# === CTR Model Loader ===
def load_ctr_model(model_path=None):
    """
    Load the CTR regression model from disk.
    Returns (model_data, error_message); error_message is None on success.
    Expected model_data keys: 'model', 'feature_names'.
    """
    if model_path is None:
        base = Path(__file__).parent.resolve()
        model_path = base / "model_output" / "ctr_regressor.pkl"

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # validate required keys
        required = {"model", "feature_names"}
        missing = required - set(model_data.keys())
        if missing:
            raise KeyError(f"Missing keys in model_data: {missing}")

        logging.info(f"Loaded CTR model from: {model_path}")
        return model_data, None

    except Exception as e:
        err = str(e)
        logging.error(f"Failed to load CTR model from {model_path}: {err}")
        return None, err
