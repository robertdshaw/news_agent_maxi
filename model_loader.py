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

    # Ensure Path object and check existence
    model_path = Path(model_path)
    if not model_path.exists():
        err = f"Model file not found: {model_path}"
        logging.error(err)
        return None, err

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Validate required keys
        required = {"model", "feature_names"}
        missing = required - set(model_data.keys())
        if missing:
            raise KeyError(f"Missing keys in model_data: {missing}")

        # Log model info if available
        num_features = len(model_data["feature_names"])
        model_type = type(model_data["model"]).__name__
        logging.info(f"Loaded CTR model from: {model_path}")
        logging.info(f"Model type: {model_type}, Features: {num_features}")

        # Log performance info if available (from updated training script)
        if "performance" in model_data:
            perf = model_data["performance"]
            logging.info(
                f"Model performance - Test RMSE: {perf.get('test_rmse', 'N/A'):.4f}"
            )

        return model_data, None

    except (pickle.PickleError, EOFError) as e:
        err = f"Corrupted model file: {str(e)}"
        logging.error(f"Failed to load CTR model from {model_path}: {err}")
        return None, err
    except Exception as e:
        err = str(e)
        logging.error(f"Failed to load CTR model from {model_path}: {err}")
        return None, err


def validate_features(model_data, feature_df):
    """
    Validate that feature DataFrame has all required columns for the model.
    Returns (is_valid, missing_features).
    """
    required_features = set(model_data["feature_names"])
    available_features = set(feature_df.columns)
    missing = required_features - available_features

    if missing:
        logging.warning(f"Missing features for model prediction: {list(missing)}")
        return False, list(missing)

    return True, []


# === Model Paths Configuration ===
def get_model_paths():
    """
    Returns a dict of file paths required by the Streamlit dashboard.
    """
    base = Path(__file__).parent.resolve()
    data_dir = base / "data" / "preprocessed" / "processed_data"
    model_dir = base / "model_output"

    paths = {
        # Processed feature matrices (parquet format)
        "train_features": data_dir / "X_train.parquet",
        "val_features": data_dir / "X_val.parquet",
        "test_features": data_dir / "X_test.parquet",
        "train_targets": data_dir / "y_train.parquet",
        "val_targets": data_dir / "y_val.parquet",
        "test_targets": data_dir / "y_test.parquet",
        # Embeddings and search index
        "embeddings": data_dir.parent / "article_embeddings.pkl",
        "faiss_index": data_dir.parent / "faiss_index.idx",
        "embedding_metadata": data_dir.parent / "embedding_metadata.pkl",
        # Model artifacts
        "ctr_model": model_dir / "ctr_regressor_1.pkl",
        "feature_metadata": data_dir / "feature_metadata.json",
        "label_encoder": data_dir / "label_encoder.json",
        "editorial_guidelines": data_dir / "editorial_guidelines.json",
        # Quality reports
        "data_quality_report": data_dir / "data_quality_report.json",
    }

    # Check which files exist and log missing ones
    missing_files = []
    existing_files = []

    for name, p in paths.items():
        if p.exists():
            existing_files.append(name)
        else:
            missing_files.append(name)
            logging.warning(f"File '{name}' not found at: {p}")

    logging.info(f"Found {len(existing_files)} of {len(paths)} required files")

    if missing_files:
        logging.warning(f"Missing files: {missing_files}")
        logging.info("Run preprocessing and training scripts to generate missing files")

    return paths


def check_pipeline_status():
    """
    Check which parts of the ML pipeline have been completed.
    Returns dict with status of each pipeline stage.
    """
    paths = get_model_paths()

    status = {
        "preprocessing_completed": all(
            [
                paths["train_features"].exists(),
                paths["val_features"].exists(),
                paths["test_features"].exists(),
                paths["feature_metadata"].exists(),
            ]
        ),
        "training_completed": paths["ctr_model"].exists(),
        "embeddings_completed": all(
            [paths["embeddings"].exists(), paths["faiss_index"].exists()]
        ),
        "quality_reports_available": paths["data_quality_report"].exists(),
        "editorial_guidelines_available": paths["editorial_guidelines"].exists(),
    }

    # Log pipeline status
    for stage, completed in status.items():
        status_msg = "✓" if completed else "✗"
        logging.info(f"{status_msg} {stage.replace('_', ' ').title()}: {completed}")

    return status


def get_critical_paths():
    """
    Returns only the essential paths needed for basic functionality.
    """
    paths = get_model_paths()

    critical = {
        "train_features": paths["train_features"],
        "train_targets": paths["train_targets"],
        "ctr_model": paths["ctr_model"],
        "feature_metadata": paths["feature_metadata"],
    }

    return critical
