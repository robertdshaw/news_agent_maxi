import logging
import pickle
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_model_version():
    """Get model version from environment variable."""
    return os.getenv("MODEL_VERSION", "1")


def get_model_paths():
    """
    Returns a dict of file paths required by the Streamlit dashboard.
    Now incorporates MODEL_VERSION from .env file.
    """
    base = Path(__file__).parent.resolve()
    model_version = get_model_version()

    # Version-aware paths
    data_dir = base / "data" / "preprocessed" / f"v{model_version}" / "processed_data"
    model_dir = base / "model_output" / f"v{model_version}"

    # Fallback to non-versioned paths if versioned don't exist
    if not data_dir.exists():
        data_dir = base / "data" / "preprocessed" / "processed_data"
        logging.warning(
            f"Versioned data directory not found, using default: {data_dir}"
        )

    if not model_dir.exists():
        model_dir = base / "model_output"
        logging.warning(
            f"Versioned model directory not found, using default: {model_dir}"
        )

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
        "ctr_model": model_dir / "ctr_regressor.pkl",
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


def load_ctr_model(model_path):
    """
    Load the CTR model from pickle file.
    Returns (model_data, error_message).
    """
    try:
        if not model_path.exists():
            return None, f"Model file not found: {model_path}"

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Validate model data structure
        if not isinstance(model_data, dict):
            return None, "Invalid model file format - expected dict"

        required_keys = ["model", "feature_names"]
        missing_keys = [key for key in required_keys if key not in model_data]
        if missing_keys:
            return None, f"Missing keys in model file: {missing_keys}"

        return model_data, None

    except Exception as e:
        return None, f"Error loading model: {str(e)}"


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
    Use this for minimal requirements checking.
    """
    paths = get_model_paths()

    critical = {
        "train_features": paths["train_features"],
        "train_targets": paths["train_targets"],
        "ctr_model": paths["ctr_model"],
        "feature_metadata": paths["feature_metadata"],
    }

    return critical


def create_minimal_test_data():
    """
    Create minimal test data files if they don't exist.
    This allows the app to launch for development/testing.
    """
    import pandas as pd
    import json
    import numpy as np

    paths = get_model_paths()

    # Create directories if they don't exist
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    # Create minimal feature data
    if not paths["train_features"].exists():
        dummy_features = pd.DataFrame(
            {
                "title_length": [50, 60, 40],
                "title_word_count": [8, 10, 6],
                "has_question": [0, 1, 0],
            }
        )
        dummy_features.to_parquet(paths["train_features"])
        logging.info("Created dummy train_features")

    # Create minimal target data
    if not paths["train_targets"].exists():
        dummy_targets = pd.DataFrame({"ctr": [0.05, 0.08, 0.03]})
        dummy_targets.to_parquet(paths["train_targets"])
        logging.info("Created dummy train_targets")

    # Create minimal metadata
    if not paths["feature_metadata"].exists():
        metadata = {
            "embedding_dim": 384,
            "feature_names": ["title_length", "title_word_count", "has_question"],
        }
        with open(paths["feature_metadata"], "w") as f:
            json.dump(metadata, f)
        logging.info("Created dummy feature_metadata")


if __name__ == "__main__":
    # Test the functions
    logging.basicConfig(level=logging.INFO)
    print(f"Model version: {get_model_version()}")
    paths = get_model_paths()
    status = check_pipeline_status()
    print(f"Pipeline status: {status}")
