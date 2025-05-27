import pickle
import logging
from pathlib import Path


# === CTR Model Loader ===
def load_ctr_model(model_path=None):
    """
    Load the XGBoost CTR prediction model from disk.
    Returns (model_data, error_message); error_message is None on success.
    Expected model_data keys: 'model', 'feature_names', 'ctr_threshold'.
    """
    if model_path is None:
        base = Path(__file__).parent.resolve()
        model_path = base / "model_output" / "ai_news_editor_model.pkl"

    # Ensure Path object and check existence
    model_path = Path(model_path)
    if not model_path.exists():
        err = f"Model file not found: {model_path}"
        logging.error(err)
        return None, err

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Validate required keys for XGBoost model
        required = {"model", "feature_names", "ctr_threshold"}
        missing = required - set(model_data.keys())
        if missing:
            raise KeyError(f"Missing keys in model_data: {missing}")

        # Log model info
        num_features = len(model_data["feature_names"])
        model_type = type(model_data["model"]).__name__
        ctr_threshold = model_data.get("ctr_threshold", 0.02)

        logging.info(f"Loaded XGBoost CTR model from: {model_path}")
        logging.info(f"Model type: {model_type}, Features: {num_features}")
        logging.info(f"CTR threshold: {ctr_threshold:.4f}")

        # Log performance info if available
        if "performance" in model_data:
            perf = model_data["performance"]
            logging.info(
                f"Model performance - Validation AUC: {perf.get('validation_auc', 'N/A'):.4f}"
            )
            logging.info(
                f"Model ready for deployment: {perf.get('deployment_ready', False)}"
            )

        return model_data, None

    except (pickle.PickleError, EOFError) as e:
        err = f"Corrupted model file: {str(e)}"
        logging.error(f"Failed to load XGBoost model from {model_path}: {err}")
        return None, err
    except Exception as e:
        err = str(e)
        logging.error(f"Failed to load XGBoost model from {model_path}: {err}")
        return None, err


def validate_features(model_data, feature_df):
    """
    Validate that feature DataFrame has all required columns for the XGBoost model.
    Returns (is_valid, missing_features).
    """
    required_features = set(model_data["feature_names"])
    available_features = set(feature_df.columns)
    missing = required_features - available_features

    if missing:
        logging.warning(f"Missing features for XGBoost prediction: {list(missing)}")
        return False, list(missing)

    return True, []


def convert_probability_to_ctr(probability, ctr_threshold, method="linear"):
    """
    Convert XGBoost engagement probability to estimated CTR.

    Args:
        probability: Engagement probability (0-1)
        ctr_threshold: CTR threshold used for binary classification
        method: Conversion method ("linear", "exponential", "threshold")
    """
    if method == "linear":
        # Linear scaling: prob * threshold * 2
        return probability * ctr_threshold * 2
    elif method == "exponential":
        # Exponential scaling for more realistic CTR curve
        import numpy as np

        return ctr_threshold * (np.exp(probability * 2) - 1) / (np.exp(2) - 1)
    elif method == "threshold":
        # Threshold-based: above/below threshold
        return ctr_threshold * 1.5 if probability > 0.5 else ctr_threshold * 0.5
    else:
        # Default to linear
        return probability * ctr_threshold * 2


# === Model Paths Configuration ===
def get_model_paths():
    """
    Returns a dict of file paths required by the AI News Editor Assistant.
    """
    base = Path(__file__).parent.resolve()
    data_dir = base / "data" / "preprocessed" / "processed_data"
    model_dir = base / "model_output"
    faiss_dir = base / "faiss_system"

    paths = {
        # Processed feature matrices (parquet format)
        "train_features": data_dir / "X_train.parquet",
        "val_features": data_dir / "X_val.parquet",
        "test_features": data_dir / "X_test.parquet",
        "train_targets": data_dir / "y_train.parquet",
        "val_targets": data_dir / "y_val.parquet",
        "test_targets": data_dir / "y_test.parquet",
        # FAISS Content System (Updated paths)
        "embeddings": faiss_dir / "article_embeddings.pkl",
        "faiss_index": faiss_dir / "faiss_index.idx",
        "content_metadata": faiss_dir / "content_metadata.pkl",
        "faiss_config": faiss_dir / "faiss_config.json",
        "content_utilities": faiss_dir / "content_utilities.py",
        # XGBoost Model artifacts (Updated)
        "ctr_model": model_dir / "ai_news_editor_model.pkl",
        "model_predictions": model_dir / "ai_news_editor_predictions.csv",
        "model_performance": model_dir / "model_performance.json",
        "feature_importance": model_dir / "feature_importance.csv",
        "topic_performance": model_dir / "topic_performance.csv",
        "concept_drift_analysis": model_dir / "concept_drift_analysis.csv",
        # Legacy preprocessing files
        "feature_metadata": data_dir / "feature_metadata.json",
        "label_encoder": data_dir / "label_encoder.json",
        "editorial_guidelines": data_dir / "editorial_guidelines.json",
        "data_quality_report": data_dir / "data_quality_report.json",
        "feature_summary": data_dir / "feature_summary.csv",
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
        logging.info("Run XGBoost training and FAISS scripts to generate missing files")

    return paths


def check_pipeline_status():
    """
    Check which parts of the AI News Editor pipeline have been completed.
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
        "xgboost_training_completed": paths["ctr_model"].exists(),
        "faiss_system_completed": all(
            [
                paths["embeddings"].exists(),
                paths["faiss_index"].exists(),
                paths["content_metadata"].exists(),
            ]
        ),
        "predictions_available": paths["model_predictions"].exists(),
        "performance_analysis_available": paths["model_performance"].exists(),
        "concept_drift_analysis_available": paths["concept_drift_analysis"].exists(),
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
    Returns only the essential paths needed for basic AI News Editor functionality.
    """
    paths = get_model_paths()

    critical = {
        "train_features": paths["train_features"],
        "train_targets": paths["train_targets"],
        "ctr_model": paths["ctr_model"],
        "feature_metadata": paths["feature_metadata"],
        "content_metadata": paths["content_metadata"],
        "faiss_index": paths["faiss_index"],
    }

    return critical


def load_faiss_system():
    """
    Load the complete FAISS content system.
    Returns (faiss_data, error_message).
    """
    paths = get_model_paths()

    try:
        # Load FAISS index
        import faiss

        index = faiss.read_index(str(paths["faiss_index"]))

        # Load content metadata
        with open(paths["content_metadata"], "rb") as f:
            metadata = pickle.load(f)

        # Load embeddings
        with open(paths["embeddings"], "rb") as f:
            embeddings = pickle.load(f)

        faiss_data = {
            "index": index,
            "metadata": metadata,
            "embeddings": embeddings,
            "total_articles": metadata.get("total_articles", 0),
            "has_engagement_predictions": metadata.get(
                "has_engagement_predictions", False
            ),
        }

        logging.info(
            f"Loaded FAISS system: {faiss_data['total_articles']} articles indexed"
        )

        return faiss_data, None

    except Exception as e:
        err = f"Failed to load FAISS system: {str(e)}"
        logging.error(err)
        return None, err
