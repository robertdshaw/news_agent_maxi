import logging
from pathlib import Path


def get_model_paths():
    """
    Returns a dict of file paths required by the Streamlit dashboard:
    - train_features: training feature matrix
    - val_features: validation feature matrix
    - test_features: test feature matrix
    - train_targets: training targets
    - val_targets: validation targets
    - test_targets: test targets
    - embeddings: pickle file with title embeddings
    - faiss_index: FAISS index file for semantic retrieval
    - ctr_model: trained CTR regressor pickle file
    - metadata: feature metadata and label encoder
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
