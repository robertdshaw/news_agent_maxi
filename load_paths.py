import logging
from pathlib import Path


def get_model_paths():
    """
    Returns a dict of file paths required by the AI News Editor Assistant:
    - Updated for XGBoost model and FAISS system
    - Includes new engagement prediction files
    - Maintains backward compatibility with existing preprocessing
    """
    base = Path(__file__).parent.resolve()
    data_dir = base / "data" / "preprocessed" / "processed_data"
    model_dir = base / "model_output"
    faiss_dir = base / "faiss_system"

    paths = {
        # === CORE DATA FILES ===
        # Processed feature matrices (parquet format)
        "train_features": data_dir / "X_train.parquet",
        "val_features": data_dir / "X_val.parquet",
        "test_features": data_dir / "X_test.parquet",
        "train_targets": data_dir / "y_train.parquet",
        "val_targets": data_dir / "y_val.parquet",
        "test_targets": data_dir / "y_test.parquet",
        # === FAISS CONTENT SYSTEM ===
        "embeddings": faiss_dir / "article_embeddings.pkl",
        "faiss_index": faiss_dir / "faiss_index.idx",
        "content_metadata": faiss_dir / "content_metadata.pkl",
        "faiss_config": faiss_dir / "faiss_config.json",
        "content_utilities": faiss_dir / "content_utilities.py",
        # Legacy embedding paths (for backward compatibility)
        "legacy_embeddings": data_dir.parent / "article_embeddings.pkl",
        "legacy_faiss_index": data_dir.parent / "faiss_index.idx",
        "legacy_embedding_metadata": data_dir.parent / "embedding_metadata.pkl",
        # === XGBOOST MODEL SYSTEM ===
        "ctr_model": model_dir / "ai_news_editor_model.pkl",
        "model_predictions": model_dir / "ai_news_editor_predictions.csv",
        "model_performance": model_dir / "model_performance.json",
        "model_integration": model_dir / "ai_news_editor_integration.json",
        # Analysis files
        "feature_importance": model_dir / "feature_importance.csv",
        "topic_performance": model_dir / "topic_performance.csv",
        "concept_drift_analysis": model_dir / "concept_drift_analysis.csv",
        "time_aware_analysis": model_dir / "time_aware_xgboost_analysis.png",
        # === PREPROCESSING METADATA ===
        "feature_metadata": data_dir / "feature_metadata.json",
        "label_encoder": data_dir / "label_encoder.json",
        "feature_summary": data_dir / "feature_summary.csv",
        "interaction_summary": data_dir / "interaction_summary.csv",
        # === QUALITY REPORTS ===
        "data_quality_report": data_dir / "data_quality_report.json",
        "editorial_guidelines": data_dir / "editorial_guidelines.json",
        # === LEGACY MODEL FILES (for compatibility) ===
        "legacy_ctr_model": model_dir / "ctr_regressor.pkl",
        "legacy_logistic_model": model_dir / "logistic_regression_model.pkl",
    }

    # Check which files exist and log missing ones
    missing_files = []
    existing_files = []
    critical_missing = []

    # Define critical files for basic functionality
    critical_files = [
        "train_features",
        "train_targets",
        "ctr_model",
        "feature_metadata",
        "content_metadata",
        "faiss_index",
    ]

    for name, p in paths.items():
        if p.exists():
            existing_files.append(name)
        else:
            missing_files.append(name)
            if name in critical_files:
                critical_missing.append(name)
            logging.warning(f"File '{name}' not found at: {p}")

    logging.info(f"Found {len(existing_files)} of {len(paths)} files")

    if critical_missing:
        logging.error(f"Critical files missing: {critical_missing}")
        logging.error("AI News Editor Assistant may not function properly")

    if missing_files:
        logging.warning(f"Missing files: {len(missing_files)} total")
        logging.info(
            "Run XGBoost training and FAISS system scripts to generate missing files"
        )

    return paths


def check_pipeline_status():
    """
    Check which parts of the AI News Editor pipeline have been completed.
    Returns dict with detailed status of each pipeline stage.
    """
    paths = get_model_paths()

    status = {
        # === CORE PREPROCESSING ===
        "preprocessing_completed": all(
            [
                paths["train_features"].exists(),
                paths["val_features"].exists(),
                paths["test_features"].exists(),
                paths["feature_metadata"].exists(),
            ]
        ),
        "feature_engineering_completed": all(
            [
                paths["feature_summary"].exists(),
                paths["interaction_summary"].exists(),
            ]
        ),
        # === MODEL TRAINING ===
        "xgboost_training_completed": paths["ctr_model"].exists(),
        "model_predictions_available": paths["model_predictions"].exists(),
        "model_performance_analyzed": paths["model_performance"].exists(),
        # === FAISS CONTENT SYSTEM ===
        "faiss_system_completed": all(
            [
                paths["embeddings"].exists(),
                paths["faiss_index"].exists(),
                paths["content_metadata"].exists(),
            ]
        ),
        "faiss_utilities_available": paths["content_utilities"].exists(),
        # === ANALYSIS & MONITORING ===
        "concept_drift_analysis_available": paths["concept_drift_analysis"].exists(),
        "topic_performance_analyzed": paths["topic_performance"].exists(),
        "feature_importance_analyzed": paths["feature_importance"].exists(),
        # === QUALITY & GUIDELINES ===
        "quality_reports_available": paths["data_quality_report"].exists(),
        "editorial_guidelines_available": paths["editorial_guidelines"].exists(),
        # === DEPLOYMENT READINESS ===
        "deployment_ready": all(
            [
                paths["ctr_model"].exists(),
                paths["faiss_index"].exists(),
                paths["content_metadata"].exists(),
                paths["model_performance"].exists(),
            ]
        ),
    }

    # Log pipeline status with detailed breakdown
    logging.info("=" * 50)
    logging.info("AI NEWS EDITOR ASSISTANT - PIPELINE STATUS")
    logging.info("=" * 50)

    for stage, completed in status.items():
        status_msg = "✓" if completed else "✗"
        stage_name = stage.replace("_", " ").title()
        logging.info(f"{status_msg} {stage_name}: {completed}")

    # Overall readiness assessment
    critical_stages = [
        "preprocessing_completed",
        "xgboost_training_completed",
        "faiss_system_completed",
    ]

    critical_ready = all(status[stage] for stage in critical_stages)
    deployment_ready = status["deployment_ready"]

    logging.info("-" * 50)
    logging.info(f"Critical Components Ready: {critical_ready}")
    logging.info(f"Full Deployment Ready: {deployment_ready}")
    logging.info("=" * 50)

    return status


def get_critical_paths():
    """
    Returns only the essential paths needed for basic AI News Editor functionality.
    These files are required for the app to start.
    """
    paths = get_model_paths()

    critical = {
        # Core data
        "train_features": paths["train_features"],
        "train_targets": paths["train_targets"],
        # XGBoost model
        "ctr_model": paths["ctr_model"],
        # Feature metadata
        "feature_metadata": paths["feature_metadata"],
        # FAISS system
        "content_metadata": paths["content_metadata"],
        "faiss_index": paths["faiss_index"],
        "embeddings": paths["embeddings"],
    }

    return critical


def get_file_sizes():
    """
    Get file sizes for monitoring and debugging.
    Returns dict with file sizes in MB.
    """
    paths = get_model_paths()
    sizes = {}

    for name, path in paths.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            sizes[name] = round(size_mb, 2)
        else:
            sizes[name] = 0

    return sizes


def validate_system_integrity():
    """
    Validate that all critical system components are present and compatible.
    Returns (is_valid, issues_list).
    """
    issues = []
    paths = get_model_paths()

    # Check critical files
    critical = get_critical_paths()
    for name, path in critical.items():
        if not path.exists():
            issues.append(f"Critical file missing: {name} at {path}")

    # Check model compatibility
    if paths["ctr_model"].exists():
        try:
            import pickle

            with open(paths["ctr_model"], "rb") as f:
                model_data = pickle.load(f)

            required_keys = ["model", "feature_names", "ctr_threshold"]
            missing_keys = [key for key in required_keys if key not in model_data]
            if missing_keys:
                issues.append(f"XGBoost model missing keys: {missing_keys}")

        except Exception as e:
            issues.append(f"XGBoost model loading error: {str(e)}")

    # Check FAISS system
    if paths["faiss_index"].exists() and paths["content_metadata"].exists():
        try:
            import faiss
            import pickle

            # Test FAISS index loading
            index = faiss.read_index(str(paths["faiss_index"]))

            # Test metadata loading
            with open(paths["content_metadata"], "rb") as f:
                metadata = pickle.load(f)

            if "articles" not in metadata:
                issues.append("FAISS metadata missing articles data")

        except Exception as e:
            issues.append(f"FAISS system error: {str(e)}")

    is_valid = len(issues) == 0
    return is_valid, issues


if __name__ == "__main__":
    # Test the path system
    logging.basicConfig(level=logging.INFO)

    print("Testing AI News Editor Assistant path system...")
    paths = get_model_paths()
    status = check_pipeline_status()
    sizes = get_file_sizes()

    print(f"\nFile sizes (MB):")
    for name, size in sizes.items():
        if size > 0:
            print(f"  {name}: {size} MB")

    is_valid, issues = validate_system_integrity()
    print(f"\nSystem integrity: {'✓ Valid' if is_valid else '✗ Issues found'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
