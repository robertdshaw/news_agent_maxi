import logging
from pathlib import Path
import json


def get_model_paths():
    """Returns file paths required by the News Editor system"""
    base = Path(__file__).parent.resolve()
    data_dir = base / "data" / "preprocessed" / "processed_data"
    model_dir = base / "models"
    faiss_dir = base / "faiss_index"

    paths = {
        # PROCESSED DATA FILES
        "train_features": data_dir / "X_train_optimized.parquet",
        "val_features": data_dir / "X_val_optimized.parquet",
        "test_features": data_dir / "X_test_optimized.parquet",
        "train_targets": data_dir / "y_train_optimized.parquet",
        "val_targets": data_dir / "y_val_optimized.parquet",
        # ARTICLE METADATA
        "article_metadata_train": data_dir / "article_metadata_train_optimized.parquet",
        "article_metadata_val": data_dir / "article_metadata_val_optimized.parquet",
        "article_metadata_test": data_dir / "article_metadata_test_optimized.parquet",
        "preprocessing_metadata": data_dir / "preprocessing_metadata.json",
        # FAISS SEARCH SYSTEM
        "faiss_index": faiss_dir / "article_index.faiss",
        "article_lookup": faiss_dir / "article_lookup.pkl",
        "article_id_mappings": faiss_dir / "article_id_mappings.pkl",
        "embedding_matrix": faiss_dir / "embedding_matrix.npy",
        "search_functions": faiss_dir / "search_functions.pkl",
        "index_metadata": faiss_dir / "index_metadata.json",
        "load_search_system": faiss_dir / "load_search_system.py",
        # REWRITE ANALYSIS
        "rewrite_analysis_data": faiss_dir
        / "rewrite_analysis"
        / "headline_rewrites.parquet",
        "rewrite_summary": faiss_dir / "rewrite_analysis" / "rewrite_summary.json",
        # MODEL FILES
        "xgboost_model": model_dir / "xgboost_optimized_model.pkl",
        "optuna_study": model_dir / "optuna_study.pkl",
        "test_predictions": model_dir / "test_predictions.parquet",
        "model_metadata": model_dir / "model_metadata.json",
        "feature_importance": model_dir / "feature_importance.csv",
        # PREPROCESSING COMPONENTS
        "category_encoder": data_dir / "category_encoder.pkl",
        "pca_transformer": data_dir / "pca_transformer.pkl",
        # PLOTS AND ANALYSIS
        "eda_plots": base
        / "data"
        / "preprocessed"
        / "plots"
        / "comprehensive_eda_analysis.png",
        "pairplot": base / "data" / "preprocessed" / "plots" / "feature_pairplot.png",
        "model_plots": model_dir / "plots" / "model_evaluation_dashboard.png",
        "readiness_plots": base
        / "data"
        / "preprocessed"
        / "plots"
        / "model_output_readiness.png",
    }

    return paths


def get_critical_paths():
    """Returns only essential paths needed for core functionality"""
    paths = get_model_paths()

    critical = {
        "train_features": paths["train_features"],
        "train_targets": paths["train_targets"],
        "xgboost_model": paths["xgboost_model"],
        "preprocessing_metadata": paths["preprocessing_metadata"],
        "faiss_index": paths["faiss_index"],
        "article_lookup": paths["article_lookup"],
        "category_encoder": paths["category_encoder"],
        "model_metadata": paths["model_metadata"],
    }

    return critical


def check_file_status(paths):
    """Check which files exist and which are missing"""
    existing_files = []
    missing_files = []

    for name, path in paths.items():
        if path.exists():
            existing_files.append(name)
        else:
            missing_files.append(name)

    return existing_files, missing_files


def get_system_status():
    """Get comprehensive system status"""
    paths = get_model_paths()
    critical_paths = get_critical_paths()

    all_existing, all_missing = check_file_status(paths)
    critical_existing, critical_missing = check_file_status(critical_paths)

    # Load metadata if available
    metadata = {}
    if paths["preprocessing_metadata"].exists():
        try:
            with open(paths["preprocessing_metadata"], "r") as f:
                metadata = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load preprocessing metadata: {e}")

    model_metadata = {}
    if paths["model_metadata"].exists():
        try:
            with open(paths["model_metadata"], "r") as f:
                model_metadata = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load model metadata: {e}")

    faiss_metadata = {}
    if paths["index_metadata"].exists():
        try:
            with open(paths["index_metadata"], "r") as f:
                faiss_metadata = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load FAISS metadata: {e}")

    status = {
        "files": {
            "total_files": len(paths),
            "existing_files": len(all_existing),
            "missing_files": len(all_missing),
            "existing_list": all_existing,
            "missing_list": all_missing,
        },
        "critical_files": {
            "total_critical": len(critical_paths),
            "existing_critical": len(critical_existing),
            "missing_critical": len(critical_missing),
            "critical_missing_list": critical_missing,
            "system_ready": len(critical_missing) == 0,
        },
        "preprocessing": {
            "completed": paths["preprocessing_metadata"].exists(),
            "features_created": metadata.get("features_created", 0),
            "dataset_sizes": metadata.get("dataset_sizes", {}),
            "target_statistics": metadata.get("target_statistics", {}),
        },
        "model": {
            "trained": paths["xgboost_model"].exists(),
            "model_type": model_metadata.get("model_type", "Unknown"),
            "performance": model_metadata.get("final_evaluation", {}),
            "training_timestamp": model_metadata.get("training_timestamp", "Unknown"),
        },
        "faiss": {
            "index_ready": paths["faiss_index"].exists(),
            "total_articles": faiss_metadata.get("total_articles", 0),
            "rewrite_variants": faiss_metadata.get("rewrite_variants", 0),
            "rewrite_analysis_available": paths["rewrite_analysis_data"].exists(),
        },
        "llm_integration": {
            "rewrite_data_available": paths["rewrite_analysis_data"].exists(),
            "rewrite_summary_available": paths["rewrite_summary"].exists(),
        },
    }

    return status


def log_system_status():
    """Log comprehensive system status"""
    status = get_system_status()

    logging.info("=" * 60)
    logging.info("NEWS EDITOR SYSTEM STATUS")
    logging.info("=" * 60)

    # File status
    files = status["files"]
    logging.info(f"FILES: {files['existing_files']}/{files['total_files']} exist")
    if files["missing_files"] > 0:
        logging.warning(f"Missing files: {files['missing_list']}")

    # Critical files
    critical = status["critical_files"]
    if critical["system_ready"]:
        logging.info("CRITICAL FILES: All present - System ready!")
    else:
        logging.error(f"CRITICAL FILES MISSING: {critical['critical_missing_list']}")

    # Preprocessing status
    prep = status["preprocessing"]
    if prep["completed"]:
        logging.info(
            f"PREPROCESSING: Complete - {prep['features_created']} features created"
        )
        if prep["dataset_sizes"]:
            for dataset, size in prep["dataset_sizes"].items():
                logging.info(f"  {dataset.title()}: {size:,} articles")
        if prep["target_statistics"]:
            stats = prep["target_statistics"]
            logging.info(
                f"  High engagement rate: {stats.get('high_engagement_rate', 0):.1%}"
            )
            logging.info(f"  Mean CTR: {stats.get('mean_ctr', 0):.6f}")
    else:
        logging.warning("PREPROCESSING: Not completed")

    # Model status
    model = status["model"]
    if model["trained"]:
        logging.info(f"MODEL: {model['model_type']} trained")
        if model["performance"]:
            perf = model["performance"]
            if "auc" in perf:
                logging.info(f"  AUC: {perf['auc']:.4f}")
            if "ctr_gain_achieved" in perf:
                logging.info(f"  CTR Gain: {perf['ctr_gain_achieved']:.4f}")
        logging.info(f"  Training time: {model['training_timestamp']}")
    else:
        logging.warning("MODEL: Not trained")

    # FAISS status
    faiss = status["faiss"]
    if faiss["index_ready"]:
        logging.info(f"FAISS: Index ready - {faiss['total_articles']:,} articles")
        if faiss["rewrite_variants"] > 0:
            logging.info(f"  Rewrite variants: {faiss['rewrite_variants']:,}")
        if faiss["rewrite_analysis_available"]:
            logging.info("  Rewrite analysis: Available")
    else:
        logging.warning("FAISS: Index not ready")

    # LLM Integration status
    llm = status["llm_integration"]
    if llm["rewrite_data_available"] and llm["rewrite_summary_available"]:
        logging.info("LLM INTEGRATION: Ready")
    else:
        logging.warning("LLM INTEGRATION: Incomplete")

    logging.info("=" * 60)

    return status


def test_system_ready():
    """Test if the system is ready for deployment"""

    print("Testing News Editor System...")

    try:
        status = get_system_status()

        # Test critical files
        if not status["critical_files"]["system_ready"]:
            print("❌ CRITICAL FILES MISSING:")
            for file in status["critical_files"]["critical_missing_list"]:
                print(f"  - {file}")
            return False

        print("✅ Critical files: All present")

        # Test preprocessing
        if not status["preprocessing"]["completed"]:
            print("❌ Preprocessing not completed")
            return False

        print(
            f"✅ Preprocessing: {status['preprocessing']['features_created']} features"
        )

        # Test model
        if not status["model"]["trained"]:
            print("❌ Model not trained")
            return False

        model_performance = status["model"]["performance"]
        auc = model_performance.get("auc", 0)
        print(f"✅ Model: {status['model']['model_type']} (AUC: {auc:.4f})")

        # Test FAISS
        if not status["faiss"]["index_ready"]:
            print("❌ FAISS index not ready")
            return False

        print(f"✅ FAISS: {status['faiss']['total_articles']:,} articles indexed")

        # Test LLM Integration (optional)
        if status["llm_integration"]["rewrite_data_available"]:
            print(
                f"✅ LLM Integration: {status['faiss']['rewrite_variants']:,} variants"
            )
        else:
            print("⚠️  LLM Integration: No rewrite data (will use fallback)")

        print(f"\n🎉 News Editor System is READY!")
        print(
            f"📊 Dataset: {sum(status['preprocessing']['dataset_sizes'].values()):,} total articles"
        )
        print(f"🎯 Target: high_engagement classification")
        print(f"📈 Performance: {auc:.4f} AUC")

        return True

    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False


def check_pipeline_status():
    """Check which parts of the pipeline have been completed"""
    paths = get_model_paths()

    pipeline_steps = {
        "data_preprocessing": {
            "completed": all(
                [
                    paths["train_features"].exists(),
                    paths["val_features"].exists(),
                    paths["test_features"].exists(),
                    paths["preprocessing_metadata"].exists(),
                ]
            ),
            "description": "EDA preprocessing with feature engineering",
        },
        "model_training": {
            "completed": paths["xgboost_model"].exists(),
            "description": "XGBoost model training with Optuna",
        },
        "model_evaluation": {
            "completed": paths["model_metadata"].exists(),
            "description": "Model performance analysis",
        },
        "faiss_indexing": {
            "completed": all(
                [
                    paths["faiss_index"].exists(),
                    paths["article_lookup"].exists(),
                    paths["index_metadata"].exists(),
                ]
            ),
            "description": "FAISS search system creation",
        },
        "llm_integration": {
            "completed": paths["rewrite_analysis_data"].exists(),
            "description": "LLM headline rewriting analysis",
        },
        "category_encoding": {
            "completed": paths["category_encoder"].exists(),
            "description": "Category label encoding",
        },
        "eda_analysis": {
            "completed": paths["eda_plots"].exists(),
            "description": "Exploratory data analysis plots",
        },
        "model_visualization": {
            "completed": paths["model_plots"].exists(),
            "description": "Model evaluation plots",
        },
        "deployment_readiness": {
            "completed": all(
                [
                    paths["xgboost_model"].exists(),
                    paths["faiss_index"].exists(),
                    paths["article_lookup"].exists(),
                    paths["preprocessing_metadata"].exists(),
                    paths["category_encoder"].exists(),
                ]
            ),
            "description": "All components for deployment",
        },
    }

    # Log pipeline status
    logging.info("=" * 60)
    logging.info("NEWS EDITOR PIPELINE STATUS")
    logging.info("=" * 60)

    completed_steps = 0
    total_steps = len(pipeline_steps)

    for step_name, step_info in pipeline_steps.items():
        status_msg = "✅ COMPLETED" if step_info["completed"] else "❌ PENDING"
        logging.info(f"{status_msg}: {step_info['description']}")

        if step_info["completed"]:
            completed_steps += 1

    # Overall progress
    progress_pct = (completed_steps / total_steps) * 100
    logging.info("-" * 60)
    logging.info(
        f"OVERALL PROGRESS: {completed_steps}/{total_steps} steps ({progress_pct:.1f}%)"
    )

    # Deployment readiness
    deployment_ready = pipeline_steps["deployment_readiness"]["completed"]
    logging.info(f"DEPLOYMENT READY: {'YES' if deployment_ready else 'NO'}")

    # Next steps
    if not deployment_ready:
        pending_critical = []
        if not pipeline_steps["data_preprocessing"]["completed"]:
            pending_critical.append("Run EDA_preprocess_features.py")
        if not pipeline_steps["model_training"]["completed"]:
            pending_critical.append("Run model_class.py")
        if not pipeline_steps["faiss_indexing"]["completed"]:
            pending_critical.append("Run build_faiss_index.py")

        if pending_critical:
            logging.info("NEXT STEPS:")
            for step in pending_critical:
                logging.info(f"  - {step}")

    logging.info("=" * 60)

    return {
        "pipeline_steps": pipeline_steps,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "progress_percentage": progress_pct,
        "deployment_ready": deployment_ready,
    }


def validate_data_consistency():
    """Validate consistency between different data files"""
    paths = get_model_paths()
    issues = []

    try:
        # Check preprocessing metadata consistency
        if paths["preprocessing_metadata"].exists():
            with open(paths["preprocessing_metadata"], "r") as f:
                prep_meta = json.load(f)

            # Check if feature counts match
            if paths["train_features"].exists():
                import pandas as pd

                train_features = pd.read_parquet(paths["train_features"])
                actual_features = len(train_features.columns)
                expected_features = prep_meta.get("features_created", 0)

                if actual_features != expected_features:
                    issues.append(
                        f"Feature count mismatch: {actual_features} vs {expected_features}"
                    )

        # Check model-FAISS consistency
        if paths["model_metadata"].exists() and paths["index_metadata"].exists():
            with open(paths["model_metadata"], "r") as f:
                model_meta = json.load(f)
            with open(paths["index_metadata"], "r") as f:
                faiss_meta = json.load(f)

            # Check if article counts are reasonable
            model_samples = model_meta.get("training_samples", 0)
            faiss_articles = faiss_meta.get("total_articles", 0)

            if faiss_articles < model_samples * 0.8:  # Allow some variation
                issues.append(
                    f"Article count inconsistency: FAISS {faiss_articles} vs Model training {model_samples}"
                )

    except Exception as e:
        issues.append(f"Validation error: {e}")

    return issues


def create_directories():
    """Create necessary directories if they don't exist"""
    base = Path(__file__).parent.resolve()

    directories = [
        base / "data" / "preprocessed" / "processed_data",
        base / "data" / "preprocessed" / "plots",
        base / "data" / "preprocessed" / "cache",
        base / "models" / "plots",
        base / "faiss_index" / "rewrite_analysis",
    ]

    created = []
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(str(directory))

    if created:
        logging.info(f"Created directories: {created}")

    return created


if __name__ == "__main__":
    # Test the system
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("Testing News Editor System...")

    # Create necessary directories
    create_directories()

    # Check pipeline status
    pipeline_status = check_pipeline_status()

    # Validate data consistency
    validation_issues = validate_data_consistency()
    if validation_issues:
        logging.warning("Data consistency issues found:")
        for issue in validation_issues:
            logging.warning(f"  - {issue}")

    # Test system readiness
    system_ready = test_system_ready()

    # Log comprehensive status
    log_system_status()

    if system_ready:
        print("\n🚀 System is ready for deployment!")
    else:
        print(f"\n⚠️  System needs attention. Check logs above.")
        print(
            f"Progress: {pipeline_status['completed_steps']}/{pipeline_status['total_steps']} steps completed"
        )
