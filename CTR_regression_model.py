import pandas as pd
import numpy as np
import pickle
import warnings
import logging
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging and suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
PREP_DIR = Path("data/preprocessed/processed_data")
OUTPUT_DIR = Path("model_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_prepare_data():
    """Load and prepare data for binary classification"""
    logger.info("🚀 Loading processed data...")

    try:
        # Load feature matrices and targets
        X_train = pd.read_parquet(PREP_DIR / "X_train.parquet")
        y_train = pd.read_parquet(PREP_DIR / "y_train.parquet")["ctr"]
        X_val = pd.read_parquet(PREP_DIR / "X_val.parquet")
        y_val = pd.read_parquet(PREP_DIR / "y_val.parquet")["ctr"]
        X_test = pd.read_parquet(PREP_DIR / "X_test.parquet")

        # Load CTR validity flags to filter real data
        try:
            train_flags = pd.read_parquet(PREP_DIR / "train_flags.parquet")[
                "ctr_is_real"
            ]
            val_flags = pd.read_parquet(PREP_DIR / "val_flags.parquet")["ctr_is_real"]

            # Filter for real CTR data only (no leakage)
            X_train = X_train[train_flags]
            y_train = y_train[train_flags]
            X_val = X_val[val_flags]
            y_val = y_val[val_flags]

            logger.info("✅ Applied CTR validity filtering")
        except FileNotFoundError:
            logger.warning("CTR flags not found - using all data")

        logger.info(
            f"📊 Data shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}"
        )

        return X_train, y_train, X_val, y_val, X_test

    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise


def convert_to_binary_classification(y_train, y_val):
    """Convert CTR values to binary classification (clicked/not clicked)"""
    logger.info("🎯 Converting to binary classification...")

    y_train_binary = (y_train > 0).astype(int)
    y_val_binary = (y_val > 0).astype(int)

    # Log class distribution statistics
    train_click_rate = y_train_binary.mean()
    val_click_rate = y_val_binary.mean()
    train_clicks = y_train_binary.sum()
    val_clicks = y_val_binary.sum()

    logger.info(
        f"📈 Training set: {train_clicks:,} clicked / {len(y_train_binary):,} total ({train_click_rate:.1%})"
    )
    logger.info(
        f"📈 Validation set: {val_clicks:,} clicked / {len(y_val_binary):,} total ({val_click_rate:.1%})"
    )

    return y_train_binary, y_val_binary


def create_parameter_search():
    """Define comprehensive parameter search space"""
    logger.info("🔍 Setting up parameter search space...")

    param_distributions = {
        # Feature selection parameters
        "feature_selection__k": [15, 20, 25, 30, 35, 40, 50, 66],
        # Core LightGBM parameters
        "classifier__n_estimators": stats.randint(200, 2000),
        "classifier__learning_rate": stats.uniform(0.01, 0.25),
        "classifier__max_depth": stats.randint(3, 15),
        "classifier__num_leaves": stats.randint(15, 200),
        # Regularization parameters
        "classifier__min_child_samples": stats.randint(5, 100),
        "classifier__min_child_weight": stats.uniform(1e-5, 50),
        "classifier__subsample": stats.uniform(0.5, 0.5),  # 0.5 to 1.0
        "classifier__colsample_bytree": stats.uniform(0.5, 0.5),  # 0.5 to 1.0
        "classifier__reg_alpha": stats.uniform(0, 30),
        "classifier__reg_lambda": stats.uniform(0, 30),
        # Advanced parameters
        "classifier__subsample_freq": stats.randint(1, 8),
        "classifier__min_split_gain": stats.uniform(0, 1),
        "classifier__max_bin": stats.randint(100, 400),
        # Classification specific
        "classifier__class_weight": [None, "balanced"],
        "classifier__is_unbalance": [True, False],
        # Boosting type (removed GOSS to avoid conflicts)
        "classifier__boosting_type": ["gbdt", "dart"],
        # DART specific parameters
        "classifier__drop_rate": stats.uniform(0.05, 0.4),
        "classifier__max_drop": stats.randint(10, 50),
        "classifier__skip_drop": stats.uniform(0.3, 0.6),
    }

    return param_distributions


def perform_model_search(X_train, y_train_binary):
    """Perform randomized search for best hyperparameters"""
    logger.info("🔍 Starting RandomizedSearchCV for binary classification...")

    # Create pipeline
    pipeline = Pipeline(
        [
            ("feature_selection", SelectKBest(score_func=f_classif)),
            (
                "classifier",
                LGBMClassifier(
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1,
                    objective="binary",
                    metric="binary_logloss",
                    force_col_wise=True,
                ),
            ),
        ]
    )

    # Get parameter distributions
    param_distributions = create_parameter_search()

    # Configure cross-validation
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Setup randomized search
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=25,  # Good balance of thoroughness and speed
        cv=cv_strategy,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    logger.info("⏱️ Training model (estimated 15-20 minutes)...")
    search.fit(X_train, y_train_binary)

    logger.info(f"✅ Best CV Score (ROC-AUC): {search.best_score_:.4f}")
    logger.info(f"🏆 Best parameters found: {search.best_params_}")

    return search


def evaluate_model(best_model, X_val, y_val_binary):
    """Comprehensive model evaluation"""
    logger.info("📊 Evaluating model performance...")

    # Make predictions
    y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
    y_val_pred = best_model.predict(X_val)

    # Calculate comprehensive metrics
    metrics = {
        "auc": roc_auc_score(y_val_binary, y_val_pred_proba),
        "accuracy": accuracy_score(y_val_binary, y_val_pred),
        "precision": precision_score(y_val_binary, y_val_pred),
        "recall": recall_score(y_val_binary, y_val_pred),
        "f1": f1_score(y_val_binary, y_val_pred),
    }

    logger.info(f"📈 Validation Results:")
    logger.info(f"   - ROC-AUC: {metrics['auc']:.4f}")
    logger.info(f"   - Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"   - Precision: {metrics['precision']:.4f}")
    logger.info(f"   - Recall: {metrics['recall']:.4f}")
    logger.info(f"   - F1-Score: {metrics['f1']:.4f}")

    return metrics, y_val_pred_proba, y_val_pred


def analyze_features(best_model, X_train):
    """Analyze feature importance and selection"""
    logger.info("🔍 Analyzing feature importance...")

    # Get selected features
    selected_mask = best_model["feature_selection"].get_support()
    selected_features = X_train.columns[selected_mask]
    feature_importances = best_model["classifier"].feature_importances_

    # Create feature importance dataframe
    feature_df = pd.DataFrame(
        {"feature": selected_features, "importance": feature_importances}
    ).sort_values("importance", ascending=False)

    logger.info(
        f"🎯 Selected {len(selected_features)} out of {len(X_train.columns)} features"
    )
    logger.info(f"🔝 Top 10 Most Important Features:")

    for i, (_, row) in enumerate(feature_df.head(10).iterrows(), 1):
        logger.info(f"   {i:2d}. {row['feature']}: {row['importance']:.4f}")

    # Analyze feature types
    feature_types = {
        "text": [
            f
            for f in selected_features
            if any(
                x in f
                for x in [
                    "length",
                    "reading",
                    "has_",
                    "word_count",
                    "upper_ratio",
                    "starts_with",
                ]
            )
        ],
        "temporal": [
            f
            for f in selected_features
            if any(
                x in f
                for x in ["hour", "day_of_week", "weekend", "time_period", "is_weekend"]
            )
        ],
        "category": [f for f in selected_features if "category" in f],
        "embedding": [f for f in selected_features if "emb_" in f],
    }

    logger.info(f"📊 Feature type breakdown:")
    for ftype, features in feature_types.items():
        if features:
            total_importance = feature_df[feature_df["feature"].isin(features)][
                "importance"
            ].sum()
            logger.info(
                f"   - {ftype.title()}: {len(features)} features (importance: {total_importance:.3f})"
            )

    return feature_df, selected_features


def create_visualizations(metrics, y_val_binary, y_val_pred_proba, feature_df):
    """Create performance visualizations"""
    logger.info("📊 Creating performance visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val_binary, y_val_pred_proba)
    axes[0, 0].plot(fpr, tpr, "b-", label=f'ROC Curve (AUC = {metrics["auc"]:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], "r--", label="Random Classifier")
    axes[0, 0].set_xlabel("False Positive Rate")
    axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Feature Importance
    top_features = feature_df.head(15)
    axes[0, 1].barh(range(len(top_features)), top_features["importance"])
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features["feature"], fontsize=8)
    axes[0, 1].set_xlabel("Feature Importance")
    axes[0, 1].set_title("Top 15 Feature Importances")

    # Prediction Distribution
    axes[1, 0].hist(y_val_pred_proba, bins=50, alpha=0.7, edgecolor="black")
    axes[1, 0].set_xlabel("Predicted Click Probability")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Distribution of Predicted Probabilities")

    # Performance Metrics
    metric_names = ["AUC", "Accuracy", "Precision", "Recall", "F1"]
    metric_values = [
        metrics["auc"],
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    ]

    bars = axes[1, 1].bar(
        metric_names,
        metric_values,
        color=["skyblue", "lightgreen", "lightcoral", "gold", "plum"],
    )
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("Model Performance Metrics")
    axes[1, 1].set_ylim(0, 1)

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "classification_performance.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    logger.info(
        f"📊 Visualizations saved to {OUTPUT_DIR / 'classification_performance.png'}"
    )


def save_model_package(
    search, metrics, feature_df, selected_features, X_train, y_train_binary
):
    """Save comprehensive model package"""
    logger.info("💾 Saving model package...")

    model_package = {
        "model": search.best_estimator_,
        "feature_names": list(selected_features),
        "selected_features": list(selected_features),
        "all_feature_names": X_train.columns.tolist(),
        "best_params": search.best_params_,
        "cv_results": pd.DataFrame(search.cv_results_),
        "performance": {
            "best_cv_score": search.best_score_,
            "val_auc": metrics["auc"],
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
        },
        "feature_importance": feature_df.to_dict("records"),
        "model_type": "binary_classification",
        "training_stats": {
            "train_articles": len(X_train),
            "train_click_rate": y_train_binary.mean(),
            "total_features": len(X_train.columns),
            "selected_features": len(selected_features),
            "search_iterations": len(search.cv_results_),
        },
    }

    # Save model
    model_path = OUTPUT_DIR / "ctr_classifier.pkl"  # This line needs to be updated!
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)

    # Save additional outputs
    performance_df = pd.DataFrame([metrics])
    performance_df.to_csv(OUTPUT_DIR / "classification_performance.csv", index=False)
    feature_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    logger.info(f"✅ Model saved to: {model_path}")
    logger.info(f"📊 Additional files saved to: {OUTPUT_DIR}")

    return model_path


def main():
    """Main execution function"""
    logger.info("🚀 Starting Binary Classification Model Training")
    logger.info("=" * 60)

    try:
        # Load and prepare data
        X_train, y_train, X_val, y_val, X_test = load_and_prepare_data()
        y_train_binary, y_val_binary = convert_to_binary_classification(y_train, y_val)

        # Perform model search
        search = perform_model_search(X_train, y_train_binary)
        best_model = search.best_estimator_

        # Evaluate model
        metrics, y_val_pred_proba, y_val_pred = evaluate_model(
            best_model, X_val, y_val_binary
        )

        # Analyze features
        feature_df, selected_features = analyze_features(best_model, X_train)

        # Create visualizations
        create_visualizations(metrics, y_val_binary, y_val_pred_proba, feature_df)

        # Save everything
        model_path = save_model_package(
            search, metrics, feature_df, selected_features, X_train, y_train_binary
        )

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("🏆 BINARY CLASSIFICATION MODEL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Best CV ROC-AUC: {search.best_score_:.4f}")
        logger.info(f"✅ Validation ROC-AUC: {metrics['auc']:.4f}")
        logger.info(
            f"✅ Features Selected: {len(selected_features)}/{len(X_train.columns)}"
        )
        logger.info(f"✅ Training Click Rate: {y_train_binary.mean():.1%}")
        logger.info(f"✅ Model saved to: {model_path}")
        logger.info("🎯 Ready for deployment in Streamlit app!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
