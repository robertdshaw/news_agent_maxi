import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json

import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    log_loss,  # Added log loss for better evaluation
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
from scipy.stats import uniform, randint, ks_2samp
import optuna

# Configuration
PREP_DIR = Path("data/preprocessed")
OUTPUT_DIR = Path("model_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("AI NEWS EDITOR ASSISTANT - CLEAN TIME-AWARE CTR PREDICTION MODEL (NO LEAKAGE)")
print("=" * 60)

print("Loading time-sequential data...")

# Load data (maintaining time-based splits)
X_train = pd.read_parquet(PREP_DIR / "processed_data" / "X_train_clean.parquet")
y_train = pd.read_parquet(PREP_DIR / "processed_data" / "y_train_clean.parquet")["ctr"]
X_val = pd.read_parquet(PREP_DIR / "processed_data" / "X_val_clean.parquet")
y_val = pd.read_parquet(PREP_DIR / "processed_data" / "y_val_clean.parquet")["ctr"]
X_test = pd.read_parquet(PREP_DIR / "processed_data" / "X_test_clean.parquet")
y_test = pd.read_parquet(PREP_DIR / "processed_data" / "y_test_clean.parquet")["ctr"]

print(f"Time-sequential splits loaded:")
print(f"  Train (oldest): {len(X_train):,} articles")
print(f"  Validation (middle): {len(X_val):,} articles")
print(f"  Test (newest): {len(X_test):,} articles")
print(f"  Features: {len(X_train.columns):,}")

# Remove low-correlation features based on EDA
low_value_features = [
    "is_weekend",
    "day_of_week",
    "month",
    "day_of_month",
    "title_reading_ease",
    "title_upper_ratio",
    "abstract_reading_ease",
    "abstract_word_count",
    "abstract_length",
]

# Filter features that exist
features_to_drop = [f for f in low_value_features if f in X_train.columns]
X_train = X_train.drop(columns=features_to_drop)
X_val = X_val.drop(columns=features_to_drop)
X_test = X_test.drop(columns=features_to_drop)

print(f"Removed {len(features_to_drop)} low-correlation features")
print(f"New feature count: {len(X_train.columns)}")

# Convert CTR to binary classification (high vs low engagement)
ctr_threshold = y_train.quantile(
    0.7
)  # Top 30% as high engagement based on TRAIN data only
y_train_binary = (y_train > ctr_threshold).astype(int)
y_val_binary = (y_val > ctr_threshold).astype(int)
# Test set labels may be unknown (NaN) - that's expected for real deployment

print(f"\nEngagement Classification Setup:")
print(f"  CTR threshold (70th percentile from TRAIN): {ctr_threshold:.4f}")
print(
    f"  Train high engagement: {y_train_binary.mean():.3f} ({y_train_binary.sum():,} / {len(y_train_binary):,})"
)
print(
    f"  Val high engagement: {y_val_binary.mean():.3f} ({y_val_binary.sum():,} / {len(y_val_binary):,})"
)

# Handle missing values
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)
X_test = X_test.fillna(0)

# Concept drift detection
print("\nConcept Drift Analysis:")
print("-" * 30)


def detect_concept_drift(X_old, X_new, feature_names, top_n=10):
    """Detect concept drift between time periods using KS test."""
    drift_results = []

    for feature in feature_names[:top_n]:  # Check top features to avoid noise
        if feature in X_old.columns and feature in X_new.columns:
            try:
                stat, p_value = ks_2samp(X_old[feature], X_new[feature])
                drift_results.append(
                    {
                        "feature": feature,
                        "ks_statistic": float(stat),
                        "p_value": float(p_value),
                        "significant_drift": bool(p_value < 0.05),
                    }
                )
            except Exception as e:
                print(f"Warning: Could not compute drift for {feature}: {e}")

    return sorted(drift_results, key=lambda x: x["ks_statistic"], reverse=True)


# Basic drift detection between train and validation
drift_analysis = detect_concept_drift(X_train, X_val, X_train.columns, top_n=20)
significant_drift_count = sum(1 for d in drift_analysis if d["significant_drift"])

print(f"Features with significant drift (p<0.05): {significant_drift_count}/20")
if significant_drift_count > 0:
    print("Top 5 features with most drift:")
    for d in drift_analysis[:5]:
        status = "⚠️ DRIFT" if d["significant_drift"] else "✅ STABLE"
        print(f"  {d['feature'][:30]:<30} KS={d['ks_statistic']:.3f} {status}")

# Class imbalance handling
print(f"\nClass Imbalance Handling:")
print("-" * 25)

# Method 1: Compute class weights from TRAIN data only
class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train_binary), y=y_train_binary
)
scale_pos_weight = class_weights[1] / class_weights[0]
print(
    f"Class weights - Negative: {class_weights[0]:.3f}, Positive: {class_weights[1]:.3f}"
)
print(f"Scale pos weight: {scale_pos_weight:.3f}")

# Method 2: Topic-specific balancing (category-based)
category_weights = {}
if "category_enc" in X_train.columns:
    unique_categories = X_train["category_enc"].unique()
    print(
        f"Analyzing {len(unique_categories)} categories for topic-specific balancing..."
    )

    for cat in unique_categories:
        cat_mask = X_train["category_enc"] == cat
        cat_labels = y_train_binary[cat_mask]

        if len(cat_labels) > 50 and len(cat_labels.unique()) > 1:
            cat_pos_rate = cat_labels.mean()
            overall_pos_rate = y_train_binary.mean()

            if cat_pos_rate > 0:
                weight_factor = overall_pos_rate / cat_pos_rate
                category_weights[cat] = {
                    "samples": len(cat_labels),
                    "positive_rate": cat_pos_rate,
                    "weight_factor": weight_factor,
                }

    print(f"Categories with sufficient data: {len(category_weights)}")

# Feature selection - remove low variance features
print(f"\nFeature Selection:")
print("-" * 17)
feature_variance = X_train.var()
low_variance_features = feature_variance[feature_variance < 1e-6].index
X_train = X_train.drop(columns=low_variance_features)
X_val = X_val.drop(columns=low_variance_features)
X_test = X_test.drop(columns=low_variance_features)

print(f"Removed {len(low_variance_features)} low variance features")
print(f"Final feature count: {len(X_train.columns)}")


# Time-aware cross-validation within training set
def time_aware_cv_split(X, y, n_splits=3):
    """Create time-aware CV splits by using earlier data to predict later data."""
    n_samples = len(X)
    split_size = n_samples // (n_splits + 1)

    for i in range(n_splits):
        train_end = split_size * (i + 1)
        val_start = train_end
        val_end = min(val_start + split_size, n_samples)

        train_idx = list(range(train_end))
        val_idx = list(range(val_start, val_end))

        yield train_idx, val_idx


# Hyperparameter optimization with time-aware CV
def objective(trial):
    """Optuna objective function with time-aware cross-validation."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",  # Changed to log loss for better optimization
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    # Time-aware cross-validation on training set only
    cv_scores = []
    for train_idx, val_idx in time_aware_cv_split(X_train, y_train_binary, n_splits=3):
        X_cv_train = X_train.iloc[train_idx]
        y_cv_train = y_train_binary.iloc[train_idx]
        X_cv_val = X_train.iloc[val_idx]
        y_cv_val = y_train_binary.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_cv_train, y_cv_train)

        val_probs = model.predict_proba(X_cv_val)[:, 1]
        # Use AUC for optimization but we'll track log loss too
        score = roc_auc_score(y_cv_val, val_probs)
        cv_scores.append(score)

    return np.mean(cv_scores)


print(f"\nHyperparameter Optimization (Time-Aware CV):")
print("-" * 45)
study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=13, show_progress_bar=True)

best_params = study.best_params
best_params.update(
    {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
    }
)

print(f"Best parameters: {best_params}")
print(f"Best time-aware CV AUC: {study.best_value:.4f}")

# Train final model on TRAINING SET ONLY
print(f"\nTraining Final Model:")
print("-" * 21)
print("Training on TRAIN set only (preserving time-based validation)")

# CORRECT approach: Set early_stopping_rounds in constructor
try:
    # Method 1: early_stopping_rounds in constructor (recommended)
    final_model = xgb.XGBClassifier(early_stopping_rounds=10, **best_params)
    final_model.fit(
        X_train,
        y_train_binary,
        eval_set=[(X_train, y_train_binary), (X_val, y_val_binary)],
        verbose=False,
    )
    print("Using early stopping (constructor method)")

except Exception as e:
    print(f"Early stopping failed: {e}")
    # Fallback: train without early stopping
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train_binary)
    print("Training without early stopping")

# Calculate training performance
train_probs = final_model.predict_proba(X_train)[:, 1]
train_preds = final_model.predict(X_train)
train_auc = roc_auc_score(y_train_binary, train_probs)
train_logloss = log_loss(y_train_binary, train_probs)

# Model evaluation on validation set (future time period)
print(f"\nModel Evaluation on Future Data (Validation Set):")
print("-" * 50)

val_probs = final_model.predict_proba(X_val)[:, 1]
val_preds = final_model.predict(X_val)
val_auc = roc_auc_score(y_val_binary, val_probs)
val_logloss = log_loss(y_val_binary, val_probs)

print(f"Validation AUC (future performance): {val_auc:.4f}")
print(f"Validation Log Loss (future performance): {val_logloss:.4f}")
print(f"Training AUC: {train_auc:.4f}")
print(f"Training Log Loss: {train_logloss:.4f}")
print(f"Time-aware CV AUC (training): {study.best_value:.4f}")
print(f"AUC Performance difference: {val_auc - study.best_value:.4f}")
print(f"Log Loss (lower is better): Train={train_logloss:.4f}, Val={val_logloss:.4f}")

if val_auc < study.best_value - 0.05:
    print(
        "  WARNING: Significant performance drop on future data (possible concept drift)"
    )
elif val_auc > study.best_value + 0.02:
    print("  ✅ Model generalizes well to future data")
else:
    print("  ✅ Stable performance across time periods")

# Classification report
print(f"\nValidation Set Performance Report:")
print("-" * 35)
print(classification_report(y_val_binary, val_preds))

# Generate predictions for test set (future data without known clicks)
print(f"\nGenerating Predictions for Future Deployment (Test Set):")
print("-" * 55)

test_probs = final_model.predict_proba(X_test)[:, 1]
test_preds = final_model.predict(X_test)

print(f"Test set predictions generated: {len(test_probs):,}")
print(f"Mean predicted engagement probability: {test_probs.mean():.3f}")
print(
    f"Predicted high engagement articles: {test_preds.sum():,} ({test_preds.mean():.1%})"
)

# Feature importance analysis
feature_importance = pd.DataFrame(
    {"feature": X_train.columns, "importance": final_model.feature_importances_}
).sort_values("importance", ascending=False)

print(f"\nTop 15 Most Important Features for Engagement Prediction:")
print("-" * 55)
for i, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:<30} {row['importance']:.4f}")

# Category-specific performance analysis on validation set (FIXED)
category_performance = []
category_df = pd.DataFrame()  # Initialize empty DataFrame

if "category_enc" in X_val.columns:
    print(f"\nTopic-Specific Performance Analysis:")
    print("-" * 35)

    unique_categories = X_val["category_enc"].unique()

    for cat in unique_categories:
        cat_mask = X_val["category_enc"] == cat
        if cat_mask.sum() > 20:  # Ensure enough samples
            cat_labels = y_val_binary[cat_mask]
            cat_probs = val_probs[cat_mask]

            # Check if we have both classes to calculate AUC
            if len(cat_labels.unique()) > 1:
                try:
                    cat_auc = roc_auc_score(cat_labels, cat_probs)
                    cat_logloss = log_loss(cat_labels, cat_probs)
                    cat_baseline = cat_labels.mean()

                    category_performance.append(
                        {
                            "category": int(cat),
                            "samples": int(cat_mask.sum()),
                            "baseline_rate": float(cat_baseline),
                            "auc": float(cat_auc),
                            "log_loss": float(cat_logloss),
                            "lift": float(cat_auc - 0.5),
                            "performance_vs_overall": float(cat_auc - val_auc),
                        }
                    )
                except ValueError as e:
                    print(
                        f"  Warning: Could not calculate metrics for category {cat}: {e}"
                    )

    if category_performance:
        category_df = pd.DataFrame(category_performance).sort_values(
            "auc", ascending=False
        )
        print(f"Top 5 performing topics:")
        for _, row in category_df.head(5).iterrows():
            print(
                f"  Category {int(row['category']):2d}: AUC={row['auc']:.3f}, LogLoss={row['log_loss']:.3f} (n={int(row['samples']):3d})"
            )
    else:
        print("  No categories with sufficient data for analysis")

# Create comprehensive visualizations (WITH ERROR HANDLING)
print(f"\nCreating Analysis Visualizations...")

try:
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))

    # 1. Feature Importance
    top_features = feature_importance.head(20)
    axes[0, 0].barh(range(len(top_features)), top_features["importance"])
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features["feature"], fontsize=8)
    axes[0, 0].set_title("Top 20 Feature Importance")
    axes[0, 0].set_xlabel("Importance")

    # 2. ROC Curve Comparison
    fpr, tpr, _ = roc_curve(y_val_binary, val_probs)
    axes[0, 1].plot(fpr, tpr, label=f"Validation AUC = {val_auc:.3f}", linewidth=2)
    axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].set_title("ROC Curve (Future Data Performance)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Time-Aware Performance
    performance_data = [train_auc, val_auc]
    time_periods = ["Train\n(Past)", "Validation\n(Future)"]

    bars = axes[1, 0].bar(
        time_periods, performance_data, color=["skyblue", "lightcoral"]
    )
    axes[1, 0].set_ylabel("AUC Score")
    axes[1, 0].set_title("Performance Across Time Periods")
    axes[1, 0].set_ylim([0.5, 1.0])

    # Add value labels on bars
    for bar, value in zip(bars, performance_data):
        axes[1, 0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 4. Probability Distribution by Time Period
    axes[1, 1].hist(
        train_probs, bins=50, alpha=0.6, label="Train Predictions", density=True
    )
    axes[1, 1].hist(
        val_probs, bins=50, alpha=0.6, label="Validation Predictions", density=True
    )
    axes[1, 1].set_xlabel("Predicted Engagement Probability")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Prediction Distribution by Time Period")
    axes[1, 1].legend()

    # 5. Confusion Matrix
    cm = confusion_matrix(y_val_binary, val_preds)
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[2, 0], cmap="Blues")
    axes[2, 0].set_title("Confusion Matrix (Validation Set)")
    axes[2, 0].set_xlabel("Predicted")
    axes[2, 0].set_ylabel("Actual")

    # 6. Category Performance (with safety check)
    if len(category_performance) > 0:
        cat_df_plot = category_df.head(10)
        bars = axes[2, 1].bar(range(len(cat_df_plot)), cat_df_plot["auc"])
        axes[2, 1].set_xticks(range(len(cat_df_plot)))
        axes[2, 1].set_xticklabels(
            [f"Cat {int(c)}" for c in cat_df_plot["category"]], rotation=45
        )
        axes[2, 1].set_ylabel("AUC")
        axes[2, 1].set_title("Topic Performance (AUC by Category)")
        axes[2, 1].axhline(
            y=0.5, color="red", linestyle="--", alpha=0.5, label="Random"
        )
        axes[2, 1].axhline(
            y=val_auc, color="green", linestyle="--", alpha=0.5, label="Overall"
        )
        axes[2, 1].legend()
    else:
        axes[2, 1].text(
            0.5,
            0.5,
            "No category data\navailable for plotting",
            ha="center",
            va="center",
            transform=axes[2, 1].transAxes,
            fontsize=12,
        )
        axes[2, 1].set_title("Topic Performance (No Data)")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "time_aware_xgboost_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("✅ Visualizations saved successfully!")

except Exception as e:
    print(f"⚠️  Warning: Could not create visualizations: {e}")
    print("Continuing without plots...")

# Model performance summary with time-awareness (FIXED JSON SERIALIZATION)
performance_summary = {
    "model_type": "Time-Aware XGBoost Classifier",
    "task": "Binary Classification (High vs Low Engagement)",
    "time_aware_setup": True,
    "threshold": float(ctr_threshold),
    "training_auc": float(train_auc),
    "training_log_loss": float(train_logloss),
    "validation_auc": float(val_auc),
    "validation_log_loss": float(val_logloss),
    "time_aware_cv_auc": float(study.best_value),
    "performance_stability": float(val_auc - study.best_value),
    "concept_drift_detected": bool(significant_drift_count > 10),
    "class_balance": {
        "negative_samples": int((y_train_binary == 0).sum()),
        "positive_samples": int((y_train_binary == 1).sum()),
        "scale_pos_weight": float(scale_pos_weight),
    },
    "feature_count": int(len(X_train.columns)),
    "best_params": {
        k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
        for k, v in best_params.items()
    },
    "deployment_ready": bool(val_auc > 0.65),
}

print(f"\n" + "=" * 60)
print("AI NEWS EDITOR ASSISTANT - MODEL SUMMARY")
print("=" * 60)
print(f" Model Type: Time-Aware XGBoost Classifier")
print(f" Training AUC: {train_auc:.4f}")
print(f" Training Log Loss: {train_logloss:.4f}")
print(f" Validation AUC (Future Performance): {val_auc:.4f}")
print(f" Validation Log Loss (Future Performance): {val_logloss:.4f}")
print(f" Performance Stability: {val_auc - study.best_value:+.4f}")
print(
    f" Deployment Ready: {'YES' if performance_summary['deployment_ready'] else 'NO'}"
)
print(f" Concept Drift: {'Detected' if significant_drift_count > 10 else 'Stable'}")

# Save model package for AI News Editor Assistant
model_package = {
    "model": final_model,
    "feature_names": list(X_train.columns),
    "ctr_threshold": float(ctr_threshold),
    "scaler": None,  # XGBoost doesn't need scaling
    "performance": performance_summary,
    "feature_importance": feature_importance.head(50).to_dict("records"),
    "category_performance": category_performance if category_performance else None,
    "class_weights": {"scale_pos_weight": float(scale_pos_weight)},
    "concept_drift_analysis": drift_analysis[:10],  # Top 10 drift features
    "model_version": "2.0_time_aware",
    "created_timestamp": pd.Timestamp.now().isoformat(),
    "time_aware_validation": True,
    "deployment_thresholds": {
        "filter_threshold": 0.4,
        "prioritize_threshold": 0.6,
        "high_confidence_threshold": 0.7,
    },
}

# Save model
try:
    with open(OUTPUT_DIR / "ai_news_editor_model.pkl", "wb") as f:
        pickle.dump(model_package, f)
    print("✅ Model saved successfully!")
except Exception as e:
    print(f"⚠️  Warning: Could not save model: {e}")

# Save predictions with detailed recommendations for AI News Editor
predictions_df = pd.DataFrame(
    {
        "engagement_probability": test_probs,
        "predicted_class": test_preds,
        "engagement_score": (test_probs * 100).round(1),  # 0-100 scale
        "confidence_level": pd.cut(
            test_probs, bins=[0, 0.3, 0.7, 1.0], labels=["Low", "Medium", "High"]
        ),
        "editorial_recommendation": pd.cut(
            test_probs,
            bins=[0, 0.4, 0.6, 1.0],
            labels=["Filter Out", "Consider", "Prioritize"],
        ),
        "ranking_score": (test_probs * 1000).astype(int),  # Integer ranking score
    }
)

try:
    predictions_df.to_csv(OUTPUT_DIR / "ai_news_editor_predictions.csv", index=False)
    print("✅ Predictions saved successfully!")
except Exception as e:
    print(f"⚠️  Warning: Could not save predictions: {e}")

# Save all artifacts
try:
    with open(OUTPUT_DIR / "model_performance.json", "w") as f:
        json.dump(performance_summary, f, indent=2)

    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    if len(category_performance) > 0:
        category_df.to_csv(OUTPUT_DIR / "topic_performance.csv", index=False)

    # Concept drift report
    drift_df = pd.DataFrame(drift_analysis)
    drift_df.to_csv(OUTPUT_DIR / "concept_drift_analysis.csv", index=False)

    print("✅ All performance reports saved!")

except Exception as e:
    print(f"⚠️  Warning: Could not save some reports: {e}")

# AI News Editor Assistant integration guide
integration_guide = {
    "model_file": "ai_news_editor_model.pkl",
    "time_aware_validation": True,
    "metrics_used": {
        "primary": "AUC (Area Under ROC Curve)",
        "secondary": "Log Loss (Cross-entropy)",
        "explanation": "AUC measures ranking ability, Log Loss measures probability calibration",
    },
    "deployment_workflow": """
    1. Load model: ai_news_editor_model.pkl
    2. Process new headlines with same features
    3. Get engagement probabilities (0-1)
    4. Apply thresholds:
       - Filter: < 0.4 (low engagement)
       - Consider: 0.4-0.6 (medium engagement)  
       - Prioritize: > 0.6 (high engagement)
    5. Rank by engagement_probability for ordering
    6. Monitor performance and retrain with new data
    """,
    "monitoring": {
        "track_actual_vs_predicted": "Compare predictions to actual CTR weekly",
        "concept_drift": "Run drift analysis monthly",
        "performance_threshold": "Retrain if validation AUC drops below 0.60",
        "log_loss_threshold": "Monitor if log loss increases significantly",
    },
    "a_b_testing": {
        "headline_comparison": "Use engagement_probability to rank variations",
        "statistical_significance": "Require 100+ impressions per variation",
        "winner_selection": "Choose headline with highest probability",
    },
}

try:
    with open(OUTPUT_DIR / "ai_news_editor_integration.json", "w") as f:
        json.dump(integration_guide, f, indent=2)
    print("✅ Integration guide saved!")
except Exception as e:
    print(f"⚠️  Warning: Could not save integration guide: {e}")

print(f"\n🎉 AI NEWS EDITOR ASSISTANT MODEL READY!")
print(f" Files created:")
print(f"   - ai_news_editor_model.pkl (main model)")
print(f"   - ai_news_editor_predictions.csv (test predictions)")
print(f"   - concept_drift_analysis.csv (time stability)")
if len(category_performance) > 0:
    print(f"   - topic_performance.csv (category insights)")
print(f"   - ai_news_editor_integration.json (deployment guide)")
print(f"   - model_performance.json (metrics summary)")
print(f"   - feature_importance.csv (feature analysis)")

print(f"\n📊 Ready for Streamlit Integration:")
print(f"   - Time-aware validation ensures real-world performance")
print(f"   - Concept drift detection for model monitoring")
print(f"   - Topic-specific insights for editorial strategy")
print(f"   - Production-ready thresholds for filtering/ranking")
print(f"   - Log loss tracking for probability calibration")

print(f"\n📈 Model Performance Summary:")
print(f"   - AUC Score: {val_auc:.3f} (>0.65 = deployment ready)")
print(f"   - Log Loss: {val_logloss:.3f} (lower = better calibration)")
print(f"   - Stability: {val_auc - study.best_value:+.3f} (time consistency)")
print(f"   - Features: {len(X_train.columns)} (after filtering)")

print("\n" + "=" * 60)
