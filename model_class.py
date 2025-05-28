import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json
import warnings
from datetime import datetime
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

print("=" * 80)
print("XGBOOST MODEL TRAINING WITH OPTUNA OPTIMIZATION")
print("=" * 80)

# Configuration
PREP_DIR = Path("data/preprocessed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR / "plots").mkdir(exist_ok=True)

CONFIG = {
    "optuna_trials": 25,
    "cv_folds": 3,
    "random_state": 42,
    "use_class_weights": False,  # Set to True if using class weights
    # "scale_pos_weight":   # Used if not using class weights
    "top_k_values": [10, 50, 100, 500],
    "ctr_gain_threshold": 0.01,  # Minimum CTR gain to consider model effective
}

# ============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================================================
print("\nStep 1: Loading preprocessed data...")

try:
    X_train = pd.read_parquet(
        PREP_DIR / "processed_data" / "X_train_with_interactions.parquet"
    )
    y_train = pd.read_parquet(PREP_DIR / "processed_data" / "y_train_optimized.parquet")
    X_val = pd.read_parquet(
        PREP_DIR / "processed_data" / "X_val_with_interactions.parquet"
    )
    y_val = pd.read_parquet(PREP_DIR / "processed_data" / "y_val_optimized.parquet")
    X_test = pd.read_parquet(
        PREP_DIR / "processed_data" / "X_test_with_interactions.parquet"
    )

    with open(PREP_DIR / "processed_data" / "preprocessing_metadata.json", "r") as f:
        preprocessing_metadata = json.load(f)

    print("Data loaded successfully!")
    print(f"  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Features: {len(X_train.columns)}")

except Exception as e:
    print(f"Error loading data: {e}")
    print("Please run the EDA preprocessing script first!")
    exit(1)

# ============================================================================
# STEP 2: PREPARE DATA FOR TRAINING
# ============================================================================
print("\nStep 2: Preparing data for XGBoost training...")

# ============================================================================
# STEP 2: PREPARE DATA FOR TRAINING
# ============================================================================
print("\nStep 2: Preparing data for XGBoost training...")

# Remove the old class weight calculation block entirely if you had it here.

# --- SMOTE APPLICATION ---
print("Applying SMOTE to the training data for balancing...")
smote = SMOTE(random_state=CONFIG["random_state"])

# Ensure y_train["high_engagement"] is a 1D array for SMOTE
y_train_labels = y_train["high_engagement"].values.ravel()

# --- KEY CHANGE: Convert X_train to float64 BEFORE applying SMOTE ---
X_train_float = X_train.astype(np.float64)
# --------------------------------------------------------------------

X_train_resampled_np, y_train_resampled = smote.fit_resample(
    X_train_float, y_train_labels
)

# Convert the NumPy array output by SMOTE back to a DataFrame with original column names
# This ensures X_train_resampled is a DataFrame, which DMatrix can handle well.
X_train_resampled = pd.DataFrame(X_train_resampled_np, columns=X_train.columns)

print(
    f"  Original training shape: X_train {X_train.shape}, y_train_labels_shape {y_train_labels.shape}"
)  # Corrected y_train_labels shape print
print(
    f"  Resampled training shape: X_train_resampled {X_train_resampled.shape}, y_train_resampled_labels_shape {y_train_resampled.shape}"
)  # Corrected y_train_resampled shape print
unique_classes_resampled, counts_resampled = np.unique(
    y_train_resampled, return_counts=True
)
print(
    f"  Class distribution after SMOTE: {dict(zip(unique_classes_resampled, counts_resampled))}"
)
# --- END OF SMOTE APPLICATION ---

# Create DMatrices: dtrain now uses resampled data.
# The 'weight' parameter is removed as SMOTE handles balancing.
dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
dval = xgb.DMatrix(
    X_val, label=y_val["high_engagement"]
)  # Validation data is NEVER resampled
dtest = xgb.DMatrix(X_test)  # Test data is NEVER resampled (assuming X_test is defined)

print("Data preparation completed")

# ============================================================================
# STEP 3: OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================================
print(
    f"\nStep 3: Starting Optuna hyperparameter optimization ({CONFIG['optuna_trials']} trials)..."
)


def objective(trial):
    """Optuna objective function for XGBoost hyperparameter optimization"""

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": CONFIG["random_state"],
        "verbosity": 0,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "scale_pos_weight": 1.0,
    }

    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=params["n_estimators"],
        nfold=CONFIG["cv_folds"],
        stratified=True,
        shuffle=True,
        seed=CONFIG["random_state"],
        verbose_eval=0,
        # Removed return_trained_models parameter
    )

    best_score = cv_results["test-auc-mean"].max()
    return best_score


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=CONFIG["random_state"]),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
)

print("Starting optimization...")
study.optimize(objective, n_trials=CONFIG["optuna_trials"], show_progress_bar=True)

print("Hyperparameter optimization completed!")
print(f"Best trial score (AUC): {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")

# ============================================================================
# STEP 4: TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================================================
print("\nStep 4: Training final model with optimized parameters...")

best_params = study.best_params.copy()
best_params.update(
    {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": CONFIG["random_state"],
        "verbosity": 1,
    }
)

final_model = xgb.XGBClassifier(**best_params)

final_model.fit(
    X_train_resampled,
    y_train_resampled,
    eval_set=[(X_val, y_val["high_engagement"])],
    verbose=0,
)

print("Final model training completed!")

# ============================================================================
# STEP 5: MODEL EVALUATION
# ============================================================================
print("\nStep 5: Comprehensive model evaluation...")


def evaluate_model_comprehensive(model, X_val, y_val):
    """Comprehensive evaluation including editorial metrics"""

    results = {}

    try:
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
    except Exception as e:
        print(f"Error making predictions: {e}")
        return {}

    results["auc"] = roc_auc_score(y_val["high_engagement"], y_pred_proba)
    results["precision"] = precision_score(y_val["high_engagement"], y_pred)
    results["recall"] = recall_score(y_val["high_engagement"], y_pred)
    results["f1"] = f1_score(y_val["high_engagement"], y_pred)

    print("  Computing editorial metrics...")

    # Top-K Precision (critical for editorial ranking)
    for k in CONFIG["top_k_values"]:
        if k <= len(y_pred_proba):
            top_k_indices = np.argsort(y_pred_proba)[-k:]
            top_k_precision = y_val["high_engagement"].iloc[top_k_indices].mean()
            results[f"top_{k}_precision"] = top_k_precision

    # CTR Gain Analysis
    predicted_high_engagement = y_pred_proba > 0.5
    actual_ctr = y_val["ctr"]

    baseline_ctr = actual_ctr.mean()
    predicted_high_ctr = actual_ctr[predicted_high_engagement].mean()
    ctr_lift = (
        (predicted_high_ctr - baseline_ctr) / baseline_ctr if baseline_ctr > 0 else 0
    )

    results["baseline_ctr"] = baseline_ctr
    results["predicted_high_ctr"] = predicted_high_ctr
    results["ctr_lift"] = ctr_lift
    results["ctr_gain_achieved"] = predicted_high_ctr - baseline_ctr

    # Editorial ranking effectiveness
    ctr_ranking_correlation = np.corrcoef(y_pred_proba, actual_ctr)[0, 1]
    results["ctr_ranking_correlation"] = ctr_ranking_correlation

    return results


evaluation_results = evaluate_model_comprehensive(final_model, X_val, y_val)

print("Model Performance Summary:")
print(f"  AUC Score: {evaluation_results['auc']:.4f}")
print(f"  Precision: {evaluation_results['precision']:.4f}")
print(f"  Recall: {evaluation_results['recall']:.4f}")
print(f"  F1 Score: {evaluation_results['f1']:.4f}")

print("\nEditorial Metrics:")
for k in CONFIG["top_k_values"]:
    if f"top_{k}_precision" in evaluation_results:
        print(f"  Top-{k} Precision: {evaluation_results[f'top_{k}_precision']:.4f}")

print("\nCTR Analysis:")
print(f"  Baseline CTR: {evaluation_results['baseline_ctr']:.6f}")
print(
    f"  Predicted High-Engagement CTR: {evaluation_results['predicted_high_ctr']:.6f}"
)
print(f"  CTR Lift: {evaluation_results['ctr_lift']:.2%}")
print(f"  CTR Gain: {evaluation_results['ctr_gain_achieved']:.6f}")
print(f"  CTR-Ranking Correlation: {evaluation_results['ctr_ranking_correlation']:.4f}")

# ============================================================================
# STEP 6: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\nStep 6: Analyzing feature importance...")

feature_importance = pd.DataFrame(
    {"feature": X_train.columns, "importance": final_model.feature_importances_}
).sort_values("importance", ascending=False)

print("Top 15 Most Important Features:")
for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
    print(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")

feature_importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

# ============================================================================
# STEP 7: SAVE MODEL AND RESULTS
# ============================================================================
print("\nStep 7: Saving model and results...")

model_path = MODEL_DIR / "xgboost_optimized_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)

study_path = MODEL_DIR / "optuna_study.pkl"
with open(study_path, "wb") as f:
    pickle.dump(study, f)

test_predictions = final_model.predict_proba(X_test)[:, 1]
test_predictions_df = pd.DataFrame(
    {
        "prediction_proba": test_predictions,
        "prediction_binary": final_model.predict(X_test),
    }
)
test_predictions_df.to_parquet(MODEL_DIR / "test_predictions.parquet")


# Helper function (can be defined before model_metadata or inline if preferred)
def convert_to_json_safe(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, Path):  # Path objects are not serializable
        return str(value)
    if isinstance(value, datetime):  # datetime objects need isoformat
        return value.isoformat()
    # Add more type checks if other non-serializable types are found
    return value  # For strings or other types already serializable


model_metadata = {
    "model_type": "XGBoost_Optuna_Optimized",
    "training_timestamp": convert_to_json_safe(
        datetime.now()
    ),  # Ensures datetime is handled
    "best_parameters": {
        k: convert_to_json_safe(v) for k, v in best_params.items()
    },  # Ensure standard types from Optuna
    "optuna_trials": int(CONFIG["optuna_trials"]),
    "best_auc_score": float(study.best_value),
    "final_evaluation": {
        k: convert_to_json_safe(v) for k, v in evaluation_results.items()
    },
    "feature_count": int(len(X_train.columns)),
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "training_samples_original": int(len(X_train)),  # Original size
    "training_samples_after_smote": int(len(X_train_resampled)),  # Size after SMOTE
    "resampling_method_used": "SMOTE",
    "editorial_criteria_met": {
        "auc_threshold_0.75": bool(evaluation_results.get("auc", 0.0) >= 0.75),
        "ctr_gain_achieved": bool(
            evaluation_results.get("ctr_gain_achieved", 0.0)
            >= CONFIG["ctr_gain_threshold"]
        ),
        "top_100_precision_0.5": bool(
            evaluation_results.get("top_100_precision", 0.0) >= 0.5
        ),
    },
    "files_created": [
        "xgboost_optimized_model.pkl",
        "optuna_study.pkl",
        "test_predictions.parquet",
        "feature_importance.csv",
        "model_metadata.json",
        "plots/model_evaluation_dashboard.png",
    ],
}

# The json.dump call remains the same:
with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)
# ============================================================================
# STEP 8: CREATE VISUALIZATION PLOTS
# ============================================================================
print("\nStep 8: Creating evaluation plots...")

plt.style.use("default")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Feature Importance
top_features = feature_importance.head(20)
axes[0, 0].barh(range(len(top_features)), top_features["importance"])
axes[0, 0].set_yticks(range(len(top_features)))
axes[0, 0].set_yticklabels(top_features["feature"])
axes[0, 0].set_xlabel("Feature Importance")
axes[0, 0].set_title("Top 20 Feature Importance")
axes[0, 0].invert_yaxis()

# Plot 2: Optuna Optimization History
trial_values = [trial.value for trial in study.trials if trial.value is not None]
axes[0, 1].plot(trial_values)
axes[0, 1].set_xlabel("Trial")
axes[0, 1].set_ylabel("AUC Score")
axes[0, 1].set_title("Optuna Optimization Progress")
axes[0, 1].grid(True)

# Plot 3: Prediction Distribution
val_predictions = final_model.predict_proba(X_val)[:, 1]
axes[1, 0].hist(val_predictions, bins=50, alpha=0.7, edgecolor="black")
axes[1, 0].set_xlabel("Prediction Probability")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].set_title("Prediction Probability Distribution")
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: CTR vs Prediction Scatter
sample_size = min(1000, len(val_predictions))
sample_idx = np.random.choice(len(val_predictions), sample_size, replace=False)
axes[1, 1].scatter(
    val_predictions[sample_idx], y_val["ctr"].iloc[sample_idx], alpha=0.6, s=20
)
axes[1, 1].set_xlabel("Predicted Engagement Probability")
axes[1, 1].set_ylabel("Actual CTR")
axes[1, 1].set_title("Predicted Engagement vs Actual CTR")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    MODEL_DIR / "plots" / "model_evaluation_dashboard.png", dpi=300, bbox_inches="tight"
)
plt.close()

print("Evaluation plots saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("XGBOOST MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 80)

print(f"\nMODEL PERFORMANCE:")
print(f"  AUC Score: {evaluation_results['auc']:.4f}")
print(f"  Best Optuna Score: {study.best_value:.4f}")
print(f"  CTR Gain Achieved: {evaluation_results['ctr_gain_achieved']:.6f}")
print(f"  Top-100 Precision: {evaluation_results.get('top_100_precision', 0):.4f}")

print(f"\nEDITORIAL CRITERIA ASSESSMENT:")
criteria_met = model_metadata["editorial_criteria_met"]
for criterion, met in criteria_met.items():
    status = "PASS" if met else "FAIL"
    print(f"  {status}: {criterion}")

print(f"\nOPTIMIZATION RESULTS:")
print(f"  Trials completed: {len(study.trials)}")
print(f"  Best trial: #{study.best_trial.number}")
print(f"  Hyperparameters optimized: {len(best_params)}")

print(f"\nFILES CREATED:")
for file in model_metadata["files_created"]:
    print(f"  {file}")

print(f"\nREADY FOR INTEGRATION:")
print(f"  Model available for editorial dashboard")
print(f"  Predictions ready for headline optimization")
print(f"  Performance metrics saved for monitoring")

print("=" * 80)
