import os
import pickle
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMRegressor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get version from environment or default to 1
VERSION = int(os.getenv("MODEL_VERSION", "1"))

# Updated paths to match preprocessing pipeline and versioning
PREP_DIR = Path("data/preprocessed")
BASE_OUTPUT_DIR = Path("model_output")
BASE_OUTPUT_DIR.mkdir(exist_ok=True)

# Create version-specific output directory
OUTPUT_DIR = BASE_OUTPUT_DIR / f"v{VERSION}"
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Training CTR model version {VERSION}")
logger.info(f"Output directory: {OUTPUT_DIR}")

# Load preprocessed data
try:
    X_train = pd.read_parquet(PREP_DIR / "processed_data" / "X_train.parquet")
    y_train = pd.read_parquet(PREP_DIR / "processed_data" / "y_train.parquet")["ctr"]
    X_val = pd.read_parquet(PREP_DIR / "processed_data" / "X_val.parquet")
    y_val = pd.read_parquet(PREP_DIR / "processed_data" / "y_val.parquet")["ctr"]
    X_test = pd.read_parquet(PREP_DIR / "processed_data" / "X_test.parquet")
    y_test = pd.read_parquet(PREP_DIR / "processed_data" / "y_test.parquet")["ctr"]
except FileNotFoundError as e:
    logger.error(f"Could not load data files: {e}")
    logger.error("Make sure preprocessing has been completed first")
    exit(1)

# Load the ctr_is_real flags if available
try:
    train_flags = pd.read_parquet(PREP_DIR / "processed_data" / "train_flags.parquet")[
        "ctr_is_real"
    ]
    val_flags = pd.read_parquet(PREP_DIR / "processed_data" / "val_flags.parquet")[
        "ctr_is_real"
    ]
    test_flags = pd.read_parquet(PREP_DIR / "processed_data" / "test_flags.parquet")[
        "ctr_is_real"
    ]
    logger.info("Loaded ctr_is_real flags from saved files")
except FileNotFoundError:
    logger.info("Reconstructing ctr_is_real flags from CTR values")
    train_flags = y_train >= 0  # Include zero CTR
    val_flags = y_val >= 0  # Include zero CTR
    test_flags = pd.Series([False] * len(y_test), index=y_test.index)

logger.info(
    f"Loaded data - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
)

# Filter data using the real CTR flags
train_mask = train_flags
val_mask = val_flags
test_mask = pd.Series([True] * len(X_test), index=X_test.index)

X_train_filtered = X_train[train_mask]
y_train_filtered = y_train[train_mask]
X_val_filtered = X_val[val_mask]
y_val_filtered = y_val[val_mask]
X_test_filtered = X_test[test_mask]
y_test_filtered = y_test[test_mask]

logger.info(f"After filtering for real CTR data:")
logger.info(f"- Train: {len(X_train_filtered)} articles with real CTR")
logger.info(f"- Val: {len(X_val_filtered)} articles with real CTR")
logger.info(f"- Test: {len(X_test_filtered)} articles (for prediction)")

logger.info(f"CTR statistics (real data only):")
logger.info(
    f"- Train CTR: mean={y_train_filtered.mean():.4f}, std={y_train_filtered.std():.4f}"
)
logger.info(
    f"- Val CTR: mean={y_val_filtered.mean():.4f}, std={y_val_filtered.std():.4f}"
)

# Cross-validation
logger.info("Performing 5-fold cross-validation")
model_cv = LGBMRegressor(
    n_estimators=750, learning_rate=0.05, random_state=42, verbose=-1
)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    model_cv,
    X_train_filtered,
    y_train_filtered,
    cv=cv,
    scoring="neg_root_mean_squared_error",
)
rmse_scores = -scores
logger.info(f"CV RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")

# CTR Distribution Analysis
print("=" * 60)
print("🔍 CTR DISTRIBUTION ANALYSIS")
print("=" * 60)

print(f"\n📊 BASIC CTR STATISTICS:")
print(f"Training CTR stats:")
print(y_train_filtered.describe())

print(f"\nValidation CTR stats:")
print(y_val_filtered.describe())


def analyze_ctr_distribution(ctr_data, name):
    total = len(ctr_data)
    print(f"\n{name} CTR Distribution:")
    print(f"- Total articles: {total:,}")
    print(f"- Zero CTR: {(ctr_data == 0).sum():,} ({(ctr_data == 0).mean()*100:.1f}%)")
    print(f"- CTR > 0: {(ctr_data > 0).sum():,} ({(ctr_data > 0).mean()*100:.1f}%)")
    print(
        f"- CTR > 0.01: {(ctr_data > 0.01).sum():,} ({(ctr_data > 0.01).mean()*100:.1f}%)"
    )
    print(
        f"- CTR > 0.05: {(ctr_data > 0.05).sum():,} ({(ctr_data > 0.05).mean()*100:.1f}%)"
    )
    print(
        f"- CTR > 0.1: {(ctr_data > 0.1).sum():,} ({(ctr_data > 0.1).mean()*100:.1f}%)"
    )
    print(
        f"- CTR > 0.2: {(ctr_data > 0.2).sum():,} ({(ctr_data > 0.2).mean()*100:.1f}%)"
    )
    print(
        f"- CTR > 0.5: {(ctr_data > 0.5).sum():,} ({(ctr_data > 0.5).mean()*100:.1f}%)"
    )
    print(
        f"- CTR = 1.0: {(ctr_data == 1.0).sum():,} ({(ctr_data == 1.0).mean()*100:.1f}%)"
    )

    percentiles = [50, 75, 90, 95, 99, 99.9]
    print(f"\nCTR Percentiles:")
    for p in percentiles:
        value = np.percentile(ctr_data, p)
        print(f"- {p}th percentile: {value:.4f}")


analyze_ctr_distribution(y_train_filtered, "TRAINING")
analyze_ctr_distribution(y_val_filtered, "VALIDATION")

print(f"\n🚨 POTENTIAL DATA ISSUES:")
extreme_high = (y_train_filtered > 0.8).sum()
extreme_low = (y_train_filtered == 0).sum()
print(f"- Articles with CTR > 80%: {extreme_high} (might be outliers)")
print(f"- Articles with CTR = 0%: {extreme_low} (might dominate model)")

print(f"\n🔧 CTR CALCULATION INTEGRITY:")
print(f"- Min CTR: {y_train_filtered.min():.6f}")
print(f"- Max CTR: {y_train_filtered.max():.6f}")
print(f"- Any negative CTR: {(y_train_filtered < 0).sum()}")
print(f"- Any CTR > 1.0: {(y_train_filtered > 1.0).sum()}")

# Combine datasets for training
X_combined = pd.concat([X_train_filtered, X_val_filtered])
y_combined = pd.concat([y_train_filtered, y_val_filtered])

# BASELINE MODEL
logger.info("Training baseline model")
model_baseline = LGBMRegressor(
    n_estimators=750,
    learning_rate=0.05,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
    objective="regression",
    metric="rmse",
    boosting_type="gbdt",
    num_leaves=31,
    max_depth=-1,
    min_data_in_leaf=20,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l1=0.1,
    lambda_l2=0.1,
)
model_baseline.fit(X_combined, y_combined)

# Evaluate baseline
val_preds_baseline = model_baseline.predict(X_val_filtered)
rmse_baseline = np.sqrt(np.mean((val_preds_baseline - y_val_filtered) ** 2))
mae_baseline = np.mean(np.abs(val_preds_baseline - y_val_filtered))
r2_baseline = 1 - np.sum((y_val_filtered - val_preds_baseline) ** 2) / np.sum(
    (y_val_filtered - y_val_filtered.mean()) ** 2
)

print(f"\n🎯 BASELINE MODEL RESULTS:")
print(f"- RMSE: {rmse_baseline:.4f}")
print(f"- MAE: {mae_baseline:.4f}")
print(f"- R²: {r2_baseline:.4f}")

# APPROACH 1: WEIGHTED TRAINING
print("\n🎯 APPROACH 1: Weighted Training")

sample_weights_train = np.where(y_train_filtered > 0, 2.0, 1.0)
sample_weights_val = np.where(y_val_filtered > 0, 2.0, 1.0)
sample_weights_combined = np.concatenate([sample_weights_train, sample_weights_val])

print(f"Zero CTR articles weight: 1.0")
print(f"Non-zero CTR articles weight: 2.0")
print(f"Zero CTR count: {(y_combined == 0).sum()}")
print(f"Non-zero CTR count: {(y_combined > 0).sum()}")

model_weighted = LGBMRegressor(
    n_estimators=750,
    learning_rate=0.05,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
    objective="regression",
    metric="rmse",
    boosting_type="gbdt",
    num_leaves=31,
    max_depth=-1,
    min_data_in_leaf=20,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l1=0.1,
    lambda_l2=0.1,
)
model_weighted.fit(X_combined, y_combined, sample_weight=sample_weights_combined)

val_preds_weighted = model_weighted.predict(X_val_filtered)
rmse_weighted = np.sqrt(np.mean((val_preds_weighted - y_val_filtered) ** 2))
mae_weighted = np.mean(np.abs(val_preds_weighted - y_val_filtered))
r2_weighted = 1 - np.sum((y_val_filtered - val_preds_weighted) ** 2) / np.sum(
    (y_val_filtered - y_val_filtered.mean()) ** 2
)

print(f"Weighted Model Results:")
print(f"- RMSE: {rmse_weighted:.4f}")
print(f"- MAE: {mae_weighted:.4f}")
print(f"- R²: {r2_weighted:.4f}")

# APPROACH 2: TWO-STAGE MODEL
print("\n🎯 APPROACH 2: Two-Stage Model")

try:
    # Stage 1: Classification (CTR > 0 or not)
    print("Stage 1: Training classification model (CTR > 0)")

    y_binary = (y_combined > 0).astype(int)
    print(f"Articles with CTR > 0: {y_binary.sum()} ({y_binary.mean()*100:.1f}%)")
    print(
        f"Articles with CTR = 0: {len(y_binary) - y_binary.sum()} ({(1-y_binary.mean())*100:.1f}%)"
    )

    classifier = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=4, max_depth=15, max_features="sqrt"
    )

    print("Training classifier...")
    classifier.fit(X_combined, y_binary)
    print("✅ Classifier training completed")

    # Check classifier performance
    val_binary_actual = (y_val_filtered > 0).astype(int)
    val_binary_pred = classifier.predict(X_val_filtered)
    val_binary_prob = classifier.predict_proba(X_val_filtered)[:, 1]

    binary_accuracy = accuracy_score(val_binary_actual, val_binary_pred)
    binary_auc = roc_auc_score(val_binary_actual, val_binary_prob)

    print(f"Binary classifier performance:")
    print(f"- Accuracy: {binary_accuracy:.4f}")
    print(f"- AUC: {binary_auc:.4f}")

    # Stage 2: Regression (on non-zero CTR only)
    print("\nStage 2: Training regression model (non-zero CTR only)")

    nonzero_mask = y_combined > 0
    X_nonzero = X_combined[nonzero_mask]
    y_nonzero = y_combined[nonzero_mask]

    print(f"Non-zero CTR training articles: {len(X_nonzero)}")
    print(f"Non-zero CTR range: {y_nonzero.min():.4f} to {y_nonzero.max():.4f}")
    print(f"Non-zero CTR mean: {y_nonzero.mean():.4f}")

    regressor = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
        n_jobs=4,
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=-1,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l1=0.1,
        lambda_l2=0.1,
    )

    print("Training regressor...")
    regressor.fit(X_nonzero, y_nonzero)
    print("✅ Regressor training completed")

    # Two-stage prediction
    print("\nMaking two-stage predictions...")

    val_binary_prob = classifier.predict_proba(X_val_filtered)[:, 1]
    val_ctr_preds = regressor.predict(X_val_filtered)
    val_preds_twostage = val_binary_prob * val_ctr_preds

    print(
        f"Binary probability range: {val_binary_prob.min():.4f} to {val_binary_prob.max():.4f}"
    )
    print(
        f"CTR prediction range: {val_ctr_preds.min():.4f} to {val_ctr_preds.max():.4f}"
    )
    print(
        f"Combined prediction range: {val_preds_twostage.min():.4f} to {val_preds_twostage.max():.4f}"
    )

    rmse_twostage = np.sqrt(np.mean((val_preds_twostage - y_val_filtered) ** 2))
    mae_twostage = np.mean(np.abs(val_preds_twostage - y_val_filtered))
    r2_twostage = 1 - np.sum((y_val_filtered - val_preds_twostage) ** 2) / np.sum(
        (y_val_filtered - y_val_filtered.mean()) ** 2
    )

    print(f"\nTwo-Stage Model Results:")
    print(f"- RMSE: {rmse_twostage:.4f}")
    print(f"- MAE: {mae_twostage:.4f}")
    print(f"- R²: {r2_twostage:.4f}")

    # Evaluate regressor alone (on non-zero CTR articles)
    val_nonzero_mask = y_val_filtered > 0
    if val_nonzero_mask.sum() > 0:
        val_nonzero_actual = y_val_filtered[val_nonzero_mask]
        val_nonzero_pred = regressor.predict(X_val_filtered[val_nonzero_mask])

        rmse_regressor = np.sqrt(np.mean((val_nonzero_pred - val_nonzero_actual) ** 2))
        r2_regressor = 1 - np.sum(
            (val_nonzero_actual - val_nonzero_pred) ** 2
        ) / np.sum((val_nonzero_actual - val_nonzero_actual.mean()) ** 2)

        print(f"\nRegressor Performance (non-zero CTR only):")
        print(f"- RMSE: {rmse_regressor:.4f}")
        print(f"- R²: {r2_regressor:.4f}")
        print(f"- Sample size: {len(val_nonzero_actual)}")

    two_stage_success = True

except Exception as e:
    print(f"❌ Two-stage model failed: {str(e)}")
    rmse_twostage = 999.0
    mae_twostage = 999.0
    r2_twostage = -999.0
    two_stage_success = False

# FINAL COMPARISON
print("\n📊 FINAL MODEL COMPARISON")
print("=" * 60)

if two_stage_success:
    models = ["Baseline", "Weighted", "Two-Stage"]
    rmse_values = [rmse_baseline, rmse_weighted, rmse_twostage]
    mae_values = [mae_baseline, mae_weighted, mae_twostage]
    r2_values = [r2_baseline, r2_weighted, r2_twostage]
    model_objects = [
        model_baseline,
        model_weighted,
        {"classifier": classifier, "regressor": regressor},
    ]
    predictions = [val_preds_baseline, val_preds_weighted, val_preds_twostage]
else:
    models = ["Baseline", "Weighted"]
    rmse_values = [rmse_baseline, rmse_weighted]
    mae_values = [mae_baseline, mae_weighted]
    r2_values = [r2_baseline, r2_weighted]
    model_objects = [model_baseline, model_weighted]
    predictions = [val_preds_baseline, val_preds_weighted]

results_comparison = pd.DataFrame(
    {"Model": models, "RMSE": rmse_values, "MAE": mae_values, "R²": r2_values}
)

print(results_comparison.round(4))

# Find best model
best_r2_idx = results_comparison["R²"].idxmax()
best_model_name = results_comparison.loc[best_r2_idx, "Model"]
best_r2_value = results_comparison.loc[best_r2_idx, "R²"]
best_model = model_objects[best_r2_idx]
best_predictions = predictions[best_r2_idx]

print(f"\n🏆 Best model by R²: {best_model_name} (R² = {best_r2_value:.4f})")

if best_r2_value > r2_baseline:
    print(f"✅ {best_model_name} model shows improvement over baseline!")
    if best_model_name == "Two-Stage":
        print("💡 Two-stage approach works well for highly imbalanced data like this.")
else:
    print(f"✅ Baseline model remains the best choice.")
    print("💡 Given 87% zero CTR, this performance is actually reasonable.")

# VISUALIZATIONS
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Predicted vs Actual
max_val = max(max(y_val_filtered), max(best_predictions))
axes[0, 0].scatter(y_val_filtered, best_predictions, alpha=0.5, s=1)
axes[0, 0].plot([0, max_val], [0, max_val], "r--", alpha=0.8)
axes[0, 0].set_xlabel("Actual CTR")
axes[0, 0].set_ylabel("Predicted CTR")
axes[0, 0].set_title(f"Predicted vs Actual CTR ({best_model_name})")

# Residuals
residuals = best_predictions - y_val_filtered
axes[0, 1].scatter(best_predictions, residuals, alpha=0.5, s=1)
axes[0, 1].axhline(y=0, color="r", linestyle="--", alpha=0.8)
axes[0, 1].set_xlabel("Predicted CTR")
axes[0, 1].set_ylabel("Residuals")
axes[0, 1].set_title("Residual Plot")

# Error distribution
axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor="black")
axes[1, 0].set_xlabel("Prediction Error")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].set_title("Error Distribution")

# CV scores
axes[1, 1].bar(range(len(rmse_scores)), rmse_scores)
axes[1, 1].axhline(y=rmse_scores.mean(), color="r", linestyle="--", alpha=0.8)
axes[1, 1].set_xlabel("CV Fold")
axes[1, 1].set_ylabel("RMSE")
axes[1, 1].set_title("Cross-Validation RMSE")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_performance.png", dpi=300, bbox_inches="tight")
plt.close()

# Save best model
if best_model_name != "Two-Stage":
    final_model = best_model
else:
    final_model = best_model  # Dictionary with classifier and regressor

model_package = {
    "version": VERSION,
    "model_type": best_model_name,
    "model": final_model,
    "feature_names": X_train.columns.tolist(),
    "performance": {
        "cv_rmse_mean": rmse_scores.mean(),
        "cv_rmse_std": rmse_scores.std(),
        "val_rmse": rmse_values[best_r2_idx],
        "val_mae": mae_values[best_r2_idx],
        "val_r2": best_r2_value,
        "best_model": best_model_name,
    },
}

# Save to version-specific directory
with open(OUTPUT_DIR / "ctr_regressor.pkl", "wb") as f:
    pickle.dump(model_package, f)

# Also save to root directory with version suffix for compatibility
with open(BASE_OUTPUT_DIR / f"ctr_regressor_{VERSION}.pkl", "wb") as f:
    pickle.dump(model_package, f)

# Save performance comparison
results_comparison.to_csv(OUTPUT_DIR / "performance_comparison.csv", index=False)

logger.info(f"Version {VERSION} model and results saved to:")
logger.info(f"- {OUTPUT_DIR}/")
logger.info(f"- {BASE_OUTPUT_DIR}/ctr_regressor_{VERSION}.pkl (compatibility)")
logger.info("Training completed successfully")

print(f"\n📋 FINAL RECOMMENDATION:")
print(f"Use the {best_model_name} model for production.")
print(f"Performance: RMSE = {rmse_values[best_r2_idx]:.4f}, R² = {best_r2_value:.4f}")
print(f"\nFiles created in {OUTPUT_DIR}:")
print("- ctr_regressor.pkl")
print("- model_performance.png")
print("- performance_comparison.csv")
print(f"\nCompatibility file: {BASE_OUTPUT_DIR}/ctr_regressor_{VERSION}.pkl")
print(f"\nTraining completed successfully for version {VERSION}!")
# End of CTR_regression_model.py
