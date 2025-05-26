import os
import pickle
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from lightgbm import LGBMRegressor

# Updated paths to match preprocessing pipeline
PREP_DIR = Path("data/preprocessed")
OUTPUT_DIR = Path("model_output")
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load preprocessed data
X_train = pd.read_parquet(PREP_DIR / "processed_data" / "X_train.parquet")
y_train = pd.read_parquet(PREP_DIR / "processed_data" / "y_train.parquet")["ctr"]
X_val = pd.read_parquet(PREP_DIR / "processed_data" / "X_val.parquet")
y_val = pd.read_parquet(PREP_DIR / "processed_data" / "y_val.parquet")["ctr"]
X_test = pd.read_parquet(PREP_DIR / "processed_data" / "X_test.parquet")
y_test = pd.read_parquet(PREP_DIR / "processed_data" / "y_test.parquet")["ctr"]

# Load the ctr_is_real flags if available
try:
    # Try to load the flags from the original dataframes if saved
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
    # Fallback: reconstruct the flags based on CTR values
    logger.info("Reconstructing ctr_is_real flags from CTR values")

    # For train/val: articles with CTR > 0 are real
    train_flags = y_train > 0
    val_flags = y_val > 0

    # For test: all CTR values are placeholders (since we filled with 0)
    test_flags = pd.Series([False] * len(y_test), index=y_test.index)

logger.info(
    f"Loaded data - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
)

# Filter data using the real CTR flags (not just CTR > 0)
train_mask = train_flags
val_mask = val_flags
# For test set: keep all articles since we want to predict for them
test_mask = pd.Series([True] * len(X_test), index=X_test.index)

X_train_filtered = X_train[train_mask]
y_train_filtered = y_train[train_mask]
X_val_filtered = X_val[val_mask]
y_val_filtered = y_val[val_mask]
# Keep all test articles for prediction
X_test_filtered = X_test[test_mask]
y_test_filtered = y_test[test_mask]  # These are placeholders (all 0)

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
logger.info(f"- Test CTR: all placeholders (mean={y_test_filtered.mean():.4f})")

# Cross-validation
logger.info("Performing 5-fold cross-validation")
model_cv = LGBMRegressor(
    n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1
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

# Train final model
X_combined = pd.concat([X_train_filtered, X_val_filtered])
y_combined = pd.concat([y_train_filtered, y_val_filtered])

logger.info("Training final model")
model = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1)
model.fit(X_combined, y_combined)

# Test set evaluation
preds = model.predict(X_test_filtered)
test_rmse = np.sqrt(np.mean((preds - y_test_filtered) ** 2))
test_mae = np.mean(np.abs(preds - y_test_filtered))
logger.info(f"Test RMSE: {test_rmse:.4f}")
logger.info(f"Test MAE: {test_mae:.4f}")

# Feature importance
importances = model.feature_importances_
feat_names = X_train.columns.tolist()
fi_df = pd.DataFrame({"feature": feat_names, "importance": importances}).nlargest(
    15, "importance"
)

# Visualization 1: Feature importance with categories
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Top features
axes[0].barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
axes[0].set_title("Top 15 Feature Importances")
axes[0].set_xlabel("Importance")

# Feature importance by type
feature_types = {
    "text": [
        f
        for f in feat_names
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
        for f in feat_names
        if any(x in f for x in ["hour", "day_of_week", "weekend", "time_of_day"])
    ],
    "category": [f for f in feat_names if "category" in f],
    "embedding": [f for f in feat_names if "emb_" in f],
}

type_importance = {}
for ftype, features in feature_types.items():
    type_importance[ftype] = fi_df[fi_df["feature"].isin(features)]["importance"].sum()

axes[1].bar(type_importance.keys(), type_importance.values())
axes[1].set_title("Feature Importance by Type")
axes[1].set_ylabel("Total Importance")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# Visualization 2: Model performance metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Predicted vs Actual
max_val = max(max(y_test_filtered), max(preds))
axes[0, 0].scatter(y_test_filtered, preds, alpha=0.5, s=1)
axes[0, 0].plot([0, max_val], [0, max_val], "r--", alpha=0.8)
axes[0, 0].set_xlabel("Actual CTR")
axes[0, 0].set_ylabel("Predicted CTR")
axes[0, 0].set_title("Predicted vs Actual CTR")

# Residuals
residuals = preds - y_test_filtered
axes[0, 1].scatter(preds, residuals, alpha=0.5, s=1)
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

val_preds = model.predict(X_val_filtered)
val_r2 = 1 - np.sum((y_val_filtered - val_preds) ** 2) / np.sum(
    (y_val_filtered - y_val_filtered.mean()) ** 2
)

# Performance summary table (FIXED)
performance_summary = pd.DataFrame(
    {
        "Metric": ["CV RMSE Mean", "CV RMSE Std", "Test RMSE", "Test MAE", "Val R²"],
        "Value": [
            rmse_scores.mean(),
            rmse_scores.std(),
            test_rmse,
            test_mae,
            val_r2,  # Use validation R² instead of test R²
        ],
    }
)

print("\nModel Performance Summary:")
print(performance_summary.round(4))

# Save model and metadata
model_package = {
    "model": model,
    "feature_names": X_train.columns.tolist(),
    "performance": {
        "cv_rmse_mean": rmse_scores.mean(),
        "cv_rmse_std": rmse_scores.std(),
        "test_rmse": test_rmse,
        "test_mae": test_mae,
    },
    "feature_importance": fi_df.to_dict("records"),
}

with open(OUTPUT_DIR / "ctr_regressor.pkl", "wb") as f:
    pickle.dump(model_package, f)

performance_summary.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
fi_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

logger.info(f"Model and results saved to {OUTPUT_DIR}")
logger.info("Training completed successfully")
