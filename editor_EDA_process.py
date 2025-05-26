import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
from scipy.stats import uniform, randint

# Paths
PREP_DIR = Path("data/preprocessed")
OUTPUT_DIR = Path("model_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load processed data
X_train = pd.read_parquet(PREP_DIR / "processed_data" / "X_train.parquet")
y_train = pd.read_parquet(PREP_DIR / "processed_data" / "y_train.parquet")["ctr"]
X_val = pd.read_parquet(PREP_DIR / "processed_data" / "X_val.parquet")
y_val = pd.read_parquet(PREP_DIR / "processed_data" / "y_val.parquet")["ctr"]
X_test = pd.read_parquet(PREP_DIR / "processed_data" / "X_test.parquet")
y_test = pd.read_parquet(PREP_DIR / "processed_data" / "y_test.parquet")["ctr"]

# Load PCA data
X_train_pca = pd.read_parquet(PREP_DIR / "processed_data" / "X_train_pca.parquet")
X_val_pca = pd.read_parquet(PREP_DIR / "processed_data" / "X_val_pca.parquet")
X_test_pca = pd.read_parquet(PREP_DIR / "processed_data" / "X_test_pca.parquet")

print(f"Loaded data - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Convert CTR to binary classification
# Use median CTR as threshold
median_ctr = y_train.median()
y_train_binary = (y_train > median_ctr).astype(int)
y_val_binary = (y_val > median_ctr).astype(int)

print(f"CTR threshold (median): {median_ctr:.4f}")
print(
    f"High CTR class distribution - Train: {y_train_binary.mean():.3f}, Val: {y_val_binary.mean():.3f}"
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_pca_scaled = scaler.fit_transform(X_train_pca)
X_val_pca_scaled = scaler.transform(X_val_pca)
X_test_pca_scaled = scaler.transform(X_test_pca)

# Class weights for imbalanced data
class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_train_binary), y=y_train_binary
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"Class weights: {class_weight_dict}")

# Category-specific balancing
categories = (
    X_train.columns[X_train.columns.str.contains("category_enc")][0]
    if any(X_train.columns.str.contains("category_enc"))
    else None
)
if categories:
    category_weights = {}
    unique_cats = X_train[categories].unique()
    for cat in unique_cats:
        cat_mask = X_train[categories] == cat
        cat_labels = y_train_binary[cat_mask]
        if len(cat_labels) > 0 and len(cat_labels.unique()) > 1:
            cat_class_weights = compute_class_weight(
                "balanced", classes=np.unique(cat_labels), y=cat_labels
            )
            category_weights[cat] = {0: cat_class_weights[0], 1: cat_class_weights[1]}

    print(f"Category-specific weights computed for {len(category_weights)} categories")

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    "C": uniform(0.01, 10),
    "penalty": ["l1", "l2", "elasticnet"],
    "solver": ["liblinear", "saga"],
    "l1_ratio": uniform(0, 1),
    "max_iter": [1000, 2000, 3000],
}

# Full feature model
print("Training full feature model...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_full = LogisticRegression(random_state=42, class_weight=class_weight_dict)
random_search_full = RandomizedSearchCV(
    lr_full,
    param_dist,
    n_iter=50,
    cv=cv,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

random_search_full.fit(X_train_scaled, y_train_binary)
best_lr_full = random_search_full.best_estimator_

print(f"Best parameters (full): {random_search_full.best_params_}")
print(f"Best CV score (full): {random_search_full.best_score_:.4f}")

# PCA model
print("Training PCA model...")
lr_pca = LogisticRegression(random_state=42, class_weight=class_weight_dict)
random_search_pca = RandomizedSearchCV(
    lr_pca,
    param_dist,
    n_iter=50,
    cv=cv,
    scoring="roc_auc",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

random_search_pca.fit(X_train_pca_scaled, y_train_binary)
best_lr_pca = random_search_pca.best_estimator_

print(f"Best parameters (PCA): {random_search_pca.best_params_}")
print(f"Best CV score (PCA): {random_search_pca.best_score_:.4f}")

# Cross-validation scores
cv_scores_full = cross_val_score(
    best_lr_full, X_train_scaled, y_train_binary, cv=cv, scoring="roc_auc"
)
cv_scores_pca = cross_val_score(
    best_lr_pca, X_train_pca_scaled, y_train_binary, cv=cv, scoring="roc_auc"
)

print(f"Full model CV AUC: {cv_scores_full.mean():.4f} ± {cv_scores_full.std():.4f}")
print(f"PCA model CV AUC: {cv_scores_pca.mean():.4f} ± {cv_scores_pca.std():.4f}")

# Train final models on combined train+val data
X_combined_scaled = np.vstack([X_train_scaled, X_val_scaled])
y_combined_binary = np.hstack([y_train_binary, y_val_binary])

X_combined_pca_scaled = np.vstack([X_train_pca_scaled, X_val_pca_scaled])

final_lr_full = LogisticRegression(
    **random_search_full.best_params_, random_state=42, class_weight=class_weight_dict
)
final_lr_pca = LogisticRegression(
    **random_search_pca.best_params_, random_state=42, class_weight=class_weight_dict
)

final_lr_full.fit(X_combined_scaled, y_combined_binary)
final_lr_pca.fit(X_combined_pca_scaled, y_combined_binary)

# Validation predictions
val_probs_full = best_lr_full.predict_proba(X_val_scaled)[:, 1]
val_preds_full = best_lr_full.predict(X_val_scaled)
val_auc_full = roc_auc_score(y_val_binary, val_probs_full)

val_probs_pca = best_lr_pca.predict_proba(X_val_pca_scaled)[:, 1]
val_preds_pca = best_lr_pca.predict(X_val_pca_scaled)
val_auc_pca = roc_auc_score(y_val_binary, val_probs_pca)

print(f"Validation AUC - Full: {val_auc_full:.4f}, PCA: {val_auc_pca:.4f}")

# Test predictions - keep as probabilities for high CTR class
test_probs_full = final_lr_full.predict_proba(X_test_scaled)[:, 1]
test_probs_pca = final_lr_pca.predict_proba(X_test_pca_scaled)[:, 1]
test_preds_full = final_lr_full.predict(X_test_scaled)
test_preds_pca = final_lr_pca.predict(X_test_pca_scaled)

# Feature importance analysis
feature_names = X_train.columns.tolist()
coefficients_full = best_lr_full.coef_[0]
feature_importance = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coefficients_full,
        "abs_coefficient": np.abs(coefficients_full),
    }
).sort_values("abs_coefficient", ascending=False)

print("\nTop 15 most important features:")
print(feature_importance.head(15))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Feature importance
top_features = feature_importance.head(15)
axes[0, 0].barh(range(len(top_features)), top_features["coefficient"])
axes[0, 0].set_yticks(range(len(top_features)))
axes[0, 0].set_yticklabels(top_features["feature"])
axes[0, 0].set_title("Top 15 Feature Coefficients")
axes[0, 0].set_xlabel("Coefficient Value")

# ROC curves
fpr_full, tpr_full, _ = roc_curve(y_val_binary, val_probs_full)
fpr_pca, tpr_pca, _ = roc_curve(y_val_binary, val_probs_pca)

axes[0, 1].plot(fpr_full, tpr_full, label=f"Full Model (AUC = {val_auc_full:.3f})")
axes[0, 1].plot(fpr_pca, tpr_pca, label=f"PCA Model (AUC = {val_auc_pca:.3f})")
axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].set_title("ROC Curves")
axes[0, 1].legend()

# Probability distributions
axes[1, 0].hist(
    val_probs_full[y_val_binary == 0], bins=30, alpha=0.5, label="Low CTR", density=True
)
axes[1, 0].hist(
    val_probs_full[y_val_binary == 1],
    bins=30,
    alpha=0.5,
    label="High CTR",
    density=True,
)
axes[1, 0].set_xlabel("Predicted Probability")
axes[1, 0].set_ylabel("Density")
axes[1, 0].set_title("Probability Distribution by True Class")
axes[1, 0].legend()

# Confusion matrix
cm = confusion_matrix(y_val_binary, val_preds_full)
sns.heatmap(cm, annot=True, fmt="d", ax=axes[1, 1])
axes[1, 1].set_title("Confusion Matrix (Full Model)")
axes[1, 1].set_xlabel("Predicted")
axes[1, 1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "logistic_regression_analysis.png", dpi=300, bbox_inches="tight"
)
plt.show()

# Model performance summary
performance_summary = pd.DataFrame(
    {
        "Model": ["Full Features", "PCA Features"],
        "CV_AUC_Mean": [cv_scores_full.mean(), cv_scores_pca.mean()],
        "CV_AUC_Std": [cv_scores_full.std(), cv_scores_pca.std()],
        "Val_AUC": [val_auc_full, val_auc_pca],
        "N_Features": [len(feature_names), X_train_pca.shape[1]],
    }
)

print("\nModel Performance Summary:")
print(performance_summary.round(4))

# Classification reports
print("\nClassification Report - Full Model:")
print(classification_report(y_val_binary, val_preds_full))

print("\nClassification Report - PCA Model:")
print(classification_report(y_val_binary, val_preds_pca))

# Save models and results
model_package = {
    "full_model": final_lr_full,
    "pca_model": final_lr_pca,
    "scaler": scaler,
    "feature_names": feature_names,
    "ctr_threshold": median_ctr,
    "class_weights": class_weight_dict,
    "best_params_full": random_search_full.best_params_,
    "best_params_pca": random_search_pca.best_params_,
    "performance": {
        "cv_auc_full": cv_scores_full.mean(),
        "cv_auc_pca": cv_scores_pca.mean(),
        "val_auc_full": val_auc_full,
        "val_auc_pca": val_auc_pca,
    },
    "feature_importance": feature_importance.to_dict("records"),
}

with open(OUTPUT_DIR / "logistic_regression_model.pkl", "wb") as f:
    pickle.dump(model_package, f)

# Save predictions
predictions_df = pd.DataFrame(
    {
        "news_id": X_test.index if hasattr(X_test, "index") else range(len(X_test)),
        "predicted_high_ctr_prob_full": test_probs_full,
        "predicted_high_ctr_prob_pca": test_probs_pca,
        "predicted_class_full": test_preds_full,
        "predicted_class_pca": test_preds_pca,
    }
)

predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
performance_summary.to_csv(OUTPUT_DIR / "model_performance.csv", index=False)
feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

# Additional analysis: Category-wise performance
if "category_enc" in X_val.columns:
    category_performance = []
    unique_categories = X_val["category_enc"].unique()

    for cat in unique_categories:
        cat_mask = X_val["category_enc"] == cat
        if cat_mask.sum() > 10:  # Only analyze categories with sufficient samples
            cat_auc = roc_auc_score(y_val_binary[cat_mask], val_probs_full[cat_mask])
            category_performance.append(
                {
                    "category": cat,
                    "auc": cat_auc,
                    "n_samples": cat_mask.sum(),
                    "high_ctr_rate": y_val_binary[cat_mask].mean(),
                }
            )

    category_df = pd.DataFrame(category_performance).sort_values("auc", ascending=False)
    category_df.to_csv(OUTPUT_DIR / "category_performance.csv", index=False)

    print("\nTop performing categories:")
    print(category_df.head(10))

print(f"\nModel training completed successfully!")
print(f"Files saved to: {OUTPUT_DIR}")
print(f"- logistic_regression_model.pkl")
print(f"- test_predictions.csv")
print(f"- model_performance.csv")
print(f"- feature_importance.csv")
print(f"- category_performance.csv")
