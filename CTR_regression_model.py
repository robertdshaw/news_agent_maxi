# ===== CTR Regression Trainer with Validation & Expanded Features =====
import os
import pickle
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from textstat import flesch_reading_ease
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor

# Paths
default_base = Path(__file__).parent
PREPROCESSED_FILE = (
    default_base / "preprocessed_data" / "processed_data" / "news_with_engagement.csv"
)
OUTPUT_DIR = default_base / "model_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1) Load Preprocessed Data
df = pd.read_csv(PREPROCESSED_FILE)
logger.info(f"Loaded {len(df)} articles from preprocessed dataset")

# 2) Filter & Prepare Data
# Ensure title/abstract lengths and reading ease
if "title_length" not in df.columns:
    df["title_length"] = df["title"].str.len()
if "abstract_length" not in df.columns:
    df["abstract_length"] = df["abstract"].fillna("").str.len()


def calc_reading(x):
    return flesch_reading_ease(x) if isinstance(x, str) and x else 0


if "title_reading_ease" not in df.columns:
    df["title_reading_ease"] = df["title"].apply(calc_reading)

# Filter by minimum impressions and cap CTR
df = df[
    (df["title_length"] >= 10)
    & (df["abstract_length"] >= 20)
    & (df["total_impressions"] >= 2)
]
df["ctr"] = df["ctr"].clip(upper=0.20)
logger.info(f"Filtered to {len(df)} articles after length, impressions, and CTR cap")

# 3) Feature Extraction
logger.info("Extracting expanded features and embeddings")
# Basic text features
features = pd.DataFrame(
    {
        "title_length": df["title_length"],
        "abstract_length": df["abstract_length"],
        "title_reading_ease": df["title_reading_ease"],
        "has_question": df["title"].str.contains(r"\?").astype(int),
        "has_exclamation": df["title"].str.contains(r"!").astype(int),
        "has_number": df["title"].str.contains(r"\d").astype(int),
        "has_colon": df["title"].str.contains(r":").astype(int),
        "has_quotes": df["title"].str.contains(r'["\']').astype(int),
    }
)
# Time features if available
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    features["hour"] = df["time"].dt.hour.fillna(0).astype(int)
    features["day_of_week"] = df["time"].dt.dayofweek.fillna(0).astype(int)
# Category encoding
le = LabelEncoder()
features["category_enc"] = le.fit_transform(df["category"])
# Embeddings (first 50 dims)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embs = embedder.encode(df["title"].tolist(), show_progress_bar=True)
for i in range(min(50, embs.shape[1])):
    features[f"emb_{i}"] = embs[:, i]

# 4) Train/Validation/Test Split
X_temp, X_test, y_temp, y_test, df_temp, df_test = train_test_split(
    features, df["ctr"], df, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val, df_train, df_val = train_test_split(
    X_temp, y_temp, df_temp, test_size=0.25, random_state=42
)  # 0.25 x 0.8 = 0.2
logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} samples")

# 5) Cross-Validation for Stability
logger.info("Performing 5-fold cross-validation")
model_cv = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    model_cv, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
)
rmse_scores = -scores
logger.info(f"CV RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")

# 6) Train final model on train+val
data_final = pd.concat([X_train, X_val])
target_final = pd.concat([y_train, y_val])
logger.info("Training final model on combined train+val set")
model = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
model.fit(data_final, target_final)

# 7) Evaluation on Test Set
preds = model.predict(X_test)
test_rmse = np.sqrt(np.mean((preds - y_test) ** 2))
logger.info(f"Test RMSE: {test_rmse:.4f}")

# 8) Performance Breakdown
# By category
df_test = df_test.copy()
df_test["preds"] = preds
cat_rmse = df_test.groupby("category").apply(
    lambda g: np.sqrt(((g["preds"] - g["ctr"]) ** 2).mean())
)
logger.info("RMSE by Category:")
for cat, rm in cat_rmse.items():
    logger.info(f"  {cat}: {rm:.4f}")

# 9) Feature Importance Analysis
importances = model.feature_importances_
feat_names = features.columns.tolist()
fi_df = pd.DataFrame({"feature": feat_names, "importance": importances}).nlargest(
    15, "importance"
)
plt.figure(figsize=(6, 8))
plt.barh(fi_df["feature"][::-1], fi_df["importance"][::-1])
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top15_feature_importances.png", dpi=300)
plt.close()

# Plot: Predicted vs Actual CTR
plt.figure()
max_val = max(max(y_test), max(preds))
plt.scatter(y_test, preds, alpha=0.3)
plt.plot([0, max_val], [0, max_val], "k--")
plt.xlabel("Actual CTR")
plt.ylabel("Predicted CTR")
plt.title("Predicted vs. Actual CTR")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "predicted_vs_actual.png", dpi=300)
plt.close()

# Plot: RMSE by Category
plt.figure(figsize=(8, 6))
cat_rmse_sorted = cat_rmse.sort_values()
plt.barh(cat_rmse_sorted.index, cat_rmse_sorted.values)
plt.xlabel("RMSE")
plt.title("RMSE by Category")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rmse_by_category.png", dpi=300)
plt.close()

# 10) Save Model
with open(OUTPUT_DIR / "ctr_regressor.pkl", "wb") as f:
    pickle.dump({"model": model, "feature_names": features.columns.tolist()}, f)
logger.info(f"Model saved to {OUTPUT_DIR / '/ctr_regressor.pkl'}")
