import pandas as pd
from pathlib import Path

print("Adding missing interaction features...")

PREP_DIR = Path("data/preprocessed")

# Load current data
X_train = pd.read_parquet(PREP_DIR / "processed_data" / "X_train.parquet")
X_val = pd.read_parquet(PREP_DIR / "processed_data" / "X_val.parquet")
X_test = pd.read_parquet(PREP_DIR / "processed_data" / "X_test.parquet")
y_train = pd.read_parquet(PREP_DIR / "processed_data" / "y_train.parquet")["ctr"]

print(f"Before: {len(X_train.columns)} features")

# Add interaction features to all datasets
for df_name, df in [("train", X_train), ("val", X_val), ("test", X_test)]:
    df["title_length_x_category"] = df["title_length"] * df["category_enc"]
    df["has_colon_x_category"] = df["has_colon"] * df["category_enc"]
    df["word_count_x_category"] = df["title_word_count"] * df["category_enc"]
    df["has_quotes_x_category"] = df["has_quotes"] * df["category_enc"]
    df["title_features_combined"] = (
        df["has_colon"] + df["has_quotes"] + df["has_number"]
    ) * df["title_length"]

interaction_features = [
    "title_length_x_category",
    "has_colon_x_category",
    "word_count_x_category",
    "has_quotes_x_category",
    "title_features_combined",
]

print(f"After: {len(X_train.columns)} features")
print(f"Added {len(interaction_features)} interaction features")

# Check correlations of new interaction features
df_analysis = X_train.copy()
df_analysis["ctr"] = y_train

interaction_corrs = (
    df_analysis[interaction_features].corrwith(y_train).sort_values(ascending=False)
)
print("\nInteraction feature correlations with CTR:")
for feat, corr in interaction_corrs.items():
    print(f"  {feat}: {corr:.4f}")

# Save updated data
X_train.to_parquet(PREP_DIR / "processed_data" / "X_train.parquet")
X_val.to_parquet(PREP_DIR / "processed_data" / "X_val.parquet")
X_test.to_parquet(PREP_DIR / "processed_data" / "X_test.parquet")

print(f"\n✅ Interaction features added!")
print(f"✅ Final feature count: {len(X_train.columns)}")
print("✅ Ready for model training and EDA visualizations!")
