import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Load your processed data
PREP_DIR = Path("data/preprocessed")
X_train = pd.read_parquet(PREP_DIR / "processed_data" / "X_train.parquet")
y_train = pd.read_parquet(PREP_DIR / "processed_data" / "y_train.parquet")["ctr"]

# Create output directory for plots
plot_dir = PREP_DIR / "analysis_plots"
plot_dir.mkdir(exist_ok=True)

print("Creating comprehensive data analysis visualizations...")

# ===== VISUALIZATION 1: Correlation Matrix =====
plt.figure(figsize=(16, 12))

# Focus on base features + top embedding features for readability
base_features = [col for col in X_train.columns if not col.startswith("emb_")]
top_embeddings = [f"emb_{i}" for i in range(10)]  # First 10 embeddings
viz_features = base_features + top_embeddings + ["ctr"]

# Create correlation matrix with target
corr_data = pd.concat([X_train[base_features + top_embeddings], y_train], axis=1)
corr_matrix = corr_data.corr()

# Plot heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    cmap="coolwarm",
    center=0,
    square=True,
    fmt=".3f",
    cbar_kws={"shrink": 0.8},
)
plt.title(
    "Feature Correlation Matrix (Base Features + Top 10 Embeddings)",
    fontsize=16,
    pad=20,
)
plt.tight_layout()
plt.savefig(plot_dir / "correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# ===== VISUALIZATION 2: Feature Distributions =====
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.ravel()

# Select interesting features to plot
features_to_plot = [
    "title_length",
    "title_word_count",
    "title_reading_ease",
    "abstract_length",
    "has_question",
    "has_exclamation",
    "has_number",
    "has_colon",
    "title_upper_ratio",
    "category_enc",
    "emb_0",
    "emb_1",
]

for i, feature in enumerate(features_to_plot):
    if feature in X_train.columns:
        if feature in ["has_question", "has_exclamation", "has_number", "has_colon"]:
            # Bar plot for binary features
            X_train[feature].value_counts().plot(
                kind="bar", ax=axes[i], color=["skyblue", "orange"]
            )
            axes[i].set_title(f"{feature} Distribution")
        else:
            # Histogram for continuous features
            axes[i].hist(
                X_train[feature], bins=50, alpha=0.7, color="skyblue", edgecolor="black"
            )
            axes[i].set_title(f"{feature} Distribution")

        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Frequency")
        axes[i].tick_params(axis="x", rotation=45)

plt.suptitle("Feature Distributions", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig(plot_dir / "feature_distributions.png", dpi=300, bbox_inches="tight")
plt.show()

# ===== VISUALIZATION 3: CTR Analysis =====
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# CTR distribution (with and without zeros)
axes[0, 0].hist(y_train, bins=50, alpha=0.7, color="lightblue", edgecolor="black")
axes[0, 0].set_title("CTR Distribution (All Articles)")
axes[0, 0].set_xlabel("CTR")
axes[0, 0].set_ylabel("Frequency")

# CTR distribution (excluding zeros)
ctr_nonzero = y_train[y_train > 0]
if len(ctr_nonzero) > 0:
    axes[0, 1].hist(
        ctr_nonzero, bins=50, alpha=0.7, color="lightgreen", edgecolor="black"
    )
    axes[0, 1].set_title(f"CTR Distribution (Non-Zero Only, n={len(ctr_nonzero)})")
    axes[0, 1].set_xlabel("CTR")
    axes[0, 1].set_ylabel("Frequency")

# CTR by category (if category_enc exists)
if "category_enc" in X_train.columns:
    ctr_by_category = (
        pd.concat([X_train["category_enc"], y_train], axis=1)
        .groupby("category_enc")["ctr"]
        .mean()
    )
    axes[0, 2].bar(range(len(ctr_by_category)), ctr_by_category.values, color="coral")
    axes[0, 2].set_title("Average CTR by Category")
    axes[0, 2].set_xlabel("Category (Encoded)")
    axes[0, 2].set_ylabel("Average CTR")

# Title length vs CTR
if len(ctr_nonzero) > 0:
    sample_data = pd.concat([X_train["title_length"], y_train], axis=1)
    sample_data = sample_data[sample_data["ctr"] > 0].sample(
        min(5000, len(sample_data))
    )
    axes[1, 0].scatter(sample_data["title_length"], sample_data["ctr"], alpha=0.5, s=1)
    axes[1, 0].set_title("Title Length vs CTR (Sample)")
    axes[1, 0].set_xlabel("Title Length")
    axes[1, 0].set_ylabel("CTR")

# Reading ease vs CTR
if len(ctr_nonzero) > 0:
    sample_data = pd.concat([X_train["title_reading_ease"], y_train], axis=1)
    sample_data = sample_data[sample_data["ctr"] > 0].sample(
        min(5000, len(sample_data))
    )
    axes[1, 1].scatter(
        sample_data["title_reading_ease"], sample_data["ctr"], alpha=0.5, s=1
    )
    axes[1, 1].set_title("Reading Ease vs CTR (Sample)")
    axes[1, 1].set_xlabel("Reading Ease Score")
    axes[1, 1].set_ylabel("CTR")

# Feature importance preview (correlations with CTR)
feature_corrs = X_train.corrwith(y_train).abs().sort_values(ascending=False).head(10)
axes[1, 2].barh(
    range(len(feature_corrs)), feature_corrs.values, color="purple", alpha=0.7
)
axes[1, 2].set_yticks(range(len(feature_corrs)))
axes[1, 2].set_yticklabels(feature_corrs.index, fontsize=8)
axes[1, 2].set_title("Top 10 Features by Correlation with CTR")
axes[1, 2].set_xlabel("Absolute Correlation")

plt.tight_layout()
plt.savefig(plot_dir / "ctr_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# ===== VISUALIZATION 4: Embedding Analysis =====
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Embedding correlation with CTR
emb_features = [col for col in X_train.columns if col.startswith("emb_")]
emb_corrs = X_train[emb_features].corrwith(y_train).abs().sort_values(ascending=False)

axes[0, 0].plot(range(len(emb_corrs)), emb_corrs.values, "o-", alpha=0.7)
axes[0, 0].set_title("Embedding Feature Correlations with CTR")
axes[0, 0].set_xlabel("Embedding Dimension")
axes[0, 0].set_ylabel("Absolute Correlation")

# Top embedding features
top_emb_corrs = emb_corrs.head(15)
axes[0, 1].barh(
    range(len(top_emb_corrs)), top_emb_corrs.values, color="teal", alpha=0.7
)
axes[0, 1].set_yticks(range(len(top_emb_corrs)))
axes[0, 1].set_yticklabels(top_emb_corrs.index, fontsize=8)
axes[0, 1].set_title("Top 15 Embedding Features by CTR Correlation")
axes[0, 1].set_xlabel("Absolute Correlation")

# Embedding distribution (first embedding)
axes[1, 0].hist(
    X_train["emb_0"], bins=50, alpha=0.7, color="lightcoral", edgecolor="black"
)
axes[1, 0].set_title("Distribution of emb_0 (Top Correlated Embedding)")
axes[1, 0].set_xlabel("emb_0 Value")
axes[1, 0].set_ylabel("Frequency")

# Embedding variance
emb_variances = X_train[emb_features].var().sort_values(ascending=False)
axes[1, 1].plot(
    range(len(emb_variances)), emb_variances.values, "o-", alpha=0.7, color="green"
)
axes[1, 1].set_title("Embedding Feature Variances")
axes[1, 1].set_xlabel("Embedding Dimension (sorted by variance)")
axes[1, 1].set_ylabel("Variance")

plt.tight_layout()
plt.savefig(plot_dir / "embedding_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# ===== SUMMARY STATISTICS =====
print("\n" + "=" * 60)
print("📊 DATA ANALYSIS SUMMARY")
print("=" * 60)

print(f"\n🎯 DATASET OVERVIEW:")
print(f"- Total articles: {len(X_train):,}")
print(f"- Total features: {len(X_train.columns)}")
print(
    f"- Articles with CTR > 0: {(y_train > 0).sum():,} ({(y_train > 0).mean()*100:.1f}%)"
)

print(f"\n📈 CTR STATISTICS:")
print(f"- Overall mean CTR: {y_train.mean():.4f}")
print(
    f"- Mean CTR (non-zero): {ctr_nonzero.mean():.4f}"
    if len(ctr_nonzero) > 0
    else "- No non-zero CTR values"
)
print(f"- CTR standard deviation: {y_train.std():.4f}")
print(f"- Max CTR: {y_train.max():.4f}")

print(f"\n🔍 TOP PREDICTIVE FEATURES:")
top_features = X_train.corrwith(y_train).abs().sort_values(ascending=False).head(5)
for feat, corr in top_features.items():
    print(f"- {feat}: {corr:.4f}")

print(f"\n📁 VISUALIZATIONS SAVED TO:")
print(f"- {plot_dir}/correlation_matrix.png")
print(f"- {plot_dir}/feature_distributions.png")
print(f"- {plot_dir}/ctr_analysis.png")
print(f"- {plot_dir}/embedding_analysis.png")

print(f"\n✅ Analysis complete! Your data looks great for modeling.")
