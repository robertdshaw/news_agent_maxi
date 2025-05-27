import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from textstat import flesch_reading_ease
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from pathlib import Path

# Configuration
TRAIN_DIR = Path("source_data/train_data")
VAL_DIR = Path("source_data/val_data")
TEST_DIR = Path("source_data/test_data")
PREP_DIR = Path("data/preprocessed")
PREP_DIR.mkdir(parents=True, exist_ok=True)
(PREP_DIR / "plots").mkdir(exist_ok=True)
(PREP_DIR / "processed_data").mkdir(exist_ok=True)

CONFIG = {
    "train_sample_size": 400_000,
    "val_sample_size": 400_000,
    "test_sample_size": 400_000,
    "embedding_dim": 384,
    "random_state": 42,
}

# Load data
news_train = pd.read_csv(
    TRAIN_DIR / "news.tsv",
    sep="\t",
    header=None,
    names=[
        "newsID",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ],
)
news_val = pd.read_csv(
    VAL_DIR / "news.tsv",
    sep="\t",
    header=None,
    names=[
        "newsID",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ],
)
news_test = pd.read_csv(
    TEST_DIR / "news.tsv",
    sep="\t",
    header=None,
    names=[
        "newsID",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ],
)

behaviors_train = pd.read_csv(
    TRAIN_DIR / "behaviors.tsv",
    sep="\t",
    header=None,
    names=["impression_id", "user_id", "time", "history", "impressions"],
)
behaviors_val = pd.read_csv(
    VAL_DIR / "behaviors.tsv",
    sep="\t",
    header=None,
    names=["impression_id", "user_id", "time", "history", "impressions"],
)
behaviors_test = pd.read_csv(
    TEST_DIR / "behaviors.tsv",
    sep="\t",
    header=None,
    names=["impression_id", "user_id", "time", "history", "impressions"],
)

# Clean data
news_train = news_train.dropna(subset=["newsID", "title", "category"]).drop_duplicates(
    subset=["newsID", "title"]
)
news_train["abstract"] = news_train["abstract"].fillna("")

news_val = news_val.dropna(subset=["newsID", "title", "category"]).drop_duplicates(
    subset=["newsID", "title"]
)
news_val["abstract"] = news_val["abstract"].fillna("")

news_test = news_test.dropna(subset=["newsID", "title", "category"]).drop_duplicates(
    subset=["newsID", "title"]
)
news_test["abstract"] = news_test["abstract"].fillna("")

behaviors_train = behaviors_train.dropna(
    subset=["impression_id", "user_id", "impressions"]
)
behaviors_train["time"] = pd.to_datetime(behaviors_train["time"])
behaviors_train = behaviors_train.sample(
    n=min(CONFIG["train_sample_size"], len(behaviors_train)), random_state=42
)

behaviors_val = behaviors_val.dropna(subset=["impression_id", "user_id", "impressions"])
behaviors_val["time"] = pd.to_datetime(behaviors_val["time"])
behaviors_val = behaviors_val.sample(
    n=min(CONFIG["val_sample_size"], len(behaviors_val)), random_state=42
)

behaviors_test = behaviors_test.dropna(
    subset=["impression_id", "user_id", "impressions"]
)
behaviors_test["time"] = pd.to_datetime(behaviors_test["time"])
behaviors_test = behaviors_test.sample(
    n=min(CONFIG["test_sample_size"], len(behaviors_test)), random_state=42
)

# Process train impressions
train_impressions = behaviors_train["impressions"].str.split().explode()
train_expanded = pd.DataFrame(
    {
        "impression": train_impressions.values,
        "user_id": behaviors_train.loc[train_impressions.index, "user_id"].values,
        "time": behaviors_train.loc[train_impressions.index, "time"].values,
    }
)
train_split = train_expanded["impression"].str.split("-", expand=True)
train_expanded["news_id"] = train_split[0]
train_expanded["clicked"] = train_split[1].astype(int)

articles_train = (
    train_expanded.groupby("news_id")
    .agg({"clicked": ["sum", "count"], "user_id": "nunique", "time": "first"})
    .reset_index()
)
articles_train.columns = [
    "news_id",
    "total_clicks",
    "total_impressions",
    "total_users",
    "first_seen",
]
articles_train["ctr"] = (
    articles_train["total_clicks"] / articles_train["total_impressions"]
)

# Process val impressions
val_impressions = behaviors_val["impressions"].str.split().explode()
val_expanded = pd.DataFrame(
    {
        "impression": val_impressions.values,
        "user_id": behaviors_val.loc[val_impressions.index, "user_id"].values,
        "time": behaviors_val.loc[val_impressions.index, "time"].values,
    }
)
val_split = val_expanded["impression"].str.split("-", expand=True)
val_expanded["news_id"] = val_split[0]
val_expanded["clicked"] = val_split[1].astype(int)

articles_val = (
    val_expanded.groupby("news_id")
    .agg({"clicked": ["sum", "count"], "user_id": "nunique", "time": "first"})
    .reset_index()
)
articles_val.columns = [
    "news_id",
    "total_clicks",
    "total_impressions",
    "total_users",
    "first_seen",
]
articles_val["ctr"] = articles_val["total_clicks"] / articles_val["total_impressions"]

# Process test impressions
test_impressions = behaviors_test["impressions"].str.split().explode()
test_expanded = pd.DataFrame(
    {
        "news_id": test_impressions.values,
        "user_id": behaviors_test.loc[test_impressions.index, "user_id"].values,
        "time": behaviors_test.loc[test_impressions.index, "time"].values,
    }
)

articles_test = (
    test_expanded.groupby("news_id")
    .agg({"user_id": "nunique", "time": "first"})
    .reset_index()
)
articles_test.columns = ["news_id", "total_users", "first_seen"]
articles_test["total_impressions"] = test_expanded.groupby("news_id").size().values
articles_test["total_clicks"] = np.nan
articles_test["ctr"] = np.nan

# Merge with news data
df_train = articles_train.merge(
    news_train, left_on="news_id", right_on="newsID", how="left"
)
df_val = articles_val.merge(news_val, left_on="news_id", right_on="newsID", how="left")
df_test = articles_test.merge(
    news_test, left_on="news_id", right_on="newsID", how="left"
)

# Add temporal features
df_train["first_seen"] = pd.to_datetime(df_train["first_seen"])
df_train["hour"] = df_train["first_seen"].dt.hour
df_train["day_of_week"] = df_train["first_seen"].dt.dayofweek
df_train["is_weekend"] = (df_train["day_of_week"] >= 5).astype(int)
df_train["time_of_day"] = pd.cut(
    df_train["hour"], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], include_lowest=True
).astype(int)

df_val["first_seen"] = pd.to_datetime(df_val["first_seen"])
df_val["hour"] = df_val["first_seen"].dt.hour
df_val["day_of_week"] = df_val["first_seen"].dt.dayofweek
df_val["is_weekend"] = (df_val["day_of_week"] >= 5).astype(int)
df_val["time_of_day"] = pd.cut(
    df_val["hour"], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], include_lowest=True
).astype(int)

df_test["first_seen"] = pd.to_datetime(df_test["first_seen"])
df_test["hour"] = df_test["first_seen"].dt.hour
df_test["day_of_week"] = df_test["first_seen"].dt.dayofweek
df_test["is_weekend"] = (df_test["day_of_week"] >= 5).astype(int)
df_test["time_of_day"] = pd.cut(
    df_test["hour"], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], include_lowest=True
).astype(int)

# Text features
df_train["title_length"] = df_train["title"].str.len()
df_train["abstract_length"] = df_train["abstract"].str.len()
df_train["title_word_count"] = df_train["title"].str.split().str.len()
df_train["abstract_word_count"] = df_train["abstract"].str.split().str.len()
df_train["title_reading_ease"] = df_train["title"].apply(
    lambda x: flesch_reading_ease(x) if pd.notna(x) and len(x) > 0 else 0
)
df_train["abstract_reading_ease"] = df_train["abstract"].apply(
    lambda x: flesch_reading_ease(x) if pd.notna(x) and len(x) > 0 else 0
)
df_train["has_question"] = df_train["title"].str.contains(r"\?", na=False).astype(int)
df_train["has_exclamation"] = df_train["title"].str.contains(r"!", na=False).astype(int)
df_train["has_number"] = df_train["title"].str.contains(r"\d", na=False).astype(int)
df_train["has_colon"] = df_train["title"].str.contains(r":", na=False).astype(int)
df_train["has_quotes"] = df_train["title"].str.contains(r'["\']', na=False).astype(int)
df_train["title_upper_ratio"] = df_train["title"].apply(
    lambda x: (
        sum(c.isupper() for c in str(x)) / len(str(x))
        if pd.notna(x) and len(str(x)) > 0
        else 0
    )
)

df_val["title_length"] = df_val["title"].str.len()
df_val["abstract_length"] = df_val["abstract"].str.len()
df_val["title_word_count"] = df_val["title"].str.split().str.len()
df_val["abstract_word_count"] = df_val["abstract"].str.split().str.len()
df_val["title_reading_ease"] = df_val["title"].apply(
    lambda x: flesch_reading_ease(x) if pd.notna(x) and len(x) > 0 else 0
)
df_val["abstract_reading_ease"] = df_val["abstract"].apply(
    lambda x: flesch_reading_ease(x) if pd.notna(x) and len(x) > 0 else 0
)
df_val["has_question"] = df_val["title"].str.contains(r"\?", na=False).astype(int)
df_val["has_exclamation"] = df_val["title"].str.contains(r"!", na=False).astype(int)
df_val["has_number"] = df_val["title"].str.contains(r"\d", na=False).astype(int)
df_val["has_colon"] = df_val["title"].str.contains(r":", na=False).astype(int)
df_val["has_quotes"] = df_val["title"].str.contains(r'["\']', na=False).astype(int)
df_val["title_upper_ratio"] = df_val["title"].apply(
    lambda x: (
        sum(c.isupper() for c in str(x)) / len(str(x))
        if pd.notna(x) and len(str(x)) > 0
        else 0
    )
)

df_test["title_length"] = df_test["title"].str.len()
df_test["abstract_length"] = df_test["abstract"].str.len()
df_test["title_word_count"] = df_test["title"].str.split().str.len()
df_test["abstract_word_count"] = df_test["abstract"].str.split().str.len()
df_test["title_reading_ease"] = df_test["title"].apply(
    lambda x: flesch_reading_ease(x) if pd.notna(x) and len(x) > 0 else 0
)
df_test["abstract_reading_ease"] = df_test["abstract"].apply(
    lambda x: flesch_reading_ease(x) if pd.notna(x) and len(x) > 0 else 0
)
df_test["has_question"] = df_test["title"].str.contains(r"\?", na=False).astype(int)
df_test["has_exclamation"] = df_test["title"].str.contains(r"!", na=False).astype(int)
df_test["has_number"] = df_test["title"].str.contains(r"\d", na=False).astype(int)
df_test["has_colon"] = df_test["title"].str.contains(r":", na=False).astype(int)
df_test["has_quotes"] = df_test["title"].str.contains(r'["\']', na=False).astype(int)
df_test["title_upper_ratio"] = df_test["title"].apply(
    lambda x: (
        sum(c.isupper() for c in str(x)) / len(str(x))
        if pd.notna(x) and len(str(x)) > 0
        else 0
    )
)

# Category encoding - handle NaN and unseen categories properly
df_train["category"] = df_train["category"].fillna("unknown")
df_val["category"] = df_val["category"].fillna("unknown")
df_test["category"] = df_test["category"].fillna("unknown")

# Fit encoder only on training data
le = LabelEncoder()
df_train["category_enc"] = le.fit_transform(df_train["category"])


# Create mapping for unseen categories
def safe_transform(categories, encoder):
    result = []
    for cat in categories:
        if cat in encoder.classes_:
            result.append(encoder.transform([cat])[0])
        else:
            # Map unseen categories to "unknown" if it exists, otherwise create new class
            if "unknown" in encoder.classes_:
                result.append(encoder.transform(["unknown"])[0])
            else:
                result.append(-1)  # Or len(encoder.classes_) for new class
    return result


df_val["category_enc"] = safe_transform(df_val["category"], le)
df_test["category_enc"] = safe_transform(df_test["category"], le)

# Embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
cache_dir = PREP_DIR / "cache"
cache_dir.mkdir(exist_ok=True)

# Train embeddings
if (cache_dir / "train_embeddings.pkl").exists():
    train_emb_map = pd.read_pickle(cache_dir / "train_embeddings.pkl")
else:
    train_unique = df_train.drop_duplicates("newsID").set_index("newsID")
    train_titles = train_unique["title"].fillna("").tolist()
    train_embs = embedder.encode(train_titles, show_progress_bar=True, batch_size=32)
    train_emb_map = pd.DataFrame(
        train_embs[:, : CONFIG["embedding_dim"]],
        index=train_unique.index,
        columns=[f"emb_{i}" for i in range(CONFIG["embedding_dim"])],
    )
    train_emb_map.to_pickle(cache_dir / "train_embeddings.pkl")
df_train = df_train.merge(train_emb_map, on="newsID", how="left")

# Val embeddings
if (cache_dir / "val_embeddings.pkl").exists():
    val_emb_map = pd.read_pickle(cache_dir / "val_embeddings.pkl")
else:
    val_unique = df_val.drop_duplicates("newsID").set_index("newsID")
    val_titles = val_unique["title"].fillna("").tolist()
    val_embs = embedder.encode(val_titles, show_progress_bar=True, batch_size=32)
    val_emb_map = pd.DataFrame(
        val_embs[:, : CONFIG["embedding_dim"]],
        index=val_unique.index,
        columns=[f"emb_{i}" for i in range(CONFIG["embedding_dim"])],
    )
    val_emb_map.to_pickle(cache_dir / "val_embeddings.pkl")
df_val = df_val.merge(val_emb_map, on="newsID", how="left")

# Test embeddings
if (cache_dir / "test_embeddings.pkl").exists():
    test_emb_map = pd.read_pickle(cache_dir / "test_embeddings.pkl")
else:
    test_unique = df_test.drop_duplicates("newsID").set_index("newsID")
    test_titles = test_unique["title"].fillna("").tolist()
    test_embs = embedder.encode(test_titles, show_progress_bar=True, batch_size=32)
    test_emb_map = pd.DataFrame(
        test_embs[:, : CONFIG["embedding_dim"]],
        index=test_unique.index,
        columns=[f"emb_{i}" for i in range(CONFIG["embedding_dim"])],
    )
    test_emb_map.to_pickle(cache_dir / "test_embeddings.pkl")
df_test = df_test.merge(test_emb_map, on="newsID", how="left")

# Feature selection for analysis
base_features = [
    "title_length",
    "abstract_length",
    "title_word_count",
    "abstract_word_count",
    "title_reading_ease",
    "abstract_reading_ease",
    "has_question",
    "has_exclamation",
    "has_number",
    "has_colon",
    "has_quotes",
    "title_upper_ratio",
    "category_enc",
    "hour",
    "day_of_week",
    "is_weekend",
    "time_of_day",
    "total_impressions",
    "total_users",
]

emb_features = [f"emb_{i}" for i in range(CONFIG["embedding_dim"])]
all_features = base_features + emb_features

# Bivariate analysis
correlations = (
    df_train[base_features + ["ctr"]]
    .corr()["ctr"]
    .drop("ctr")
    .sort_values(ascending=False)
)
print("Top correlations with CTR:")
print(correlations.head(10))

# Feature importance using correlation
top_corr_features = (
    correlations.abs().sort_values(ascending=False).head(10).index.tolist()
)

# Correlation matrix heatmap
plt.figure(figsize=(12, 10))
feature_corr_matrix = df_train[top_corr_features + ["ctr"]].corr()
sns.heatmap(feature_corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix (Top 10 Features + CTR)")
plt.tight_layout()
plt.savefig(
    PREP_DIR / "plots" / "correlation_heatmap.png", dpi=300, bbox_inches="tight"
)
plt.show()

# Bivariate plots for top features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(top_corr_features[:6]):
    axes[i].scatter(df_train[feature], df_train["ctr"], alpha=0.5, s=1)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("CTR")
    axes[i].set_title(f"{feature} vs CTR (r={correlations[feature]:.3f})")

plt.tight_layout()
plt.savefig(PREP_DIR / "plots" / "bivariate_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# Pair plot of most correlated features
top_5_features = top_corr_features[:5] + ["ctr"]
sample_data = df_train[top_5_features].sample(
    n=min(5000, len(df_train)), random_state=42
)
sns.pairplot(sample_data, diag_kind="hist")
plt.savefig(PREP_DIR / "plots" / "pairplot.png", dpi=300, bbox_inches="tight")
plt.show()

# Category analysis
category_stats = (
    df_train.groupby("category")
    .agg({"ctr": ["mean", "std", "count"], "total_impressions": "sum"})
    .round(4)
)
category_stats.columns = ["ctr_mean", "ctr_std", "article_count", "total_impressions"]
category_stats = category_stats.sort_values("ctr_mean", ascending=False)
print("\nCategory performance:")
print(category_stats)

# Time-based analysis
time_stats = df_train.groupby("hour")["ctr"].mean().sort_values(ascending=False)
print("\nBest performing hours:")
print(time_stats.head(10))

# Prepare final feature matrices
X_train = df_train[all_features].fillna(0)
y_train = df_train["ctr"]

X_val = df_val[all_features].fillna(0)
y_val = df_val["ctr"]

X_test = df_test[all_features].fillna(0)
# Test CTR remains nan (unknown)

# # PCA for dimensionality reduction
# pca = PCA(n_components=50, random_state=42)
# X_train_pca = pca.fit_transform(X_train)
# X_val_pca = pca.transform(X_val)
# X_test_pca = pca.transform(X_test)

# explained_variance = pca.explained_variance_ratio_
# cumulative_variance = np.cumsum(explained_variance)

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 51), cumulative_variance, "bo-")
# plt.xlabel("Number of Components")
# plt.ylabel("Cumulative Explained Variance Ratio")
# plt.title("PCA Explained Variance")
# plt.grid(True)
# plt.savefig(PREP_DIR / "plots" / "pca_variance.png", dpi=300, bbox_inches="tight")
# plt.show()

# Save processed data
X_train_df = pd.DataFrame(X_train, columns=all_features)
X_val_df = pd.DataFrame(X_val, columns=all_features)
X_test_df = pd.DataFrame(X_test, columns=all_features)

X_train_df.to_parquet(PREP_DIR / "processed_data" / "X_train.parquet")
y_train.to_frame("ctr").to_parquet(PREP_DIR / "processed_data" / "y_train.parquet")

X_val_df.to_parquet(PREP_DIR / "processed_data" / "X_val.parquet")
y_val.to_frame("ctr").to_parquet(PREP_DIR / "processed_data" / "y_val.parquet")

X_test_df.to_parquet(PREP_DIR / "processed_data" / "X_test.parquet")
df_test[["ctr"]].to_parquet(PREP_DIR / "processed_data" / "y_test.parquet")

# # Save PCA data
# pd.DataFrame(X_train_pca).to_parquet(
#     PREP_DIR / "processed_data" / "X_train_pca.parquet"
# )
# pd.DataFrame(X_val_pca).to_parquet(PREP_DIR / "processed_data" / "X_val_pca.parquet")
# pd.DataFrame(X_test_pca).to_parquet(PREP_DIR / "processed_data" / "X_test_pca.parquet")

# print(f"Dataset sizes:")
# print(f"Train: {len(df_train)} articles")
# print(f"Val: {len(df_val)} articles")
# print(f"Test: {len(df_test)} articles")
# print(f"Features: {len(all_features)}")
# print(f"PCA components: 50")
# print(f"PCA explained variance: {cumulative_variance[-1]:.3f}")

print("EDA processing complete - data saved and ready for modeling")
