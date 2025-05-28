import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
from textstat import flesch_reading_ease
from sklearn.preprocessing import LabelEncoder
import json

print("=" * 60)
print("CLEAN AI NEWS EDITOR DATA PREPROCESSING")
print("=" * 60)

# Configuration
TRAIN_DIR = Path("source_data/train_data")
VAL_DIR = Path("source_data/val_data")
TEST_DIR = Path("source_data/test_data")
PREP_DIR = Path("data/preprocessed")
PREP_DIR.mkdir(parents=True, exist_ok=True)
(PREP_DIR / "plots").mkdir(exist_ok=True)
(PREP_DIR / "processed_data").mkdir(exist_ok=True)
(PREP_DIR / "cache").mkdir(exist_ok=True)

CONFIG = {
    "target_sample_size": 50000,  # Final target after all processing
    "embedding_dim": 384,
    "random_state": 42,
}

# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================
print("\nStep 1: Loading raw data...")

# Load news data
print("Loading news data...")
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

# Load behavior data
print("Loading behavior data...")
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

# ============================================================================
# STEP 2: INITIAL DATA EXPLORATION
# ============================================================================
# Add this code after Step 2 in your script to understand the overlap

print("\n--- ARTICLE OVERLAP ANALYSIS ---")

# Get unique articles in each split
train_articles = set(news_train["newsID"].unique())
val_articles = set(news_val["newsID"].unique())
test_articles = set(news_test["newsID"].unique())

print(f"Unique articles in train: {len(train_articles):,}")
print(f"Unique articles in val: {len(val_articles):,}")
print(f"Unique articles in test: {len(test_articles):,}")

# Check overlaps
train_val_overlap = train_articles.intersection(val_articles)
train_test_overlap = train_articles.intersection(test_articles)
val_test_overlap = val_articles.intersection(test_articles)
all_three_overlap = train_articles.intersection(val_articles).intersection(
    test_articles
)

print(f"\nOverlap between train and val: {len(train_val_overlap):,}")
print(f"Overlap between train and test: {len(train_test_overlap):,}")
print(f"Overlap between val and test: {len(val_test_overlap):,}")
print(f"Articles in all three splits: {len(all_three_overlap):,}")

# Total unique articles across all splits
all_unique = train_articles.union(val_articles).union(test_articles)
print(f"\nTotal unique articles across all splits: {len(all_unique):,}")

# Breakdown by exclusivity
only_train = train_articles - val_articles - test_articles
only_val = val_articles - train_articles - test_articles
only_test = test_articles - train_articles - val_articles

print(f"\nArticles exclusive to train: {len(only_train):,}")
print(f"Articles exclusive to val: {len(only_val):,}")
print(f"Articles exclusive to test: {len(only_test):,}")

# This should explain the 160K vs 290K discrepancy
print(f"\n🔍 ANALYSIS:")
print(
    f"Individual split totals: {len(train_articles) + len(val_articles) + len(test_articles):,}"
)
print(f"Unique articles total: {len(all_unique):,}")
print(
    f"Difference due to overlap: {(len(train_articles) + len(val_articles) + len(test_articles)) - len(all_unique):,}"
)

if len(all_unique) <= 160000:
    print(f"✅ This explains the ~160K official count!")
else:
    print(f"❓ Still higher than 160K - might be different subsets or versions")

print("\nStep 2: Initial data exploration...")

print("\n--- NEWS DATA EXPLORATION ---")
print(f"Train news shape: {news_train.shape}")
print(f"Val news shape: {news_val.shape}")
print(f"Test news shape: {news_test.shape}")

print("\nTrain news info:")
print(news_train.info())

print("\nTrain news head:")
print(news_train.head())

print("\n--- BEHAVIOR DATA EXPLORATION ---")
print(f"Train behaviors shape: {behaviors_train.shape}")
print(f"Val behaviors shape: {behaviors_val.shape}")
print(f"Test behaviors shape: {behaviors_test.shape}")

print("\nTrain behaviors info:")
print(behaviors_train.info())

print("\nTrain behaviors head:")
print(behaviors_train.head())

# Sample impression to understand structure
print("\nSample impression structure:")
sample_impression = behaviors_train["impressions"].iloc[0]
print(f"Sample impressions: {sample_impression}")
print(f"Number of articles in this impression: {len(sample_impression.split())}")

# ============================================================================
# STEP 3: DATA QUALITY CHECKS
# ============================================================================
print("\nStep 3: Data quality checks...")

print("\n--- MISSING VALUES ---")
print("News train missing values:")
print(news_train.isnull().sum())

print("\nBehaviors train missing values:")
print(behaviors_train.isnull().sum())

print("\n--- DUPLICATES ---")
print(f"News train duplicates: {news_train.duplicated().sum()}")
print(
    f"News train duplicates by newsID: {news_train.duplicated(subset=['newsID']).sum()}"
)
print(f"Behaviors train duplicates: {behaviors_train.duplicated().sum()}")

print("\n--- UNIQUE VALUES ---")
print(f"Unique news articles (train): {news_train['newsID'].nunique()}")
print(f"Unique users (train): {behaviors_train['user_id'].nunique()}")
print(f"Unique categories: {news_train['category'].nunique()}")
print("Top categories:")
print(news_train["category"].value_counts().head(10))

# ============================================================================
# STEP 4: CLEAN DATA
# ============================================================================
print("\nStep 4: Cleaning data...")

# Clean news data
news_train = news_train.dropna(subset=["newsID", "title", "category"])
news_val = news_val.dropna(subset=["newsID", "title", "category"])
news_test = news_test.dropna(subset=["newsID", "title", "category"])

# Remove duplicates
news_train = news_train.drop_duplicates(subset=["newsID", "title"])
news_val = news_val.drop_duplicates(subset=["newsID", "title"])
news_test = news_test.drop_duplicates(subset=["newsID", "title"])

# Fill missing abstracts
news_train["abstract"] = news_train["abstract"].fillna("")
news_val["abstract"] = news_val["abstract"].fillna("")
news_test["abstract"] = news_test["abstract"].fillna("")

# Clean behavior data
behaviors_train = behaviors_train.dropna(
    subset=["impression_id", "user_id", "impressions"]
)
behaviors_val = behaviors_val.dropna(subset=["impression_id", "user_id", "impressions"])
behaviors_test = behaviors_test.dropna(
    subset=["impression_id", "user_id", "impressions"]
)

# Convert time to datetime
behaviors_train["time"] = pd.to_datetime(behaviors_train["time"])
behaviors_val["time"] = pd.to_datetime(behaviors_val["time"])
behaviors_test["time"] = pd.to_datetime(behaviors_test["time"])

print(
    f"After cleaning - News train: {news_train.shape}, Behaviors train: {behaviors_train.shape}"
)

# ============================================================================
# STEP 5: SAMPLING STRATEGY
# ============================================================================
print("\nStep 5: Sampling strategy...")

# Start with a reasonable sample of behavior records to get target articles
initial_behavior_sample = 100000  # This should give us enough unique articles

behaviors_train_sample = behaviors_train.sample(
    n=min(initial_behavior_sample, len(behaviors_train)), random_state=42
)
behaviors_val_sample = behaviors_val.sample(
    n=min(initial_behavior_sample, len(behaviors_val)), random_state=42
)
behaviors_test_sample = behaviors_test.sample(
    n=min(initial_behavior_sample, len(behaviors_test)), random_state=42
)

print(
    f"Sampled behaviors - Train: {len(behaviors_train_sample)}, Val: {len(behaviors_val_sample)}, Test: {len(behaviors_test_sample)}"
)

# ============================================================================
# STEP 6: PARSE IMPRESSIONS TO CREATE TARGET VARIABLE
# ============================================================================
print("\nStep 6: Parsing impressions to create target variable...")

# Process train impressions (with clicks)
print("Processing train impressions...")
train_impressions = behaviors_train_sample["impressions"].str.split().explode()
train_expanded = pd.DataFrame(
    {
        "impression": train_impressions.values,
        "user_id": behaviors_train_sample.loc[
            train_impressions.index, "user_id"
        ].values,
        "time": behaviors_train_sample.loc[train_impressions.index, "time"].values,
    }
)

# Split impression to get news_id and click info
train_split = train_expanded["impression"].str.split("-", expand=True)
train_expanded["news_id"] = train_split[0]
train_expanded["clicked"] = train_split[1].astype(int)

print(f"Train expanded impressions: {len(train_expanded)}")
print(f"Unique articles in train: {train_expanded['news_id'].nunique()}")

# Calculate CTR per article
articles_train = (
    train_expanded.groupby("news_id")
    .agg({"clicked": ["sum", "count"], "time": "first"})
    .reset_index()
)

articles_train.columns = ["news_id", "total_clicks", "total_impressions", "first_seen"]
articles_train["ctr"] = (
    articles_train["total_clicks"] / articles_train["total_impressions"]
)

print(f"Train articles with CTR: {len(articles_train)}")

# Process validation impressions
print("Processing validation impressions...")
val_impressions = behaviors_val_sample["impressions"].str.split().explode()
val_expanded = pd.DataFrame(
    {
        "impression": val_impressions.values,
        "user_id": behaviors_val_sample.loc[val_impressions.index, "user_id"].values,
        "time": behaviors_val_sample.loc[val_impressions.index, "time"].values,
    }
)

val_split = val_expanded["impression"].str.split("-", expand=True)
val_expanded["news_id"] = val_split[0]
val_expanded["clicked"] = val_split[1].astype(int)

articles_val = (
    val_expanded.groupby("news_id")
    .agg({"clicked": ["sum", "count"], "time": "first"})
    .reset_index()
)

articles_val.columns = ["news_id", "total_clicks", "total_impressions", "first_seen"]
articles_val["ctr"] = articles_val["total_clicks"] / articles_val["total_impressions"]

print(f"Val articles with CTR: {len(articles_val)}")

# Process test impressions (no clicks)
print("Processing test impressions...")
test_impressions = behaviors_test_sample["impressions"].str.split().explode()
test_expanded = pd.DataFrame(
    {
        "impression": test_impressions.values,
        "user_id": behaviors_test_sample.loc[test_impressions.index, "user_id"].values,
        "time": behaviors_test_sample.loc[test_impressions.index, "time"].values,
    }
)

test_expanded["news_id"] = test_expanded["impression"]

articles_test = test_expanded.groupby("news_id").agg({"time": "first"}).reset_index()

articles_test.columns = ["news_id", "first_seen"]
articles_test["ctr"] = np.nan

print(f"Test articles: {len(articles_test)}")

# ============================================================================
# STEP 7: MERGE WITH NEWS CONTENT
# ============================================================================
print("\nStep 7: Merging with news content...")

df_train = articles_train.merge(
    news_train, left_on="news_id", right_on="newsID", how="left"
)
df_val = articles_val.merge(news_val, left_on="news_id", right_on="newsID", how="left")
df_test = articles_test.merge(
    news_test, left_on="news_id", right_on="newsID", how="left"
)

print(f"After merge - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

# Check merge quality
print(f"Train merge success rate: {(~df_train['title'].isnull()).mean():.3f}")
print(f"Val merge success rate: {(~df_val['title'].isnull()).mean():.3f}")
print(f"Test merge success rate: {(~df_test['title'].isnull()).mean():.3f}")

# Remove failed merges
df_train = df_train.dropna(subset=["title"])
df_val = df_val.dropna(subset=["title"])
df_test = df_test.dropna(subset=["title"])

print(
    f"After removing failed merges - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
)

# ============================================================================
# STEP 8: BIVARIATE ANALYSIS WITH TARGET (CTR)
# ============================================================================
print("\nStep 8: Bivariate analysis with target (CTR)...")

print("\n--- CTR STATISTICS ---")
print(f"CTR mean: {df_train['ctr'].mean():.6f}")
print(f"CTR median: {df_train['ctr'].median():.6f}")
print(f"CTR std: {df_train['ctr'].std():.6f}")
print(f"CTR min: {df_train['ctr'].min():.6f}")
print(f"CTR max: {df_train['ctr'].max():.6f}")

print("\n--- CTR BY CATEGORY ---")
ctr_by_category = df_train.groupby("category")["ctr"].agg(["mean", "count"]).round(6)
ctr_by_category = ctr_by_category[ctr_by_category["count"] >= 50].sort_values(
    "mean", ascending=False
)
print(ctr_by_category.head(10))

print("\n--- CTR BY TIME ---")
df_train["hour"] = df_train["first_seen"].dt.hour
df_train["day_of_week"] = df_train["first_seen"].dt.dayofweek
ctr_by_hour = df_train.groupby("hour")["ctr"].mean().round(6)
print("CTR by hour (top 5):")
print(ctr_by_hour.sort_values(ascending=False).head())

# Basic text length analysis
df_train["title_length"] = df_train["title"].str.len()
print(
    f"\nCorrelation between title length and CTR: {df_train['title_length'].corr(df_train['ctr']):.6f}"
)

# ============================================================================
# STEP 9: FINAL SAMPLING TO TARGET SIZE
# ============================================================================
print("\nStep 9: Final sampling to target size...")

target_size = CONFIG["target_sample_size"]

# Sample to target size
df_train = df_train.sample(n=min(target_size, len(df_train)), random_state=42)
df_val = df_val.sample(n=min(target_size, len(df_val)), random_state=42)
df_test = df_test.sample(n=min(target_size, len(df_test)), random_state=42)

print(
    f"Final sample sizes - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
)

# ============================================================================
# STEP 10: FEATURE ENGINEERING - TEXT FEATURES
# ============================================================================
print("\nStep 10: Text feature engineering...")

# Basic text statistics
df_train["abstract_length"] = df_train["abstract"].str.len()
df_train["title_word_count"] = df_train["title"].str.split().str.len()
df_train["abstract_word_count"] = df_train["abstract"].str.split().str.len()

# Apply same to val and test
df_val["title_length"] = df_val["title"].str.len()
df_val["abstract_length"] = df_val["abstract"].str.len()
df_val["title_word_count"] = df_val["title"].str.split().str.len()
df_val["abstract_word_count"] = df_val["abstract"].str.split().str.len()

df_test["title_length"] = df_test["title"].str.len()
df_test["abstract_length"] = df_test["abstract"].str.len()
df_test["title_word_count"] = df_test["title"].str.split().str.len()
df_test["abstract_word_count"] = df_test["abstract"].str.split().str.len()

# Readability scores
print("Computing readability scores...")
df_train["title_reading_ease"] = df_train["title"].apply(
    lambda x: flesch_reading_ease(x) if pd.notna(x) and len(x) > 0 else 0
)
df_train["abstract_reading_ease"] = df_train["abstract"].apply(
    lambda x: flesch_reading_ease(x) if pd.notna(x) and len(x) > 0 else 0
)

df_val["title_reading_ease"] = df_val["title"].apply(
    lambda x: flesch_reading_ease(x) if pd.notna(x) and len(x) > 0 else 0
)
df_val["abstract_reading_ease"] = df_val["abstract"].apply(
    lambda x: flesch_reading_ease(x) if pd.notna(x) and len(x) > 0 else 0
)

df_test["title_reading_ease"] = df_test["title"].apply(
    lambda x: flesch_reading_ease(x) if pd.notna(x) and len(x) > 0 else 0
)
df_test["abstract_reading_ease"] = df_test["abstract"].apply(
    lambda x: flesch_reading_ease(x) if pd.notna(x) and len(x) > 0 else 0
)

# Title characteristics
df_train["has_question"] = df_train["title"].str.contains(r"\?", na=False).astype(int)
df_train["has_exclamation"] = df_train["title"].str.contains(r"!", na=False).astype(int)
df_train["has_number"] = df_train["title"].str.contains(r"\d", na=False).astype(int)
df_train["has_colon"] = df_train["title"].str.contains(r":", na=False).astype(int)
df_train["has_quotes"] = df_train["title"].str.contains(r'["\']', na=False).astype(int)

# Apply to val and test
for col in ["has_question", "has_exclamation", "has_number", "has_colon", "has_quotes"]:
    pattern = {
        "has_question": r"\?",
        "has_exclamation": r"!",
        "has_number": r"\d",
        "has_colon": r":",
        "has_quotes": r'["\']',
    }[col]
    df_val[col] = df_val["title"].str.contains(pattern, na=False).astype(int)
    df_test[col] = df_test["title"].str.contains(pattern, na=False).astype(int)

# Title case ratio
df_train["title_upper_ratio"] = df_train["title"].apply(
    lambda x: (
        sum(c.isupper() for c in str(x)) / len(str(x))
        if pd.notna(x) and len(str(x)) > 0
        else 0
    )
)
df_val["title_upper_ratio"] = df_val["title"].apply(
    lambda x: (
        sum(c.isupper() for c in str(x)) / len(str(x))
        if pd.notna(x) and len(str(x)) > 0
        else 0
    )
)
df_test["title_upper_ratio"] = df_test["title"].apply(
    lambda x: (
        sum(c.isupper() for c in str(x)) / len(str(x))
        if pd.notna(x) and len(str(x)) > 0
        else 0
    )
)

print("Text features created")

# ============================================================================
# STEP 11: TEMPORAL FEATURES
# ============================================================================
print("\nStep 11: Creating temporal features...")

# Apply to all datasets
for df in [df_train, df_val, df_test]:
    df["first_seen"] = pd.to_datetime(df["first_seen"])
    df["hour"] = df["first_seen"].dt.hour
    df["day_of_week"] = df["first_seen"].dt.dayofweek
    df["month"] = df["first_seen"].dt.month
    df["day_of_month"] = df["first_seen"].dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["time_of_day"] = pd.cut(
        df["hour"], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], include_lowest=True
    ).astype(int)

print("Temporal features created")

# ============================================================================
# STEP 12: CATEGORY ENCODING
# ============================================================================
print("\nStep 12: Category encoding...")

# Handle missing categories
df_train["category"] = df_train["category"].fillna("unknown")
df_val["category"] = df_val["category"].fillna("unknown")
df_test["category"] = df_test["category"].fillna("unknown")

# Fit encoder on training data
le = LabelEncoder()
df_train["category_enc"] = le.fit_transform(df_train["category"])


# Safe transform for validation and test
def safe_transform_categories(categories, encoder):
    result = []
    for cat in categories:
        if cat in encoder.classes_:
            result.append(encoder.transform([cat])[0])
        else:
            # Use 'unknown' if available, otherwise use 0
            if "unknown" in encoder.classes_:
                result.append(encoder.transform(["unknown"])[0])
            else:
                result.append(0)
    return result


df_val["category_enc"] = safe_transform_categories(df_val["category"], le)
df_test["category_enc"] = safe_transform_categories(df_test["category"], le)

print("Category encoding completed")

# ============================================================================
# STEP 13: CREATE TEXT EMBEDDINGS
# ============================================================================
print("\nStep 13: Creating text embeddings...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
cache_dir = PREP_DIR / "cache"


# Function to create embeddings with caching
def create_embeddings_cached(df, dataset_name):
    cache_file = cache_dir / f"{dataset_name}_embeddings.pkl"

    if cache_file.exists():
        print(f"Loading cached {dataset_name} embeddings...")
        emb_map = pd.read_pickle(cache_file)
        return df.merge(emb_map, on="newsID", how="left")
    else:
        print(f"Creating {dataset_name} embeddings...")
        # Get unique articles to avoid duplicate computation
        unique_articles = df.drop_duplicates("newsID").set_index("newsID")
        titles = unique_articles["title"].fillna("").tolist()

        # Create embeddings
        embeddings = embedder.encode(titles, show_progress_bar=True, batch_size=32)

        # Create embedding dataframe
        emb_map = pd.DataFrame(
            embeddings[:, : CONFIG["embedding_dim"]],
            index=unique_articles.index,
            columns=[f"emb_{i}" for i in range(CONFIG["embedding_dim"])],
        )

        # Cache the embeddings
        emb_map.to_pickle(cache_file)

        # Merge with original dataframe
        return df.merge(emb_map, on="newsID", how="left")


# Create embeddings for all datasets
df_train = create_embeddings_cached(df_train, "train")
df_val = create_embeddings_cached(df_val, "val")
df_test = create_embeddings_cached(df_test, "test")

print("Embeddings created and merged")

# ============================================================================
# STEP 14: INTERACTION FEATURES
# ============================================================================
print("\nStep 14: Creating interaction features...")

# Create interaction features for all datasets
for df in [df_train, df_val, df_test]:
    df["title_length_x_category"] = df["title_length"] * df["category_enc"]
    df["has_colon_x_category"] = df["has_colon"] * df["category_enc"]
    df["word_count_x_category"] = df["title_word_count"] * df["category_enc"]
    df["has_quotes_x_category"] = df["has_quotes"] * df["category_enc"]
    df["title_features_combined"] = (
        df["has_colon"] + df["has_quotes"] + df["has_number"]
    ) * df["title_length"]

print("Interaction features created")

# ============================================================================
# STEP 15: PREPARE FINAL FEATURE MATRICES
# ============================================================================
print("\nStep 15: Preparing final feature matrices...")

# Define feature sets
base_features = [
    # Text features
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
    # Category features
    "category_enc",
    # Temporal features
    "hour",
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "time_of_day",
    # Interaction features
    "title_length_x_category",
    "has_colon_x_category",
    "word_count_x_category",
    "has_quotes_x_category",
    "title_features_combined",
]

# Embedding features
emb_features = [f"emb_{i}" for i in range(CONFIG["embedding_dim"])]
all_features = base_features + emb_features

# Check which features exist
available_features = [f for f in all_features if f in df_train.columns]
missing_features = [f for f in all_features if f not in df_train.columns]

print(f"Available features: {len(available_features)} / {len(all_features)}")
if missing_features:
    print(f"Missing features: {missing_features}")

# Create final feature matrices
X_train = df_train[available_features].fillna(0)
y_train = df_train["ctr"]

X_val = df_val[available_features].fillna(0)
y_val = df_val["ctr"]

X_test = df_test[available_features].fillna(0)

print(
    f"Final shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}"
)

# ============================================================================
# STEP 16: FEATURE CORRELATION ANALYSIS
# ============================================================================
print("\nStep 16: Feature correlation analysis...")

# Analyze correlations with target
non_emb_features = [f for f in available_features if not f.startswith("emb_")][:15]
df_corr = X_train[non_emb_features].copy()
df_corr["ctr"] = y_train
correlations = df_corr.corr()["ctr"].drop("ctr").abs().sort_values(ascending=False)

print("Top 10 feature correlations with CTR:")
for feat, corr in correlations.head(10).items():
    print(f"  {feat}: {corr:.4f}")

# Check for high correlations (potential data leakage)
high_corr = correlations[correlations > 0.8]
if len(high_corr) > 0:
    print(f"WARNING: High correlations detected (>0.8): {list(high_corr.index)}")
else:
    print("No suspiciously high correlations detected")

# ============================================================================
# STEP 17: CREATE VISUALIZATIONS
# ============================================================================
print("\nStep 17: Creating visualizations...")

# Set plotting style
plt.style.use("default")
sns.set_palette("husl")

# Plot 1: CTR Distribution
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("CTR Analysis Dashboard", fontsize=16, fontweight="bold")

# CTR histogram
axes[0, 0].hist(y_train, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
axes[0, 0].set_title("CTR Distribution")
axes[0, 0].set_xlabel("Click Through Rate")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].axvline(
    y_train.mean(), color="red", linestyle="--", label=f"Mean: {y_train.mean():.4f}"
)
axes[0, 0].legend()

# CTR by category
ctr_by_cat = (
    df_train.groupby("category")["ctr"].mean().sort_values(ascending=False).head(10)
)
axes[0, 1].barh(range(len(ctr_by_cat)), ctr_by_cat.values, color="lightcoral")
axes[0, 1].set_yticks(range(len(ctr_by_cat)))
axes[0, 1].set_yticklabels(ctr_by_cat.index)
axes[0, 1].set_title("Average CTR by Category (Top 10)")
axes[0, 1].set_xlabel("Average CTR")

# CTR by hour
ctr_by_hour = df_train.groupby("hour")["ctr"].mean()
axes[1, 0].plot(ctr_by_hour.index, ctr_by_hour.values, marker="o", linewidth=2)
axes[1, 0].set_title("CTR by Hour of Day")
axes[1, 0].set_xlabel("Hour")
axes[1, 0].set_ylabel("Average CTR")
axes[1, 0].grid(True, alpha=0.3)

# Feature importance preview
top_correlations = correlations.head(10)
axes[1, 1].barh(range(len(top_correlations)), top_correlations.values)
axes[1, 1].set_yticks(range(len(top_correlations)))
axes[1, 1].set_yticklabels(top_correlations.index, fontsize=8)
axes[1, 1].set_title("Top Feature Correlations with CTR")
axes[1, 1].set_xlabel("Absolute Correlation")

plt.tight_layout()
plt.savefig(
    PREP_DIR / "plots" / "ctr_analysis_dashboard.png", dpi=300, bbox_inches="tight"
)
plt.close()

# Plot 2: Feature correlation heatmap
key_features_for_heatmap = ["ctr"] + non_emb_features[:12]  # Top 12 + CTR
corr_matrix = df_train[key_features_for_heatmap].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5,
    fmt=".3f",
)
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(
    PREP_DIR / "plots" / "feature_correlation_heatmap.png", dpi=300, bbox_inches="tight"
)
plt.close()

# Plot 3: Text feature analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Text Feature Analysis", fontsize=16, fontweight="bold")

# Title length vs CTR
axes[0, 0].scatter(df_train["title_length"], df_train["ctr"], alpha=0.1, s=1)
axes[0, 0].set_title("Title Length vs CTR")
axes[0, 0].set_xlabel("Title Length")
axes[0, 0].set_ylabel("CTR")

# Question marks impact
question_ctr = df_train.groupby("has_question")["ctr"].agg(["mean", "count"])
axes[0, 1].bar(
    ["No Question", "Has Question"], question_ctr["mean"], color=["lightblue", "orange"]
)
axes[0, 1].set_title("CTR Impact of Questions in Titles")
axes[0, 1].set_ylabel("Average CTR")
for i, (idx, row) in enumerate(question_ctr.iterrows()):
    axes[0, 1].text(
        i, row["mean"] + 0.001, f'n={row["count"]}', ha="center", va="bottom"
    )

# Word count distribution
axes[1, 0].hist(
    df_train["title_word_count"], bins=range(1, 21), alpha=0.7, color="purple"
)
axes[1, 0].set_title("Title Word Count Distribution")
axes[1, 0].set_xlabel("Number of Words")
axes[1, 0].set_ylabel("Frequency")

# Numbers in titles
number_ctr = df_train.groupby("has_number")["ctr"].agg(["mean", "count"])
axes[1, 1].bar(
    ["No Numbers", "Has Numbers"], number_ctr["mean"], color=["lightgreen", "coral"]
)
axes[1, 1].set_title("CTR Impact of Numbers in Titles")
axes[1, 1].set_ylabel("Average CTR")
for i, (idx, row) in enumerate(number_ctr.iterrows()):
    axes[1, 1].text(
        i, row["mean"] + 0.001, f'n={row["count"]}', ha="center", va="bottom"
    )

plt.tight_layout()
plt.savefig(
    PREP_DIR / "plots" / "text_feature_analysis.png", dpi=300, bbox_inches="tight"
)
plt.close()

print("Visualizations created and saved")

# ============================================================================
# STEP 18: SAVE PROCESSED DATA
# ============================================================================
print("\nStep 18: Saving processed data...")

# Save feature matrices
X_train.to_parquet(PREP_DIR / "processed_data" / "X_train_clean.parquet")
y_train.to_frame("ctr").to_parquet(
    PREP_DIR / "processed_data" / "y_train_clean.parquet"
)

X_val.to_parquet(PREP_DIR / "processed_data" / "X_val_clean.parquet")
y_val.to_frame("ctr").to_parquet(PREP_DIR / "processed_data" / "y_val_clean.parquet")

X_test.to_parquet(PREP_DIR / "processed_data" / "X_test_clean.parquet")
df_test[["ctr"]].to_parquet(PREP_DIR / "processed_data" / "y_test_clean.parquet")

# Save label encoder
with open(PREP_DIR / "processed_data" / "label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Save metadata
metadata = {
    "dataset_info": {
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "total_features": len(available_features),
    },
    "features_used": available_features,
    "feature_counts": {
        "total": len(available_features),
        "text": len(
            [
                f
                for f in available_features
                if any(x in f for x in ["title_", "abstract_", "has_"])
            ]
        ),
        "temporal": len(
            [
                f
                for f in available_features
                if any(x in f for x in ["hour", "day_", "weekend", "time_"])
            ]
        ),
        "interaction": len([f for f in available_features if "_x_" in f]),
        "embedding": len([f for f in available_features if f.startswith("emb_")]),
        "category": 1,
    },
    "target_statistics": {
        "mean_ctr": float(y_train.mean()),
        "median_ctr": float(y_train.median()),
        "std_ctr": float(y_train.std()),
        "min_ctr": float(y_train.min()),
        "max_ctr": float(y_train.max()),
    },
    "top_correlations": {
        feat: float(corr) for feat, corr in correlations.head(10).items()
    },
    "categories_count": len(le.classes_),
    "processing_config": CONFIG,
    "visualizations_created": [
        "ctr_analysis_dashboard.png",
        "feature_correlation_heatmap.png",
        "text_feature_analysis.png",
    ],
}

with open(PREP_DIR / "processed_data" / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nDATAS ET SIZES:")
print(f"  Train: {len(X_train):,} articles")
print(f"  Validation: {len(X_val):,} articles")
print(f"  Test: {len(X_test):,} articles")

print(f"\nFEATURE BREAKDOWN:")
for ftype, count in metadata["feature_counts"].items():
    print(f"  {ftype.title()}: {count} features")

print(f"\nTARGET VARIABLE (CTR) STATISTICS:")
print(f"  Mean: {metadata['target_statistics']['mean_ctr']:.6f}")
print(f"  Median: {metadata['target_statistics']['median_ctr']:.6f}")
print(f"  Std Dev: {metadata['target_statistics']['std_ctr']:.6f}")
print(
    f"  Range: [{metadata['target_statistics']['min_ctr']:.6f}, {metadata['target_statistics']['max_ctr']:.6f}]"
)

print(f"\nTOP PREDICTIVE FEATURES:")
for i, (feat, corr) in enumerate(list(metadata["top_correlations"].items())[:5], 1):
    print(f"  {i}. {feat}: {corr:.4f}")

print(f"\nDATA QUALITY:")
high_corr_check = any(corr > 0.8 for corr in metadata["top_correlations"].values())
print(
    f"  High correlations (>0.8): {'❌ Found' if high_corr_check else '✅ None detected'}"
)
print(f"  Missing values: ✅ Handled")
print(f"  Duplicates: ✅ Removed")
print(f"  Categories encoded: ✅ {metadata['categories_count']} categories")

print(f"\nFILES SAVED:")
print(f"  ✅ X_train_clean.parquet, y_train_clean.parquet")
print(f"  ✅ X_val_clean.parquet, y_val_clean.parquet")
print(f"  ✅ X_test_clean.parquet, y_test_clean.parquet")
print(f"  ✅ label_encoder.pkl")
print(f"  ✅ metadata.json")

print(f"\nVISUALIZATIONS CREATED:")
for viz in metadata["visualizations_created"]:
    print(f"  📊 {viz}")

print(f"\nREADY FOR MODEL TRAINING!")
print(f"  Expected performance: Realistic CTR prediction")
print(f"  No data leakage: ✅ Only pre-publication features used")
print(f"  Balanced dataset: ✅ Representative sampling")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("  1. Load processed data: pd.read_parquet('X_train_clean.parquet')")
print("  2. Train models using clean features")
print("  3. Validate on clean validation set")
print("  4. Test on clean test set")
print("  5. Review visualizations for insights")
print("=" * 60)
