import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

print("Completing EDA process...")

# Paths
PREP_DIR = Path("data/preprocessed")
cache_dir = PREP_DIR / "cache"

# Check what we have so far
print("Checking existing data...")

# Load basic processed data (should exist from your correlations output)
try:
    # Try to find the dataframes from memory or recreate basic processing
    print("Need to recreate basic processing first...")

    # Quick data processing (same as before but faster)
    TRAIN_DIR = Path("source_data/train_data")
    VAL_DIR = Path("source_data/val_data")
    TEST_DIR = Path("source_data/test_data")

    CONFIG = {
        "train_sample_size": 400_000,
        "val_sample_size": 400_000,
        "test_sample_size": 400_000,
        "embedding_dim": 384,
        "random_state": 42,
    }

    # Load data (quick version)
    print("Loading and processing data...")

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
    behaviors_train = pd.read_csv(
        TRAIN_DIR / "behaviors.tsv",
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"],
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
    behaviors_val = pd.read_csv(
        VAL_DIR / "behaviors.tsv",
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"],
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
    behaviors_test = pd.read_csv(
        TEST_DIR / "behaviors.tsv",
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"],
    )

    # Clean and process (abbreviated version)
    for news_df in [news_train, news_val, news_test]:
        news_df.dropna(subset=["newsID", "title", "category"], inplace=True)
        news_df.drop_duplicates(subset=["newsID", "title"], inplace=True)
        news_df["abstract"] = news_df["abstract"].fillna("")

    for behav_df in [behaviors_train, behaviors_val, behaviors_test]:
        behav_df.dropna(
            subset=["impression_id", "user_id", "impressions"], inplace=True
        )
        behav_df["time"] = pd.to_datetime(behav_df["time"])

    # Sample behaviors
    behaviors_train = behaviors_train.sample(
        n=min(CONFIG["train_sample_size"], len(behaviors_train)), random_state=42
    )
    behaviors_val = behaviors_val.sample(
        n=min(CONFIG["val_sample_size"], len(behaviors_val)), random_state=42
    )
    behaviors_test = behaviors_test.sample(
        n=min(CONFIG["test_sample_size"], len(behaviors_test)), random_state=42
    )

    print("Processing impressions to articles...")

    # Process train
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

    # Process val
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
    articles_val["ctr"] = (
        articles_val["total_clicks"] / articles_val["total_impressions"]
    )

    # Process test
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

    # Merge with news
    df_train = articles_train.merge(
        news_train, left_on="news_id", right_on="newsID", how="left"
    )
    df_val = articles_val.merge(
        news_val, left_on="news_id", right_on="newsID", how="left"
    )
    df_test = articles_test.merge(
        news_test, left_on="news_id", right_on="newsID", how="left"
    )

    print("Adding features...")

    # Add all features (text, temporal, category)
    from textstat import flesch_reading_ease
    from sklearn.preprocessing import LabelEncoder

    # Text features for all datasets
    for df in [df_train, df_val, df_test]:
        df["title_length"] = df["title"].str.len()
        df["abstract_length"] = df["abstract"].str.len()
        df["title_word_count"] = df["title"].str.split().str.len()
        df["abstract_word_count"] = df["abstract"].str.split().str.len()
        df["title_reading_ease"] = df["title"].apply(
            lambda x: flesch_reading_ease(x) if pd.notna(x) and len(x) > 0 else 0
        )
        df["abstract_reading_ease"] = df["abstract"].apply(
            lambda x: flesch_reading_ease(x) if pd.notna(x) and len(x) > 0 else 0
        )
        df["has_question"] = df["title"].str.contains(r"\?", na=False).astype(int)
        df["has_exclamation"] = df["title"].str.contains(r"!", na=False).astype(int)
        df["has_number"] = df["title"].str.contains(r"\d", na=False).astype(int)
        df["has_colon"] = df["title"].str.contains(r":", na=False).astype(int)
        df["has_quotes"] = df["title"].str.contains(r'["\']', na=False).astype(int)
        df["title_upper_ratio"] = df["title"].apply(
            lambda x: (
                sum(c.isupper() for c in str(x)) / len(str(x))
                if pd.notna(x) and len(str(x)) > 0
                else 0
            )
        )

        # Temporal features
        df["first_seen"] = pd.to_datetime(df["first_seen"])
        df["hour"] = df["first_seen"].dt.hour
        df["day_of_week"] = df["first_seen"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["time_of_day"] = pd.cut(
            df["hour"],
            bins=[0, 6, 12, 18, 24],
            labels=[0, 1, 2, 3],
            include_lowest=True,
        ).astype(int)

    # Category encoding
    df_train["category"] = df_train["category"].fillna("unknown")
    df_val["category"] = df_val["category"].fillna("unknown")
    df_test["category"] = df_test["category"].fillna("unknown")

    le = LabelEncoder()
    df_train["category_enc"] = le.fit_transform(df_train["category"])

    def safe_transform(categories, encoder):
        result = []
        for cat in categories:
            if cat in encoder.classes_:
                result.append(encoder.transform([cat])[0])
            else:
                if "unknown" in encoder.classes_:
                    result.append(encoder.transform(["unknown"])[0])
                else:
                    result.append(-1)
        return result

    df_val["category_enc"] = safe_transform(df_val["category"], le)
    df_test["category_enc"] = safe_transform(df_test["category"], le)

    print("Loading embeddings...")

    # Load embeddings (should be created by the batch script)
    emb_files = list(cache_dir.glob("*embeddings*.pkl"))
    if not emb_files:
        print("ERROR: No embedding files found! Run the batch embedding script first.")
        exit()

    # Load train embeddings
    train_emb_file = cache_dir / "train_embeddings.pkl"
    if train_emb_file.exists():
        train_emb_map = pd.read_pickle(train_emb_file)
        df_train = df_train.merge(train_emb_map, on="newsID", how="left")

    # Load val embeddings
    val_emb_file = cache_dir / "val_embeddings.pkl"
    if val_emb_file.exists():
        val_emb_map = pd.read_pickle(val_emb_file)
        df_val = df_val.merge(val_emb_map, on="newsID", how="left")

    # Load test embeddings
    test_emb_file = cache_dir / "test_embeddings.pkl"
    if test_emb_file.exists():
        test_emb_map = pd.read_pickle(test_emb_file)
        df_test = df_test.merge(test_emb_map, on="newsID", how="left")

    print("Creating final feature matrices...")

    # Define feature sets
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

    # Check which features actually exist
    available_features = [f for f in all_features if f in df_train.columns]
    print(f"Available features: {len(available_features)} / {len(all_features)}")

    # Create final matrices
    X_train = df_train[available_features].fillna(0)
    y_train = df_train["ctr"]
    X_val = df_val[available_features].fillna(0)
    y_val = df_val["ctr"]
    X_test = df_test[available_features].fillna(0)

    print("Saving final processed data...")

    # Save processed data
    X_train.to_parquet(PREP_DIR / "processed_data" / "X_train.parquet")
    y_train.to_frame("ctr").to_parquet(PREP_DIR / "processed_data" / "y_train.parquet")
    X_val.to_parquet(PREP_DIR / "processed_data" / "X_val.parquet")
    y_val.to_frame("ctr").to_parquet(PREP_DIR / "processed_data" / "y_val.parquet")
    X_test.to_parquet(PREP_DIR / "processed_data" / "X_test.parquet")
    df_test[["ctr"]].to_parquet(PREP_DIR / "processed_data" / "y_test.parquet")

    print("EDA completion successful!")
    print(f"Final dataset sizes:")
    print(f"- Train: {len(df_train)} articles, {len(available_features)} features")
    print(f"- Val: {len(df_val)} articles")
    print(f"- Test: {len(df_test)} articles")
    print("Ready for model training!")

except Exception as e:
    print(f"Error: {e}")
    print("Make sure to run the batch embedding script first!")
