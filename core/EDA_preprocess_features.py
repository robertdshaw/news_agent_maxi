import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from sentence_transformers import SentenceTransformer
from textstat import flesch_reading_ease
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("EDA PREPROCESSING WITH FEATURE ENGINEERING")
print("=" * 80)

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
    "high_engagement_threshold": 0.05,
    "embedding_dim": 384,
    "random_state": 42,
    "chunk_size": 50000,
    "use_pca": True,
    "pca_components": 150,
}

EDITORIAL_CRITERIA = {
    "target_reading_ease": 60,
    "readability_weight": 0.3,
    "engagement_weight": 0.4,
    "headline_quality_weight": 0.2,
    "timeliness_weight": 0.1,
    "target_ctr_gain": 0.05,
    "optimal_word_count": (8, 12),
    "max_title_length": 75,
}

# ============================================================================
# STEP 1: LOAD RAW DATA WITH MEMORY OPTIMIZATION
# ============================================================================
print("\nStep 1: Loading raw data with memory optimization...")


def load_data_efficiently():
    """Load data with optimized data types to reduce memory usage"""

    print("Loading news data...")

    news_columns = [
        "newsID",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]

    behaviors_columns = ["impression_id", "user_id", "time", "history", "impressions"]

    news_train = pd.read_csv(
        TRAIN_DIR / "news.tsv",
        sep="\t",
        header=None,
        names=news_columns,
        dtype={
            "newsID": "category",
            "category": "category",
            "title": "string",
            "abstract": "string",
        },
    )

    news_val = pd.read_csv(
        VAL_DIR / "news.tsv",
        sep="\t",
        header=None,
        names=news_columns,
        dtype={
            "newsID": "category",
            "category": "category",
            "title": "string",
            "abstract": "string",
        },
    )

    news_test = pd.read_csv(
        TEST_DIR / "news.tsv",
        sep="\t",
        header=None,
        names=news_columns,
        dtype={
            "newsID": "category",
            "category": "category",
            "title": "string",
            "abstract": "string",
        },
    )

    print("Loading behavior data...")

    behaviors_train = pd.read_csv(
        TRAIN_DIR / "behaviors.tsv",
        sep="\t",
        header=None,
        names=behaviors_columns,
        dtype={"impression_id": "int32", "user_id": "category"},
    )

    behaviors_val = pd.read_csv(
        VAL_DIR / "behaviors.tsv",
        sep="\t",
        header=None,
        names=behaviors_columns,
        dtype={"impression_id": "int32", "user_id": "category"},
    )

    behaviors_test = pd.read_csv(
        TEST_DIR / "behaviors.tsv",
        sep="\t",
        header=None,
        names=behaviors_columns,
        dtype={"impression_id": "int32", "user_id": "category"},
    )

    return (
        news_train,
        news_val,
        news_test,
        behaviors_train,
        behaviors_val,
        behaviors_test,
    )


try:
    (
        news_train,
        news_val,
        news_test,
        behaviors_train,
        behaviors_val,
        behaviors_test,
    ) = load_data_efficiently()

    print(f"Loaded successfully:")
    print(
        f"  News train: {news_train.shape}, Val: {news_val.shape}, Test: {news_test.shape}"
    )
    print(
        f"  Behaviors train: {behaviors_train.shape}, Val: {behaviors_val.shape}, Test: {behaviors_test.shape}"
    )

except Exception as e:
    print(f"Error loading data: {e}")
    print("Please ensure source data files exist in the correct directories")
    exit(1)

# ============================================================================
# STEP 2: CHUNKED IMPRESSION PROCESSING
# ============================================================================
print(
    f"\nStep 2: Processing impressions with chunking (chunk size: {CONFIG['chunk_size']:,})..."
)


def process_impressions_chunked(behaviors_df, is_test=False, chunk_size=50000):
    """Process large impression datasets in chunks to avoid memory issues"""

    print(
        f"Processing {len(behaviors_df):,} behavior records in chunks of {chunk_size:,}"
    )

    all_articles = []
    total_chunks = (len(behaviors_df) + chunk_size - 1) // chunk_size

    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(behaviors_df))
        chunk = behaviors_df.iloc[start_idx:end_idx].copy()

        print(
            f"  Processing chunk {chunk_idx + 1}/{total_chunks}: rows {start_idx:,} to {end_idx:,}"
        )

        impressions = chunk["impressions"].str.split().explode()
        expanded = pd.DataFrame(
            {
                "impression": impressions.values,
                "time": chunk.loc[impressions.index, "time"].values,
            }
        )

        if not is_test:
            split = expanded["impression"].str.split("-", expand=True)
            expanded["newsID"] = split[0]
            expanded["clicked"] = split[1].astype("int8")

            chunk_articles = (
                expanded.groupby("newsID")
                .agg({"clicked": ["sum", "count"], "time": "first"})
                .reset_index()
            )
            chunk_articles.columns = [
                "newsID",
                "total_clicks",
                "total_impressions",
                "first_seen",
            ]

        else:
            expanded["newsID"] = expanded["impression"]
            chunk_articles = (
                expanded.groupby("newsID")
                .agg({"time": ["first", "count"]})
                .reset_index()
            )
            chunk_articles.columns = ["newsID", "first_seen", "total_impressions"]
            chunk_articles["total_clicks"] = 0

        all_articles.append(chunk_articles)

        del expanded, chunk

        if (chunk_idx + 1) % 5 == 0:
            print(f"    Completed {chunk_idx + 1}/{total_chunks} chunks")

    print("  Combining chunks and performing final aggregation...")
    combined = pd.concat(all_articles, ignore_index=True)

    if not is_test:
        final_articles = (
            combined.groupby("newsID")
            .agg(
                {
                    "total_clicks": "sum",
                    "total_impressions": "sum",
                    "first_seen": "first",
                }
            )
            .reset_index()
        )
        final_articles["ctr"] = (
            final_articles["total_clicks"] / final_articles["total_impressions"]
        )
    else:
        final_articles = (
            combined.groupby("newsID")
            .agg({"total_impressions": "sum", "first_seen": "first"})
            .reset_index()
        )
        final_articles["ctr"] = np.nan
        final_articles["total_clicks"] = 0

    print(f"  Final result: {len(final_articles):,} unique articles")
    return final_articles


print("Starting chunked processing...")
articles_train = process_impressions_chunked(
    behaviors_train, is_test=False, chunk_size=CONFIG["chunk_size"]
)
articles_val = process_impressions_chunked(
    behaviors_val, is_test=False, chunk_size=CONFIG["chunk_size"]
)
articles_test = process_impressions_chunked(
    behaviors_test, is_test=True, chunk_size=CONFIG["chunk_size"]
)

print("Chunked processing completed successfully!")

# ============================================================================
# STEP 3: MERGE WITH NEWS CONTENT
# ============================================================================
print("\nStep 3: Merging article engagement with news content...")

df_train = articles_train.merge(news_train, on="newsID", how="inner")
df_val = articles_val.merge(news_val, on="newsID", how="inner")
df_test = articles_test.merge(news_test, on="newsID", how="inner")

print(f"After merge:")
print(f"  Train: {len(df_train):,} articles")
print(f"  Val: {len(df_val):,} articles")
print(f"  Test: {len(df_test):,} articles")

# ============================================================================
# STEP 4: COMPREHENSIVE FEATURE ENGINEERING
# ============================================================================
print("\nStep 4: Creating comprehensive features with editorial criteria...")

training_median_ctr = df_train["ctr"].median()
print(f"Training median CTR: {training_median_ctr:.6f}")


def create_editorial_features(df, median_ctr_value):
    """Create comprehensive features including editorial criteria for engagement prediction"""

    print(f"  Creating basic text features...")
    df["title_length"] = df["title"].str.len()
    df["abstract_length"] = df["abstract"].fillna("").str.len()
    df["title_word_count"] = df["title"].str.split().str.len()
    df["abstract_word_count"] = df["abstract"].fillna("").str.split().str.len()

    print(f"  Computing Flesch Reading Ease scores...")
    df["title_reading_ease"] = df["title"].apply(
        lambda x: flesch_reading_ease(x) if pd.notna(x) and len(str(x)) > 0 else 0
    )
    df["abstract_reading_ease"] = (
        df["abstract"]
        .fillna("")
        .apply(lambda x: flesch_reading_ease(x) if len(str(x)) > 0 else 0)
    )

    print(f"  Creating headline quality indicators...")
    df["has_question"] = df["title"].str.contains(r"\?", na=False).astype(int)
    df["has_exclamation"] = df["title"].str.contains(r"!", na=False).astype(int)
    df["has_number"] = df["title"].str.contains(r"\d", na=False).astype(int)
    df["has_colon"] = df["title"].str.contains(r":", na=False).astype(int)
    df["has_quotes"] = df["title"].str.contains(r'["\']', na=False).astype(int)
    df["has_dash"] = df["title"].str.contains(r"[-–—]", na=False).astype(int)

    print(f"  Computing advanced headline metrics...")
    df["title_upper_ratio"] = df["title"].apply(
        lambda x: (
            sum(c.isupper() for c in str(x)) / len(str(x))
            if pd.notna(x) and len(str(x)) > 0
            else 0
        )
    )
    df["title_caps_words"] = df["title"].str.count(r"\b[A-Z][A-Z]+\b")
    df["avg_word_length"] = df["title"].apply(
        lambda x: (
            np.mean([len(word) for word in str(x).split()])
            if pd.notna(x) and len(str(x).split()) > 0
            else 0
        )
    )

    df["has_abstract"] = (df["abstract_length"] > 0).fillna(False).astype(int)
    df["title_abstract_ratio"] = df["title_length"] / (df["abstract_length"] + 1)

    print(f"  Creating editorial scoring metrics...")
    df["editorial_readability_score"] = (
        np.clip(df["title_reading_ease"] / 100, 0, 1)
        * EDITORIAL_CRITERIA["readability_weight"]
    )

    df["editorial_headline_score"] = (
        (df["has_question"] + df["has_number"] + df["has_colon"])
        / 3
        * EDITORIAL_CRITERIA["headline_quality_weight"]
    )

    # CTR GAIN POTENTIAL
    if "ctr" in df.columns and not df["ctr"].isna().all():
        df["ctr_gain_potential"] = np.maximum(
            0, EDITORIAL_CRITERIA["target_ctr_gain"] - (df["ctr"] - median_ctr_value)
        )
        df["below_median_ctr"] = (df["ctr"] < median_ctr_value).astype(int)
    else:
        df["ctr_gain_potential"] = EDITORIAL_CRITERIA["target_ctr_gain"]
        df["below_median_ctr"] = 0

    df["needs_readability_improvement"] = (
        df["title_reading_ease"] < EDITORIAL_CRITERIA["target_reading_ease"]
    ).astype(int)

    df["suboptimal_word_count"] = (
        (df["title_word_count"] < EDITORIAL_CRITERIA["optimal_word_count"][0])
        | (df["title_word_count"] > EDITORIAL_CRITERIA["optimal_word_count"][1])
    ).astype(int)

    df["too_long_title"] = (
        df["title_length"] > EDITORIAL_CRITERIA["max_title_length"]
    ).astype(int)

    return df


df_train = create_editorial_features(df_train, training_median_ctr)
df_val = create_editorial_features(df_val, training_median_ctr)
df_test = create_editorial_features(df_test, training_median_ctr)

print("Feature engineering completed")

# ============================================================================
# STEP 5: CREATE TARGETS FOR CLASSIFICATION
# ============================================================================
print("\nStep 5: Creating classification targets...")

threshold = CONFIG["high_engagement_threshold"]
df_train["high_engagement"] = (df_train["ctr"] >= threshold).astype(int)
df_val["high_engagement"] = (df_val["ctr"] >= threshold).astype(int)

# Additional target for headline rewriting - Use training median CTR
df_train["needs_rewrite"] = (df_train["ctr"] < training_median_ctr).astype(int)
df_val["needs_rewrite"] = (df_val["ctr"] < training_median_ctr).astype(int)

print(f"Target distribution:")
print(
    f"  High engagement rate: {df_train['high_engagement'].mean():.3f} ({df_train['high_engagement'].mean()*100:.1f}%)"
)
print(
    f"  Articles needing rewrite: {df_train['needs_rewrite'].mean():.3f} ({df_train['needs_rewrite'].mean()*100:.1f}%)"
)

# ============================================================================
# STEP 6: COMPREHENSIVE EDA ANALYSIS
# ============================================================================
print("\nStep 6: Comprehensive EDA Analysis...")


def create_comprehensive_eda_plots(df_train, df_val):
    """Create comprehensive EDA plots for analysis"""

    plt.style.use("default")
    fig = plt.figure(figsize=(20, 25))

    threshold = CONFIG["high_engagement_threshold"]

    # 1. Target Distribution
    ax1 = plt.subplot(5, 3, 1)
    target_counts = df_train["high_engagement"].value_counts()
    plt.pie(
        target_counts.values,
        labels=["Low Engagement", "High Engagement"],
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Target Distribution (High Engagement)", fontsize=12, fontweight="bold")

    # 2. CTR Distribution
    ax2 = plt.subplot(5, 3, 2)
    plt.hist(df_train["ctr"], bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    plt.axvline(
        df_train["ctr"].median(),
        color="red",
        linestyle="--",
        label=f'Median: {df_train["ctr"].median():.6f}',
    )
    plt.axvline(
        threshold, color="orange", linestyle="--", label=f"Threshold: {threshold}"
    )
    plt.xlabel("Click-Through Rate (CTR)")
    plt.ylabel("Frequency")
    plt.title("CTR Distribution", fontsize=12, fontweight="bold")
    plt.legend()

    # 3. Category vs High Engagement
    ax3 = plt.subplot(5, 3, 3)
    category_engagement = (
        df_train.groupby("category")["high_engagement"]
        .agg(["mean", "count"])
        .reset_index()
    )
    category_engagement = category_engagement[category_engagement["count"] >= 50]
    plt.bar(range(len(category_engagement)), category_engagement["mean"])
    plt.xticks(
        range(len(category_engagement)),
        category_engagement["category"],
        rotation=45,
        ha="right",
    )
    plt.ylabel("High Engagement Rate")
    plt.title("High Engagement Rate by Category", fontsize=12, fontweight="bold")

    # 4. Title Length vs CTR
    ax4 = plt.subplot(5, 3, 4)
    plt.scatter(df_train["title_length"], df_train["ctr"], alpha=0.5, s=10)
    plt.xlabel("Title Length (characters)")
    plt.ylabel("CTR")
    plt.title("Title Length vs CTR", fontsize=12, fontweight="bold")

    # 5. Word Count vs CTR
    ax5 = plt.subplot(5, 3, 5)
    plt.scatter(df_train["title_word_count"], df_train["ctr"], alpha=0.5, s=10)
    plt.xlabel("Title Word Count")
    plt.ylabel("CTR")
    plt.title("Word Count vs CTR", fontsize=12, fontweight="bold")

    # 6. Reading Ease vs CTR
    ax6 = plt.subplot(5, 3, 6)
    plt.scatter(df_train["title_reading_ease"], df_train["ctr"], alpha=0.5, s=10)
    plt.xlabel("Flesch Reading Ease Score")
    plt.ylabel("CTR")
    plt.title("Readability vs CTR", fontsize=12, fontweight="bold")

    # 7. Engagement Features Comparison
    ax7 = plt.subplot(5, 3, 7)
    engagement_features = [
        "has_question",
        "has_number",
        "has_exclamation",
        "has_colon",
        "has_quotes",
    ]
    high_eng = df_train[df_train["high_engagement"] == 1][engagement_features].mean()
    low_eng = df_train[df_train["high_engagement"] == 0][engagement_features].mean()

    x = np.arange(len(engagement_features))
    width = 0.35
    plt.bar(x - width / 2, high_eng, width, label="High Engagement", alpha=0.8)
    plt.bar(x + width / 2, low_eng, width, label="Low Engagement", alpha=0.8)
    plt.xlabel("Features")
    plt.ylabel("Proportion")
    plt.title("Engagement Features by Target", fontsize=12, fontweight="bold")
    plt.xticks(x, engagement_features, rotation=45, ha="right")
    plt.legend()

    # 8. CTR by Title Word Count (Boxplot)
    ax8 = plt.subplot(5, 3, 8)
    word_count_bins = pd.cut(df_train["title_word_count"], bins=6)
    df_train["word_count_bin"] = pd.cut(
        df_train["title_word_count"],
        bins=[0, 10, 20, 30, 40, 50],
        labels=["0-10", "11-20", "21-30", "31-40", "41-50"],
    )
    df_train.boxplot(column="ctr", by="word_count_bin", ax=ax8)

    plt.title("CTR Distribution by Word Count Bins", fontsize=12, fontweight="bold")
    plt.suptitle("")

    # 9. High Engagement by Reading Ease Categories
    ax9 = plt.subplot(5, 3, 9)
    df_train["readability_category"] = pd.cut(
        df_train["title_reading_ease"],
        bins=[0, 30, 50, 70, 90, 100],
        labels=["Very Hard", "Hard", "Medium", "Easy", "Very Easy"],
    )
    readability_engagement = df_train.groupby("readability_category")[
        "high_engagement"
    ].mean()
    plt.bar(range(len(readability_engagement)), readability_engagement.values)
    plt.xticks(
        range(len(readability_engagement)), readability_engagement.index, rotation=45
    )
    plt.ylabel("High Engagement Rate")
    plt.title("Engagement by Readability Category", fontsize=12, fontweight="bold")

    # 10. Correlation Heatmap
    ax10 = plt.subplot(5, 3, 10)
    numeric_cols = [
        "ctr",
        "high_engagement",
        "title_length",
        "title_word_count",
        "title_reading_ease",
        "has_question",
        "has_number",
        "has_exclamation",
        "total_impressions",
    ]
    corr_matrix = df_train[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, ax=ax10, fmt=".2f")
    plt.title("Feature Correlation Matrix", fontsize=12, fontweight="bold")

    # 11. Impressions vs CTR
    ax11 = plt.subplot(5, 3, 11)
    plt.scatter(
        np.log10(df_train["total_impressions"]), df_train["ctr"], alpha=0.5, s=10
    )
    plt.xlabel("Log10(Total Impressions)")
    plt.ylabel("CTR")
    plt.title("Impressions vs CTR", fontsize=12, fontweight="bold")

    # 12. Editorial Score Distribution
    ax12 = plt.subplot(5, 3, 12)
    df_train["editorial_total_score"] = (
        df_train["editorial_readability_score"] + df_train["editorial_headline_score"]
    )
    high_scores = df_train[df_train["high_engagement"] == 1]["editorial_total_score"]
    low_scores = df_train[df_train["high_engagement"] == 0]["editorial_total_score"]

    plt.hist(low_scores, bins=30, alpha=0.7, label="Low Engagement", color="lightcoral")
    plt.hist(
        high_scores, bins=30, alpha=0.7, label="High Engagement", color="lightgreen"
    )
    plt.xlabel("Editorial Score")
    plt.ylabel("Frequency")
    plt.title(
        "Editorial Score Distribution by Engagement", fontsize=12, fontweight="bold"
    )
    plt.legend()

    # 13. Bivariate: Title Length vs Word Count (colored by engagement)
    ax13 = plt.subplot(5, 3, 13)
    scatter = plt.scatter(
        df_train["title_length"],
        df_train["title_word_count"],
        c=df_train["high_engagement"],
        cmap="viridis",
        alpha=0.6,
        s=10,
    )
    plt.colorbar(scatter, label="High Engagement")
    plt.xlabel("Title Length (characters)")
    plt.ylabel("Title Word Count")
    plt.title("Title Length vs Word Count", fontsize=12, fontweight="bold")

    # 14. CTR vs Reading Ease (colored by engagement)
    ax14 = plt.subplot(5, 3, 14)
    scatter2 = plt.scatter(
        df_train["title_reading_ease"],
        df_train["ctr"],
        c=df_train["high_engagement"],
        cmap="viridis",
        alpha=0.6,
        s=10,
    )
    plt.colorbar(scatter2, label="High Engagement")
    plt.xlabel("Flesch Reading Ease Score")
    plt.ylabel("CTR")
    plt.title("Readability vs CTR (by Engagement)", fontsize=12, fontweight="bold")

    # 15. Feature Importance Preview
    ax15 = plt.subplot(5, 3, 15)

    features_available_at_step6_for_corr = [
        "title_length",
        "abstract_length",
        "title_word_count",
        "abstract_word_count",
        "title_reading_ease",
        "abstract_reading_ease",
        "avg_word_length",
        "has_question",
        "has_exclamation",
        "has_number",
        "has_colon",
        "has_quotes",
        "has_dash",
        "title_upper_ratio",
        "title_caps_words",
        "has_abstract",
        "title_abstract_ratio",
        "editorial_readability_score",
        "editorial_headline_score",
        "ctr_gain_potential",
        "needs_readability_improvement",
        "suboptimal_word_count",
        "too_long_title",
        "below_median_ctr",
        "ctr",
        "total_impressions",
        "total_clicks",
    ]

    features_to_examine = [
        col
        for col in features_available_at_step6_for_corr
        if col in df_train.columns and col != "high_engagement"
    ]

    cols_for_corr_matrix = features_to_examine + ["high_engagement"]

    df_subset_for_plot15 = pd.DataFrame()
    valid_cols_for_this_plot = []

    for col in cols_for_corr_matrix:
        if col in df_train:
            numeric_series = pd.to_numeric(df_train[col], errors="coerce")
            if numeric_series.isnull().sum() > df_train[col].isnull().sum():
                print(
                    f"Plot 15 Warning: Column '{col}' contained non-numeric values that were coerced to NaN."
                )
            df_subset_for_plot15[col] = numeric_series
            valid_cols_for_this_plot.append(col)
        else:
            print(f"Plot 15 Warning: Expected column '{col}' not found in df_train.")

    if (
        "high_engagement" not in df_subset_for_plot15.columns
        or df_subset_for_plot15["high_engagement"].isnull().all()
    ):
        print(
            "Plot 15 Error: 'high_engagement' target column is missing or all NaN after coercion. Cannot generate correlation plot."
        )
        ax15.set_title(
            "Feature Correlation (Target Error)", fontsize=12, fontweight="bold"
        )
    elif (
        len(valid_cols_for_this_plot) < 2
    ):  # Need at least target and one other feature
        print(
            "Plot 15 Warning: Not enough valid numeric columns for correlation. Skipping plot content."
        )
        ax15.set_title("Feature Correlation (No Data)", fontsize=12, fontweight="bold")
    else:
        correlation_matrix = df_subset_for_plot15[valid_cols_for_this_plot].corr()

        if "high_engagement" in correlation_matrix:
            corr_with_target = (
                correlation_matrix["high_engagement"]
                .drop("high_engagement", errors="ignore")
                .dropna()
            )

            if not corr_with_target.empty:
                feature_corr_plot15 = (
                    corr_with_target.abs().sort_values(ascending=False).head(10)
                )

                if not feature_corr_plot15.empty:
                    plt.barh(
                        range(len(feature_corr_plot15)), feature_corr_plot15.values
                    )
                    plt.yticks(
                        range(len(feature_corr_plot15)), feature_corr_plot15.index
                    )
                    plt.xlabel("Absolute Correlation with High Engagement")
                    plt.title(
                        "Top 10 Features by Correlation", fontsize=12, fontweight="bold"
                    )
                else:
                    print(
                        "Plot 15 Warning: No features found to correlate with target after processing."
                    )
                    ax15.set_title(
                        "Feature Correlation (No Features)",
                        fontsize=12,
                        fontweight="bold",
                    )
            else:
                print(
                    "Plot 15 Warning: Correlation with target resulted in an empty series. Skipping plot content."
                )
                ax15.set_title(
                    "Feature Correlation (No Valid Correlations)",
                    fontsize=12,
                    fontweight="bold",
                )
        else:
            print(
                "Plot 15 Warning: 'high_engagement' column not in correlation matrix. Skipping plot content."
            )
            ax15.set_title(
                "Feature Correlation (Target Missing in Matrix)",
                fontsize=12,
                fontweight="bold",
            )

    print("  Creating pairplot for key features...")
    key_features = [
        "ctr",
        "title_length",
        "title_word_count",
        "title_reading_ease",
        "high_engagement",
    ]
    sample_size = min(5000, len(df_train))
    sample_df = df_train[key_features].sample(n=sample_size, random_state=42)

    plt.figure(figsize=(12, 10))
    pairplot = sns.pairplot(
        sample_df, hue="high_engagement", diag_kind="hist", plot_kws={"alpha": 0.6}
    )
    pairplot.fig.suptitle(
        "Pairplot of Key Features", y=1.02, fontsize=16, fontweight="bold"
    )
    plt.savefig(
        PREP_DIR / "plots" / "feature_pairplot.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


print("Creating comprehensive EDA plots...")
create_comprehensive_eda_plots(df_train, df_val)

# ============================================================================
# STEP 7: CATEGORY ENCODING
# ============================================================================
print("\nStep 7: Encoding categories...")

df_train["category"] = (
    df_train["category"].astype(str).replace("nan", "unknown").fillna("unknown")
)
df_val["category"] = (
    df_val["category"].astype(str).replace("nan", "unknown").fillna("unknown")
)
df_test["category"] = (
    df_test["category"].astype(str).replace("nan", "unknown").fillna("unknown")
)

le = LabelEncoder()
df_train["category_enc"] = le.fit_transform(df_train["category"])


def safe_transform_categories(categories, encoder):
    """Safely transform categories, handling unknown values"""
    result = []
    for cat in categories:
        if cat in encoder.classes_:
            result.append(encoder.transform([cat])[0])
        else:
            result.append(
                encoder.transform(["unknown"])[0]
                if "unknown" in encoder.classes_
                else 0
            )
    return result


df_val["category_enc"] = safe_transform_categories(df_val["category"], le)
df_test["category_enc"] = safe_transform_categories(df_test["category"], le)

print(f"Encoded {len(le.classes_)} categories: {list(le.classes_)}")

# ============================================================================
# STEP 8: CREATE TITLE EMBEDDINGS WITH PCA OPTIMIZATION
# ============================================================================
print("\nStep 8: Creating title embeddings with PCA optimization...")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
cache_dir = PREP_DIR / "cache"


def create_embeddings_with_pca(df, dataset_name):
    """Create embeddings and optionally apply PCA for dimensionality reduction"""

    cache_file = cache_dir / f"embeddings_{dataset_name}.pkl"
    pca_cache_file = cache_dir / f"embeddings_pca_{dataset_name}.pkl"

    # Check for PCA cache first
    if CONFIG["use_pca"] and pca_cache_file.exists():
        print(f"Loading cached PCA embeddings for {dataset_name}...")
        emb_df = pd.read_pickle(pca_cache_file)
        return emb_df

    # Check for regular embeddings cache
    if cache_file.exists():
        print(f"Loading cached embeddings for {dataset_name}...")
        emb_df = pd.read_pickle(cache_file)
    else:
        print(f"Creating {dataset_name} embeddings...")
        unique_articles = (
            df[["newsID", "title"]].drop_duplicates("newsID").set_index("newsID")
        )
        titles = unique_articles["title"].fillna("").tolist()

        embeddings = embedder.encode(titles, show_progress_bar=True, batch_size=32)

        emb_df = pd.DataFrame(
            embeddings,
            index=unique_articles.index,
            columns=[f"title_emb_{i}" for i in range(CONFIG["embedding_dim"])],
        )
        emb_df.reset_index(inplace=True)
        emb_df.to_pickle(cache_file)

    return emb_df


# Create embeddings for all datasets
print("Creating embeddings...")
train_embeddings = create_embeddings_with_pca(df_train, "train")
val_embeddings = create_embeddings_with_pca(df_val, "val")
test_embeddings = create_embeddings_with_pca(df_test, "test")

# Apply PCA if requested
if CONFIG["use_pca"]:
    print(
        f"Applying PCA to reduce embedding dimensions from {CONFIG['embedding_dim']} to {CONFIG['pca_components']}..."
    )

    pca = PCA(
        n_components=CONFIG["pca_components"], random_state=CONFIG["random_state"]
    )

    # Fit PCA on training embeddings
    emb_cols = [f"title_emb_{i}" for i in range(CONFIG["embedding_dim"])]
    train_emb_matrix = train_embeddings[emb_cols].fillna(0).values
    train_emb_pca = pca.fit_transform(train_emb_matrix)

    # Transform validation and test
    val_emb_matrix = val_embeddings[emb_cols].fillna(0).values
    val_emb_pca = pca.transform(val_emb_matrix)

    test_emb_matrix = test_embeddings[emb_cols].fillna(0).values
    test_emb_pca = pca.transform(test_emb_matrix)

    # Create PCA feature dataframes
    pca_cols = [f"title_pca_{i}" for i in range(CONFIG["pca_components"])]

    train_pca_df = pd.DataFrame(
        train_emb_pca, columns=pca_cols, index=train_embeddings.index
    )
    train_pca_df["newsID"] = train_embeddings["newsID"].values

    val_pca_df = pd.DataFrame(val_emb_pca, columns=pca_cols, index=val_embeddings.index)
    val_pca_df["newsID"] = val_embeddings["newsID"].values

    test_pca_df = pd.DataFrame(
        test_emb_pca, columns=pca_cols, index=test_embeddings.index
    )
    test_pca_df["newsID"] = test_embeddings["newsID"].values

    # Add PCA features to main dataframes
    df_train = df_train.merge(train_pca_df, on="newsID", how="left")
    df_val = df_val.merge(val_pca_df, on="newsID", how="left")
    df_test = df_test.merge(test_pca_df, on="newsID", how="left")

    # Save PCA transformer
    with open(PREP_DIR / "processed_data" / "pca_transformer.pkl", "wb") as f:
        pickle.dump(pca, f)

    print(f"PCA applied successfully:")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    print(
        f"  Dimensionality reduced by: {((CONFIG['embedding_dim'] - CONFIG['pca_components']) / CONFIG['embedding_dim'] * 100):.1f}%"
    )

    # Use PCA features instead of full embeddings
    embedding_features = pca_cols
else:
    # Merge full embeddings
    df_train = df_train.merge(train_embeddings, on="newsID", how="left")
    df_val = df_val.merge(val_embeddings, on="newsID", how="left")
    df_test = df_test.merge(test_embeddings, on="newsID", how="left")

    embedding_features = [f"title_emb_{i}" for i in range(CONFIG["embedding_dim"])]

print("Embeddings processing completed")

# ============================================================================
# STEP 9: PREPARE FINAL FEATURE SETS
# ============================================================================
print("\nStep 9: Preparing final feature sets...")

# Define comprehensive feature sets - EXCLUDE DATA LEAKAGE FEATURES
editorial_features = [
    "title_length",
    "abstract_length",
    "title_word_count",
    "abstract_word_count",
    "title_reading_ease",
    "abstract_reading_ease",
    "avg_word_length",
    "has_question",
    "has_exclamation",
    "has_number",
    "has_colon",
    "has_quotes",
    "has_dash",
    "title_upper_ratio",
    "title_caps_words",
    "has_abstract",
    "title_abstract_ratio",
    "editorial_readability_score",
    "editorial_headline_score",
    # REMOVED DATA LEAKAGE FEATURES:
    # "ctr_gain_potential",  # ❌ Depends on historical CTR performance
    # "below_median_ctr",    # ❌ Depends on historical CTR performance
    "needs_readability_improvement",
    "suboptimal_word_count",
    "too_long_title",
    "category_enc",
]

# Combine all features - ONLY include features available at publication time
all_features = editorial_features + embedding_features

# Check availability
available_features = [f for f in all_features if f in df_train.columns]
missing_features = [f for f in all_features if f not in df_train.columns]

print(f"Feature summary (EXCLUDING data leakage features):")
print(f"  Editorial features: {len(editorial_features)}")
print(f"  Embedding features: {len(embedding_features)}")
print(f"  Total available: {len(available_features)}")
print(f"  EXCLUDED for data leakage prevention:")
print(f"    - ctr_gain_potential (depends on historical CTR)")
print(f"    - below_median_ctr (depends on historical CTR)")
print(f"    - ctr (raw performance metric)")
if missing_features:
    print(f"  Missing: {missing_features}")

# ============================================================================
# STEP 10: CREATE MODEL OUTPUT VISUALIZATION PLOTS
# ============================================================================
print("\nStep 10: Creating model output visualization plots...")


def create_model_output_plots(df_train, df_val, features):
    """Create plots to visualize model readiness and feature distributions"""

    plt.style.use("default")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # 1. Feature Distribution Comparison
    ax1 = axes[0, 0]
    sample_features = ["title_length", "title_word_count", "title_reading_ease"]
    for i, feature in enumerate(sample_features):
        ax1.hist(df_train[feature], bins=30, alpha=0.5, label=f"{feature} (train)")
        ax1.hist(df_val[feature], bins=30, alpha=0.5, label=f"{feature} (val)")
    ax1.set_title("Feature Distributions: Train vs Val")
    ax1.legend()

    # 2. Target Balance Across Datasets
    ax2 = axes[0, 1]
    train_balance = df_train["high_engagement"].value_counts(normalize=True)
    val_balance = df_val["high_engagement"].value_counts(normalize=True)

    x = np.arange(2)
    width = 0.35
    ax2.bar(x - width / 2, train_balance.values, width, label="Train", alpha=0.8)
    ax2.bar(x + width / 2, val_balance.values, width, label="Val", alpha=0.8)
    ax2.set_xlabel("Engagement Level")
    ax2.set_ylabel("Proportion")
    ax2.set_title("Target Balance: Train vs Val")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Low", "High"])
    ax2.legend()

    # 3. CTR Distribution by Dataset
    ax3 = axes[0, 2]
    ax3.hist(df_train["ctr"], bins=50, alpha=0.7, label="Train", color="blue")
    ax3.hist(df_val["ctr"], bins=50, alpha=0.7, label="Val", color="orange")
    ax3.axvline(
        CONFIG["high_engagement_threshold"],
        color="red",
        linestyle="--",
        label=f'Threshold: {CONFIG["high_engagement_threshold"]}',
    )
    ax3.set_xlabel("CTR")
    ax3.set_ylabel("Frequency")
    ax3.set_title("CTR Distribution by Dataset")
    ax3.legend()

    # 4. Feature Correlation with Target
    ax4 = axes[1, 0]
    target_corr = (
        df_train[editorial_features + ["high_engagement"]]
        .corr()["high_engagement"]
        .abs()
        .sort_values(ascending=False)[1:]
    )
    top_corr = target_corr.head(10)
    ax4.barh(range(len(top_corr)), top_corr.values)
    ax4.set_yticks(range(len(top_corr)))
    ax4.set_yticklabels(top_corr.index)
    ax4.set_xlabel("Absolute Correlation")
    ax4.set_title("Top 10 Features: Correlation with Target")

    # 5. Editorial Score Distributions
    ax5 = axes[1, 1]
    high_eng = df_train[df_train["high_engagement"] == 1]["editorial_readability_score"]
    low_eng = df_train[df_train["high_engagement"] == 0]["editorial_readability_score"]
    ax5.hist(low_eng, bins=30, alpha=0.7, label="Low Engagement", color="lightcoral")
    ax5.hist(high_eng, bins=30, alpha=0.7, label="High Engagement", color="lightgreen")
    ax5.set_xlabel("Editorial Readability Score")
    ax5.set_ylabel("Frequency")
    ax5.set_title("Editorial Readability by Engagement")
    ax5.legend()

    # 6. Missing Values Analysis
    ax6 = axes[1, 2]
    missing_counts = (
        df_train[available_features]
        .isnull()
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    if len(missing_counts[missing_counts > 0]) > 0:
        ax6.bar(range(len(missing_counts)), missing_counts.values)
        ax6.set_xticks(range(len(missing_counts)))
        ax6.set_xticklabels(missing_counts.index, rotation=45, ha="right")
        ax6.set_ylabel("Missing Count")
        ax6.set_title("Missing Values by Feature")
    else:
        ax6.text(
            0.5,
            0.5,
            "No Missing Values!",
            ha="center",
            va="center",
            transform=ax6.transAxes,
            fontsize=14,
        )
        ax6.set_title("Missing Values Analysis")

    # 7. Category Performance
    ax7 = axes[2, 0]
    cat_performance = (
        df_train.groupby("category")
        .agg({"high_engagement": "mean", "ctr": "mean", "newsID": "count"})
        .reset_index()
    )
    cat_performance = cat_performance[cat_performance["newsID"] >= 50]

    ax7.scatter(
        cat_performance["ctr"],
        cat_performance["high_engagement"],
        s=cat_performance["newsID"] * 2,
        alpha=0.6,
    )
    ax7.set_xlabel("Average CTR")
    ax7.set_ylabel("High Engagement Rate")
    ax7.set_title("Category Performance (size = article count)")

    # Add category labels for top performers
    for idx, row in cat_performance.nlargest(3, "high_engagement").iterrows():
        ax7.annotate(
            row["category"],
            (row["ctr"], row["high_engagement"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # 8. Feature Engineering Success
    ax8 = axes[2, 1]
    feature_types = {
        "Editorial Features": len(
            [f for f in editorial_features if f in available_features]
        ),
        "Embedding Features": len(
            [f for f in embedding_features if f in available_features]
        ),
        "Missing Features": len(missing_features),
    }

    colors = ["lightgreen", "lightblue", "lightcoral"]
    ax8.pie(
        feature_types.values(),
        labels=feature_types.keys(),
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    ax8.set_title("Feature Engineering Summary")

    # 9. Model Readiness Checklist
    ax9 = axes[2, 2]
    checklist = {
        "Features Created": len(available_features) >= 50,
        "Target Balanced": 0.1 <= df_train["high_engagement"].mean() <= 0.3,
        "No Missing Values": df_train[available_features].isnull().sum().sum() == 0,
        "Embeddings Ready": len(embedding_features) > 0,
        "Editorial Criteria": all(
            f in available_features
            for f in ["editorial_readability_score", "editorial_headline_score"]
        ),
        "Categories Encoded": "category_enc" in available_features,
    }

    passed = sum(checklist.values())
    total = len(checklist)

    y_pos = np.arange(len(checklist))
    colors = ["green" if v else "red" for v in checklist.values()]
    ax9.barh(y_pos, [1] * len(checklist), color=colors, alpha=0.7)
    ax9.set_yticks(y_pos)
    ax9.set_yticklabels(checklist.keys())
    ax9.set_xlabel("Status")
    ax9.set_title(f"Model Readiness: {passed}/{total} Checks Passed")
    ax9.set_xlim(0, 1)

    # Add status text
    for i, (check, passed) in enumerate(checklist.items()):
        ax9.text(
            0.5,
            i,
            "✓" if passed else "✗",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=12,
            color="white",
        )

    plt.tight_layout()
    plt.savefig(
        PREP_DIR / "plots" / "model_output_readiness.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


create_model_output_plots(df_train, df_val, available_features)

# ============================================================================
# STEP 11: SAVE PROCESSED DATA
# ============================================================================
print("\nStep 11: Saving processed data...")


def sanitize_features_for_modeling(df, features_list, df_name):
    """
    Ensures that all specified features in the DataFrame are numeric.
    Converts object/string types to numeric, coercing errors to NaN.
    Converts boolean types to int.
    """
    print(f"Sanitizing features in DataFrame: {df_name} for modeling...")
    df_copy = df.copy()
    for col in features_list:
        if col in df_copy.columns:
            if df_copy[col].dtype == "object" or pd.api.types.is_string_dtype(
                df_copy[col]
            ):
                print(
                    f"  Feature '{col}' in {df_name} is {df_copy[col].dtype}. Attempting conversion to numeric."
                )
                original_nan_count = df_copy[col].isnull().sum()
                df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
                coerced_nan_count = df_copy[col].isnull().sum()
                if coerced_nan_count > original_nan_count:
                    print(
                        f"    Warning: Coercion of '{col}' in {df_name} introduced {coerced_nan_count - original_nan_count} new NaN(s). Non-numeric strings were present."
                    )
            elif pd.api.types.is_bool_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype(int)
        else:
            print(
                f"  Warning: Expected feature '{col}' not found in {df_name} during sanitization for modeling."
            )
    return df_copy


df_train = sanitize_features_for_modeling(df_train, available_features, "df_train")
df_val = sanitize_features_for_modeling(df_val, available_features, "df_val")
df_test = sanitize_features_for_modeling(df_test, available_features, "df_test")


X_train = df_train[available_features].fillna(0)
y_train = pd.DataFrame(
    {
        "ctr": df_train["ctr"],
        "high_engagement": df_train["high_engagement"],
        "needs_rewrite": df_train["needs_rewrite"],
    }
)

X_val = df_val[available_features].fillna(0)
y_val = pd.DataFrame(
    {
        "ctr": df_val["ctr"],
        "high_engagement": df_val["high_engagement"],
        "needs_rewrite": df_val["needs_rewrite"],
    }
)

X_test = df_test[available_features].fillna(0)

X_train.to_parquet(PREP_DIR / "processed_data" / "X_train_optimized.parquet")
y_train.to_parquet(PREP_DIR / "processed_data" / "y_train_optimized.parquet")

X_val.to_parquet(PREP_DIR / "processed_data" / "X_val_optimized.parquet")
y_val.to_parquet(PREP_DIR / "processed_data" / "y_val_optimized.parquet")

X_test.to_parquet(PREP_DIR / "processed_data" / "X_test_optimized.parquet")

article_metadata_columns = [
    "newsID",
    "title",
    "category",
    "abstract",
    "ctr",
    "high_engagement",
    "needs_rewrite",
    "title_reading_ease",
    "editorial_readability_score",
    "editorial_headline_score",
]

train_meta_columns = [
    col for col in article_metadata_columns if col in df_train.columns
]
val_meta_columns = [col for col in article_metadata_columns if col in df_val.columns]
test_meta_columns = [col for col in article_metadata_columns if col in df_test.columns]

df_train[train_meta_columns].to_parquet(
    PREP_DIR / "processed_data" / "article_metadata_train_optimized.parquet"
)
df_val[val_meta_columns].to_parquet(
    PREP_DIR / "processed_data" / "article_metadata_val_optimized.parquet"
)
df_test[test_meta_columns].to_parquet(
    PREP_DIR / "processed_data" / "article_metadata_test_optimized.parquet"
)

# Save preprocessing components
with open(PREP_DIR / "processed_data" / "category_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Save comprehensive metadata INCLUDING TRAINING MEDIAN CTR (but not as model feature)
processing_metadata = {
    "processing_type": "chunked_optimized_with_eda",
    "chunk_size": CONFIG["chunk_size"],
    "features_created": len(available_features),
    "editorial_criteria": EDITORIAL_CRITERIA,
    "training_median_ctr": float(
        training_median_ctr
    ),  # Save for reference only - NOT used in model
    "pca_applied": CONFIG["use_pca"],
    "pca_components": CONFIG["pca_components"] if CONFIG["use_pca"] else None,
    "pca_variance_explained": (
        float(pca.explained_variance_ratio_.sum()) if CONFIG["use_pca"] else None
    ),
    "feature_categories": {
        "editorial_features": len(editorial_features),
        "embedding_features": len(embedding_features),
        "total_features": len(available_features),
    },
    "dataset_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
    "target_statistics": {
        "high_engagement_rate": float(df_train["high_engagement"].mean()),
        "needs_rewrite_rate": float(df_train["needs_rewrite"].mean()),
        "mean_ctr": float(df_train["ctr"].mean()),
        "median_ctr": float(training_median_ctr),  # For reference only
        "ctr_threshold": CONFIG["high_engagement_threshold"],
    },
    "data_leakage_prevention": {
        "excluded_features": ["ctr_gain_potential", "below_median_ctr", "ctr"],
        "exclusion_reason": "These features depend on historical performance data not available at publication time",
    },
    "eda_analysis_completed": True,
    "available_features": available_features,  # Save feature order (WITHOUT leaky features)
    "plots_created": [
        "comprehensive_eda_analysis.png",
        "feature_pairplot.png",
        "model_output_readiness.png",
    ],
    "files_created": [
        "X_train_optimized.parquet",
        "y_train_optimized.parquet",
        "X_val_optimized.parquet",
        "y_val_optimized.parquet",
        "X_test_optimized.parquet",
        "article_metadata_train_optimized.parquet",
        "article_metadata_val_optimized.parquet",
        "article_metadata_test_optimized.parquet",
        "category_encoder.pkl",
        "preprocessing_metadata.json",
    ],
}

if CONFIG["use_pca"]:
    processing_metadata["files_created"].append("pca_transformer.pkl")

with open(PREP_DIR / "processed_data" / "preprocessing_metadata.json", "w") as f:
    json.dump(processing_metadata, f, indent=2)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("EDA & PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 80)

print(f"\nKEY OPTIMIZATIONS APPLIED:")
print(f"  Chunked processing: {CONFIG['chunk_size']:,} records per chunk")
print(f"  Memory optimization: Efficient data types")
print(f"  NaN handling: Fixed boolean conversion errors")
print(
    f"  PCA dimensionality reduction: {CONFIG['embedding_dim']} → {CONFIG['pca_components'] if CONFIG['use_pca'] else CONFIG['embedding_dim']}"
)
print(f"  Editorial criteria integration: Flesch Reading Ease + CTR gain")
print(
    f"  Training median CTR: {training_median_ctr:.6f} (reference only - NOT used as model feature)"
)
print(
    f"  DATA LEAKAGE PREVENTION: Excluded ctr_gain_potential, below_median_ctr, raw ctr"
)

print(f"\nDATASET SUMMARY:")
print(f"  Train: {len(X_train):,} articles with {len(available_features)} features")
print(f"  Val: {len(X_val):,} articles")
print(f"  Test: {len(X_test):,} articles")
print(f"  NO DATA LEAKAGE: Model uses only features available at publication time")

print(f"\nEDITORIAL METRICS:")
print(
    f"  High engagement rate: {processing_metadata['target_statistics']['high_engagement_rate']:.1%}"
)
print(
    f"  Articles needing rewrite: {processing_metadata['target_statistics']['needs_rewrite_rate']:.1%}"
)
print(f"  Average CTR: {processing_metadata['target_statistics']['mean_ctr']:.6f}")
print(f"  CTR Threshold: {processing_metadata['target_statistics']['ctr_threshold']}")

print(f"\nEDA ANALYSIS COMPLETED:")
print(f"  Comprehensive bivariate analysis")
print(f"  Target distribution analysis")
print(f"  Feature correlation matrix")
print(f"  Pairplot for key features")
print(f"  Model readiness assessment")

print(f"\nFILES CREATED:")
for file in processing_metadata["files_created"]:
    print(f"  {file}")

print(f"\nPLOTS CREATED:")
for plot in processing_metadata["plots_created"]:
    print(f"  {plot}")

print(f"\nREADY FOR NEXT STEP:")
print(f"  Run XGBoost model training with Optuna optimization")
print(f"  Features ready for classification and editorial analysis")
print(f"  Article metadata prepared for LLM headline rewriting")
print(f"  VALID MODEL: No data leakage - only publication-time features used")

print("=" * 80)
