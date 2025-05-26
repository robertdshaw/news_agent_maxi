# UPWORTHY EDITORIAL ASSISTANT - DIRECT COMMANDS PIPELINE
# Complete EDA to modeling pipeline as executable commands

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

plt.style.use("default")
sns.set_palette("husl")

# =================================================================
# 1. DATA ACQUISITION
# =================================================================

print("UPWORTHY RESEARCH ARCHIVE DATASET")
print("=" * 50)
print("Loading 32,000+ headline A/B tests...")

df = pd.read_csv(r"upworthy_data\upworthy-archive-exploratory-packages-03.12.2020.csv")
print(df.head())
print(f"Dataset size: {len(df)} headlines")
print(f"Dataset shape: {df.shape}")

# =================================================================
# 2. BASIC DATA EXPLORATION
# =================================================================

print("\n" + "=" * 60)
print("BASIC DATA EXPLORATION")
print("=" * 60)

# Dataset overview
print(f"Dataset shape: {df.shape}")
memory_usage = df.memory_usage(deep=True).sum() / 1024**2
print(f"Memory usage: {memory_usage:.2f} MB")

# Display first few rows
print("\nFirst few rows:")
print(df.head())

# Dataset info
print("\nDataset info:")
print(df.info())

# Column types analysis
print("\nColumn types breakdown:")
dtype_counts = df.dtypes.value_counts()
for dtype in dtype_counts.index:
    cols = df.select_dtypes(include=[dtype]).columns.tolist()
    print(f"{dtype}: {len(cols)} columns")
    display_cols = cols[:5] if len(cols) > 10 else cols
    remaining = len(cols) - 5 if len(cols) > 10 else 0
    print(
        f"  {display_cols}{'... (and ' + str(remaining) + ' more)' if remaining > 0 else ''}"
    )

# Basic statistics
print("\nBasic statistics:")
print(df.describe(include="all"))

# =================================================================
# 3. MISSING VALUES ANALYSIS
# =================================================================

print("\n" + "=" * 60)
print("MISSING VALUES ANALYSIS")
print("=" * 60)

# Calculate missing values
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100

# Create missing values summary
missing_df = pd.DataFrame(
    {
        "Column": missing_data.index,
        "Missing_Count": missing_data.values,
        "Missing_Percentage": missing_percent.values,
    }
).sort_values("Missing_Count", ascending=False)

print("Missing values summary:")
columns_with_missing = missing_df[missing_df["Missing_Count"] > 0]
print(columns_with_missing)

# Total missing values across dataset
total_missing = np.sum(missing_data.values)
print(f"Total missing values across dataset: {total_missing}")

# Visualize missing values
missing_cols = missing_df[missing_df["Missing_Count"] > 0]
missing_count = len(missing_cols)
print(f"Columns with missing values: {missing_count}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.barh(missing_cols["Column"], missing_cols["Missing_Count"])
plt.title("Missing Values Count")
plt.xlabel("Count")

plt.subplot(1, 2, 2)
plt.barh(missing_cols["Column"], missing_cols["Missing_Percentage"])
plt.title("Missing Values Percentage")
plt.xlabel("Percentage")

plt.tight_layout()
plt.show()

# =================================================================
# 4. DUPLICATES ANALYSIS
# =================================================================

print("\n" + "=" * 60)
print("DUPLICATES ANALYSIS")
print("=" * 60)

# Overall duplicates
total_duplicates = df.duplicated().sum()
print(f"Total duplicate rows: {total_duplicates}")
duplicate_percentage = (total_duplicates / len(df)) * 100
print(f"Percentage of duplicates: {duplicate_percentage:.2f}%")

# Sample duplicate rows
duplicate_rows = df[df.duplicated(keep=False)].head(10)
print(f"Sample duplicate rows count: {len(duplicate_rows)}")
print(duplicate_rows)

# Check duplicates in key columns
key_columns = ["headline", "clickability_test_id"]
for col in key_columns:
    col_duplicates = df[col].duplicated().sum()
    unique_values = df[col].nunique()
    total_values = len(df[col])
    print(f"Duplicates in '{col}': {col_duplicates}")
    print(f"Unique values in '{col}': {unique_values}")
    print(f"Total values in '{col}': {total_values}")

# =================================================================
# 5. CREATE TARGET VARIABLE
# =================================================================

print("\n" + "=" * 60)
print("TARGET VARIABLE CREATION")
print("=" * 60)

# Create CTR (Click-Through Rate)
df["ctr"] = df["clicks"] / df["impressions"]

# Handle infinite values
infinite_count = np.sum(np.isinf(df["ctr"]))
print(f"Infinite values before replacement: {infinite_count}")
df["ctr"] = df["ctr"].replace([np.inf, -np.inf], np.nan)

# Remove invalid CTR values
initial_size = len(df)
df = df[(df["ctr"] >= 0) & (df["ctr"] <= 1)]
removed_count = initial_size - len(df)

print(f"Created CTR target variable")
print(f"Removed {removed_count} rows with invalid CTR values")
print(f"Final dataset size: {len(df)}")

# Analyze CTR distribution
ctr_mean = df["ctr"].mean()
ctr_median = df["ctr"].median()
ctr_std = df["ctr"].std()
ctr_min = df["ctr"].min()
ctr_max = df["ctr"].max()

print(f"Mean CTR: {ctr_mean:.4f}")
print(f"Median CTR: {ctr_median:.4f}")
print(f"Std CTR: {ctr_std:.4f}")
print(f"Min CTR: {ctr_min:.4f}")
print(f"Max CTR: {ctr_max:.4f}")

# CTR percentiles
percentiles = [10, 25, 50, 75, 90, 95, 99]
print("CTR Percentiles:")
for p in percentiles:
    percentile_value = np.percentile(df["ctr"], p)
    print(f"  {p}th percentile: {percentile_value:.4f}")

# =================================================================
# 6. UNIVARIATE ANALYSIS
# =================================================================

print("\n" + "=" * 60)
print("UNIVARIATE ANALYSIS")
print("=" * 60)

# Identify column types
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

numeric_count = len(numeric_cols)
categorical_count = len(categorical_cols)
print(f"Numeric columns ({numeric_count}): {numeric_cols}")
print(f"Categorical columns ({categorical_count}): {categorical_cols}")

# Analyze numeric columns
print("NUMERIC VARIABLES ANALYSIS")
print("-" * 40)

analysis_cols = numeric_cols[:5]  # Limit to first 5 for readability
for col in analysis_cols:
    col_mean = df[col].mean()
    col_median = df[col].median()
    col_std = df[col].std()
    col_min = df[col].min()
    col_max = df[col].max()
    col_skew = df[col].skew()
    col_kurtosis = df[col].kurtosis()

    print(f"\n{col.upper()}:")
    print(f"  Mean: {col_mean:.4f}")
    print(f"  Median: {col_median:.4f}")
    print(f"  Std: {col_std:.4f}")
    print(f"  Min: {col_min:.4f}")
    print(f"  Max: {col_max:.4f}")
    print(f"  Skewness: {col_skew:.4f}")
    print(f"  Kurtosis: {col_kurtosis:.4f}")

# Analyze categorical columns
print("CATEGORICAL VARIABLES ANALYSIS")
print("-" * 40)

analysis_cats = categorical_cols[:5]  # Limit to first 5
for col in analysis_cats:
    unique_count = df[col].nunique()
    print(f"\n{col.upper()}:")
    print(f"  Unique values: {unique_count}")
    print("  Most common values:")
    value_counts = df[col].value_counts().head()
    print(value_counts)

# =================================================================
# 7. HEADLINE FEATURE ENGINEERING
# =================================================================

print("\n" + "=" * 60)
print("HEADLINE FEATURE ENGINEERING")
print("=" * 60)

print("Creating headline features...")

# Basic text features
df["headline_length"] = df["headline"].str.len()
df["headline_word_count"] = df["headline"].str.split().str.len()

# Punctuation features
df["has_question_mark"] = df["headline"].str.contains("\\?").astype(int)
df["has_exclamation"] = df["headline"].str.contains("!").astype(int)
df["has_colon"] = df["headline"].str.contains(":").astype(int)
df["has_quotes"] = df["headline"].str.contains("[\"']").astype(int)

# Number and capitalization features
df["has_numbers"] = df["headline"].str.contains("\\d").astype(int)
df["capital_letters_count"] = df["headline"].str.count("[A-Z]")
df["capital_ratio"] = df["capital_letters_count"] / df["headline_length"]

# Common clickbait patterns
df["starts_with_this"] = df["headline"].str.lower().str.startswith("this").astype(int)
df["starts_with_you"] = df["headline"].str.lower().str.startswith("you").astype(int)
df["contains_will"] = df["headline"].str.lower().str.contains(" will ").astype(int)
df["contains_what"] = df["headline"].str.lower().str.contains("what").astype(int)
df["contains_how"] = df["headline"].str.lower().str.contains("how").astype(int)
df["contains_why"] = df["headline"].str.lower().str.contains("why").astype(int)

# Emotional words
positive_words = [
    "amazing",
    "incredible",
    "awesome",
    "fantastic",
    "wonderful",
    "brilliant",
]
negative_words = ["shocking", "terrible", "awful", "horrible", "devastating", "tragic"]

df["positive_words_count"] = (
    df["headline"].str.lower().str.count("|".join(positive_words))
)
df["negative_words_count"] = (
    df["headline"].str.lower().str.count("|".join(negative_words))
)

# Count created features
excluded_cols = [
    "headline",
    "clicks",
    "impressions",
    "ctr",
    "clickability_test_id",
    "dataset_source",
]
feature_cols = [col for col in df.columns if col not in excluded_cols]
features_created = len(feature_cols)
print(f"Total features created: {features_created}")

# =================================================================
# 8. CORRELATION ANALYSIS
# =================================================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

# Update numeric columns list after feature engineering
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Calculate correlation matrix
corr_matrix = df[numeric_cols].corr()

print("Correlation matrix:")
correlation_display = corr_matrix.round(3)
print(correlation_display)

# Find strong correlations with target variable (CTR)
ctr_correlations = corr_matrix["ctr"].drop("ctr").abs().sort_values(ascending=False)
top_correlations = ctr_correlations.head(10)
print("Strongest correlations with CTR:")
print(top_correlations)

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    cmap="coolwarm",
    center=0,
    square=True,
    fmt=".2f",
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# =================================================================
# 9. BIVARIATE ANALYSIS
# =================================================================

print("\n" + "=" * 60)
print("BIVARIATE ANALYSIS")
print("=" * 60)

# Select key variables for scatter plots
key_vars = [col for col in numeric_cols if col != "ctr"][:4]  # Limit to 4 variables

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, var in enumerate(key_vars):
    axes[i].scatter(df[var], df["ctr"], alpha=0.6)
    axes[i].set_xlabel(var)
    axes[i].set_ylabel("CTR")
    axes[i].set_title(f"CTR vs {var}")

    # Add trend line
    valid_data = df[var].notna()
    z = np.polyfit(df.loc[valid_data, var], df.loc[valid_data, "ctr"], 1)
    p = np.poly1d(z)
    axes[i].plot(df[var], p(df[var]), "r--", alpha=0.8)

plt.tight_layout()
plt.show()

# Analyze A/B test patterns
print("A/B TEST ANALYSIS")
print("-" * 20)

test_stats = (
    df.groupby("clickability_test_id")
    .agg({"ctr": ["count", "mean", "std"], "clicks": "sum", "impressions": "sum"})
    .round(4)
)

# Flatten column names
test_stats.columns = ["_".join(col).strip() for col in test_stats.columns]

# Find tests with multiple variants
multi_variant_tests = test_stats[test_stats["ctr_count"] > 1]
multi_variant_count = len(multi_variant_tests)
print(f"Tests with multiple variants: {multi_variant_count}")

print("Sample A/B test performance:")
sample_tests = multi_variant_tests.head()
print(sample_tests)

# Calculate CTR differences in A/B tests
test_ctr_ranges = multi_variant_tests.groupby(multi_variant_tests.index)[
    "ctr_mean"
].agg(["min", "max"])
test_ctr_ranges["ctr_range"] = test_ctr_ranges["max"] - test_ctr_ranges["min"]

mean_ctr_diff = test_ctr_ranges["ctr_range"].mean()
max_ctr_diff = test_ctr_ranges["ctr_range"].max()
print(f"Mean CTR difference: {mean_ctr_diff:.4f}")
print(f"Max CTR difference: {max_ctr_diff:.4f}")

# =================================================================
# 10. PREPROCESSING PIPELINE
# =================================================================

print("\n" + "=" * 60)
print("PREPROCESSING PIPELINE")
print("=" * 60)

# Remove rows with missing critical values
critical_cols = ["headline", "ctr"]
initial_size = len(df)
df = df.dropna(subset=critical_cols)
critical_removed = initial_size - len(df)
print(f"Removed {critical_removed} rows with missing critical values")

# Remove extreme outliers in CTR
q1 = df["ctr"].quantile(0.01)
q99 = df["ctr"].quantile(0.99)
initial_size = len(df)
df = df[(df["ctr"] >= q1) & (df["ctr"] <= q99)]
outliers_removed = initial_size - len(df)
print(f"Removed {outliers_removed} rows with extreme CTR outliers")

# Handle remaining missing values in features
numeric_cols = df.select_dtypes(include=[np.number]).columns
missing_before = np.sum(df[numeric_cols].isnull().sum())
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
missing_after = np.sum(df[numeric_cols].isnull().sum())

print(f"Missing values before imputation: {missing_before}")
print(f"Missing values after imputation: {missing_after}")
print(f"Final dataset size: {len(df)}")

# =================================================================
# 11. DISTRIBUTION ANALYSIS
# =================================================================

print("\n" + "=" * 60)
print("DISTRIBUTION ANALYSIS")
print("=" * 60)

# Create distribution plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# CTR histogram
axes[0, 0].hist(df["ctr"], bins=50, alpha=0.7, edgecolor="black")
axes[0, 0].set_title("CTR Distribution")
axes[0, 0].set_xlabel("Click-Through Rate")
axes[0, 0].set_ylabel("Frequency")

# CTR log distribution
log_ctr = np.log(df["ctr"])
axes[0, 1].hist(log_ctr, bins=50, alpha=0.7, edgecolor="black")
axes[0, 1].set_title("Log CTR Distribution")
axes[0, 1].set_xlabel("Log Click-Through Rate")
axes[0, 1].set_ylabel("Frequency")

# Headline length distribution
headline_lengths = df["headline_length"]
axes[1, 0].hist(headline_lengths, bins=50, alpha=0.7, edgecolor="black")
axes[1, 0].set_title("Headline Length Distribution")
axes[1, 0].set_xlabel("Character Count")
axes[1, 0].set_ylabel("Frequency")

# Word count distribution
word_counts = df["headline_word_count"]
axes[1, 1].hist(word_counts, bins=30, alpha=0.7, edgecolor="black")
axes[1, 1].set_title("Headline Word Count Distribution")
axes[1, 1].set_xlabel("Word Count")
axes[1, 1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# =================================================================
# 12. FEATURE IMPORTANCE ANALYSIS
# =================================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Prepare features for modeling
excluded_cols = [
    "headline",
    "clicks",
    "impressions",
    "ctr",
    "clickability_test_id",
    "dataset_source",
]
feature_cols = [col for col in df.columns if col not in excluded_cols]

X = df[feature_cols]
y = df["ctr"]

# Remove any remaining NaN values
valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[valid_mask]
y = y[valid_mask]

valid_samples = len(X)
total_features = len(feature_cols)
print(f"Valid samples for analysis: {valid_samples}")
print(f"Total features: {total_features}")

# Train Random Forest for feature importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importance
feature_importance = rf.feature_importances_
importance_df = pd.DataFrame(
    {"feature": feature_cols, "importance": feature_importance}
).sort_values("importance", ascending=False)

print("Top 15 most important features:")
top
