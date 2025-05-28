import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("COMPREHENSIVE BIVARIATE ANALYSIS - FEATURES vs CTR")
print("=" * 70)

# Load the preprocessed data
PREP_DIR = Path("data/preprocessed")

# Load processed data
X_train = pd.read_parquet(PREP_DIR / "processed_data" / "X_train_clean.parquet")
y_train = pd.read_parquet(PREP_DIR / "processed_data" / "y_train_clean.parquet")["ctr"]

# Combine for analysis
df_analysis = X_train.copy()
df_analysis["ctr"] = y_train

print(f"Dataset: {len(df_analysis)} articles with {len(X_train.columns)} features")
print(f"CTR Statistics: Mean={y_train.mean():.4f}, Std={y_train.std():.4f}")

# Create output directory for plots
bivariate_plots_dir = PREP_DIR / "plots" / "bivariate_analysis"
bivariate_plots_dir.mkdir(parents=True, exist_ok=True)

# Separate features by type
embedding_features = [col for col in X_train.columns if col.startswith("emb_")]
text_features = [
    col
    for col in X_train.columns
    if any(x in col for x in ["title_", "abstract_", "has_", "reading"])
]
temporal_features = [
    col
    for col in X_train.columns
    if any(x in col for x in ["hour", "day_", "month", "weekend", "time_"])
]
interaction_features = [col for col in X_train.columns if "_x_" in col]
category_features = ["category_enc"]

print(f"\nFeature breakdown:")
print(f"- Text features: {len(text_features)}")
print(f"- Temporal features: {len(temporal_features)}")
print(f"- Interaction features: {len(interaction_features)}")
print(f"- Category features: {len(category_features)}")
print(f"- Embedding features: {len(embedding_features)}")

# ================================
# 1. COMPREHENSIVE CORRELATION ANALYSIS
# ================================
print("\n" + "=" * 70)
print("1. COMPREHENSIVE CORRELATION ANALYSIS")
print("=" * 70)

# Calculate correlations with CTR for all non-embedding features
non_embedding_features = (
    text_features + temporal_features + interaction_features + category_features
)

correlation_results = []

for feature in non_embedding_features:
    # Pearson correlation
    pearson_corr, pearson_p = pearsonr(df_analysis[feature], df_analysis["ctr"])

    # Spearman correlation (rank-based, captures non-linear relationships)
    spearman_corr, spearman_p = spearmanr(df_analysis[feature], df_analysis["ctr"])

    correlation_results.append(
        {
            "feature": feature,
            "pearson_corr": pearson_corr,
            "pearson_p": pearson_p,
            "spearman_corr": spearman_corr,
            "spearman_p": spearman_p,
            "abs_pearson": abs(pearson_corr),
            "abs_spearman": abs(spearman_corr),
        }
    )

corr_df = pd.DataFrame(correlation_results).sort_values("abs_pearson", ascending=False)

print("TOP 15 FEATURES BY CORRELATION WITH CTR:")
print("=" * 70)
print(f"{'Feature':<25} {'Pearson':<8} {'P-val':<8} {'Spearman':<8} {'P-val':<8}")
print("-" * 70)

for _, row in corr_df.head(15).iterrows():
    significance = (
        "***"
        if row["pearson_p"] < 0.001
        else "**" if row["pearson_p"] < 0.01 else "*" if row["pearson_p"] < 0.05 else ""
    )
    print(
        f"{row['feature']:<25} {row['pearson_corr']:<8.4f} {row['pearson_p']:<8.4f} {row['spearman_corr']:<8.4f} {row['spearman_p']:<8.4f} {significance}"
    )

# ================================
# 2. BIVARIATE VISUALIZATIONS
# ================================
print("\n" + "=" * 70)
print("2. CREATING BIVARIATE VISUALIZATIONS")
print("=" * 70)

# Plot 1: Top correlations visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Feature-CTR Correlation Analysis", fontsize=16, fontweight="bold")

# Correlation comparison
top_features = corr_df.head(20)
x_pos = np.arange(len(top_features))

axes[0, 0].barh(
    x_pos, top_features["pearson_corr"], alpha=0.7, color="skyblue", label="Pearson"
)
axes[0, 0].barh(
    x_pos, top_features["spearman_corr"], alpha=0.7, color="orange", label="Spearman"
)
axes[0, 0].set_yticks(x_pos)
axes[0, 0].set_yticklabels(top_features["feature"], fontsize=8)
axes[0, 0].set_xlabel("Correlation with CTR")
axes[0, 0].set_title("Top 20 Features: Pearson vs Spearman Correlation")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Correlation magnitude distribution
axes[0, 1].hist(
    corr_df["abs_pearson"], bins=20, alpha=0.7, color="lightgreen", edgecolor="black"
)
axes[0, 1].set_xlabel("Absolute Correlation")
axes[0, 1].set_ylabel("Number of Features")
axes[0, 1].set_title("Distribution of Feature-CTR Correlations")
axes[0, 1].axvline(0.1, color="red", linestyle="--", label="|r| = 0.1")
axes[0, 1].axvline(0.05, color="orange", linestyle="--", label="|r| = 0.05")
axes[0, 1].legend()

# Statistical significance analysis
significant_features = corr_df[corr_df["pearson_p"] < 0.05]
axes[1, 0].bar(
    ["Significant\n(p<0.05)", "Non-significant\n(p≥0.05)"],
    [len(significant_features), len(corr_df) - len(significant_features)],
    color=["green", "red"],
    alpha=0.7,
)
axes[1, 0].set_ylabel("Number of Features")
axes[1, 0].set_title("Statistical Significance of Correlations")

# Effect size categorization
weak_corr = len(
    corr_df[(corr_df["abs_pearson"] >= 0.01) & (corr_df["abs_pearson"] < 0.3)]
)
moderate_corr = len(
    corr_df[(corr_df["abs_pearson"] >= 0.3) & (corr_df["abs_pearson"] < 0.5)]
)
strong_corr = len(corr_df[corr_df["abs_pearson"] >= 0.5])
very_weak_corr = len(corr_df[corr_df["abs_pearson"] < 0.01])

axes[1, 1].pie(
    [very_weak_corr, weak_corr, moderate_corr, strong_corr],
    labels=[
        "Very Weak\n(<0.01)",
        "Weak\n(0.01-0.3)",
        "Moderate\n(0.3-0.5)",
        "Strong\n(≥0.5)",
    ],
    autopct="%1.1f%%",
    startangle=90,
)
axes[1, 1].set_title("Distribution of Correlation Strengths")

plt.tight_layout()
plt.savefig(
    bivariate_plots_dir / "01_correlation_analysis.png", dpi=300, bbox_inches="tight"
)
plt.show()

# ================================
# 3. DETAILED BIVARIATE PLOTS FOR TOP FEATURES
# ================================
print("\nCreating detailed bivariate plots for top correlated features...")

top_8_features = corr_df.head(8)["feature"].tolist()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
fig.suptitle(
    "Detailed Bivariate Analysis - Top 8 Features vs CTR",
    fontsize=16,
    fontweight="bold",
)

for i, feature in enumerate(top_8_features):
    ax = axes[i]

    # Create scatter plot with trend line
    x = df_analysis[feature]
    y = df_analysis["ctr"]

    # Sample data if too many points (for better visualization)
    if len(x) > 5000:
        sample_idx = np.random.choice(len(x), 5000, replace=False)
        x_sample = x.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
    else:
        x_sample = x
        y_sample = y

    ax.scatter(x_sample, y_sample, alpha=0.3, s=1)

    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)

    # Add correlation info
    corr_val = corr_df[corr_df["feature"] == feature]["pearson_corr"].iloc[0]
    ax.set_title(f"{feature}\nCorr: {corr_val:.4f}", fontsize=10)
    ax.set_xlabel(feature, fontsize=8)
    ax.set_ylabel("CTR", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    bivariate_plots_dir / "02_top_features_scatter.png", dpi=300, bbox_inches="tight"
)
plt.show()

# ================================
# 4. CATEGORICAL FEATURE ANALYSIS
# ================================
print("\n" + "=" * 70)
print("4. CATEGORICAL FEATURE ANALYSIS")
print("=" * 70)

# Analyze binary features (has_question, has_exclamation, etc.)
binary_features = [f for f in text_features if f.startswith("has_")]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
fig.suptitle("Binary Features vs CTR Analysis", fontsize=16, fontweight="bold")

binary_analysis_results = []

for i, feature in enumerate(binary_features[:6]):  # Top 6 binary features
    ax = axes[i]

    # Calculate CTR by binary feature
    ctr_by_feature = (
        df_analysis.groupby(feature)["ctr"].agg(["mean", "count", "std"]).reset_index()
    )

    # Bar plot
    labels = (
        ["No", "Yes"]
        if len(ctr_by_feature) == 2
        else [str(x) for x in ctr_by_feature[feature]]
    )
    ax.bar(
        labels,
        ctr_by_feature["mean"],
        color=["lightcoral", "lightblue"] if len(labels) == 2 else "lightgreen",
        alpha=0.7,
        edgecolor="black",
    )

    # Add count annotations
    for j, (_, row) in enumerate(ctr_by_feature.iterrows()):
        ax.text(
            j,
            row["mean"] + 0.002,
            f'n={row["count"]}',
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Statistical test (t-test for binary features)
    if len(ctr_by_feature) == 2:
        group_0 = df_analysis[df_analysis[feature] == 0]["ctr"]
        group_1 = df_analysis[df_analysis[feature] == 1]["ctr"]
        t_stat, p_val = stats.ttest_ind(group_1, group_0)

        binary_analysis_results.append(
            {
                "feature": feature,
                "no_ctr": ctr_by_feature.iloc[0]["mean"],
                "yes_ctr": ctr_by_feature.iloc[1]["mean"],
                "difference": ctr_by_feature.iloc[1]["mean"]
                - ctr_by_feature.iloc[0]["mean"],
                "p_value": p_val,
                "significant": p_val < 0.05,
            }
        )

        significance = (
            "***"
            if p_val < 0.001
            else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        )
        ax.set_title(
            f'{feature.replace("has_", "")}\np={p_val:.4f} {significance}', fontsize=10
        )
    else:
        ax.set_title(feature.replace("has_", ""), fontsize=10)

    ax.set_ylabel("Average CTR")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    bivariate_plots_dir / "03_binary_features_analysis.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Print binary feature analysis results
if binary_analysis_results:
    print("\nBINARY FEATURE IMPACT ON CTR:")
    print("=" * 70)
    print(
        f"{'Feature':<20} {'No CTR':<8} {'Yes CTR':<8} {'Diff':<8} {'P-value':<10} {'Sig'}"
    )
    print("-" * 70)

    for result in sorted(
        binary_analysis_results, key=lambda x: abs(x["difference"]), reverse=True
    ):
        sig_marker = (
            "***"
            if result["p_value"] < 0.001
            else (
                "**"
                if result["p_value"] < 0.01
                else "*" if result["p_value"] < 0.05 else ""
            )
        )
        print(
            f"{result['feature']:<20} {result['no_ctr']:<8.4f} {result['yes_ctr']:<8.4f} {result['difference']:<8.4f} {result['p_value']:<10.4f} {sig_marker}"
        )

# ================================
# 5. TEMPORAL PATTERN ANALYSIS
# ================================
print("\n" + "=" * 70)
print("5. TEMPORAL PATTERN ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "Temporal Features vs CTR - Detailed Analysis", fontsize=16, fontweight="bold"
)

# Hour analysis with confidence intervals
hour_stats = (
    df_analysis.groupby("hour")["ctr"].agg(["mean", "std", "count"]).reset_index()
)
hour_stats["se"] = hour_stats["std"] / np.sqrt(hour_stats["count"])
hour_stats["ci_lower"] = hour_stats["mean"] - 1.96 * hour_stats["se"]
hour_stats["ci_upper"] = hour_stats["mean"] + 1.96 * hour_stats["se"]

axes[0, 0].plot(hour_stats["hour"], hour_stats["mean"], "o-", linewidth=2, markersize=6)
axes[0, 0].fill_between(
    hour_stats["hour"], hour_stats["ci_lower"], hour_stats["ci_upper"], alpha=0.3
)
axes[0, 0].set_xlabel("Hour of Day")
axes[0, 0].set_ylabel("Average CTR")
axes[0, 0].set_title("CTR by Hour (with 95% CI)")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(range(0, 24, 2))

# Day of week with statistical significance
day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_stats = (
    df_analysis.groupby("day_of_week")["ctr"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

# ANOVA test for day of week
dow_groups = [df_analysis[df_analysis["day_of_week"] == i]["ctr"] for i in range(7)]
f_stat, p_val = stats.f_oneway(*dow_groups)

axes[0, 1].bar(range(7), dow_stats["mean"], color="lightseagreen", alpha=0.7)
axes[0, 1].set_xticks(range(7))
axes[0, 1].set_xticklabels(day_names)
axes[0, 1].set_ylabel("Average CTR")
axes[0, 1].set_title(f"CTR by Day of Week\nANOVA p={p_val:.4f}")
axes[0, 1].grid(True, alpha=0.3)

# Month analysis
month_stats = df_analysis.groupby("month")["ctr"].agg(["mean", "count"]).reset_index()
month_names = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

axes[1, 0].bar(month_stats["month"], month_stats["mean"], color="darkgreen", alpha=0.7)
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_ylabel("Average CTR")
axes[1, 0].set_title("CTR by Month")
axes[1, 0].set_xticks(month_stats["month"])
axes[1, 0].set_xticklabels(
    [month_names[i - 1] for i in month_stats["month"]], rotation=45
)
axes[1, 0].grid(True, alpha=0.3)

# Weekend vs Weekday with effect size
weekend_stats = (
    df_analysis.groupby("is_weekend")["ctr"].agg(["mean", "count", "std"]).reset_index()
)
weekday_ctr = df_analysis[df_analysis["is_weekend"] == 0]["ctr"]
weekend_ctr = df_analysis[df_analysis["is_weekend"] == 1]["ctr"]

# Effect size (Cohen's d)
pooled_std = np.sqrt(
    (
        (len(weekday_ctr) - 1) * weekday_ctr.std() ** 2
        + (len(weekend_ctr) - 1) * weekend_ctr.std() ** 2
    )
    / (len(weekday_ctr) + len(weekend_ctr) - 2)
)
cohens_d = (weekend_ctr.mean() - weekday_ctr.mean()) / pooled_std

t_stat, p_val = stats.ttest_ind(weekend_ctr, weekday_ctr)

axes[1, 1].bar(
    ["Weekday", "Weekend"],
    weekend_stats["mean"],
    color=["skyblue", "salmon"],
    alpha=0.7,
)
axes[1, 1].set_ylabel("Average CTR")
axes[1, 1].set_title(f"Weekday vs Weekend\np={p_val:.4f}, d={cohens_d:.3f}")
for i, (_, row) in enumerate(weekend_stats.iterrows()):
    axes[1, 1].text(
        i,
        row["mean"] + 0.002,
        f'n={row["count"]}',
        ha="center",
        va="bottom",
        fontweight="bold",
    )
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    bivariate_plots_dir / "04_temporal_analysis_detailed.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# ================================
# 6. INTERACTION EFFECTS ANALYSIS
# ================================
print("\n" + "=" * 70)
print("6. INTERACTION EFFECTS ANALYSIS")
print("=" * 70)

if interaction_features:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Interaction Features vs CTR Analysis", fontsize=16, fontweight="bold")

    for i, feature in enumerate(interaction_features[:4]):
        ax = axes[i // 2, i % 2]

        # Scatter plot for interaction features
        x = df_analysis[feature]
        y = df_analysis["ctr"]

        ax.scatter(x, y, alpha=0.3, s=1)

        # Add correlation
        corr_val = corr_df[corr_df["feature"] == feature]["pearson_corr"].iloc[0]
        ax.set_title(f"{feature}\nCorr: {corr_val:.4f}")
        ax.set_xlabel(feature)
        ax.set_ylabel("CTR")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        bivariate_plots_dir / "05_interaction_features.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

# ================================
# 7. SUMMARY REPORT
# ================================
print("\n" + "=" * 70)
print("7. BIVARIATE ANALYSIS SUMMARY REPORT")
print("=" * 70)

# Save detailed correlation results
corr_df.to_csv(bivariate_plots_dir / "correlation_results.csv", index=False)

print(f"CORRELATION ANALYSIS SUMMARY:")
print(f"- Features analyzed: {len(corr_df)}")
print(f"- Statistically significant (p<0.05): {len(significant_features)}")
print(f"- Strong correlations (|r|≥0.1): {len(corr_df[corr_df['abs_pearson'] >= 0.1])}")
print(
    f"- Moderate correlations (0.05≤|r|<0.1): {len(corr_df[(corr_df['abs_pearson'] >= 0.05) & (corr_df['abs_pearson'] < 0.1)])}"
)

print(f"\nTOP 5 PREDICTIVE FEATURES:")
for i, (_, row) in enumerate(corr_df.head(5).iterrows(), 1):
    print(
        f"{i}. {row['feature']}: r={row['pearson_corr']:.4f} (p={row['pearson_p']:.4f})"
    )

print(f"\nKEY INSIGHTS:")
print(
    f"- Category encoding is the strongest predictor (r={corr_df.iloc[0]['pearson_corr']:.4f})"
)
print(f"- Text features show weak but significant correlations")
print(f"- Temporal patterns exist but are subtle")
print(f"- Most correlations are weak (|r|<0.1), indicating complex relationships")

print(f"\nFILES SAVED:")
print(f"- Detailed correlation results: correlation_results.csv")
print(
    f"- Bivariate plots: {len(list(bivariate_plots_dir.glob('*.png')))} visualization files"
)

print("\n" + "=" * 70)
print("✅ COMPREHENSIVE BIVARIATE ANALYSIS COMPLETE!")
print("=" * 70)
