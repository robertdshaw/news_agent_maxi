import pandas as pd
import numpy as np
import json
from pathlib import Path
from textstat import flesch_reading_ease
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class HeadlineEDAAnalyzer:
    """Generate data-driven insights for headline optimization prompts using your actual preprocessed data"""

    def __init__(self, preprocessed_data_dir="data/preprocessed/processed_data"):
        """Load your preprocessed data from the EDA preprocessing pipeline"""
        self.data_dir = Path(preprocessed_data_dir)
        self.insights = {}

        # Load the preprocessed data
        print("Loading preprocessed data...")
        try:
            # Load training data (your main analysis dataset)
            self.df_train = pd.read_parquet(self.data_dir / "X_train_optimized.parquet")
            self.y_train = pd.read_parquet(self.data_dir / "y_train_optimized.parquet")
            self.metadata_train = pd.read_parquet(
                self.data_dir / "article_metadata_train_optimized.parquet"
            )

            # Combine features with targets and metadata
            self.df = pd.concat(
                [
                    self.df_train,
                    self.y_train,
                    self.metadata_train[["newsID", "title", "category"]],
                ],
                axis=1,
            )

            # Load preprocessing metadata
            with open(self.data_dir / "preprocessing_metadata.json", "r") as f:
                self.preprocessing_meta = json.load(f)

            print(
                f"✅ Loaded {len(self.df):,} articles with {len(self.df_train.columns)} features"
            )
            print(
                f"Training median CTR: {self.preprocessing_meta['target_statistics']['median_ctr']:.6f}"
            )

        except FileNotFoundError as e:
            print(f"❌ Error loading preprocessed data: {e}")
            print("Please run the EDA_preprocess_features.py script first")
            raise

    def analyze_engagement_features(self):
        """Analyze which features correlate with high CTR using your actual features"""

        print("Analyzing engagement features from your preprocessed data...")

        # Your actual boolean features from the preprocessing
        boolean_features = [
            "has_question",
            "has_exclamation",
            "has_number",
            "has_colon",
            "has_quotes",
            "has_dash",
            "has_abstract",
            "needs_readability_improvement",
            "suboptimal_word_count",
            "too_long_title",
        ]

        # Your actual continuous features
        continuous_features = [
            "title_length",
            "title_word_count",
            "title_reading_ease",
            "abstract_length",
            "avg_word_length",
            "title_upper_ratio",
            "editorial_readability_score",
            "editorial_headline_score",
        ]

        # Calculate CTR improvements for boolean features
        ctr_improvements = {}
        for feature in boolean_features:
            if feature in self.df.columns:
                with_feature = self.df[self.df[feature] == 1]["ctr"].mean()
                without_feature = self.df[self.df[feature] == 0]["ctr"].mean()
                improvement = (with_feature - without_feature) / without_feature * 100

                ctr_improvements[feature] = {
                    "with_feature_ctr": with_feature,
                    "without_feature_ctr": without_feature,
                    "improvement_percent": improvement,
                    "sample_size_with": (self.df[feature] == 1).sum(),
                    "sample_size_without": (self.df[feature] == 0).sum(),
                }

        # Calculate correlations for continuous features
        correlations = {}
        for feature in continuous_features:
            if feature in self.df.columns:
                corr = self.df[feature].corr(self.df["ctr"])
                correlations[feature] = corr

        self.insights["feature_improvements"] = ctr_improvements
        self.insights["feature_correlations"] = correlations

        return ctr_improvements, correlations

    def analyze_optimal_ranges(self):
        """Find optimal ranges using your actual data"""

        print("Analyzing optimal ranges for key metrics...")

        # Word count analysis (based on your editorial criteria)
        word_count_bins = [(3, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 20)]
        word_count_performance = {}

        for min_words, max_words in word_count_bins:
            mask = (self.df["title_word_count"] >= min_words) & (
                self.df["title_word_count"] < max_words
            )
            if mask.sum() > 50:  # Only if enough samples
                avg_ctr = self.df[mask]["ctr"].mean()
                median_ctr = self.df[mask]["ctr"].median()
                count = mask.sum()
                high_engagement_rate = self.df[mask]["high_engagement"].mean()

                word_count_performance[f"{min_words}-{max_words}"] = {
                    "avg_ctr": avg_ctr,
                    "median_ctr": median_ctr,
                    "count": count,
                    "high_engagement_rate": high_engagement_rate,
                }

        # Character length analysis
        length_bins = [(20, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]
        length_performance = {}

        for min_len, max_len in length_bins:
            mask = (self.df["title_length"] >= min_len) & (
                self.df["title_length"] < max_len
            )
            if mask.sum() > 50:
                avg_ctr = self.df[mask]["ctr"].mean()
                median_ctr = self.df[mask]["ctr"].median()
                count = mask.sum()
                high_engagement_rate = self.df[mask]["high_engagement"].mean()

                length_performance[f"{min_len}-{max_len}"] = {
                    "avg_ctr": avg_ctr,
                    "median_ctr": median_ctr,
                    "count": count,
                    "high_engagement_rate": high_engagement_rate,
                }

        # Readability analysis (using your Flesch Reading Ease scores)
        readability_bins = [(0, 30), (30, 50), (50, 70), (70, 90), (90, 100)]
        readability_performance = {}

        for min_read, max_read in readability_bins:
            mask = (self.df["title_reading_ease"] >= min_read) & (
                self.df["title_reading_ease"] < max_read
            )
            if mask.sum() > 30:
                avg_ctr = self.df[mask]["ctr"].mean()
                median_ctr = self.df[mask]["ctr"].median()
                count = mask.sum()
                high_engagement_rate = self.df[mask]["high_engagement"].mean()

                readability_performance[f"{min_read}-{max_read}"] = {
                    "avg_ctr": avg_ctr,
                    "median_ctr": median_ctr,
                    "count": count,
                    "high_engagement_rate": high_engagement_rate,
                }

        self.insights["optimal_ranges"] = {
            "word_count": word_count_performance,
            "character_length": length_performance,
            "readability": readability_performance,
        }

        return word_count_performance, length_performance, readability_performance

    def analyze_category_performance(self):
        """Analyze performance by category using your encoded categories"""

        print("Analyzing category performance...")

        category_stats = {}

        # Group by category (use string category, not encoded)
        for category in self.df["category"].unique():
            if pd.isna(category):
                continue

            cat_data = self.df[self.df["category"] == category]

            if len(cat_data) >= 20:  # Only analyze categories with sufficient data
                category_stats[category] = {
                    "avg_ctr": cat_data["ctr"].mean(),
                    "median_ctr": cat_data["ctr"].median(),
                    "std_ctr": cat_data["ctr"].std(),
                    "count": len(cat_data),
                    "high_engagement_rate": cat_data["high_engagement"].mean(),
                    "top_10_percent_threshold": cat_data["ctr"].quantile(0.9),
                    "needs_rewrite_rate": cat_data["needs_rewrite"].mean(),
                }

        self.insights["category_performance"] = category_stats
        return category_stats

    def analyze_high_performers(self, top_percentile=0.1):
        """Analyze characteristics of top-performing headlines"""

        print(f"Analyzing top {top_percentile*100:.0f}% performers...")

        threshold = self.df["ctr"].quantile(1 - top_percentile)
        top_performers = self.df[self.df["ctr"] >= threshold]
        low_performers = self.df[
            self.df["ctr"] <= self.df["ctr"].quantile(top_percentile)
        ]

        print(f"High performance threshold: {threshold:.6f}")
        print(f"Top performers: {len(top_performers):,} articles")
        print(f"Low performers: {len(low_performers):,} articles")

        high_performer_characteristics = {}

        # Analyze feature prevalence in top vs bottom performers
        features_to_analyze = [
            "has_number",
            "has_question",
            "has_exclamation",
            "has_colon",
            "has_quotes",
            "has_dash",
            "needs_readability_improvement",
        ]

        for feature in features_to_analyze:
            if feature in self.df.columns:
                top_rate = top_performers[feature].mean()
                low_rate = low_performers[feature].mean()
                lift = (top_rate - low_rate) / low_rate * 100 if low_rate > 0 else 0

                high_performer_characteristics[feature] = {
                    "top_performer_rate": top_rate,
                    "low_performer_rate": low_rate,
                    "lift_percent": lift,
                }

        # Analyze optimal ranges for top performers
        top_word_count = {
            "mean": top_performers["title_word_count"].mean(),
            "median": top_performers["title_word_count"].median(),
            "std": top_performers["title_word_count"].std(),
            "mode": (
                top_performers["title_word_count"].mode().iloc[0]
                if not top_performers["title_word_count"].mode().empty
                else None
            ),
        }

        top_length = {
            "mean": top_performers["title_length"].mean(),
            "median": top_performers["title_length"].median(),
            "std": top_performers["title_length"].std(),
        }

        top_readability = {
            "mean": top_performers["title_reading_ease"].mean(),
            "median": top_performers["title_reading_ease"].median(),
            "std": top_performers["title_reading_ease"].std(),
        }

        # Get sample headlines
        sample_headlines = top_performers.nlargest(10, "ctr")["title"].tolist()

        self.insights["high_performers"] = {
            "characteristics": high_performer_characteristics,
            "optimal_word_count": top_word_count,
            "optimal_length": top_length,
            "optimal_readability": top_readability,
            "threshold_ctr": threshold,
            "sample_headlines": sample_headlines,
            "count": len(top_performers),
        }

        return (
            high_performer_characteristics,
            top_word_count,
            top_length,
            top_readability,
        )

    def analyze_editorial_scoring_effectiveness(self):
        """Analyze how well your editorial scoring correlates with actual performance"""

        print("Analyzing editorial scoring effectiveness...")

        # Analyze editorial readability score vs actual CTR
        readability_corr = self.df["editorial_readability_score"].corr(self.df["ctr"])
        headline_score_corr = self.df["editorial_headline_score"].corr(self.df["ctr"])

        # Analyze combined editorial score
        self.df["combined_editorial_score"] = (
            self.df["editorial_readability_score"] + self.df["editorial_headline_score"]
        )
        combined_corr = self.df["combined_editorial_score"].corr(self.df["ctr"])

        # Analyze how editorial criteria map to high engagement
        high_editorial_score = self.df[
            self.df["combined_editorial_score"]
            > self.df["combined_editorial_score"].median()
        ]
        low_editorial_score = self.df[
            self.df["combined_editorial_score"]
            <= self.df["combined_editorial_score"].median()
        ]

        editorial_effectiveness = {
            "readability_score_correlation": readability_corr,
            "headline_score_correlation": headline_score_corr,
            "combined_score_correlation": combined_corr,
            "high_editorial_engagement_rate": high_editorial_score[
                "high_engagement"
            ].mean(),
            "low_editorial_engagement_rate": low_editorial_score[
                "high_engagement"
            ].mean(),
            "editorial_score_lift": (
                high_editorial_score["high_engagement"].mean()
                - low_editorial_score["high_engagement"].mean()
            )
            / low_editorial_score["high_engagement"].mean()
            * 100,
        }

        self.insights["editorial_effectiveness"] = editorial_effectiveness
        return editorial_effectiveness

    def analyze_common_patterns(self):
        """Analyze common word patterns and structures in high-performing headlines"""

        print("Analyzing common patterns...")

        # Extract starting words from titles
        starting_words = defaultdict(list)
        for idx, title in enumerate(self.df["title"]):
            if pd.notna(title) and isinstance(title, str):
                first_word = title.split()[0].lower() if title.split() else ""
                if len(first_word) > 1:  # Skip single characters
                    starting_words[first_word].append(self.df.iloc[idx]["ctr"])

        # Get performance by starting word (min 20 occurrences)
        starting_word_performance = {}
        for word, ctrs in starting_words.items():
            if len(ctrs) >= 20:
                starting_word_performance[word] = {
                    "avg_ctr": np.mean(ctrs),
                    "median_ctr": np.median(ctrs),
                    "count": len(ctrs),
                    "std_ctr": np.std(ctrs),
                }

        # Sort by performance
        top_starting_words = sorted(
            starting_word_performance.items(),
            key=lambda x: x[1]["avg_ctr"],
            reverse=True,
        )[:15]

        # Analyze punctuation patterns
        punctuation_patterns = {}
        patterns = {
            "ends_with_question": lambda x: x.strip().endswith("?"),
            "ends_with_exclamation": lambda x: x.strip().endswith("!"),
            "contains_colon": lambda x: ":" in x,
            "contains_dash": lambda x: any(dash in x for dash in ["-", "–", "—"]),
            "starts_with_number": lambda x: (
                x.strip()[0].isdigit() if x.strip() else False
            ),
            "all_caps_words": lambda x: len(
                [word for word in x.split() if word.isupper() and len(word) > 1]
            ),
        }

        for pattern_name, pattern_func in patterns.items():
            try:
                matching_articles = self.df[self.df["title"].apply(pattern_func)]
                if len(matching_articles) > 10:
                    punctuation_patterns[pattern_name] = {
                        "avg_ctr": matching_articles["ctr"].mean(),
                        "count": len(matching_articles),
                        "high_engagement_rate": matching_articles[
                            "high_engagement"
                        ].mean(),
                    }
            except:
                continue

        self.insights["patterns"] = {
            "top_starting_words": dict(top_starting_words),
            "punctuation_patterns": punctuation_patterns,
            "overall_avg_ctr": self.df["ctr"].mean(),
            "overall_median_ctr": self.df["ctr"].median(),
        }

        return starting_word_performance, punctuation_patterns

    def generate_prompt_insights(self):
        """Generate comprehensive insights for prompt engineering"""

        print("Generating prompt insights...")

        # Run all analyses
        self.analyze_engagement_features()
        self.analyze_optimal_ranges()
        self.analyze_category_performance()
        self.analyze_high_performers()
        self.analyze_editorial_scoring_effectiveness()
        self.analyze_common_patterns()

        # Extract key insights for prompts
        prompt_data = {
            "top_features_by_impact": [],
            "optimal_specifications": {},
            "category_benchmarks": self.insights["category_performance"],
            "high_performer_patterns": self.insights["high_performers"],
            "proven_starters": self.insights["patterns"]["top_starting_words"],
            "editorial_validation": self.insights["editorial_effectiveness"],
            "data_source": "preprocessed_eda_analysis",
            "baseline_metrics": {
                "overall_avg_ctr": self.insights["patterns"]["overall_avg_ctr"],
                "overall_median_ctr": self.insights["patterns"]["overall_median_ctr"],
                "high_engagement_threshold": self.preprocessing_meta[
                    "target_statistics"
                ]["ctr_threshold"],
                "training_median_ctr": self.preprocessing_meta["target_statistics"][
                    "median_ctr"
                ],
            },
        }

        # Rank features by impact
        feature_impacts = []
        for feature, data in self.insights["feature_improvements"].items():
            if data["improvement_percent"] > 0 and data["sample_size_with"] > 100:
                feature_impacts.append(
                    (
                        feature,
                        data["improvement_percent"],
                        data["with_feature_ctr"],
                        data["sample_size_with"],
                    )
                )

        # Sort by improvement percentage
        feature_impacts.sort(key=lambda x: x[1], reverse=True)
        prompt_data["top_features_by_impact"] = feature_impacts[:8]

        # Find optimal ranges
        if self.insights["optimal_ranges"]["word_count"]:
            best_word_range = max(
                self.insights["optimal_ranges"]["word_count"].items(),
                key=lambda x: x[1]["avg_ctr"],
            )
            prompt_data["optimal_specifications"]["word_count_range"] = best_word_range[
                0
            ]
            prompt_data["optimal_specifications"]["word_count_ctr"] = best_word_range[
                1
            ]["avg_ctr"]

        if self.insights["optimal_ranges"]["character_length"]:
            best_length_range = max(
                self.insights["optimal_ranges"]["character_length"].items(),
                key=lambda x: x[1]["avg_ctr"],
            )
            prompt_data["optimal_specifications"]["length_range"] = best_length_range[0]
            prompt_data["optimal_specifications"]["length_ctr"] = best_length_range[1][
                "avg_ctr"
            ]

        if self.insights["optimal_ranges"]["readability"]:
            best_readability_range = max(
                self.insights["optimal_ranges"]["readability"].items(),
                key=lambda x: x[1]["avg_ctr"],
            )
            prompt_data["optimal_specifications"]["readability_range"] = (
                best_readability_range[0]
            )
            prompt_data["optimal_specifications"]["readability_ctr"] = (
                best_readability_range[1]["avg_ctr"]
            )

        return prompt_data

    def save_insights(self, output_path="headline_eda_insights.json"):
        """Save all insights to JSON file for use in prompt generation"""

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return obj

        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [clean_dict(v) for v in d]
            else:
                return convert_numpy(d)

        clean_insights = clean_dict(self.insights)

        with open(output_path, "w") as f:
            json.dump(clean_insights, f, indent=2)

        print(f"✅ EDA insights saved to {output_path}")
        return clean_insights

    def create_visualization_report(self, output_dir="eda_reports"):
        """Create visualizations of key insights"""

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 1. Feature impact visualization
        if "feature_improvements" in self.insights:
            improvements = []
            features = []
            sample_sizes = []

            for feature, data in self.insights["feature_improvements"].items():
                if data["improvement_percent"] > 0 and data["sample_size_with"] > 50:
                    improvements.append(data["improvement_percent"])
                    features.append(feature.replace("_", " ").title())
                    sample_sizes.append(data["sample_size_with"])

            if improvements:
                plt.figure(figsize=(12, 8))
                bars = plt.bar(features, improvements)
                plt.title(
                    "CTR Improvement by Feature (%)", fontsize=14, fontweight="bold"
                )
                plt.xlabel("Features")
                plt.ylabel("CTR Improvement (%)")
                plt.xticks(rotation=45, ha="right")

                # Add value labels on bars
                for bar, value, size in zip(bars, improvements, sample_sizes):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{value:.1f}%\n(n={size:,})",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

                plt.tight_layout()
                plt.savefig(
                    output_dir / "feature_impact.png", dpi=300, bbox_inches="tight"
                )
                plt.close()

        # 2. Optimal ranges visualization
        if "optimal_ranges" in self.insights:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Word count performance
            if self.insights["optimal_ranges"]["word_count"]:
                word_data = self.insights["optimal_ranges"]["word_count"]
                ranges = list(word_data.keys())
                ctrs = [data["avg_ctr"] for data in word_data.values()]
                counts = [data["count"] for data in word_data.values()]

                bars = axes[0].bar(ranges, ctrs)
                axes[0].set_title("Average CTR by Word Count Range")
                axes[0].set_xlabel("Word Count Range")
                axes[0].set_ylabel("Average CTR")
                axes[0].tick_params(axis="x", rotation=45)

                # Add count labels
                for bar, ctr, count in zip(bars, ctrs, counts):
                    axes[0].text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.0001,
                        f"{ctr:.4f}\n(n={count:,})",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            # Character length performance
            if self.insights["optimal_ranges"]["character_length"]:
                length_data = self.insights["optimal_ranges"]["character_length"]
                ranges = list(length_data.keys())
                ctrs = [data["avg_ctr"] for data in length_data.values()]
                counts = [data["count"] for data in length_data.values()]

                bars = axes[1].bar(ranges, ctrs)
                axes[1].set_title("Average CTR by Character Length Range")
                axes[1].set_xlabel("Character Length Range")
                axes[1].set_ylabel("Average CTR")
                axes[1].tick_params(axis="x", rotation=45)

                for bar, ctr, count in zip(bars, ctrs, counts):
                    axes[1].text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.0001,
                        f"{ctr:.4f}\n(n={count:,})",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            # Readability performance
            if self.insights["optimal_ranges"]["readability"]:
                read_data = self.insights["optimal_ranges"]["readability"]
                ranges = list(read_data.keys())
                ctrs = [data["avg_ctr"] for data in read_data.values()]
                counts = [data["count"] for data in read_data.values()]

                bars = axes[2].bar(ranges, ctrs)
                axes[2].set_title("Average CTR by Readability Range")
                axes[2].set_xlabel("Readability Range (Flesch Score)")
                axes[2].set_ylabel("Average CTR")
                axes[2].tick_params(axis="x", rotation=45)

                for bar, ctr, count in zip(bars, ctrs, counts):
                    axes[2].text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.0001,
                        f"{ctr:.4f}\n(n={count:,})",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            plt.tight_layout()
            plt.savefig(output_dir / "optimal_ranges.png", dpi=300, bbox_inches="tight")
            plt.close()

        # 3. Category performance
        if "category_performance" in self.insights:
            cat_data = self.insights["category_performance"]
            categories = list(cat_data.keys())[:10]  # Top 10 categories
            avg_ctrs = [cat_data[cat]["avg_ctr"] for cat in categories]
            counts = [cat_data[cat]["count"] for cat in categories]

            plt.figure(figsize=(12, 8))
            bars = plt.bar(categories, avg_ctrs)
            plt.title(
                "Average CTR by Category (Top 10 by Sample Size)",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Category")
            plt.ylabel("Average CTR")
            plt.xticks(rotation=45, ha="right")

            for bar, ctr, count in zip(bars, avg_ctrs, counts):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0001,
                    f"{ctr:.4f}\n(n={count:,})",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            plt.tight_layout()
            plt.savefig(
                output_dir / "category_performance.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        print(f"✅ Visualization report saved to {output_dir}")


# Usage example
if __name__ == "__main__":
    # Initialize analyzer with your preprocessed data
    analyzer = HeadlineEDAAnalyzer("data/preprocessed/processed_data")

    # Generate comprehensive insights
    prompt_insights = analyzer.generate_prompt_insights()

    # Save insights for use in prompts
    analyzer.save_insights("headline_eda_insights.json")

    # Create visualizations
    analyzer.create_visualization_report("eda_reports")

    # Print key findings for immediate use
    print("\n" + "=" * 60)
    print("KEY INSIGHTS FOR PROMPT ENGINEERING")
    print("=" * 60)

    print(f"\n📊 DATA SOURCE:")
    print(f"  Analyzed {len(analyzer.df):,} articles from preprocessed data")
    print(
        f"  Training median CTR: {prompt_insights['baseline_metrics']['training_median_ctr']:.6f}"
    )
    print(
        f"  Overall average CTR: {prompt_insights['baseline_metrics']['overall_avg_ctr']:.6f}"
    )

    print(f"\n🚀 TOP FEATURES BY CTR IMPACT:")
    for feature, improvement, ctr, sample_size in prompt_insights[
        "top_features_by_impact"
    ][:5]:
        print(
            f"  {feature.replace('_', ' ').title()}: +{improvement:.1f}% improvement (CTR: {ctr:.4f}, n={sample_size:,})"
        )

    if "optimal_specifications" in prompt_insights:
        print(f"\n📏 OPTIMAL SPECIFICATIONS:")
        specs = prompt_insights["optimal_specifications"]
        if "word_count_range" in specs:
            print(
                f"  Word Count: {specs['word_count_range']} words (CTR: {specs['word_count_ctr']:.4f})"
            )
        if "length_range" in specs:
            print(
                f"  Character Length: {specs['length_range']} chars (CTR: {specs['length_ctr']:.4f})"
            )
        if "readability_range" in specs:
            print(
                f"  Readability: {specs['readability_range']} (CTR: {specs['readability_ctr']:.4f})"
            )

    print(f"\n🏆 TOP STARTING WORDS:")
    for word, data in list(prompt_insights["proven_starters"].items())[:5]:
        print(f"  '{word}': {data['avg_ctr']:.4f} CTR ({data['count']} samples)")

    print(f"\n✅ Editorial scoring effectiveness:")
    if "editorial_validation" in prompt_insights:
        editorial = prompt_insights["editorial_validation"]
        print(
            f"  Editorial score correlation with CTR: {editorial.get('combined_score_correlation', 0):.3f}"
        )
        print(
            f"  High editorial score engagement lift: {editorial.get('editorial_score_lift', 0):.1f}%"
        )

    print(f"\n📁 Files created:")
    print(f"  - headline_eda_insights.json (for prompt generation)")
    print(f"  - eda_reports/feature_impact.png")
    print(f"  - eda_reports/optimal_ranges.png")
    print(f"  - eda_reports/category_performance.png")

    print(f"\n✅ Ready for enhanced headline rewriter with data-driven prompts!")
