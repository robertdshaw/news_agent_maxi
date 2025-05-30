import pandas as pd
import numpy as np
import json
from pathlib import Path
from textstat import flesch_reading_ease
import pickle
import warnings

warnings.filterwarnings("ignore")


class CompleteEDAInsightsGenerator:
    """Generate complete EDA insights file for EfficientLLMHeadlineRewriter"""

    def __init__(self, preprocessed_data_dir="data/preprocessed/processed_data"):
        """Initialize with your preprocessed data directory"""
        self.data_dir = Path(preprocessed_data_dir)
        self.output_path = self.data_dir / "headline_eda_insights.json"
        self.insights = {}

        print(f"📁 Data directory: {self.data_dir}")
        print(f"📄 Output file: {self.output_path}")

        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_preprocessed_data(self):
        """Load all your preprocessed data"""
        try:
            # Load training data
            self.df_train = pd.read_parquet(self.data_dir / "X_train_optimized.parquet")
            self.y_train = pd.read_parquet(self.data_dir / "y_train_optimized.parquet")
            self.metadata_train = pd.read_parquet(
                self.data_dir / "article_metadata_train_optimized.parquet"
            )

            # Combine all data
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
            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def generate_baseline_metrics(self):
        """Generate baseline metrics from your actual data"""
        target_stats = self.preprocessing_meta.get("target_statistics", {})

        baseline_metrics = {
            "overall_avg_ctr": target_stats.get("mean_ctr", self.df["ctr"].mean()),
            "overall_median_ctr": target_stats.get(
                "median_ctr", self.df["ctr"].median()
            ),
            "training_median_ctr": target_stats.get(
                "median_ctr", self.df["ctr"].median()
            ),
            "median_ctr": target_stats.get("median_ctr", self.df["ctr"].median()),
            "high_engagement_avg_ctr": self.df[self.df["high_engagement"] == 1][
                "ctr"
            ].mean(),
            "low_engagement_avg_ctr": self.df[self.df["high_engagement"] == 0][
                "ctr"
            ].mean(),
            "high_engagement_rate": target_stats.get(
                "high_engagement_rate", self.df["high_engagement"].mean()
            ),
            "ctr_threshold": target_stats.get("ctr_threshold", 0.05),
            "std_ctr": self.df["ctr"].std(),
            "min_ctr": self.df["ctr"].min(),
            "max_ctr": self.df["ctr"].max(),
        }

        self.insights["baseline_metrics"] = baseline_metrics
        return baseline_metrics

    def analyze_feature_improvements(self):
        """Analyze CTR improvements by feature"""
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

        feature_improvements = {}
        top_features_by_impact = []

        for feature in boolean_features:
            if feature in self.df.columns:
                with_feature = self.df[self.df[feature] == 1]["ctr"].mean()
                without_feature = self.df[self.df[feature] == 0]["ctr"].mean()

                if without_feature > 0:
                    improvement_percent = (
                        (with_feature - without_feature) / without_feature * 100
                    )
                else:
                    improvement_percent = 0

                sample_size_with = (self.df[feature] == 1).sum()
                sample_size_without = (self.df[feature] == 0).sum()

                feature_improvements[feature] = {
                    "with_feature_ctr": with_feature,
                    "without_feature_ctr": without_feature,
                    "improvement_percent": improvement_percent,
                    "sample_size_with": int(sample_size_with),
                    "sample_size_without": int(sample_size_without),
                }

                # Add to top features if improvement is positive and sample size is sufficient
                if improvement_percent > 0 and sample_size_with >= 100:
                    top_features_by_impact.append(
                        (
                            feature,
                            improvement_percent,
                            with_feature,
                            int(sample_size_with),
                        )
                    )

        # Sort by improvement percentage
        top_features_by_impact.sort(key=lambda x: x[1], reverse=True)

        self.insights["feature_improvements"] = feature_improvements
        self.insights["top_features_by_impact"] = top_features_by_impact[:8]

        return feature_improvements, top_features_by_impact

    def generate_editorial_strategies(self):
        """Generate editorial strategies from feature analysis"""
        feature_improvements = self.insights.get("feature_improvements", {})

        editorial_strategies = {}

        # Map features to strategy names
        feature_strategy_map = {
            "has_question": "question_headlines",
            "has_number": "number_headlines",
            "has_colon": "colon_headlines",
            "has_dash": "dash_headlines",
            "has_exclamation": "exclamation_headlines",
        }

        for feature_name, strategy_name in feature_strategy_map.items():
            if feature_name in feature_improvements:
                data = feature_improvements[feature_name]
                improvement_pct = data["improvement_percent"]

                if improvement_pct > 0:
                    # Calculate success rate (how often this feature leads to high engagement)
                    feature_mask = self.df[feature_name] == 1
                    success_rate = self.df[feature_mask]["high_engagement"].mean()

                    editorial_strategies[strategy_name] = {
                        "avg_ctr_improvement": improvement_pct
                        / 100,  # Convert to decimal
                        "success_rate": success_rate,
                        "feature_name": feature_name,
                        "sample_size": data["sample_size_with"],
                        "with_feature_ctr": data["with_feature_ctr"],
                        "without_feature_ctr": data["without_feature_ctr"],
                    }

        self.insights["editorial_strategies"] = editorial_strategies
        return editorial_strategies

    def analyze_optimal_ranges(self):
        """Analyze optimal ranges for word count, length, and readability"""
        optimal_ranges = {}

        # Word count analysis
        word_count_bins = [(3, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 20)]
        word_count_performance = {}

        for min_words, max_words in word_count_bins:
            mask = (self.df["title_word_count"] >= min_words) & (
                self.df["title_word_count"] < max_words
            )
            if mask.sum() > 50:
                word_count_performance[f"{min_words}-{max_words}"] = {
                    "avg_ctr": self.df[mask]["ctr"].mean(),
                    "median_ctr": self.df[mask]["ctr"].median(),
                    "count": int(mask.sum()),
                    "high_engagement_rate": self.df[mask]["high_engagement"].mean(),
                }

        optimal_ranges["word_count"] = word_count_performance

        # Character length analysis
        length_bins = [(20, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]
        length_performance = {}

        for min_len, max_len in length_bins:
            mask = (self.df["title_length"] >= min_len) & (
                self.df["title_length"] < max_len
            )
            if mask.sum() > 50:
                length_performance[f"{min_len}-{max_len}"] = {
                    "avg_ctr": self.df[mask]["ctr"].mean(),
                    "median_ctr": self.df[mask]["ctr"].median(),
                    "count": int(mask.sum()),
                    "high_engagement_rate": self.df[mask]["high_engagement"].mean(),
                }

        optimal_ranges["character_length"] = length_performance

        # Readability analysis
        readability_bins = [(0, 30), (30, 50), (50, 70), (70, 90), (90, 100)]
        readability_performance = {}

        for min_read, max_read in readability_bins:
            mask = (self.df["title_reading_ease"] >= min_read) & (
                self.df["title_reading_ease"] < max_read
            )
            if mask.sum() > 30:
                readability_performance[f"{min_read}-{max_read}"] = {
                    "avg_ctr": self.df[mask]["ctr"].mean(),
                    "median_ctr": self.df[mask]["ctr"].median(),
                    "count": int(mask.sum()),
                    "high_engagement_rate": self.df[mask]["high_engagement"].mean(),
                }

        optimal_ranges["readability"] = readability_performance
        self.insights["optimal_ranges"] = optimal_ranges

        return optimal_ranges

    def analyze_category_performance(self):
        """Analyze performance by category"""
        category_performance = {}

        for category in self.df["category"].unique():
            if pd.isna(category):
                continue

            cat_data = self.df[self.df["category"] == category]

            if len(cat_data) >= 20:
                category_performance[category] = {
                    "avg_ctr": cat_data["ctr"].mean(),
                    "median_ctr": cat_data["ctr"].median(),
                    "std_ctr": cat_data["ctr"].std(),
                    "count": len(cat_data),
                    "high_engagement_rate": cat_data["high_engagement"].mean(),
                    "top_10_percent_threshold": cat_data["ctr"].quantile(0.9),
                    "needs_rewrite_rate": cat_data["needs_rewrite"].mean(),
                }

        self.insights["category_performance"] = category_performance
        return category_performance

    def analyze_high_performers(self):
        """Analyze characteristics of top performers"""
        threshold = self.df["ctr"].quantile(0.9)
        top_performers = self.df[self.df["ctr"] >= threshold]

        # Get sample headlines
        sample_headlines = top_performers.nlargest(10, "ctr")["title"].tolist()

        high_performers = {
            "threshold_ctr": threshold,
            "count": len(top_performers),
            "avg_ctr": top_performers["ctr"].mean(),
            "median_word_count": top_performers["title_word_count"].median(),
            "median_length": top_performers["title_length"].median(),
            "avg_readability": top_performers["title_reading_ease"].mean(),
            "sample_headlines": sample_headlines,
            "question_rate": top_performers["has_question"].mean(),
            "number_rate": top_performers["has_number"].mean(),
            "colon_rate": top_performers["has_colon"].mean(),
        }

        self.insights["high_performers"] = high_performers
        return high_performers

    def generate_additional_insights(self):
        """Generate additional insights that might be needed"""

        # Patterns analysis
        patterns = {
            "overall_avg_ctr": self.df["ctr"].mean(),
            "overall_median_ctr": self.df["ctr"].median(),
            "total_articles": len(self.df),
            "high_engagement_count": (self.df["high_engagement"] == 1).sum(),
            "low_engagement_count": (self.df["high_engagement"] == 0).sum(),
        }

        # Editorial effectiveness
        editorial_effectiveness = {
            "readability_score_correlation": self.df[
                "editorial_readability_score"
            ].corr(self.df["ctr"]),
            "headline_score_correlation": self.df["editorial_headline_score"].corr(
                self.df["ctr"]
            ),
            "combined_score_correlation": (
                self.df["editorial_readability_score"]
                + self.df["editorial_headline_score"]
            ).corr(self.df["ctr"]),
        }

        # Readability insights
        readability_insights = {
            "optimal_flesch_score": 65,
            "optimal_word_count": {"min": 8, "max": 12},
            "optimal_char_count": {"min": 45, "max": 75},
            "readability_improvement_threshold": 10,
        }

        # Engagement patterns
        engagement_patterns = {
            "high_engagement_features": [
                "has_question",
                "has_number",
                "has_colon",
                "optimal_word_count",
            ],
            "low_engagement_patterns": [
                "too_long_title",
                "suboptimal_word_count",
                "needs_readability_improvement",
            ],
        }

        self.insights["patterns"] = patterns
        self.insights["editorial_effectiveness"] = editorial_effectiveness
        self.insights["readability_insights"] = readability_insights
        self.insights["engagement_patterns"] = engagement_patterns

    def generate_complete_insights(self):
        """Generate all insights required by EfficientLLMHeadlineRewriter"""

        print("🔄 Loading preprocessed data...")
        if not self.load_preprocessed_data():
            return None

        print("📊 Generating baseline metrics...")
        self.generate_baseline_metrics()

        print("📈 Analyzing feature improvements...")
        self.analyze_feature_improvements()

        print("✏️ Generating editorial strategies...")
        self.generate_editorial_strategies()

        print("📏 Analyzing optimal ranges...")
        self.analyze_optimal_ranges()

        print("📂 Analyzing category performance...")
        self.analyze_category_performance()

        print("🏆 Analyzing high performers...")
        self.analyze_high_performers()

        print("🔍 Generating additional insights...")
        self.generate_additional_insights()

        return self.insights

    def save_insights(self):
        """Save insights to the correct location"""

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

        # Save to the correct location
        with open(self.output_path, "w") as f:
            json.dump(clean_insights, f, indent=2)

        print(f"✅ Complete EDA insights saved to: {self.output_path}")

        # Also save a copy in the project root for backward compatibility
        root_path = Path("headline_eda_insights.json")
        with open(root_path, "w") as f:
            json.dump(clean_insights, f, indent=2)
        print(f"✅ Copy saved to: {root_path}")

        return clean_insights

    def validate_insights(self):
        """Validate that all required keys exist"""
        required_keys = {
            "baseline_metrics": [
                "overall_avg_ctr",
                "training_median_ctr",
                "high_engagement_avg_ctr",
            ],
            "editorial_strategies": [],
            "top_features_by_impact": [],
            "optimal_ranges": [],
            "category_performance": [],
            "high_performers": [],
            "patterns": [],
        }

        print("\n🔍 Validating insights structure...")
        all_valid = True

        for section, sub_keys in required_keys.items():
            if section in self.insights:
                print(f"  ✅ {section}")
                for sub_key in sub_keys:
                    if sub_key in self.insights[section]:
                        print(f"    ✅ {sub_key}")
                    else:
                        print(f"    ❌ Missing: {sub_key}")
                        all_valid = False
            else:
                print(f"  ❌ Missing section: {section}")
                all_valid = False

        if all_valid:
            print("✅ All required structures present!")
        else:
            print("❌ Some required structures missing!")

        return all_valid


def main():
    """Generate complete EDA insights for your project"""

    # Initialize generator
    generator = CompleteEDAInsightsGenerator("data/preprocessed/processed_data")

    # Generate all insights
    insights = generator.generate_complete_insights()

    if insights:
        # Save to correct location
        generator.save_insights()

        # Validate structure
        generator.validate_insights()

        # Print summary
        print("\n" + "=" * 60)
        print("📊 EDA INSIGHTS SUMMARY")
        print("=" * 60)

        baseline = insights["baseline_metrics"]
        print(f"📈 Overall avg CTR: {baseline['overall_avg_ctr']:.6f}")
        print(f"📈 Training median CTR: {baseline['training_median_ctr']:.6f}")
        print(f"📈 High engagement rate: {baseline['high_engagement_rate']:.3f}")

        print(f"\n🚀 Top features by impact:")
        for feature, improvement, ctr, size in insights["top_features_by_impact"][:5]:
            print(f"  • {feature}: +{improvement:.1f}% (CTR: {ctr:.4f}, n={size:,})")

        print(
            f"\n✏️ Editorial strategies: {len(insights['editorial_strategies'])} strategies"
        )
        print(
            f"📂 Categories analyzed: {len(insights['category_performance'])} categories"
        )
        print(f"🏆 High performers: {insights['high_performers']['count']} articles")

        print(f"\n✅ Files created:")
        print(f"  📄 {generator.output_path}")
        print(f"  📄 headline_eda_insights.json (project root)")

        return True
    else:
        print("❌ Failed to generate insights")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Ready for EfficientLLMHeadlineRewriter!")
    else:
        print("\n❌ Please check your data files and try again.")
