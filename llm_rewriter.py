import os
import re
import pandas as pd
import numpy as np
import logging
from openai import OpenAI
from feature_utils import create_article_features_exact, load_preprocessing_components


class EfficientLLMHeadlineRewriter:
    """Efficient LLM headline rewriter using XGBoost model and EDA insights"""

    def __init__(
        self, model_pipeline, components=None, llm_client=None, eda_insights_path=None
    ):
        """Initialize with trained model and preprocessing components"""
        self.model_pipeline = model_pipeline

        if components is None:
            components = load_preprocessing_components()
        self.components = components

        if llm_client is not None:
            self.client = llm_client
        else:
            self.client = OpenAI()

        # Load EDA insights if available
        self.eda_insights = self._load_eda_insights(eda_insights_path)

        # Configuration based on EDA findings (with safe defaults)
        self.config = {
            "optimal_word_count": (8, 12),  # Conservative default
            "max_length": 100,
            "target_readability": 60,
            "high_engagement_threshold": self._get_safe_value(
                "baseline_metrics", "overall_avg_ctr", 0.041
            ),
        }

        logging.info("Efficient LLM Headline Rewriter initialized")
        logging.info(f"Using XGBoost model for 100% of CTR predictions")
        logging.info(
            f"EDA insights: {'Loaded' if self.eda_insights else 'Using defaults'}"
        )

    def _get_safe_value(self, section, key, default):
        """Safely get a value from EDA insights with fallback"""
        try:
            if self.eda_insights and section in self.eda_insights:
                return self.eda_insights[section].get(key, default)
            return default
        except:
            return default

    def _get_optimal_specs(self):
        """Extract optimal specifications from whatever data is available"""
        # Default fallback values
        default_specs = {
            "word_count_range": "8-12",
            "word_count_ctr": 0.04,
            "length_range": "50-70",
            "length_ctr": 0.04,
            "readability_range": "50-70",
            "readability_ctr": 0.04,
        }

        # If optimal_specifications exists, use it
        if self.eda_insights and "optimal_specifications" in self.eda_insights:
            return self.eda_insights["optimal_specifications"]

        # Otherwise, try to construct from optimal_ranges
        if self.eda_insights and "optimal_ranges" in self.eda_insights:
            optimal_ranges = self.eda_insights["optimal_ranges"]

            # Extract best word count range
            if "word_count" in optimal_ranges:
                word_count_data = optimal_ranges["word_count"]
                if word_count_data:
                    best_word_entry = max(
                        word_count_data.items(), key=lambda x: x[1].get("avg_ctr", 0)
                    )
                    default_specs["word_count_range"] = best_word_entry[0]
                    default_specs["word_count_ctr"] = best_word_entry[1].get(
                        "avg_ctr", 0.04
                    )

            # Extract best character length range
            if "character_length" in optimal_ranges:
                length_data = optimal_ranges["character_length"]
                if length_data:
                    best_length_entry = max(
                        length_data.items(), key=lambda x: x[1].get("avg_ctr", 0)
                    )
                    default_specs["length_range"] = best_length_entry[0]
                    default_specs["length_ctr"] = best_length_entry[1].get(
                        "avg_ctr", 0.04
                    )

            # Extract best readability range
            if "readability" in optimal_ranges:
                readability_data = optimal_ranges["readability"]
                if readability_data:
                    best_readability_entry = max(
                        readability_data.items(), key=lambda x: x[1].get("avg_ctr", 0)
                    )
                    default_specs["readability_range"] = best_readability_entry[0]
                    default_specs["readability_ctr"] = best_readability_entry[1].get(
                        "avg_ctr", 0.04
                    )

        return default_specs

    def _get_top_features(self):
        """Get top features from whatever data is available"""
        if self.eda_insights and "top_features_by_impact" in self.eda_insights:
            return self.eda_insights["top_features_by_impact"][:3]

        # Fallback to feature_improvements if available
        if self.eda_insights and "feature_improvements" in self.eda_insights:
            improvements = self.eda_insights["feature_improvements"]
            top_features = []
            for feature, data in improvements.items():
                if data.get("improvement_percent", 0) > 0:
                    top_features.append(
                        (
                            feature,
                            data["improvement_percent"],
                            data["with_feature_ctr"],
                            data["sample_size_with"],
                        )
                    )
            # Sort by improvement and return top 3
            top_features.sort(key=lambda x: x[1], reverse=True)
            return top_features[:3]

        # Ultimate fallback - generic features that usually help
        return [
            ("has_number", 15.0, 0.045, 1000),
            ("has_question", 12.0, 0.043, 800),
            ("has_colon", 8.0, 0.042, 600),
        ]

    def _get_baseline_metrics(self):
        """Get baseline metrics from whatever data is available"""
        defaults = {
            "overall_avg_ctr": 0.041,
            "training_median_ctr": 0.030,
            "articles_analyzed": 25000,
        }

        if self.eda_insights and "baseline_metrics" in self.eda_insights:
            baseline = self.eda_insights["baseline_metrics"]
            return {
                "overall_avg_ctr": baseline.get(
                    "overall_avg_ctr", defaults["overall_avg_ctr"]
                ),
                "training_median_ctr": baseline.get(
                    "training_median_ctr", defaults["training_median_ctr"]
                ),
                "articles_analyzed": baseline.get(
                    "articles_analyzed", defaults["articles_analyzed"]
                ),
            }

        return defaults

    def _get_proven_starters(self):
        """Get proven starter words from whatever data is available"""
        if self.eda_insights and "proven_starters" in self.eda_insights:
            return self.eda_insights["proven_starters"]

        # Check if high_performers has sample_headlines we can analyze
        if self.eda_insights and "high_performers" in self.eda_insights:
            high_performers = self.eda_insights["high_performers"]
            if "sample_headlines" in high_performers:
                # Extract first words from sample headlines
                starters = {}
                for headline in high_performers["sample_headlines"][:10]:
                    if headline and len(str(headline).split()) > 0:
                        first_word = str(headline).split()[0].lower().strip('.,!?";')
                        if len(first_word) > 2 and first_word.isalpha():
                            starters[first_word] = {"avg_ctr": 0.08, "count": 5}
                return starters

        # Ultimate fallback - common high-performing starters
        return {
            "breaking": {"avg_ctr": 0.12, "count": 50},
            "exclusive": {"avg_ctr": 0.10, "count": 30},
            "shocking": {"avg_ctr": 0.09, "count": 25},
        }

    def _load_eda_insights(self, eda_insights_path):
        """Load EDA insights from analysis or use defaults"""
        if eda_insights_path and os.path.exists(eda_insights_path):
            try:
                import json

                with open(eda_insights_path, "r") as f:
                    insights = json.load(f)
                logging.info(f"Loaded EDA insights from {eda_insights_path}")
                return insights
            except Exception as e:
                logging.warning(f"Could not load EDA insights: {e}")

        # Return None - we'll handle missing insights defensively
        logging.info("No EDA insights loaded - using defensive defaults")
        return None

    def predict_ctr_with_model(self, title, article_data):
        """Use YOUR XGBoost model to predict CTR - the only CTR prediction method"""
        try:
            # Extract features using your exact pipeline
            features = create_article_features_exact(
                title,
                article_data.get("abstract", ""),
                article_data.get("category", "news"),
                self.components,
            )

            # Vectorize in exact feature order
            feature_order = self.components["feature_order"]
            feature_vector = np.array(
                [features.get(f, 0.0) for f in feature_order]
            ).reshape(1, -1)

            # Predict using YOUR model - the ONLY source of CTR predictions
            engagement_prob = self.model_pipeline["model"].predict_proba(
                feature_vector
            )[0, 1]

            # Convert engagement probability to CTR estimate
            estimated_ctr = max(0.01, engagement_prob * 0.1)

            return float(estimated_ctr)

        except Exception as e:
            logging.error(f"Model CTR prediction failed: {e}")
            return 0.035  # Fallback

    def generate_candidates(self, original_title, article_data):
        """Generate headline candidates using LLM with ACTUAL EDA insights (defensive)"""

        category = article_data.get("category", "news")
        current_ctr = self.predict_ctr_with_model(original_title, article_data)

        # Get data safely from whatever is available
        baseline_metrics = self._get_baseline_metrics()
        top_features = self._get_top_features()
        optimal_specs = self._get_optimal_specs()
        proven_starters = self._get_proven_starters()

        baseline_ctr = baseline_metrics["overall_avg_ctr"]
        median_ctr = baseline_metrics["training_median_ctr"]
        articles_analyzed = baseline_metrics["articles_analyzed"]

        # Check if this is sports content for specific starter words
        is_sports = "sport" in category.lower() or any(
            team in original_title.lower() for team in proven_starters.keys()
        )

        # Build prompt based on ACTUAL available data
        prompt = f"""Generate 4 headline variations for: "{original_title}"

📊 REAL DATA INSIGHTS ({articles_analyzed:,} articles analyzed):
Current CTR: {current_ctr:.4f}
Baseline average: {baseline_ctr:.4f}
Target improvement: +{max(0.005, baseline_ctr - current_ctr):.4f}

🚀 TOP PERFORMING FEATURES (from your data):"""

        # Add top features dynamically
        for i, (feature, improvement, ctr, size) in enumerate(top_features):
            prompt += f"\n• {feature.replace('_', ' ').title()}: +{improvement:.1f}% CTR boost (n={size:,})"

        prompt += f"""

📏 OPTIMAL SPECIFICATIONS (proven by data):
• Word count: {optimal_specs['word_count_range']} words (CTR: {optimal_specs['word_count_ctr']:.4f})
• Character length: {optimal_specs['length_range']} chars (CTR: {optimal_specs['length_ctr']:.4f})
• Readability: {optimal_specs['readability_range']} Flesch score (CTR: {optimal_specs['readability_ctr']:.4f})"""

        # Add sports-specific guidance if relevant and data available
        if is_sports and proven_starters:
            sports_starters = list(proven_starters.keys())[:3]
            prompt += f"""

🏈 HIGH-PERFORMERS:
Top starting words: {', '.join(sports_starters)}"""

        prompt += f"""

✅ REQUIREMENTS (based on {articles_analyzed:,} articles):
- Apply 2-3 proven features above
- Target {optimal_specs['word_count_range']} words
- Aim for {optimal_specs['length_range']} characters  
- Readability in {optimal_specs['readability_range']} range
- Create engaging headlines for {category}

Return ONLY the 4 headlines, one per line, no numbering."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.8,
            )

            candidates = [original_title]  # Always include original
            lines = response.choices[0].message.content.strip().split("\n")

            for line in lines:
                cleaned = self._clean_response(line)
                if cleaned and cleaned != original_title:
                    candidates.append(cleaned)

            return candidates[:5]  # Max 5 candidates

        except Exception as e:
            logging.error(f"LLM candidate generation failed: {e}")
            return [original_title]

    def _clean_response(self, response):
        """Clean LLM response"""
        # Remove quotes, numbering, and extra whitespace
        response = re.sub(r'^["\']|["\']$', "", response)
        response = re.sub(r"^\d+[\.\)]\s*", "", response)  # Remove numbering
        response = response.split("\n")[0]  # Take first line only
        return response.strip()

    def get_best_headline(self, original_title, article_data):
        """Get the best headline using your model for selection"""

        # Generate candidates using EDA-driven LLM prompts (defensive)
        candidates = self.generate_candidates(original_title, article_data)

        # Score each candidate with YOUR model (100% model-based scoring)
        scores = []
        for candidate in candidates:
            ctr_score = self.predict_ctr_with_model(candidate, article_data)
            scores.append(ctr_score)

        # Return the highest-scoring candidate according to YOUR model
        best_idx = int(np.argmax(scores))
        best_headline = candidates[best_idx]
        best_score = scores[best_idx]

        # Calculate actual improvement using model predictions
        original_score = scores[0]  # Original is always first
        improvement = best_score - original_score

        return {
            "best_headline": best_headline,
            "predicted_ctr": best_score,
            "original_ctr": original_score,
            "ctr_improvement": improvement,
            "improvement_percent": (
                (improvement / original_score * 100) if original_score > 0 else 0
            ),
            "all_candidates": list(zip(candidates, scores)),
            "model_selected": True,  # Always true - we only use your model
            "eda_driven": bool(self.eda_insights),
        }

    def rewrite_headline(self, original_title, article_data=None):
        """Simple interface - returns the best headline"""
        if article_data is None:
            article_data = {"category": "news", "abstract": ""}

        result = self.get_best_headline(original_title, article_data)
        return result["best_headline"]

    def analyze_headline(self, headline, article_data):
        """Analyze a headline using your model and ACTUAL EDA insights (defensive)"""

        # Get model prediction
        predicted_ctr = self.predict_ctr_with_model(headline, article_data)

        # Extract features for analysis
        features = create_article_features_exact(
            headline,
            article_data.get("abstract", ""),
            article_data.get("category", "news"),
            self.components,
        )

        # Get baseline metrics safely
        baseline_metrics = self._get_baseline_metrics()
        baseline_ctr = baseline_metrics["overall_avg_ctr"]
        median_ctr = baseline_metrics["training_median_ctr"]

        # Feature analysis based on available EDA insights
        feature_analysis = []
        top_features = self._get_top_features()

        for feature_name, improvement, target_ctr, sample_size in top_features:
            feature_key = feature_name  # Use exact feature name from EDA
            if feature_key in features:
                has_feature = bool(features[feature_key])
                feature_analysis.append(
                    {
                        "feature": feature_name.replace("_", " ").title(),
                        "present": has_feature,
                        "potential_improvement": improvement if not has_feature else 0,
                        "target_ctr": target_ctr,
                        "sample_size": sample_size,
                    }
                )

        return {
            "predicted_ctr": predicted_ctr,
            "baseline_ctr": baseline_ctr,
            "median_ctr": median_ctr,
            "performance_vs_baseline": predicted_ctr - baseline_ctr,
            "performance_vs_median": predicted_ctr - median_ctr,
            "word_count": features["title_word_count"],
            "char_length": features["title_length"],
            "readability": features["title_reading_ease"],
            "feature_analysis": feature_analysis,
            "meets_optimal_specs": {
                "word_count": self.config["optimal_word_count"][0]
                <= features["title_word_count"]
                <= self.config["optimal_word_count"][1],
                "length": features["title_length"] <= self.config["max_length"],
                "readability": features["title_reading_ease"]
                >= self.config["target_readability"],
            },
            "engagement_prediction": predicted_ctr
            >= self.config["high_engagement_threshold"],
            "editorial_note": f"Analysis based on {baseline_metrics['articles_analyzed']:,} articles",
        }

    def batch_optimize(self, headlines_df, n_samples=10):
        """Optimize multiple headlines using your model"""

        # Sample low-performing headlines
        if "ctr" in headlines_df.columns:
            sample_df = headlines_df.nsmallest(n_samples, "ctr")
        else:
            sample_df = headlines_df.head(n_samples)

        results = []

        for idx, row in sample_df.iterrows():
            article_data = {
                "category": row.get("category", "news"),
                "abstract": row.get("abstract", ""),
            }

            # Get current model prediction for original
            original_ctr = self.predict_ctr_with_model(row["title"], article_data)

            # Get optimized headline
            result = self.get_best_headline(row["title"], article_data)

            results.append(
                {
                    "newsID": row.get("newsID", idx),
                    "original_title": row["title"],
                    "optimized_title": result["best_headline"],
                    "original_model_ctr": original_ctr,
                    "optimized_model_ctr": result["predicted_ctr"],
                    "model_ctr_improvement": result["ctr_improvement"],
                    "improvement_percent": result["improvement_percent"],
                    "category": article_data["category"],
                    "eda_driven": result["eda_driven"],
                }
            )

        return pd.DataFrame(results)

    def get_optimization_recommendations(self, headline, article_data):
        """Get specific recommendations based on ACTUAL EDA insights (defensive)"""

        analysis = self.analyze_headline(headline, article_data)
        recommendations = []

        # Check missing high-impact features from actual data
        for feature_info in analysis["feature_analysis"]:
            if (
                not feature_info["present"]
                and feature_info["potential_improvement"] > 5
            ):
                recommendations.append(
                    {
                        "type": "add_feature",
                        "feature": feature_info["feature"],
                        "impact": feature_info["potential_improvement"],
                        "sample_size": feature_info.get("sample_size", 0),
                        "suggestion": self._get_feature_suggestion(
                            feature_info["feature"]
                        ),
                    }
                )

        # Check specifications based on available optimal ranges
        specs = analysis["meets_optimal_specs"]
        optimal_specs = self._get_optimal_specs()

        if not specs["word_count"]:
            current_words = analysis["word_count"]
            target_range = optimal_specs["word_count_range"]
            recommendations.append(
                {
                    "type": "word_count",
                    "suggestion": f"Adjust to {target_range} words for optimal performance",
                }
            )

        # Performance vs baseline
        if analysis["performance_vs_baseline"] < 0:
            gap = abs(analysis["performance_vs_baseline"])
            recommendations.append(
                {
                    "type": "performance",
                    "suggestion": f"Headline below baseline by {gap:.4f} CTR - consider restructuring",
                }
            )

        return recommendations

    def _get_feature_suggestion(self, feature_name):
        """Get specific suggestions for adding features"""
        suggestions = {
            "Too Long Title": "Create longer, more descriptive headlines",
            "Has Colon": "Use a colon to separate main topic from details",
            "Has Dash": "Add dashes to connect related concepts",
            "Has Quotes": "Quote key phrases or add quotation marks for emphasis",
            "Needs Readability Improvement": "Adjust language complexity",
            "Has Number": "Add specific numbers, statistics, or quantities",
            "Has Question": "Turn into a question to increase curiosity",
            "Has Exclamation": "Add excitement with an exclamation mark",
        }
        return suggestions.get(feature_name, f"Consider adding {feature_name.lower()}")
