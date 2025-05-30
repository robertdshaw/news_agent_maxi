import os
import re
import pandas as pd
import numpy as np
import logging
from openai import OpenAI
from streamlit_app import create_article_features_exact, load_preprocessing_components


class EfficientLLMHeadlineRewriter:
    """Efficient LLM headline rewriter using your XGBoost model and real EDA insights"""

    def __init__(
        self, model_pipeline, components=None, llm_client=None, eda_insights_path=None
    ):
        """Initialize with your trained model and preprocessing components"""
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

        # Configuration based on your ACTUAL EDA findings
        self.config = {
            "optimal_word_count": (15, 20),  # Your data shows 15-20 words optimal
            "max_length": 100,  # 80-100 chars optimal in your data
            "target_readability": 40,  # 30-50 readability range optimal
            "high_engagement_threshold": 0.041238,  # Your overall average CTR
        }

        logging.info("✅ Efficient LLM Headline Rewriter initialized")
        logging.info(f"📊 Using XGBoost model for 100% of CTR predictions")
        logging.info(
            f"🎯 EDA insights: {'Loaded' if self.eda_insights else 'Using defaults'}"
        )

    def _load_eda_insights(self, eda_insights_path):
        """Load EDA insights from analysis or use defaults"""
        if eda_insights_path and os.path.exists(eda_insights_path):
            try:
                import json

                with open(eda_insights_path, "r") as f:
                    insights = json.load(f)
                logging.info(f"✅ Loaded EDA insights from {eda_insights_path}")
                return insights
            except Exception as e:
                logging.warning(f"Could not load EDA insights: {e}")

        # Your ACTUAL EDA findings as defaults
        return {
            "top_features_by_impact": [
                ("too_long_title", 22.0, 0.0474, 7438),
                ("has_colon", 16.6, 0.0465, 5539),
                ("has_dash", 5.7, 0.0432, 4187),
                ("has_quotes", 5.1, 0.0427, 8030),
                ("needs_readability_improvement", 4.6, 0.0421, 13814),
            ],
            "optimal_specifications": {
                "word_count_range": "15-20",
                "word_count_ctr": 0.0481,
                "length_range": "80-100",
                "length_ctr": 0.0469,
                "readability_range": "30-50",
                "readability_ctr": 0.0433,
            },
            "baseline_metrics": {
                "overall_avg_ctr": 0.041238,
                "training_median_ctr": 0.019231,
                "articles_analyzed": 27046,
            },
            "proven_starters": {
                "patriots": {"avg_ctr": 0.1384, "count": 20},
                "steelers": {"avg_ctr": 0.1207, "count": 35},
                "eagles": {"avg_ctr": 0.1175, "count": 23},
                "browns": {"avg_ctr": 0.1129, "count": 39},
                "mike": {"avg_ctr": 0.1098, "count": 25},
            },
            "editorial_effectiveness": {
                "correlation_with_ctr": -0.002,
                "high_score_engagement_lift": -4.6,
            },
        }

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
        """Generate headline candidates using LLM with YOUR ACTUAL EDA insights"""

        category = article_data.get("category", "news")
        current_ctr = self.predict_ctr_with_model(original_title, article_data)

        # Use your actual baseline metrics
        baseline_ctr = self.eda_insights["baseline_metrics"]["overall_avg_ctr"]
        median_ctr = self.eda_insights["baseline_metrics"]["training_median_ctr"]

        # Get your top performing features
        top_features = self.eda_insights["top_features_by_impact"][:3]

        # Get your optimal specifications
        optimal_specs = self.eda_insights["optimal_specifications"]

        # Check if this is sports content for specific starter words
        is_sports = "sport" in category.lower() or any(
            team in original_title.lower()
            for team in self.eda_insights["proven_starters"].keys()
        )

        # Build prompt based on YOUR ACTUAL findings
        prompt = f"""Generate 4 headline variations for: "{original_title}"

📊 REAL DATA INSIGHTS (27,046 articles analyzed):
Current CTR: {current_ctr:.4f}
Baseline average: {baseline_ctr:.4f}
Target improvement: +{max(0.005, baseline_ctr - current_ctr):.4f}

🚀 TOP PERFORMING FEATURES (from your data):
• {top_features[0][0].replace('_', ' ').title()}: +{top_features[0][1]:.1f}% CTR boost (n={top_features[0][3]:,})
• {top_features[1][0].replace('_', ' ').title()}: +{top_features[1][1]:.1f}% CTR boost (n={top_features[1][3]:,})
• {top_features[2][0].replace('_', ' ').title()}: +{top_features[2][1]:.1f}% CTR boost (n={top_features[2][3]:,})

📏 OPTIMAL SPECIFICATIONS (proven by data):
• Word count: {optimal_specs['word_count_range']} words (CTR: {optimal_specs['word_count_ctr']:.4f})
• Character length: {optimal_specs['length_range']} chars (CTR: {optimal_specs['length_ctr']:.4f})
• Readability: {optimal_specs['readability_range']} Flesch score (CTR: {optimal_specs['readability_ctr']:.4f})"""

        # Add sports-specific guidance if relevant
        if is_sports:
            sports_starters = list(self.eda_insights["proven_starters"].keys())[:3]
            prompt += f"""

🏈 SPORTS HIGH-PERFORMERS:
Top starting words: {', '.join(sports_starters)} (CTR: 0.11-0.14)"""

        prompt += f"""

✅ REQUIREMENTS (based on {self.eda_insights['baseline_metrics']['articles_analyzed']:,} articles):
- Apply 2-3 proven features above
- Target {optimal_specs['word_count_range']} words (longer headlines perform better)
- Aim for {optimal_specs['length_range']} characters  
- Lower readability OK (30-50 range optimal)
- Create descriptive, detailed headlines for {category}

Return ONLY the 4 headlines, one per line, no numbering."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,  # Increased for longer headlines
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

        # Generate candidates using EDA-driven LLM prompts
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
        """Analyze a headline using your model and ACTUAL EDA insights"""

        # Get model prediction
        predicted_ctr = self.predict_ctr_with_model(headline, article_data)

        # Extract features for analysis
        features = create_article_features_exact(
            headline,
            article_data.get("abstract", ""),
            article_data.get("category", "news"),
            self.components,
        )

        # Use your actual baseline metrics
        baseline_ctr = self.eda_insights["baseline_metrics"]["overall_avg_ctr"]
        median_ctr = self.eda_insights["baseline_metrics"]["training_median_ctr"]

        # Feature analysis based on YOUR EDA insights
        feature_analysis = []
        for feature_name, improvement, target_ctr, sample_size in self.eda_insights[
            "top_features_by_impact"
        ]:
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
            "editorial_note": f"Editorial scoring shows {self.eda_insights['editorial_effectiveness']['correlation_with_ctr']:.3f} correlation with CTR",
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
        """Get specific recommendations based on YOUR ACTUAL EDA insights"""

        analysis = self.analyze_headline(headline, article_data)
        recommendations = []

        # Check missing high-impact features from YOUR data
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

        # Check specifications based on YOUR optimal ranges
        specs = analysis["meets_optimal_specs"]
        if not specs["word_count"]:
            current_words = analysis["word_count"]
            optimal_range = self.config["optimal_word_count"]
            if current_words < optimal_range[0]:
                recommendations.append(
                    {
                        "type": "word_count",
                        "suggestion": f"Expand to {optimal_range[0]}-{optimal_range[1]} words (your data shows longer headlines perform better)",
                    }
                )
            elif current_words > optimal_range[1]:
                recommendations.append(
                    {
                        "type": "word_count",
                        "suggestion": f"Consider keeping longer format - your data shows {optimal_range[0]}-{optimal_range[1]} words optimal",
                    }
                )

        # Readability recommendation based on YOUR data (30-50 optimal, not 60+)
        if analysis["readability"] > 50:
            recommendations.append(
                {
                    "type": "readability",
                    "suggestion": f"Consider more complex language - your data shows 30-50 readability optimal (current: {analysis['readability']:.1f})",
                }
            )

        # Performance vs baseline
        if analysis["performance_vs_baseline"] < 0:
            gap = abs(analysis["performance_vs_baseline"])
            recommendations.append(
                {
                    "type": "performance",
                    "suggestion": f"Headline below baseline by {gap:.4f} CTR - consider major restructuring",
                }
            )

        return recommendations

    def _get_feature_suggestion(self, feature_name):
        """Get specific suggestions for adding features based on YOUR data"""
        suggestions = {
            "Too Long Title": "Create longer, more descriptive headlines (80-100+ characters)",
            "Has Colon": "Use a colon to separate main topic from details or explanation",
            "Has Dash": "Add dashes to connect related concepts or create emphasis",
            "Has Quotes": "Quote key phrases, statements, or add quotation marks for emphasis",
            "Needs Readability Improvement": "Use more complex, detailed language (30-50 readability optimal)",
            "Has Number": "Add specific numbers, statistics, or quantities",
            "Has Question": "Turn into a question to increase curiosity",
            "Has Exclamation": "Add excitement with an exclamation mark",
        }
        return suggestions.get(feature_name, f"Consider adding {feature_name.lower()}")


# import os
# import re
# import time
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from textstat import flesch_reading_ease
# import logging
# from pathlib import Path
# import json
# from openai import OpenAI


# class LLMHeadlineRewriter:
#     """LLM-powered headline rewriting based on EDA findings and editorial guidelines"""

#     def __init__(self):
#         self.api_available = self._check_openai_api_and_init_client()
#         self.request_count = 0
#         self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
#         self.editorial_guidelines = self._load_editorial_guidelines()
#         self.eda_insights = self._load_eda_insights()

#     def _load_editorial_guidelines(self):
#         """Load editorial guidelines based on EDA preprocessing findings"""
#         return {
#             "target_reading_ease": 60,
#             "optimal_word_count": (8, 12),
#             "max_title_length": 75,
#             "high_engagement_threshold": 0.05,
#             "engagement_features": {
#                 "questions": 0.15,
#                 "numbers": 0.20,
#                 "colons": 0.10,
#                 "quotes": 0.08,
#                 "exclamation": 0.05,
#             },
#             "category_performance": {
#                 "sports": {"avg_ctr": 0.045, "high_engagement_rate": 0.18},
#                 "entertainment": {"avg_ctr": 0.042, "high_engagement_rate": 0.16},
#                 "finance": {"avg_ctr": 0.038, "high_engagement_rate": 0.14},
#                 "news": {"avg_ctr": 0.035, "high_engagement_rate": 0.12},
#             },
#         }

#     def _load_eda_insights(self):
#         """Load EDA insights from preprocessing metadata"""
#         try:
#             prep_dir = Path("data/preprocessed/processed_data")
#             metadata_file = prep_dir / "preprocessing_metadata.json"

#             if metadata_file.exists():
#                 with open(metadata_file, "r") as f:
#                     metadata = json.load(f)

#                 return {
#                     "high_engagement_rate": metadata.get("target_statistics", {}).get(
#                         "high_engagement_rate", 0.15
#                     ),
#                     "mean_ctr": metadata.get("target_statistics", {}).get(
#                         "mean_ctr", 0.035
#                     ),
#                     "median_ctr": metadata.get("target_statistics", {}).get(
#                         "median_ctr", 0.030
#                     ),
#                     "ctr_threshold": metadata.get("target_statistics", {}).get(
#                         "ctr_threshold", 0.15
#                     ),
#                     "features_count": metadata.get("feature_categories", {}).get(
#                         "total_features", 100
#                     ),
#                     "pca_variance_explained": metadata.get(
#                         "pca_variance_explained", 0.8
#                     ),
#                 }
#             else:
#                 logging.warning("EDA metadata not found, using defaults")
#                 return self._default_eda_insights()

#         except Exception as e:
#             logging.error(f"Error loading EDA insights: {e}")
#             return self._default_eda_insights()

#     def _default_eda_insights(self):
#         """Default EDA insights when metadata unavailable"""
#         return {
#             "high_engagement_rate": 0.15,
#             "mean_ctr": 0.035,
#             "median_ctr": 0.030,
#             "ctr_threshold": 0.15,
#             "features_count": 100,
#             "pca_variance_explained": 0.8,
#         }

#     def _check_openai_api_and_init_client(self):
#         """Check if OpenAI API is available and initialize client"""
#         try:
#             api_key = os.getenv("OPENAI_API_KEY")
#             if api_key:
#                 self.client = OpenAI(api_key=api_key)
#                 logging.info("OpenAI client initialized successfully.")
#                 return True
#             else:
#                 logging.warning(
#                     "OpenAI API key not found in environment variables. LLM rewriting will use fallback."
#                 )
#                 return False
#         except ImportError:
#             logging.warning(
#                 "OpenAI library not installed. LLM rewriting will use fallback."
#             )
#             return False
#         except Exception as e:
#             logging.error(
#                 f"Error initializing OpenAI client: {e}. LLM rewriting will use fallback."
#             )
#             return False

#     def rewrite_headline(
#         self, original_title, strategy="comprehensive", article_data=None
#     ):
#         """Rewrite a headline using specified strategy"""

#         if not self.api_available or not self.client:
#             logging.info(
#                 f"API not available or client not initialized for '{original_title}'. Using fallback rewrite for strategy '{strategy}'."
#             )
#             return self._fallback_rewrite(original_title, strategy)

#         try:
#             article_data = article_data or {}
#             prompt = self._create_prompt(original_title, strategy, article_data)

#             response = self.client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are an expert news editor. Create engaging headlines while maintaining accuracy.",
#                     },
#                     {"role": "user", "content": prompt},
#                 ],
#                 max_tokens=100,
#                 temperature=0.7,
#             )

#             self.request_count += 1
#             rewritten = response.choices[0].message.content.strip()
#             rewritten = self._clean_response(rewritten)

#             time.sleep(0.5)  # Rate limiting
#             return rewritten

#         except Exception as e:
#             logging.error(
#                 f"Error rewriting headline '{original_title}' with OpenAI: {e}"
#             )
#             logging.info(
#                 f"Using fallback rewrite for '{original_title}' strategy '{strategy}'."
#             )
#             return self._fallback_rewrite(original_title, strategy)

#     def _create_prompt(self, original_title, strategy, article_data):
#         """Create data-driven prompts based on EDA findings and editorial guidelines"""

#         category = article_data.get("category", "news")
#         # Use model-predicted CTR if available, otherwise use provided CTR
#         current_ctr = article_data.get("ctr", self.eda_insights["mean_ctr"])
#         current_readability = article_data.get(
#             "readability", flesch_reading_ease(original_title)
#         )

#         # Get category-specific benchmarks
#         category_benchmark = self.editorial_guidelines["category_performance"].get(
#             category, self.editorial_guidelines["category_performance"]["news"]
#         )

#         # Calculate improvement targets based on EDA insights and model predictions
#         target_ctr_improvement = max(
#             0.01, self.eda_insights["median_ctr"] - current_ctr
#         )
#         readability_gap = max(
#             0, self.editorial_guidelines["target_reading_ease"] - current_readability
#         )

#         base_context = f"""
# EDITORIAL CONTEXT:
# - Category: {category} (avg CTR: {category_benchmark['avg_ctr']:.3f})
# - Current headline CTR: {current_ctr:.4f} vs median {self.eda_insights['median_ctr']:.4f}
# - Current readability: {current_readability:.1f} (target: {self.editorial_guidelines['target_reading_ease']})
# - High engagement threshold: {self.eda_insights['ctr_threshold']:.3f}

# OPTIMIZATION TARGETS:
# - CTR improvement needed: +{target_ctr_improvement:.4f}
# - Readability improvement: +{readability_gap:.1f} points
# - Word count: {self.editorial_guidelines['optimal_word_count'][0]}-{self.editorial_guidelines['optimal_word_count'][1]} words
# - Character limit: {self.editorial_guidelines['max_title_length']} chars
# """

#         strategy_prompts = {
#             "readability": f"""
# {base_context}

# READABILITY OPTIMIZATION:
# Original: "{original_title}"

# Based on our data analysis, headlines with reading ease scores above {self.editorial_guidelines['target_reading_ease']} perform {((self.editorial_guidelines['engagement_features']['numbers'] * 100)):.0f}% better.

# Requirements:
# - Simplify complex words and phrases
# - Target reading ease: {self.editorial_guidelines['target_reading_ease']}+
# - Maintain {self.editorial_guidelines['optimal_word_count'][0]}-{self.editorial_guidelines['optimal_word_count'][1]} words
# - Keep factual accuracy for {category} content

# Return only the optimized headline.
# """,
#             "engagement": f"""
# {base_context}

# ENGAGEMENT OPTIMIZATION:
# Original: "{original_title}"

# Our analysis shows these features boost engagement:
# - Numbers: +{self.editorial_guidelines['engagement_features']['numbers']*100:.0f}% CTR
# - Questions: +{self.editorial_guidelines['engagement_features']['questions']*100:.0f}% CTR
# - Colons: +{self.editorial_guidelines['engagement_features']['colons']*100:.0f}% CTR

# Current performance gap: {target_ctr_improvement:.4f} CTR points to reach median.

# Requirements:
# - Add high-impact engagement elements (numbers, questions)
# - Create curiosity while staying truthful
# - Target CTR improvement: +{target_ctr_improvement:.4f}
# - Under {self.editorial_guidelines['max_title_length']} characters

# Return only the optimized headline.
# """,
#             "structure": f"""
# {base_context}

# STRUCTURE OPTIMIZATION:
# Original: "{original_title}"

# Data shows optimal structure patterns:
# - {self.editorial_guidelines['optimal_word_count'][0]}-{self.editorial_guidelines['optimal_word_count'][1]} words perform best
# - Front-loaded key information increases CTR by 15%
# - Active voice correlates with higher engagement

# Requirements:
# - Lead with most important information
# - Use active voice and strong verbs
# - Eliminate filler words
# - Target: {self.editorial_guidelines['optimal_word_count'][0]}-{self.editorial_guidelines['optimal_word_count'][1]} words exactly

# Return only the optimized headline.
# """,
#             "comprehensive": f"""
# {base_context}

# COMPREHENSIVE OPTIMIZATION:
# Original: "{original_title}"

# Multi-factor optimization based on our {self.eda_insights['features_count']} feature analysis:

# HIGH-IMPACT CHANGES:
# - Add numbers (CTR boost: +{self.editorial_guidelines['engagement_features']['numbers']*100:.0f}%)
# - Improve readability to {self.editorial_guidelines['target_reading_ease']}+
# - Optimize to {self.editorial_guidelines['optimal_word_count'][0]}-{self.editorial_guidelines['optimal_word_count'][1]} words
# - Front-load key information

# CATEGORY-SPECIFIC ({category}):
# - Target CTR: {category_benchmark['avg_ctr']:.4f}
# - High engagement rate: {category_benchmark['high_engagement_rate']*100:.0f}%

# Requirements:
# - Maximize engagement while maintaining accuracy
# - Hit all structural and readability targets
# - Create urgency/curiosity appropriate for {category}
# - Stay under {self.editorial_guidelines['max_title_length']} characters

# Return only the final optimized headline.
# """,
#         }

#         return strategy_prompts.get(strategy, strategy_prompts["comprehensive"])

#     def _clean_response(self, response):
#         """Clean LLM response"""
#         response = re.sub(r'^["\']|["\']$', "", response)
#         response = response.split("\n")[0]
#         return response.strip()

#     def _fallback_rewrite(self, original_title, strategy):
#         """Data-driven rule-based fallback using EDA insights"""

#         words = original_title.split()
#         word_count = len(words)
#         target_min, target_max = self.editorial_guidelines["optimal_word_count"]

#         # Apply EDA-based transformations
#         if strategy == "readability":
#             # Simplify and truncate to optimal length
#             simplified_words = []
#             for word in words[:target_max]:
#                 # Replace complex words with simpler alternatives
#                 if len(word) > 12:  # Long words hurt readability
#                     continue
#                 simplified_words.append(word.lower() if word.isupper() else word)
#             return " ".join(simplified_words[:target_max])

#         elif strategy == "engagement":
#             # Add engagement elements based on EDA findings
#             if word_count < target_min:
#                 return original_title  # Too short already

#             # Add number if missing (20% CTR boost from EDA)
#             if not any(c.isdigit() for c in original_title):
#                 return (
#                     f"5 {original_title}"
#                     if len(f"5 {original_title}")
#                     <= self.editorial_guidelines["max_title_length"]
#                     else original_title
#                 )

#             # Add question if missing (15% CTR boost from EDA)
#             if not original_title.endswith("?") and word_count <= target_max:
#                 return (
#                     f"Why {original_title}?"
#                     if len(f"Why {original_title}?")
#                     <= self.editorial_guidelines["max_title_length"]
#                     else original_title
#                 )

#             return original_title

#         elif strategy == "structure":
#             # Optimize word order and count based on EDA structure analysis
#             if word_count > target_max:
#                 # Keep most important words (assume first are most important)
#                 return " ".join(words[:target_max])
#             elif word_count < target_min:
#                 # Add descriptive words
#                 return (
#                     f"{original_title} - Key Details"
#                     if len(f"{original_title} - Key Details")
#                     <= self.editorial_guidelines["max_title_length"]
#                     else original_title
#                 )
#             return original_title

#         else:  # comprehensive
#             # Apply multiple optimizations based on EDA feature importance
#             result = original_title

#             # Length optimization
#             if word_count > target_max:
#                 result = " ".join(words[:target_max])

#             # Add engagement elements if missing and space allows
#             if not any(c.isdigit() for c in result) and len(result) < 60:
#                 result = f"5 {result}"

#             return result[: self.editorial_guidelines["max_title_length"]]

#     def create_rewrite_variants(self, original_title, article_data=None):
#         """Create multiple rewrite variants using different strategies"""

#         strategies = ["readability", "engagement", "structure", "comprehensive"]
#         variants = {}

#         for strategy in strategies:
#             variant = self.rewrite_headline(original_title, strategy, article_data)
#             variants[strategy] = variant

#         return variants

#     def evaluate_rewrite_quality(self, original_title, rewritten_title):
#         """Evaluate quality improvements based on EDA feature importance"""

#         def extract_eda_features(title):
#             """Extract features that correlate with engagement from EDA"""
#             return {
#                 "length": len(title),
#                 "word_count": len(title.split()),
#                 "readability": flesch_reading_ease(title),
#                 "has_question": "?" in title,
#                 "has_number": any(c.isdigit() for c in title),
#                 "has_colon": ":" in title,
#                 "has_quotes": any(q in title for q in ['"', "'", '"', '"']),
#                 "has_exclamation": "!" in title,
#                 "title_upper_ratio": (
#                     sum(c.isupper() for c in title) / len(title) if title else 0
#                 ),
#                 "avg_word_length": (
#                     np.mean([len(word) for word in title.split()])
#                     if title.split()
#                     else 0
#                 ),
#                 "embedding": self.embedder.encode([title])[0],
#             }

#         original_features = extract_eda_features(original_title)
#         rewritten_features = extract_eda_features(rewritten_title)

#         # Calculate semantic similarity
#         similarity = np.dot(
#             original_features["embedding"], rewritten_features["embedding"]
#         )

#         # Calculate engagement score improvement based on EDA feature weights
#         engagement_weights = self.editorial_guidelines["engagement_features"]

#         original_engagement = (
#             original_features["has_question"] * engagement_weights["questions"]
#             + original_features["has_number"] * engagement_weights["numbers"]
#             + original_features["has_colon"] * engagement_weights["colons"]
#             + original_features["has_quotes"] * engagement_weights["quotes"]
#             + original_features["has_exclamation"] * engagement_weights["exclamation"]
#         )

#         rewritten_engagement = (
#             rewritten_features["has_question"] * engagement_weights["questions"]
#             + rewritten_features["has_number"] * engagement_weights["numbers"]
#             + rewritten_features["has_colon"] * engagement_weights["colons"]
#             + rewritten_features["has_quotes"] * engagement_weights["quotes"]
#             + rewritten_features["has_exclamation"] * engagement_weights["exclamation"]
#         )

#         # Word count optimization score
#         target_min, target_max = self.editorial_guidelines["optimal_word_count"]
#         original_word_score = (
#             1.0 if target_min <= original_features["word_count"] <= target_max else 0.5
#         )
#         rewritten_word_score = (
#             1.0 if target_min <= rewritten_features["word_count"] <= target_max else 0.5
#         )

#         improvements = {
#             "readability_improvement": rewritten_features["readability"]
#             - original_features["readability"],
#             "engagement_score_improvement": rewritten_engagement - original_engagement,
#             "word_count_optimization": rewritten_word_score - original_word_score,
#             "length_change": rewritten_features["length"] - original_features["length"],
#             "semantic_similarity": float(similarity),
#             "meets_length_target": rewritten_features["length"]
#             <= self.editorial_guidelines["max_title_length"],
#             "meets_word_target": target_min
#             <= rewritten_features["word_count"]
#             <= target_max,
#             "meets_readability_target": rewritten_features["readability"]
#             >= self.editorial_guidelines["target_reading_ease"],
#             "overall_quality_score": self._calculate_eda_quality_score(
#                 rewritten_features
#             ),
#             "predicted_ctr_improvement": self._estimate_ctr_improvement(
#                 original_features, rewritten_features
#             ),
#         }

#         return improvements

#     def _calculate_eda_quality_score(self, features):
#         """Calculate quality score based on EDA feature importance rankings"""

#         score = 0

#         # Word count optimization (highest weight from EDA)
#         target_min, target_max = self.editorial_guidelines["optimal_word_count"]
#         if target_min <= features["word_count"] <= target_max:
#             score += 35  # Highest impact from EDA
#         elif target_min - 2 <= features["word_count"] <= target_max + 2:
#             score += 25
#         else:
#             score += 10

#         # Readability (second highest correlation from EDA)
#         if features["readability"] >= self.editorial_guidelines["target_reading_ease"]:
#             score += 25
#         elif features["readability"] >= 30:
#             score += 15
#         else:
#             score += 5

#         # Engagement features (weighted by EDA findings)
#         engagement_weights = self.editorial_guidelines["engagement_features"]

#         if features["has_number"]:
#             score += int(engagement_weights["numbers"] * 100)  # ~20 points
#         if features["has_question"]:
#             score += int(engagement_weights["questions"] * 100)  # ~15 points
#         if features["has_colon"]:
#             score += int(engagement_weights["colons"] * 100)  # ~10 points

#         # Length penalty (from EDA analysis)
#         if features["length"] > self.editorial_guidelines["max_title_length"]:
#             score -= 15
#         elif features["length"] > 100:
#             score -= 25

#         return min(100, max(0, score))

#     def _estimate_ctr_improvement(self, original_features, rewritten_features):
#         """Estimate CTR improvement based on EDA correlations and model insights"""

#         improvement = 0
#         weights = self.editorial_guidelines["engagement_features"]

#         # Feature-based improvements from EDA analysis
#         if rewritten_features["has_number"] and not original_features["has_number"]:
#             improvement += weights["numbers"] * self.eda_insights["mean_ctr"]

#         if rewritten_features["has_question"] and not original_features["has_question"]:
#             improvement += weights["questions"] * self.eda_insights["mean_ctr"]

#         if rewritten_features["has_colon"] and not original_features["has_colon"]:
#             improvement += weights["colons"] * self.eda_insights["mean_ctr"]

#         # Readability improvement (correlates with model performance)
#         readability_improvement = (
#             rewritten_features["readability"] - original_features["readability"]
#         )
#         if readability_improvement > 0:
#             # Scale based on how readability improvements correlate with CTR
#             improvement += (readability_improvement / 100) * 0.01

#         # Word count optimization (model shows this improves predictions)
#         target_min, target_max = self.editorial_guidelines["optimal_word_count"]
#         original_optimal = target_min <= original_features["word_count"] <= target_max
#         rewritten_optimal = target_min <= rewritten_features["word_count"] <= target_max

#         if rewritten_optimal and not original_optimal:
#             improvement += 0.005  # 0.5% CTR improvement for optimal word count

#         return improvement

#     def batch_rewrite_headlines(self, headlines_df, n_samples=10):
#         """Batch rewrite headlines for analysis"""

#         if not self.api_available:
#             logging.warning("API not available - using fallback rewrites")

#         # Sample low-performing articles
#         sample_df = (
#             headlines_df.nsmallest(n_samples, "ctr")
#             if "ctr" in headlines_df.columns
#             else headlines_df.head(n_samples)
#         )

#         results = []

#         for idx, row in sample_df.iterrows():
#             # Use model-predicted CTR if available, otherwise use actual CTR
#             article_ctr = row.get("ctr", self.eda_insights["mean_ctr"])

#             article_data = {
#                 "category": row.get("category", "news"),
#                 "ctr": article_ctr,  # This could be model-predicted or actual
#                 "abstract": row.get("abstract", ""),
#             }

#             variants = self.create_rewrite_variants(row["title"], article_data)

#             for strategy, rewritten in variants.items():
#                 quality_metrics = self.evaluate_rewrite_quality(row["title"], rewritten)

#                 results.append(
#                     {
#                         "newsID": row.get("newsID", idx),
#                         "original_title": row["title"],
#                         "strategy": strategy,
#                         "rewritten_title": rewritten,
#                         "original_ctr": article_ctr,
#                         "category": article_data["category"],
#                         **quality_metrics,
#                     }
#                 )

#         return pd.DataFrame(results)

#     def get_best_rewrite(self, original_title, article_data=None):
#         """Get the best rewrite based on EDA-driven quality metrics"""

#         variants = self.create_rewrite_variants(original_title, article_data)
#         best_variant = original_title
#         best_score = 0
#         best_metrics = {}

#         for strategy, rewritten in variants.items():
#             quality_metrics = self.evaluate_rewrite_quality(original_title, rewritten)

#             # Composite score based on EDA priorities
#             composite_score = (
#                 quality_metrics["overall_quality_score"] * 0.4  # Base quality
#                 + quality_metrics["predicted_ctr_improvement"]
#                 * 1000
#                 * 0.3  # CTR impact (scaled)
#                 + quality_metrics["engagement_score_improvement"]
#                 * 100
#                 * 0.2  # Engagement features
#                 + quality_metrics["semantic_similarity"] * 100 * 0.1  # Maintain meaning
#             )

#             if composite_score > best_score:
#                 best_score = composite_score
#                 best_variant = rewritten
#                 best_metrics = quality_metrics
#                 best_metrics["strategy"] = strategy
#                 best_metrics["composite_score"] = composite_score

#         return {
#             "best_rewrite": best_variant,
#             "original_title": original_title,
#             "improvement_metrics": best_metrics,
#             "all_variants": variants,
#         }
