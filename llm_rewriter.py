import os
import re
import pandas as pd
import numpy as np
import logging
from openai import OpenAI
from feature_utils import create_article_features_exact, load_preprocessing_components

import os

# Load environment variables (works locally, ignored on Streamlit Cloud)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # dotenv not available (like on Streamlit Cloud)
    pass
except Exception:
    # Any other error, continue without dotenv
    pass


class EnhancedLLMHeadlineRewriter:
    """Enhanced multi-layer LLM headline rewriter with psychological triggers and feedback loops"""

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
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Load EDA insights if available
        self.eda_insights = self._load_eda_insights(eda_insights_path)

        # Configuration based on EDA findings (with safe defaults)
        self.config = {
            "optimal_word_count": (8, 12),
            "max_length": 100,
            "target_readability": 60,
            "high_engagement_threshold": self._get_safe_value(
                "baseline_metrics", "overall_avg_ctr", 0.041
            ),
            "enable_multi_layer": True,
            "max_iterations": 3,
        }

        # Psychological trigger mapping
        self.psychological_triggers = self._build_psychological_framework()

        # Persona profiles
        self.persona_profiles = self._build_persona_profiles()

        logging.info("Enhanced LLM Headline Rewriter initialized")
        logging.info(
            f"Multi-layer optimization: {'Enabled' if self.config['enable_multi_layer'] else 'Disabled'}"
        )

    def _build_psychological_framework(self):
        """Categorize EDA features by psychological triggers"""
        return {
            "CURIOSITY": {
                "triggers": ["has_question", "has_colon", "incomplete_info"],
                "psychology": "Creates information gaps that demand closure",
                "examples": [
                    "What happens when...",
                    "The secret behind:",
                    "5 things you don't know about...",
                ],
            },
            "URGENCY": {
                "triggers": ["has_number", "time_words", "breaking_news"],
                "psychology": "Scarcity and time pressure drive immediate action",
                "examples": ["Breaking:", "Just announced:", "24 hours left:"],
            },
            "CREDIBILITY": {
                "triggers": ["has_quotes", "has_number", "expert_source"],
                "psychology": "Authority and specificity build trust",
                "examples": ["Expert reveals:", "Study shows:", "Data confirms:"],
            },
            "EMOTION": {
                "triggers": ["has_exclamation", "emotional_words", "surprise_factor"],
                "psychology": "Emotional arousal increases engagement",
                "examples": ["Shocking:", "Amazing:", "Heartbreaking:"],
            },
        }

    def _build_persona_profiles(self):
        """Build audience persona profiles from EDA insights"""
        return {
            "sports_fans": {
                "keywords": ["sport", "game", "team", "player", "match"],
                "preferences": {
                    "style": "Direct, action-oriented, outcome-focused",
                    "triggers": ["URGENCY", "EMOTION"],
                    "language": "scores, wins, defeats, championships",
                },
            },
            "business_readers": {
                "keywords": ["business", "market", "company", "economy", "finance"],
                "preferences": {
                    "style": "Professional, data-driven, impact-focused",
                    "triggers": ["CREDIBILITY", "URGENCY"],
                    "language": "percentages, growth, analysis, trends",
                },
            },
            "general_news": {
                "keywords": ["news", "politics", "world", "society"],
                "preferences": {
                    "style": "Balanced, informative, comprehensive",
                    "triggers": ["CURIOSITY", "CREDIBILITY"],
                    "language": "reports, reveals, investigation, impact",
                },
            },
            "lifestyle": {
                "keywords": ["health", "lifestyle", "food", "travel", "entertainment"],
                "preferences": {
                    "style": "Personal, relatable, benefit-focused",
                    "triggers": ["CURIOSITY", "EMOTION"],
                    "language": "discover, transform, experience, enjoy",
                },
            },
        }

    def get_best_headline(self, original_headline, article_data):
        """Get best headline with stricter quality requirements"""

        # Generate candidates
        candidates = self.generate_candidates(original_headline, article_data)

        # Filter for quality
        meaningful_candidates = self.filter_meaningful_rewrites(
            original_headline,
            candidates,
            min_improvement=0.02,  # At least 2% CTR improvement
            min_word_diff=0.3,  # At least 30% different words
        )

        # If no meaningful improvements, return original
        if not meaningful_candidates:
            return {
                "best_headline": original_headline,
                "improvement_reason": "original_already_optimized",
                "all_candidates": candidates[:3],  # Show what was tried
            }

        # Return best meaningful candidate
        best = max(meaningful_candidates, key=lambda x: x[1])

        return {
            "best_headline": best[0],
            "predicted_ctr": best[1],
            "improvement_reason": "significant_optimization",
            "all_candidates": meaningful_candidates,
        }

    def filter_meaningful_rewrites(
        self, original, candidates, min_improvement=0.02, min_word_diff=0.3
    ):
        """Filter out low-quality rewrites"""
        filtered = []

        original_words = set(original.lower().split())

        for candidate, predicted_ctr in candidates:
            # Skip if too similar
            candidate_words = set(candidate.lower().split())
            word_overlap = len(original_words & candidate_words) / len(
                original_words | candidate_words
            )

            if word_overlap > (1 - min_word_diff):  # Too similar
                continue

            # Skip if improvement is too small
            if predicted_ctr < self.baseline_ctr + min_improvement:
                continue

            filtered.append((candidate, predicted_ctr))

        return filtered

    def _identify_persona(self, article_data):
        """Identify target persona from article data"""
        category = article_data.get("category", "news").lower()
        title = article_data.get("title", "").lower()
        abstract = article_data.get("abstract", "").lower()

        content = f"{category} {title} {abstract}"

        persona_scores = {}
        for persona, profile in self.persona_profiles.items():
            score = sum(1 for keyword in profile["keywords"] if keyword in content)
            persona_scores[persona] = score

        # Return highest scoring persona or default
        best_persona = max(persona_scores.items(), key=lambda x: x[1])
        return best_persona[0] if best_persona[1] > 0 else "general_news"

    def _categorize_features_by_psychology(self):
        """Categorize your EDA features by psychological impact"""
        top_features = self._get_top_features()
        categorized = {trigger: [] for trigger in self.psychological_triggers.keys()}

        # Map your EDA features to psychological categories
        feature_mapping = {
            "has_number": "URGENCY",
            "has_question": "CURIOSITY",
            "has_colon": "CURIOSITY",
            "has_quotes": "CREDIBILITY",
            "has_exclamation": "EMOTION",
            "has_dash": "CURIOSITY",
            "breaking_news": "URGENCY",
            "expert_mention": "CREDIBILITY",
        }

        for feature, improvement, ctr, size in top_features:
            psychological_category = feature_mapping.get(feature, "CREDIBILITY")
            categorized[psychological_category].append(
                {
                    "feature": feature,
                    "improvement": improvement,
                    "ctr": ctr,
                    "size": size,
                    "readable_name": feature.replace("_", " ").title(),
                }
            )

        return categorized

    def _generate_layer_1_candidates(self, original_title, article_data, persona):
        """Layer 1: Structure and EDA-based optimization"""

        current_ctr = self.predict_ctr_with_model(original_title, article_data)
        baseline_metrics = self._get_baseline_metrics()
        optimal_specs = self._get_optimal_specs()
        psychological_features = self._categorize_features_by_psychology()

        persona_profile = self.persona_profiles[persona]
        preferred_triggers = persona_profile["preferences"]["triggers"]

        prompt = f"""You are an expert headline optimizer specializing in {persona.replace('_', ' ')} content.

## TASK
Transform: "{original_title}"
Target audience: {persona_profile['preferences']['style']}
Current CTR: {current_ctr:.4f} | Target: {current_ctr + 0.01:.4f}

## PSYCHOLOGICAL FRAMEWORK (from {baseline_metrics['articles_analyzed']:,} articles)"""

        # Add psychological triggers relevant to this persona
        for trigger in preferred_triggers:
            if trigger in psychological_features and psychological_features[trigger]:
                prompt += f"\n\n**{trigger} TRIGGERS** ({self.psychological_triggers[trigger]['psychology']}):"
                for feature_data in psychological_features[trigger][
                    :2
                ]:  # Top 2 features per trigger
                    prompt += f"\n• {feature_data['readable_name']}: +{feature_data['improvement']:.1f}% CTR (n={feature_data['size']:,})"

                # Add examples
                prompt += f"\n• Examples: {', '.join(self.psychological_triggers[trigger]['examples'][:2])}"

        prompt += f"""

## STRUCTURAL REQUIREMENTS
• Word count: {optimal_specs['word_count_range']} words
• Character length: {optimal_specs['length_range']} chars
• Readability: {optimal_specs['readability_range']} Flesch score

## PERSONA-SPECIFIC GUIDANCE
Style: {persona_profile['preferences']['style']}
Language patterns: {persona_profile['preferences']['language']}

## LAYER 1 TASK
Generate 4 headlines focusing on STRUCTURE and proven features:
1. Apply 2-3 psychological triggers from above
2. Meet structural requirements
3. Maintain factual accuracy
4. Match persona style

Return ONLY 4 headlines, one per line:"""

        return self._call_llm(prompt, temperature=0.8)

    def _generate_layer_2_candidates(
        self, layer_1_results, original_title, article_data, persona
    ):
        """Layer 2: Emotional refinement based on model feedback"""

        # Get best performer from layer 1
        best_l1_headline = layer_1_results["best_headline"]
        best_l1_score = layer_1_results["best_score"]

        persona_profile = self.persona_profiles[persona]

        prompt = f"""You are optimizing emotional impact for {persona.replace('_', ' ')} audience.

## LAYER 1 RESULTS
Best headline: "{best_l1_headline}"
Predicted CTR: {best_l1_score:.4f}
Improvement over original: +{layer_1_results['improvement']:.4f}

## LAYER 2 TASK: BEAT THIS SCORE
Target audience psychology: {persona_profile['preferences']['style']}

Create 4 NEW variations that could score higher by:

**EMOTIONAL OPTIMIZATION:**
• Test different emotional hooks (curiosity vs urgency vs credibility)
• Vary intensity levels (subtle vs bold)
• Experiment with emotional triggers specific to {persona.replace('_', ' ')}

**REFINEMENT STRATEGIES:**
• Lead with strongest psychological trigger
• Optimize word choice for {persona.replace('_', ' ')} vocabulary
• Balance information vs intrigue
• Test alternative sentence structures

**CONSTRAINTS:**
• Must be factually equivalent to original
• Keep structural specs from Layer 1
• Don't repeat Layer 1 headlines

Generate 4 emotionally optimized headlines that beat {best_l1_score:.4f} CTR:"""

        return self._call_llm(prompt, temperature=0.8)

    def _generate_layer_3_candidates(
        self, layer_2_results, original_title, article_data, persona
    ):
        """Layer 3: Final polish and A/B test variants"""

        best_l2_headline = layer_2_results["best_headline"]
        best_l2_score = layer_2_results["best_score"]

        prompt = f"""You are creating final A/B test variants for {persona.replace('_', ' ')} audience.

## CURRENT CHAMPION
Headline: "{best_l2_headline}"
Predicted CTR: {best_l2_score:.4f}

## LAYER 3 TASK: CREATE A/B TEST VARIANTS
Generate 4 polished variants that could potentially beat the champion:

**VARIANT STRATEGIES:**
1. **POWER WORD VARIANT**: Replace key words with more powerful alternatives
2. **STRUCTURE VARIANT**: Rearrange same elements for different impact
3. **SPECIFICITY VARIANT**: Add/modify numbers, details, or precision
4. **ANGLE VARIANT**: Same story, different perspective or focus

**POLISH CRITERIA:**
• Publication-ready quality
• No grammatical issues
• Clear value proposition
• Compelling first 3 words
• Natural reading flow

Each variant should be distinctly different while maintaining the core message.

Generate 4 polished A/B test variants:"""

        return self._call_llm(prompt, temperature=0.6)

    def _call_llm(self, prompt, temperature=0.7):
        """Make LLM call with error handling"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=temperature,
            )

            lines = response.choices[0].message.content.strip().split("\n")
            candidates = []

            for line in lines:
                cleaned = self._clean_response(line)
                if cleaned:
                    candidates.append(cleaned)

            return candidates[:4]  # Ensure max 4 candidates

        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            return []

    def _evaluate_candidates(self, candidates, article_data):
        """Evaluate candidates with your XGBoost model"""
        if not candidates:
            return {
                "best_headline": "",
                "best_score": 0,
                "improvement": 0,
                "all_scores": [],
            }

        scores = []
        for candidate in candidates:
            score = self.predict_ctr_with_model(candidate, article_data)
            scores.append(score)

        best_idx = int(np.argmax(scores))
        best_headline = candidates[best_idx]
        best_score = scores[best_idx]

        return {
            "best_headline": best_headline,
            "best_score": best_score,
            "improvement": best_score,
            "all_scores": scores,
            "all_candidates": candidates,
        }

    def generate_candidates_multi_layer(self, original_title, article_data):
        """Multi-layer candidate generation with feedback loops"""

        if not self.config["enable_multi_layer"]:
            # Fallback to original single-layer approach
            return self.generate_candidates(original_title, article_data)

        # Identify target persona
        persona = self._identify_persona(article_data)
        original_ctr = self.predict_ctr_with_model(original_title, article_data)

        results = {"persona": persona, "original_ctr": original_ctr, "layers": {}}

        # Layer 1: Structure & EDA optimization
        layer_1_candidates = self._generate_layer_1_candidates(
            original_title, article_data, persona
        )
        layer_1_results = self._evaluate_candidates(layer_1_candidates, article_data)
        layer_1_results["improvement"] = layer_1_results["best_score"] - original_ctr
        results["layers"]["layer_1"] = layer_1_results

        logging.info(
            f"Layer 1 best: {layer_1_results['best_score']:.4f} (+{layer_1_results['improvement']:.4f})"
        )

        # Layer 2: Emotional refinement (if Layer 1 improved)
        if (
            layer_1_results["improvement"] > 0.001
        ):  # Only proceed if meaningful improvement
            layer_2_candidates = self._generate_layer_2_candidates(
                layer_1_results, original_title, article_data, persona
            )
            layer_2_results = self._evaluate_candidates(
                layer_2_candidates, article_data
            )
            layer_2_results["improvement"] = (
                layer_2_results["best_score"] - original_ctr
            )
            results["layers"]["layer_2"] = layer_2_results

            logging.info(
                f"Layer 2 best: {layer_2_results['best_score']:.4f} (+{layer_2_results['improvement']:.4f})"
            )

            # Layer 3: Final polish (if Layer 2 improved further)
            if layer_2_results["best_score"] > layer_1_results["best_score"]:
                layer_3_candidates = self._generate_layer_3_candidates(
                    layer_2_results, original_title, article_data, persona
                )
                layer_3_results = self._evaluate_candidates(
                    layer_3_candidates, article_data
                )
                layer_3_results["improvement"] = (
                    layer_3_results["best_score"] - original_ctr
                )
                results["layers"]["layer_3"] = layer_3_results

                logging.info(
                    f"Layer 3 best: {layer_3_results['best_score']:.4f} (+{layer_3_results['improvement']:.4f})"
                )

        return results

    def get_best_headline_enhanced(self, original_title, article_data):
        """Enhanced headline optimization with multi-layer approach"""

        if self.config["enable_multi_layer"]:
            multi_layer_results = self.generate_candidates_multi_layer(
                original_title, article_data
            )

            # Find the best result across all layers
            best_result = None
            best_score = multi_layer_results["original_ctr"]
            best_layer = "original"

            for layer_name, layer_results in multi_layer_results["layers"].items():
                if layer_results["best_score"] > best_score:
                    best_score = layer_results["best_score"]
                    best_result = layer_results
                    best_layer = layer_name

            if best_result:
                return {
                    "best_headline": best_result["best_headline"],
                    "predicted_ctr": best_score,
                    "original_ctr": multi_layer_results["original_ctr"],
                    "ctr_improvement": best_score - multi_layer_results["original_ctr"],
                    "improvement_percent": (
                        (best_score - multi_layer_results["original_ctr"])
                        / multi_layer_results["original_ctr"]
                        * 100
                    ),
                    "best_layer": best_layer,
                    "persona": multi_layer_results["persona"],
                    "layer_results": multi_layer_results["layers"],
                    "multi_layer_used": True,
                }
            else:
                # No improvement found, return original
                return {
                    "best_headline": original_title,
                    "predicted_ctr": multi_layer_results["original_ctr"],
                    "original_ctr": multi_layer_results["original_ctr"],
                    "ctr_improvement": 0,
                    "improvement_percent": 0,
                    "best_layer": "original",
                    "persona": multi_layer_results["persona"],
                    "multi_layer_used": True,
                }
        else:
            # Fallback to original approach
            return self.get_best_headline(original_title, article_data)

    def get_best_headline(self, original_title, article_data):
        """Original method for backwards compatibility"""

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
        """Simple interface - returns the best headline using enhanced method"""
        if article_data is None:
            article_data = {"category": "news", "abstract": ""}

        result = self.get_best_headline_enhanced(original_title, article_data)
        return result["best_headline"]

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
        """Original single-layer generation method (for backwards compatibility)"""

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

    def create_rewrite_variants(self, original_title, article_data):
        """Create rewrite variants using different strategies - for compatibility with FAISS script"""
        strategies = {
            "comprehensive": "Apply multiple optimization techniques",
            "engagement": "Focus on emotional hooks and curiosity",
            "readability": "Improve clarity and readability",
            "structure": "Optimize word count and structure",
        }

        variants = {}

        # Generate candidates using the main method
        candidates = self.generate_candidates(original_title, article_data)

        # Assign candidates to strategies (simplified approach)
        strategy_names = list(strategies.keys())
        for i, candidate in enumerate(
            candidates[1:], 0
        ):  # Skip original (first candidate)
            if i < len(strategy_names):
                strategy = strategy_names[i]
                variants[strategy] = candidate
            else:
                # If more candidates than strategies, use comprehensive
                variants["comprehensive"] = candidate
                break

        # Ensure we always have at least one variant
        if not variants:
            variants["comprehensive"] = original_title

        return variants

    def evaluate_rewrite_quality(self, original_title, rewritten_title):
        """Evaluate rewrite quality - for compatibility with FAISS script"""
        from textstat import flesch_reading_ease

        try:
            # Basic quality metrics
            original_readability = flesch_reading_ease(original_title)
            rewritten_readability = flesch_reading_ease(rewritten_title)

            # Simple similarity check (character-based)
            similarity = len(
                set(original_title.lower().split())
                & set(rewritten_title.lower().split())
            ) / max(len(original_title.split()), len(rewritten_title.split()))

            # Predicted CTR improvement (simplified heuristic)
            word_count_diff = abs(
                len(rewritten_title.split()) - len(original_title.split())
            )
            readability_improvement = rewritten_readability - original_readability

            # Simple heuristic for CTR improvement
            predicted_ctr_improvement = max(
                0, (readability_improvement / 100) * 0.01 - word_count_diff * 0.002
            )

            # Overall quality score
            quality_score = (
                min(1.0, similarity) * 0.4  # Semantic similarity
                + min(1.0, max(0, readability_improvement / 20))
                * 0.3  # Readability improvement
                + min(1.0, max(0, predicted_ctr_improvement * 100))
                * 0.3  # CTR improvement
            )

            return {
                "overall_quality_score": quality_score,
                "readability_improvement": readability_improvement,
                "predicted_ctr_improvement": predicted_ctr_improvement,
                "semantic_similarity": similarity,
            }

        except Exception as e:
            logging.error(f"Quality evaluation failed: {e}")
            return {
                "overall_quality_score": 0.5,
                "readability_improvement": 0,
                "predicted_ctr_improvement": 0,
                "semantic_similarity": 0.5,
            }

    # All the utility methods
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


# Backward compatibility alias
EfficientLLMHeadlineRewriter = EnhancedLLMHeadlineRewriter

# import os
# import re
# import pandas as pd
# import numpy as np
# import logging
# from openai import OpenAI
# from feature_utils import create_article_features_exact, load_preprocessing_components


# class EfficientLLMHeadlineRewriter:
#     """Efficient LLM headline rewriter using XGBoost model and EDA insights"""

#     def __init__(
#         self, model_pipeline, components=None, llm_client=None, eda_insights_path=None
#     ):
#         """Initialize with trained model and preprocessing components"""
#         self.model_pipeline = model_pipeline

#         if components is None:
#             components = load_preprocessing_components()
#         self.components = components

#         if llm_client is not None:
#             self.client = llm_client
#         else:
#             self.client = OpenAI()

#         # Load EDA insights if available
#         self.eda_insights = self._load_eda_insights(eda_insights_path)

#         # Configuration based on EDA findings (with safe defaults)
#         self.config = {
#             "optimal_word_count": (8, 12),  # Conservative default
#             "max_length": 100,
#             "target_readability": 60,
#             "high_engagement_threshold": self._get_safe_value(
#                 "baseline_metrics", "overall_avg_ctr", 0.041
#             ),
#         }

#         logging.info("Efficient LLM Headline Rewriter initialized")
#         logging.info(f"Using XGBoost model for 100% of CTR predictions")
#         logging.info(
#             f"EDA insights: {'Loaded' if self.eda_insights else 'Using defaults'}"
#         )

#     def _get_safe_value(self, section, key, default):
#         """Safely get a value from EDA insights with fallback"""
#         try:
#             if self.eda_insights and section in self.eda_insights:
#                 return self.eda_insights[section].get(key, default)
#             return default
#         except:
#             return default

#     def _get_optimal_specs(self):
#         """Extract optimal specifications from whatever data is available"""
#         # Default fallback values
#         default_specs = {
#             "word_count_range": "8-12",
#             "word_count_ctr": 0.04,
#             "length_range": "50-70",
#             "length_ctr": 0.04,
#             "readability_range": "50-70",
#             "readability_ctr": 0.04,
#         }

#         # If optimal_specifications exists, use it
#         if self.eda_insights and "optimal_specifications" in self.eda_insights:
#             return self.eda_insights["optimal_specifications"]

#         # Otherwise, try to construct from optimal_ranges
#         if self.eda_insights and "optimal_ranges" in self.eda_insights:
#             optimal_ranges = self.eda_insights["optimal_ranges"]

#             # Extract best word count range
#             if "word_count" in optimal_ranges:
#                 word_count_data = optimal_ranges["word_count"]
#                 if word_count_data:
#                     best_word_entry = max(
#                         word_count_data.items(), key=lambda x: x[1].get("avg_ctr", 0)
#                     )
#                     default_specs["word_count_range"] = best_word_entry[0]
#                     default_specs["word_count_ctr"] = best_word_entry[1].get(
#                         "avg_ctr", 0.04
#                     )

#             # Extract best character length range
#             if "character_length" in optimal_ranges:
#                 length_data = optimal_ranges["character_length"]
#                 if length_data:
#                     best_length_entry = max(
#                         length_data.items(), key=lambda x: x[1].get("avg_ctr", 0)
#                     )
#                     default_specs["length_range"] = best_length_entry[0]
#                     default_specs["length_ctr"] = best_length_entry[1].get(
#                         "avg_ctr", 0.04
#                     )

#             # Extract best readability range
#             if "readability" in optimal_ranges:
#                 readability_data = optimal_ranges["readability"]
#                 if readability_data:
#                     best_readability_entry = max(
#                         readability_data.items(), key=lambda x: x[1].get("avg_ctr", 0)
#                     )
#                     default_specs["readability_range"] = best_readability_entry[0]
#                     default_specs["readability_ctr"] = best_readability_entry[1].get(
#                         "avg_ctr", 0.04
#                     )

#         return default_specs

#     def _get_top_features(self):
#         """Get top features from whatever data is available"""
#         if self.eda_insights and "top_features_by_impact" in self.eda_insights:
#             return self.eda_insights["top_features_by_impact"][:3]

#         # Fallback to feature_improvements if available
#         if self.eda_insights and "feature_improvements" in self.eda_insights:
#             improvements = self.eda_insights["feature_improvements"]
#             top_features = []
#             for feature, data in improvements.items():
#                 if data.get("improvement_percent", 0) > 0:
#                     top_features.append(
#                         (
#                             feature,
#                             data["improvement_percent"],
#                             data["with_feature_ctr"],
#                             data["sample_size_with"],
#                         )
#                     )
#             # Sort by improvement and return top 3
#             top_features.sort(key=lambda x: x[1], reverse=True)
#             return top_features[:3]

#         # Ultimate fallback - generic features that usually help
#         return [
#             ("has_number", 15.0, 0.045, 1000),
#             ("has_question", 12.0, 0.043, 800),
#             ("has_colon", 8.0, 0.042, 600),
#         ]

#     def _get_baseline_metrics(self):
#         """Get baseline metrics from whatever data is available"""
#         defaults = {
#             "overall_avg_ctr": 0.041,
#             "training_median_ctr": 0.030,
#             "articles_analyzed": 25000,
#         }

#         if self.eda_insights and "baseline_metrics" in self.eda_insights:
#             baseline = self.eda_insights["baseline_metrics"]
#             return {
#                 "overall_avg_ctr": baseline.get(
#                     "overall_avg_ctr", defaults["overall_avg_ctr"]
#                 ),
#                 "training_median_ctr": baseline.get(
#                     "training_median_ctr", defaults["training_median_ctr"]
#                 ),
#                 "articles_analyzed": baseline.get(
#                     "articles_analyzed", defaults["articles_analyzed"]
#                 ),
#             }

#         return defaults

#     def _get_proven_starters(self):
#         """Get proven starter words from whatever data is available"""
#         if self.eda_insights and "proven_starters" in self.eda_insights:
#             return self.eda_insights["proven_starters"]

#         # Check if high_performers has sample_headlines we can analyze
#         if self.eda_insights and "high_performers" in self.eda_insights:
#             high_performers = self.eda_insights["high_performers"]
#             if "sample_headlines" in high_performers:
#                 # Extract first words from sample headlines
#                 starters = {}
#                 for headline in high_performers["sample_headlines"][:10]:
#                     if headline and len(str(headline).split()) > 0:
#                         first_word = str(headline).split()[0].lower().strip('.,!?";')
#                         if len(first_word) > 2 and first_word.isalpha():
#                             starters[first_word] = {"avg_ctr": 0.08, "count": 5}
#                 return starters

#         # Ultimate fallback - common high-performing starters
#         return {
#             "breaking": {"avg_ctr": 0.12, "count": 50},
#             "exclusive": {"avg_ctr": 0.10, "count": 30},
#             "shocking": {"avg_ctr": 0.09, "count": 25},
#         }

#     def _load_eda_insights(self, eda_insights_path):
#         """Load EDA insights from analysis or use defaults"""
#         if eda_insights_path and os.path.exists(eda_insights_path):
#             try:
#                 import json

#                 with open(eda_insights_path, "r") as f:
#                     insights = json.load(f)
#                 logging.info(f"Loaded EDA insights from {eda_insights_path}")
#                 return insights
#             except Exception as e:
#                 logging.warning(f"Could not load EDA insights: {e}")

#         # Return None - we'll handle missing insights defensively
#         logging.info("No EDA insights loaded - using defensive defaults")
#         return None

#     def predict_ctr_with_model(self, title, article_data):
#         """Use YOUR XGBoost model to predict CTR - the only CTR prediction method"""
#         try:
#             # Extract features using your exact pipeline
#             features = create_article_features_exact(
#                 title,
#                 article_data.get("abstract", ""),
#                 article_data.get("category", "news"),
#                 self.components,
#             )

#             # Vectorize in exact feature order
#             feature_order = self.components["feature_order"]
#             feature_vector = np.array(
#                 [features.get(f, 0.0) for f in feature_order]
#             ).reshape(1, -1)

#             # Predict using YOUR model - the ONLY source of CTR predictions
#             engagement_prob = self.model_pipeline["model"].predict_proba(
#                 feature_vector
#             )[0, 1]

#             # Convert engagement probability to CTR estimate
#             estimated_ctr = max(0.01, engagement_prob * 0.1)

#             return float(estimated_ctr)

#         except Exception as e:
#             logging.error(f"Model CTR prediction failed: {e}")
#             return 0.035  # Fallback

#     def generate_candidates(self, original_title, article_data):
#         """Generate headline candidates using LLM with ACTUAL EDA insights (defensive)"""

#         category = article_data.get("category", "news")
#         current_ctr = self.predict_ctr_with_model(original_title, article_data)

#         # Get data safely from whatever is available
#         baseline_metrics = self._get_baseline_metrics()
#         top_features = self._get_top_features()
#         optimal_specs = self._get_optimal_specs()
#         proven_starters = self._get_proven_starters()

#         baseline_ctr = baseline_metrics["overall_avg_ctr"]
#         median_ctr = baseline_metrics["training_median_ctr"]
#         articles_analyzed = baseline_metrics["articles_analyzed"]

#         # Check if this is sports content for specific starter words
#         is_sports = "sport" in category.lower() or any(
#             team in original_title.lower() for team in proven_starters.keys()
#         )

#         # Build prompt based on ACTUAL available data
#         prompt = f"""Generate 4 headline variations for: "{original_title}"

# 📊 REAL DATA INSIGHTS ({articles_analyzed:,} articles analyzed):
# Current CTR: {current_ctr:.4f}
# Baseline average: {baseline_ctr:.4f}
# Target improvement: +{max(0.005, baseline_ctr - current_ctr):.4f}

# 🚀 TOP PERFORMING FEATURES (from your data):"""

#         # Add top features dynamically
#         for i, (feature, improvement, ctr, size) in enumerate(top_features):
#             prompt += f"\n• {feature.replace('_', ' ').title()}: +{improvement:.1f}% CTR boost (n={size:,})"

#         prompt += f"""

# 📏 OPTIMAL SPECIFICATIONS (proven by data):
# • Word count: {optimal_specs['word_count_range']} words (CTR: {optimal_specs['word_count_ctr']:.4f})
# • Character length: {optimal_specs['length_range']} chars (CTR: {optimal_specs['length_ctr']:.4f})
# • Readability: {optimal_specs['readability_range']} Flesch score (CTR: {optimal_specs['readability_ctr']:.4f})"""

#         # Add sports-specific guidance if relevant and data available
#         if is_sports and proven_starters:
#             sports_starters = list(proven_starters.keys())[:3]
#             prompt += f"""

# 🏈 HIGH-PERFORMERS:
# Top starting words: {', '.join(sports_starters)}"""

#         prompt += f"""

# ✅ REQUIREMENTS (based on {articles_analyzed:,} articles):
# - Apply 2-3 proven features above
# - Target {optimal_specs['word_count_range']} words
# - Aim for {optimal_specs['length_range']} characters
# - Readability in {optimal_specs['readability_range']} range
# - Create engaging headlines for {category}

# Return ONLY the 4 headlines, one per line, no numbering."""

#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=250,
#                 temperature=0.8,
#             )

#             candidates = [original_title]  # Always include original
#             lines = response.choices[0].message.content.strip().split("\n")

#             for line in lines:
#                 cleaned = self._clean_response(line)
#                 if cleaned and cleaned != original_title:
#                     candidates.append(cleaned)

#             return candidates[:5]  # Max 5 candidates

#         except Exception as e:
#             logging.error(f"LLM candidate generation failed: {e}")
#             return [original_title]

#     def _clean_response(self, response):
#         """Clean LLM response"""
#         # Remove quotes, numbering, and extra whitespace
#         response = re.sub(r'^["\']|["\']$', "", response)
#         response = re.sub(r"^\d+[\.\)]\s*", "", response)  # Remove numbering
#         response = response.split("\n")[0]  # Take first line only
#         return response.strip()

#     def get_best_headline(self, original_title, article_data):
#         """Get the best headline using your model for selection"""

#         # Generate candidates using EDA-driven LLM prompts (defensive)
#         candidates = self.generate_candidates(original_title, article_data)

#         # Score each candidate with YOUR model (100% model-based scoring)
#         scores = []
#         for candidate in candidates:
#             ctr_score = self.predict_ctr_with_model(candidate, article_data)
#             scores.append(ctr_score)

#         # Return the highest-scoring candidate according to YOUR model
#         best_idx = int(np.argmax(scores))
#         best_headline = candidates[best_idx]
#         best_score = scores[best_idx]

#         # Calculate actual improvement using model predictions
#         original_score = scores[0]  # Original is always first
#         improvement = best_score - original_score

#         return {
#             "best_headline": best_headline,
#             "predicted_ctr": best_score,
#             "original_ctr": original_score,
#             "ctr_improvement": improvement,
#             "improvement_percent": (
#                 (improvement / original_score * 100) if original_score > 0 else 0
#             ),
#             "all_candidates": list(zip(candidates, scores)),
#             "model_selected": True,  # Always true - we only use your model
#             "eda_driven": bool(self.eda_insights),
#         }

#     def rewrite_headline(self, original_title, article_data=None):
#         """Simple interface - returns the best headline"""
#         if article_data is None:
#             article_data = {"category": "news", "abstract": ""}

#         result = self.get_best_headline(original_title, article_data)
#         return result["best_headline"]

#     def analyze_headline(self, headline, article_data):
#         """Analyze a headline using your model and ACTUAL EDA insights (defensive)"""

#         # Get model prediction
#         predicted_ctr = self.predict_ctr_with_model(headline, article_data)

#         # Extract features for analysis
#         features = create_article_features_exact(
#             headline,
#             article_data.get("abstract", ""),
#             article_data.get("category", "news"),
#             self.components,
#         )

#         # Get baseline metrics safely
#         baseline_metrics = self._get_baseline_metrics()
#         baseline_ctr = baseline_metrics["overall_avg_ctr"]
#         median_ctr = baseline_metrics["training_median_ctr"]

#         # Feature analysis based on available EDA insights
#         feature_analysis = []
#         top_features = self._get_top_features()

#         for feature_name, improvement, target_ctr, sample_size in top_features:
#             feature_key = feature_name  # Use exact feature name from EDA
#             if feature_key in features:
#                 has_feature = bool(features[feature_key])
#                 feature_analysis.append(
#                     {
#                         "feature": feature_name.replace("_", " ").title(),
#                         "present": has_feature,
#                         "potential_improvement": improvement if not has_feature else 0,
#                         "target_ctr": target_ctr,
#                         "sample_size": sample_size,
#                     }
#                 )

#         return {
#             "predicted_ctr": predicted_ctr,
#             "baseline_ctr": baseline_ctr,
#             "median_ctr": median_ctr,
#             "performance_vs_baseline": predicted_ctr - baseline_ctr,
#             "performance_vs_median": predicted_ctr - median_ctr,
#             "word_count": features["title_word_count"],
#             "char_length": features["title_length"],
#             "readability": features["title_reading_ease"],
#             "feature_analysis": feature_analysis,
#             "meets_optimal_specs": {
#                 "word_count": self.config["optimal_word_count"][0]
#                 <= features["title_word_count"]
#                 <= self.config["optimal_word_count"][1],
#                 "length": features["title_length"] <= self.config["max_length"],
#                 "readability": features["title_reading_ease"]
#                 >= self.config["target_readability"],
#             },
#             "engagement_prediction": predicted_ctr
#             >= self.config["high_engagement_threshold"],
#             "editorial_note": f"Analysis based on {baseline_metrics['articles_analyzed']:,} articles",
#         }

#     def batch_optimize(self, headlines_df, n_samples=10):
#         """Optimize multiple headlines using your model"""

#         # Sample low-performing headlines
#         if "ctr" in headlines_df.columns:
#             sample_df = headlines_df.nsmallest(n_samples, "ctr")
#         else:
#             sample_df = headlines_df.head(n_samples)

#         results = []

#         for idx, row in sample_df.iterrows():
#             article_data = {
#                 "category": row.get("category", "news"),
#                 "abstract": row.get("abstract", ""),
#             }

#             # Get current model prediction for original
#             original_ctr = self.predict_ctr_with_model(row["title"], article_data)

#             # Get optimized headline
#             result = self.get_best_headline(row["title"], article_data)

#             results.append(
#                 {
#                     "newsID": row.get("newsID", idx),
#                     "original_title": row["title"],
#                     "optimized_title": result["best_headline"],
#                     "original_model_ctr": original_ctr,
#                     "optimized_model_ctr": result["predicted_ctr"],
#                     "model_ctr_improvement": result["ctr_improvement"],
#                     "improvement_percent": result["improvement_percent"],
#                     "category": article_data["category"],
#                     "eda_driven": result["eda_driven"],
#                 }
#             )

#         return pd.DataFrame(results)

#     def get_optimization_recommendations(self, headline, article_data):
#         """Get specific recommendations based on ACTUAL EDA insights (defensive)"""

#         analysis = self.analyze_headline(headline, article_data)
#         recommendations = []

#         # Check missing high-impact features from actual data
#         for feature_info in analysis["feature_analysis"]:
#             if (
#                 not feature_info["present"]
#                 and feature_info["potential_improvement"] > 5
#             ):
#                 recommendations.append(
#                     {
#                         "type": "add_feature",
#                         "feature": feature_info["feature"],
#                         "impact": feature_info["potential_improvement"],
#                         "sample_size": feature_info.get("sample_size", 0),
#                         "suggestion": self._get_feature_suggestion(
#                             feature_info["feature"]
#                         ),
#                     }
#                 )

#         # Check specifications based on available optimal ranges
#         specs = analysis["meets_optimal_specs"]
#         optimal_specs = self._get_optimal_specs()

#         if not specs["word_count"]:
#             current_words = analysis["word_count"]
#             target_range = optimal_specs["word_count_range"]
#             recommendations.append(
#                 {
#                     "type": "word_count",
#                     "suggestion": f"Adjust to {target_range} words for optimal performance",
#                 }
#             )

#         # Performance vs baseline
#         if analysis["performance_vs_baseline"] < 0:
#             gap = abs(analysis["performance_vs_baseline"])
#             recommendations.append(
#                 {
#                     "type": "performance",
#                     "suggestion": f"Headline below baseline by {gap:.4f} CTR - consider restructuring",
#                 }
#             )

#         return recommendations

#     def _get_feature_suggestion(self, feature_name):
#         """Get specific suggestions for adding features"""
#         suggestions = {
#             "Too Long Title": "Create longer, more descriptive headlines",
#             "Has Colon": "Use a colon to separate main topic from details",
#             "Has Dash": "Add dashes to connect related concepts",
#             "Has Quotes": "Quote key phrases or add quotation marks for emphasis",
#             "Needs Readability Improvement": "Adjust language complexity",
#             "Has Number": "Add specific numbers, statistics, or quantities",
#             "Has Question": "Turn into a question to increase curiosity",
#             "Has Exclamation": "Add excitement with an exclamation mark",
#         }
#         return suggestions.get(feature_name, f"Consider adding {feature_name.lower()}")
