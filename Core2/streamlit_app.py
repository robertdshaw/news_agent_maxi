import os
import torch

# Fix for Streamlit's local_sources_watcher crashing on torch.classes
try:
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
except Exception:
    pass

import streamlit as st
import json
import pandas as pd
import numpy as np
import pickle
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from textstat import flesch_reading_ease
import warnings
from llm_rewriter import EfficientLLMHeadlineRewriter
from feature_utils import create_article_features_exact, load_preprocessing_components

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Article Engagement Predictor & Rewriter",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
MODEL_DIR = Path("model_output")
FAISS_DIR = Path("faiss_index")
PREP_DIR = Path("data/preprocessed")

# CSS Styles
st.markdown(
    """
    <style>
      .card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
      }
      .full-width { width: 100% !important; }
      .guidelines-box {
        background-color: #f0f2f6;
        border: 2px solid #4f8bf9;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
      }
      .guidelines-title {
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 15px;
        color: #1f4e79;
        text-align: center;
      }
      .guidelines-content {
        font-size: 15px;
        line-height: 1.6;
        color: #2c3e50;
      }
      .guidelines-content ul {
        margin: 10px 0;
        padding-left: 20px;
      }
      .guidelines-content li {
        margin: 8px 0;
        color: #34495e;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    """Load the trained engagement prediction model"""
    try:
        model_files = list(MODEL_DIR.glob("*_optimized_model.pkl"))
        model_file = (
            model_files[0] if model_files else MODEL_DIR / "xgboost_optimized_model.pkl"
        )

        with open(model_file, "rb") as f:
            model = pickle.load(f)

        metadata_file = MODEL_DIR / "model_metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

        # Load baseline metrics from your actual EDA insights
        baseline_metrics = {}
        try:
            # Try the preprocessed data directory first
            eda_path = "data/preprocessed/processed_data/headline_eda_insights.json"
            if not Path(eda_path).exists():
                eda_path = "headline_eda_insights.json"

            with open(eda_path, "r") as f:
                eda_insights = json.load(f)
                baseline_metrics = eda_insights.get(
                    "baseline_metrics",
                    {
                        "overall_avg_ctr": 0.041238101089957464,
                        "training_median_ctr": 0.019230769230769232,
                        "ctr_threshold": 0.05,
                    },
                )
        except:
            # Fallback to metadata values
            baseline_metrics = {
                "overall_avg_ctr": metadata.get("target_statistics", {}).get(
                    "mean_ctr", 0.041
                ),
                "training_median_ctr": metadata.get("target_statistics", {}).get(
                    "median_ctr", 0.019
                ),
                "ctr_threshold": metadata.get("target_statistics", {}).get(
                    "ctr_threshold", 0.05
                ),
            }

        model_pipeline = {
            "model": model,
            "model_name": metadata.get("model_type", "XGBoost"),
            "target": "high_engagement",
            "performance": metadata.get("final_evaluation", {}),
            "feature_names": (
                list(model.feature_names_in_)
                if hasattr(model, "feature_names_in_")
                else []
            ),
            "scaler": None,
            "baseline_metrics": baseline_metrics,
            "metadata": metadata,
        }

        return model_pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_search_system():
    """Load the enhanced FAISS search system with rewrite capabilities"""
    try:
        index = faiss.read_index(str(FAISS_DIR / "article_index.faiss"))

        with open(FAISS_DIR / "article_lookup.pkl", "rb") as f:
            article_lookup = pickle.load(f)

        with open(FAISS_DIR / "article_id_mappings.pkl", "rb") as f:
            mappings = pickle.load(f)

        with open(FAISS_DIR / "index_metadata.json", "r") as f:
            metadata = json.load(f)

        return {
            "index": index,
            "article_lookup": article_lookup,
            "mappings": mappings,
            "metadata": metadata,
        }
    except Exception as e:
        st.error(f"Error loading search system: {e}")
        return None


@st.cache_resource
def load_llm_rewriter():
    """Load the efficient LLM headline rewriter with correct EDA insights path"""
    try:
        model_pipeline = load_model()
        components = load_preprocessing_components()

        if model_pipeline and components:
            try:
                # Try the preprocessed data directory first
                eda_insights_path = (
                    "data/preprocessed/processed_data/headline_eda_insights.json"
                )
                if not Path(eda_insights_path).exists():
                    # Fall back to project root
                    eda_insights_path = "headline_eda_insights.json"

                return EfficientLLMHeadlineRewriter(
                    model_pipeline=model_pipeline,
                    components=components,
                    eda_insights_path=eda_insights_path,
                )
            except Exception as e:
                st.warning(f"EfficientLLMHeadlineRewriter failed: {e}")
                # Fall back to basic LLM rewriter
                try:
                    from llm_rewriter import LLMHeadlineRewriter

                    return LLMHeadlineRewriter(
                        model_pipeline=model_pipeline, components=components
                    )
                except Exception as e2:
                    st.warning(f"Basic LLMHeadlineRewriter also failed: {e2}")
                    return None
        else:
            st.warning("Could not load model pipeline or components for rewriter")
            return None
    except Exception as e:
        st.error(f"Error loading LLM rewriter: {e}")
        return None


@st.cache_resource
def load_embedder():
    """Load the sentence transformer model"""
    try:
        from feature_utils import get_embedder

        return get_embedder()
    except Exception as e:
        st.error(f"Error loading embedder: {e}")
        return SentenceTransformer("all-MiniLM-L6-v2")


def predict_engagement(
    title, abstract="", category="news", model_pipeline=None, components=None
):
    """Predict engagement for a single article using exact feature replication"""
    if model_pipeline is None or components is None:
        return None

    try:
        # Create features using exact replication of preprocessing pipeline
        features = create_article_features_exact(title, abstract, category, components)

        # Create feature vector in the exact order expected by the model
        feature_order = components.get("feature_order", [])

        if feature_order:
            feature_vector = [
                features.get(feat_name, 0.0) for feat_name in feature_order
            ]
        elif hasattr(model_pipeline["model"], "feature_names_in_"):
            expected_features = list(model_pipeline["model"].feature_names_in_)
            feature_vector = [
                features.get(feat_name, 0.0) for feat_name in expected_features
            ]
        else:
            feature_vector = [features[k] for k in sorted(features.keys())]

        feature_vector = np.array(feature_vector).reshape(1, -1)

        # Predict
        prediction = model_pipeline["model"].predict(feature_vector)[0]
        prediction_proba = model_pipeline["model"].predict_proba(feature_vector)[0]

        # Convert engagement probability to estimated CTR
        engagement_prob = float(prediction_proba[1])
        estimated_ctr = max(0.01, engagement_prob * 0.1)

        return {
            "high_engagement": bool(prediction),
            "engagement_probability": engagement_prob,
            "estimated_ctr": estimated_ctr,
            "confidence": float(max(prediction_proba)),
            "features": features,
        }

    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None


def main():
    st.title("📰 AI-Assisted Headline Hunter")
    st.write("**Predict engagement and optimize headlines with AI-powered rewriting**")

    # Load all systems once at startup
    with st.spinner("Loading AI systems..."):
        model_pipeline = load_model()
        preprocessing_components = load_preprocessing_components()
        search_system = load_search_system()
        llm_rewriter = load_llm_rewriter()

    # # Sidebar status
    # st.sidebar.header("🎯 System Status")
    # if model_pipeline:
    #     st.sidebar.success(f"✅ Model: {model_pipeline['model_name']}")
    #     if "auc" in model_pipeline.get("performance", {}):
    #         st.sidebar.info(f"📊 AUC: {model_pipeline['performance']['auc']:.4f}")
    # else:
    #     st.sidebar.error("❌ Model not loaded")

    # if preprocessing_components:
    #     st.sidebar.success("✅ Preprocessing: Components loaded")
    #     st.sidebar.info(
    #         f"📈 Features: {len(preprocessing_components.get('feature_order', []))}"
    #     )
    # else:
    #     st.sidebar.error("❌ Preprocessing components not loaded")

    # if search_system:
    #     st.sidebar.success(
    #         f"✅ Search: {search_system['metadata']['total_articles']:,} articles"
    #     )
    # else:
    #     st.sidebar.error("❌ Search system not loaded")

    # if llm_rewriter:
    #     st.sidebar.success("✅ AI Rewriter: Available")
    # else:
    #     st.sidebar.warning("⚠️ AI Rewriter: Not available")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(
        ["🔮 Predict & Rewrite", "🔍 Search Articles", "📊 Headline Rewrite Analysis"]
    )

    # Tab 1: Predict & Rewrite
    with tab1:
        # Main layout with input on left and guidelines on right
        col1, col2 = st.columns([2, 1])

        with col1:
            title = st.text_area(
                "Article Title",
                placeholder="Enter your article headline here...",
                height=100,
                help="Enter the headline you want to test and optimize",
            )

            categories = [
                "news",
                "sports",
                "finance",
                "travel",
                "lifestyle",
                "video",
                "foodanddrink",
                "weather",
                "autos",
                "health",
                "entertainment",
                "tv",
                "music",
                "movies",
                "kids",
                "northamerica",
                "middleeast",
                "unknown",
            ]

            category = st.selectbox("Article Category", categories, index=0)

            predict_and_rewrite = st.button("🤖 Analyze & Optimize", type="primary")

        with col2:
            # Editorial Guidelines in right column
            st.markdown(
                """
            <div class="guidelines-box">
                <div class="guidelines-title">💡 Editorial Guidelines</div>
                <div class="guidelines-content">
                    <strong>High-engagement headlines:</strong>
                    <ul>
                        <li>8-12 words optimal</li>
                        <li>Include numbers/questions</li>
                        <li>High readability (60+ score)</li>
                        <li>Front-load key information</li>
                        <li>Under 75 characters</li>
                    </ul>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Process prediction and rewriting
        # Initialize button state
        # predict_and_rewrite = False
        if predict_and_rewrite:
            if title.strip():
                if model_pipeline and preprocessing_components:
                    # Step 1: Predict engagement
                    with st.spinner("🔮 Analyzing engagement potential..."):
                        result = predict_engagement(
                            title,
                            "",
                            category,
                            model_pipeline,
                            preprocessing_components,
                        )

                    if result and isinstance(result, dict):
                        # Get threshold from model pipeline
                        threshold = model_pipeline["baseline_metrics"].get(
                            "ctr_threshold", 0.05
                        )

                        # Display prediction results
                        st.subheader("📊 Engagement Analysis")

                        col_pred1, col_pred2, col_pred3 = st.columns(3)

                        with col_pred1:
                            # Show engagement level with threshold
                            engagement_level = (
                                "High Engagement"
                                if result["high_engagement"]
                                else "Low Engagement"
                            )
                            engagement_emoji = (
                                "🔥" if result["high_engagement"] else "📉"
                            )
                            st.metric(
                                "Engagement Level",
                                f"{engagement_emoji} {engagement_level}",
                            )

                        with col_pred2:
                            # Show CTR as percentage with threshold
                            ctr_percentage = result["estimated_ctr"] * 100
                            st.metric("Estimated CTR", f"{ctr_percentage:.2f}%")
                            st.caption(f"Threshold: {threshold*100:.1f}%")

                        # with col_pred3:
                        #     st.metric("Confidence", f"{result['confidence']:.1%}")

                        # Step 2: Generate AI rewrites
                        st.subheader("✨ AI-Optimized Headlines")

                        if llm_rewriter:
                            with st.spinner("🤖 Generating AI-optimized headlines..."):
                                try:
                                    article_data = {
                                        "category": category,
                                        "abstract": "",
                                        "current_ctr": result["estimated_ctr"],
                                    }

                                    rewrite_result = llm_rewriter.get_best_headline(
                                        title, article_data
                                    )

                                    if (
                                        rewrite_result
                                        and "best_headline" in rewrite_result
                                    ):
                                        best_headline = rewrite_result["best_headline"]

                                        if best_headline.strip() != title.strip():
                                            # Predict engagement for the rewritten headline
                                            rewritten_result = predict_engagement(
                                                best_headline,
                                                "",
                                                category,
                                                model_pipeline,
                                                preprocessing_components,
                                            )

                                            # Display comparison
                                            col_orig, col_rewrite = st.columns(2)

                                            with col_orig:
                                                st.markdown("**Original Headline:**")
                                                st.info(f"📝 {title}")
                                                st.write(
                                                    f"CTR: {result['estimated_ctr']*100:.2f}%"
                                                )
                                                # st.write(
                                                #     f"Engagement: {result['engagement_probability']:.1%}"
                                                # )

                                            with col_rewrite:
                                                st.markdown(
                                                    "**AI-Optimized Headline:**"
                                                )
                                                st.success(f"✨ {best_headline}")

                                                if rewritten_result:
                                                    st.write(
                                                        f"CTR: {rewritten_result['estimated_ctr']*100:.2f}%"
                                                    )
                                                    # st.write(
                                                    #     f"Engagement: {rewritten_result['engagement_probability']:.1%}"
                                                    # )

                                                    # Show improvement
                                                    ctr_improvement = (
                                                        rewritten_result[
                                                            "estimated_ctr"
                                                        ]
                                                        - result["estimated_ctr"]
                                                    ) * 100
                                                    if ctr_improvement > 0:
                                                        st.success(
                                                            f"📈 CTR Improvement: +{ctr_improvement:.2f}%"
                                                        )
                                                    elif ctr_improvement < 0:
                                                        st.warning(
                                                            f"📉 CTR Change: {ctr_improvement:.2f}%"
                                                        )
                                                    else:
                                                        st.info(
                                                            "📊 No significant change"
                                                        )

                                            # Show all candidates
                                            if "all_candidates" in rewrite_result:
                                                with st.expander(
                                                    "🔍 View All AI-Generated Candidates"
                                                ):
                                                    for i, (
                                                        candidate,
                                                        ctr,
                                                    ) in enumerate(
                                                        rewrite_result[
                                                            "all_candidates"
                                                        ],
                                                        1,
                                                    ):
                                                        st.write(
                                                            f"{i}. **{candidate}** (CTR: {ctr*100:.2f}%)"
                                                        )
                                        else:
                                            st.info(
                                                "🎯 Original headline is already well-optimized!"
                                            )

                                            # Show the analysis anyway
                                            if "all_candidates" in rewrite_result:
                                                with st.expander(
                                                    "🔍 View Alternative Suggestions"
                                                ):
                                                    for i, (
                                                        candidate,
                                                        ctr,
                                                    ) in enumerate(
                                                        rewrite_result[
                                                            "all_candidates"
                                                        ],
                                                        1,
                                                    ):
                                                        if candidate != title:
                                                            st.write(
                                                                f"{i}. **{candidate}** (CTR: {ctr*100:.2f}%)"
                                                            )
                                    else:
                                        st.warning(
                                            "🤔 No rewrite suggestions generated"
                                        )

                                except Exception as e:
                                    st.error(f"❌ Rewriting failed: {e}")

                                    # Show the exact error for debugging
                                    with st.expander("🔍 Debug Info"):
                                        import traceback

                                        st.code(traceback.format_exc())
                        else:
                            st.error(
                                "❌ AI Rewriter not available. Please check your OpenAI API key."
                            )

                    else:
                        st.error("❌ Prediction failed. Please check your inputs.")
                else:
                    st.error("❌ Model or preprocessing components not loaded.")
            else:
                st.warning("⚠️ Please enter an article title.")

    # Tab 2: Search Articles
    with tab2:
        st.header("Search Articles")

        search_query = st.text_input(
            "Search Query",
            placeholder="Enter keywords or describe the topic...",
            help="Search through articles by keywords, title, or content.",
        )

        col_search1, col_search2 = st.columns(2)
        with col_search1:
            num_results = st.slider("Number of results", 5, 20, 10)

        if st.button("🔍 Search", type="primary"):
            if search_query.strip() and search_system:
                with st.spinner("Searching articles..."):
                    try:
                        embedder = load_embedder()
                        query_embedding = embedder.encode([search_query])
                        query_embedding = query_embedding.astype(np.float32)
                        faiss.normalize_L2(query_embedding)

                        search_k = num_results * 3
                        distances, indices = search_system["index"].search(
                            query_embedding, search_k
                        )

                        results = []
                        for dist, idx in zip(distances[0], indices[0]):
                            if idx in search_system["mappings"]["idx_to_article_id"]:
                                article_id = search_system["mappings"][
                                    "idx_to_article_id"
                                ][idx]
                                if article_id in search_system["article_lookup"]:
                                    article_info = search_system["article_lookup"][
                                        article_id
                                    ].copy()
                                    l2_distance = float(dist)
                                    similarity_score = 1.0 / (1.0 + l2_distance)
                                    article_info["similarity_score"] = similarity_score
                                    results.append(article_info)

                                    if len(results) >= num_results:
                                        break

                        if results:
                            st.subheader(f"📰 Found {len(results)} articles")
                            for i, article in enumerate(results, 1):
                                with st.expander(f"{i}. {article['title'][:70]}..."):
                                    col_art1, col_art2 = st.columns([3, 1])

                                    with col_art1:
                                        st.write(f"**Title:** {article['title']}")
                                        if article.get("abstract"):
                                            abstract_preview = (
                                                article["abstract"][:200] + "..."
                                                if len(article["abstract"]) > 200
                                                else article["abstract"]
                                            )
                                            st.write(
                                                f"**Abstract:** {abstract_preview}"
                                            )

                                    with col_art2:
                                        st.metric(
                                            "Similarity",
                                            f"{article['similarity_score']:.3f}",
                                        )
                                        st.write(f"**Category:** {article['category']}")

                                        if not pd.isna(article.get("ctr")):
                                            st.write(
                                                f"**CTR:** {article['ctr']*100:.2f}%"
                                            )

                                        if not pd.isna(article.get("high_engagement")):
                                            engagement_status = (
                                                "🔥 High"
                                                if article["high_engagement"]
                                                else "📉 Low"
                                            )
                                            st.write(
                                                f"**Engagement:** {engagement_status}"
                                            )
                        else:
                            st.info("No articles found. Try different keywords.")
                    except Exception as e:
                        st.error(f"Search failed: {e}")
            else:
                if not search_query.strip():
                    st.warning("Please enter a search query.")
                else:
                    st.error("Search system not available.")

    # Tab 3: Rewrite Analysis
    with tab3:
        st.header("Headline Rewrite Analysis")

        if search_system and search_system["metadata"].get("rewrite_analysis"):
            rewrite_stats = search_system["metadata"]["rewrite_analysis"]

            # Load detailed rewrite results if available
            rewrite_file = FAISS_DIR / "rewrite_analysis" / "headline_rewrites.parquet"
            if rewrite_file.exists():
                try:
                    rewrite_df = pd.read_parquet(rewrite_file)

                    st.subheader("📈 Strategy Performance")

                    # Strategy comparison
                    if "model_ctr_improvement" in rewrite_df.columns:
                        strategy_performance = (
                            rewrite_df.groupby("strategy")
                            .agg(
                                {
                                    "quality_score": "mean",
                                    "readability_improvement": "mean",
                                    "model_ctr_improvement": "mean",
                                }
                            )
                            .round(4)
                        )

                        fig = px.bar(
                            strategy_performance.reset_index(),
                            x="strategy",
                            y="model_ctr_improvement",
                            title="Average Model-Based CTR Improvement by Strategy",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        strategy_performance = (
                            rewrite_df.groupby("strategy")
                            .agg(
                                {
                                    "quality_score": "mean",
                                    "readability_improvement": "mean",
                                    "predicted_ctr_improvement": "mean",
                                }
                            )
                            .round(3)
                        )

                        fig = px.bar(
                            strategy_performance.reset_index(),
                            x="strategy",
                            y="quality_score",
                            title="Average Quality Score by Strategy",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Show detailed results
                    st.subheader("🔍 Detailed Results")

                    display_columns = [
                        "original_title",
                        "strategy",
                        "rewritten_title",
                        "quality_score",
                        "readability_improvement",
                        "predicted_ctr_improvement",
                    ]

                    if "model_ctr_improvement" in rewrite_df.columns:
                        display_columns.extend(
                            ["model_ctr_improvement", "original_ctr", "rewritten_ctr"]
                        )

                    available_columns = [
                        col for col in display_columns if col in rewrite_df.columns
                    ]
                    st.dataframe(
                        rewrite_df[available_columns].head(20), use_container_width=True
                    )

                except Exception as e:
                    st.error(f"Error loading rewrite analysis: {e}")
            else:
                st.info(
                    "Detailed rewrite analysis not available. Run the FAISS index creation script to generate analysis."
                )
        else:
            st.info(
                "No rewrite analysis available. The system may be running in offline mode."
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "**Article Engagement Predictor & AI Rewriter** | Built with Streamlit, XGBoost, and OpenAI"
    )


if __name__ == "__main__":
    main()
