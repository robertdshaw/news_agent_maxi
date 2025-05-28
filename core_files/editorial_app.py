import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from textstat import flesch_reading_ease
import warnings
from llm_rewriter import LLMHeadlineRewriter

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Article Engagement Predictor & Rewriter",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
MODEL_DIR = Path("models")
FAISS_DIR = Path("faiss_index")
PREP_DIR = Path("data/preprocessed")


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
    """Load the LLM headline rewriter"""
    return LLMHeadlineRewriter()


@st.cache_resource
def load_embedder():
    """Load the sentence transformer model"""
    return SentenceTransformer("all-MiniLM-L6-v2")


def create_article_features(title, abstract="", category="news"):
    """Create features for a single article (simplified version)"""
    features = {}
    features["title_length"] = len(title)
    features["abstract_length"] = len(abstract)
    features["title_word_count"] = len(title.split())
    features["abstract_word_count"] = len(abstract.split())
    features["has_question"] = 1 if "?" in title else 0
    features["has_exclamation"] = 1 if "!" in title else 0
    features["has_number"] = 1 if any(c.isdigit() for c in title) else 0
    features["has_colon"] = 1 if ":" in title else 0
    features["has_quotes"] = 1 if any(q in title for q in ['"', "'", '"', '"']) else 0
    features["has_dash"] = 1 if any(d in title for d in ["-", "–", "—"]) else 0
    features["title_upper_ratio"] = (
        sum(c.isupper() for c in title) / len(title) if title else 0
    )
    features["title_caps_words"] = len(
        [w for w in title.split() if w.isupper() and len(w) > 1]
    )
    features["title_reading_ease"] = flesch_reading_ease(title) if title else 0
    features["abstract_reading_ease"] = flesch_reading_ease(abstract) if abstract else 0
    features["has_abstract"] = 1 if len(abstract) > 0 else 0
    features["title_abstract_ratio"] = features["title_length"] / (
        features["abstract_length"] + 1
    )
    features["avg_word_length"] = (
        np.mean([len(word) for word in title.split()])
        if title and len(title.split()) > 0
        else 0
    )

    return features


def predict_engagement(title, abstract="", category="news", model_pipeline=None):
    """Predict engagement for a single article"""
    if model_pipeline is None:
        return None

    try:
        features = create_article_features(title, abstract, category)

        # Category encoding
        with open(PREP_DIR / "processed_data" / "category_encoder.pkl", "rb") as f:
            le = pickle.load(f)

        if category in le.classes_:
            features["category_enc"] = le.transform([category])[0]
        else:
            features["category_enc"] = (
                le.transform(["unknown"])[0] if "unknown" in le.classes_ else 0
            )

        # Editorial features (simplified)
        features["editorial_readability_score"] = (
            features["title_reading_ease"] / 100 * 0.3
        )
        features["editorial_headline_score"] = (
            (features["has_question"] + features["has_number"] + features["has_colon"])
            / 3
            * 0.2
        )
        features["ctr_gain_potential"] = 0.05
        features["needs_readability_improvement"] = (
            1 if features["title_reading_ease"] < 60 else 0
        )
        features["suboptimal_word_count"] = (
            1
            if (features["title_word_count"] < 8 or features["title_word_count"] > 12)
            else 0
        )
        features["too_long_title"] = 1 if features["title_length"] > 75 else 0

        # Create title embedding
        embedder = load_embedder()
        title_embedding = embedder.encode([title])[0]

        # Add embeddings to features
        for i, emb_val in enumerate(title_embedding[:384]):
            features[f"title_emb_{i}"] = emb_val

        # Get available feature names from model
        available_features = model_pipeline.get("feature_names", [])
        if not available_features and hasattr(
            model_pipeline["model"], "feature_names_in_"
        ):
            available_features = list(model_pipeline["model"].feature_names_in_)

        # Create feature vector
        if available_features:
            feature_vector = [
                features.get(feat_name, 0) for feat_name in available_features
            ]
        else:
            feature_vector = [features[k] for k in sorted(features.keys())]

        feature_vector = np.array(feature_vector).reshape(1, -1)

        # Predict
        prediction = model_pipeline["model"].predict(feature_vector)[0]
        prediction_proba = model_pipeline["model"].predict_proba(feature_vector)[0]

        # Convert engagement probability to estimated CTR
        engagement_prob = float(prediction_proba[1])
        estimated_ctr = max(
            0.01, engagement_prob * 0.1
        )  # Scale to reasonable CTR range

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
    # Load models and systems
    model_pipeline = load_model()
    search_system = load_search_system()
    llm_rewriter = load_llm_rewriter()

    # Header
    st.title("📰 Article Engagement Predictor & AI Rewriter")
    st.markdown(
        "**Predict engagement and optimize headlines with AI-powered rewriting**"
    )

    # Sidebar
    st.sidebar.header("🎯 System Status")
    if model_pipeline:
        st.sidebar.success(f"✅ Model: {model_pipeline['model_name']}")
        if "auc" in model_pipeline.get("performance", {}):
            st.sidebar.info(f"📊 AUC: {model_pipeline['performance']['auc']:.4f}")
    else:
        st.sidebar.error("❌ Model not loaded")

    if search_system:
        st.sidebar.success(
            f"✅ Search: {search_system['metadata']['total_articles']:,} articles"
        )
        if search_system["metadata"].get("rewrite_variants", 0) > 0:
            st.sidebar.info(
                f"🔄 Rewrites: {search_system['metadata']['rewrite_variants']} variants"
            )
    else:
        st.sidebar.error("❌ Search system not loaded")

    if llm_rewriter.api_available:
        st.sidebar.success("✅ AI Rewriter: Available")
    else:
        st.sidebar.warning("⚠️ AI Rewriter: Offline mode")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔮 Predict & Rewrite",
            "🔍 Search Articles",
            "📊 Rewrite Analysis",
            "🏆 Top Articles",
        ]
    )

    # Tab 1: Predict & Rewrite
    with tab1:
        st.header("Predict Engagement & Generate AI Rewrites")

        col1, col2 = st.columns([3, 2])

        with col1:
            title = st.text_area(
                "Article Title",
                placeholder="Enter your article headline here...",
                height=100,
                help="Enter the headline you want to test and optimize",
            )

            abstract = st.text_area(
                "Abstract (Optional)",
                placeholder="Enter article abstract...",
                height=80,
                help="Optional: Add abstract for better predictions",
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

            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                predict_only = st.button("🎯 Predict Only", type="secondary")

            with col_btn2:
                predict_and_rewrite = st.button(
                    "🤖 Predict & AI Rewrite", type="primary"
                )

            if predict_only or predict_and_rewrite:
                if title.strip():
                    with st.spinner("Analyzing your article..."):
                        result = predict_engagement(
                            title, abstract, category, model_pipeline
                        )

                    if result:
                        # Display prediction results
                        st.subheader("📊 Engagement Prediction")

                        col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)

                        with col_pred1:
                            engagement_status = (
                                "🔥 High Engagement"
                                if result["high_engagement"]
                                else "📉 Low Engagement"
                            )
                            st.metric("Prediction", engagement_status)

                        with col_pred2:
                            st.metric(
                                "Probability", f"{result['engagement_probability']:.1%}"
                            )

                        with col_pred3:
                            st.metric("Confidence", f"{result['confidence']:.1%}")

                        with col_pred4:
                            st.metric("Est. CTR", f"{result['estimated_ctr']:.4f}")

                        # AI Rewriting section
                        if predict_and_rewrite:
                            st.subheader("🤖 AI-Powered Headline Rewrites")

                            with st.spinner("Generating optimized headlines..."):
                                article_data = {
                                    "category": category,
                                    "ctr": result[
                                        "estimated_ctr"
                                    ],  # Use model-predicted CTR
                                    "readability": result["features"][
                                        "title_reading_ease"
                                    ],
                                    "abstract": abstract,
                                }

                                rewrite_result = llm_rewriter.get_best_rewrite(
                                    title, article_data
                                )

                            if (
                                rewrite_result
                                and rewrite_result["best_rewrite"] != title
                            ):
                                # Show best rewrite
                                st.success("✨ Optimized Headline Generated")

                                col_orig, col_new = st.columns(2)

                                with col_orig:
                                    st.write("**Original:**")
                                    st.info(title)

                                with col_new:
                                    st.write("**AI Optimized:**")
                                    st.success(rewrite_result["best_rewrite"])

                                # Show improvement metrics
                                if "improvement_metrics" in rewrite_result:
                                    metrics = rewrite_result["improvement_metrics"]

                                    st.write("**Improvement Analysis:**")
                                    col_met1, col_met2, col_met3 = st.columns(3)

                                    with col_met1:
                                        st.metric(
                                            "Quality Score",
                                            f"{metrics.get('overall_quality_score', 0):.0f}/100",
                                        )

                                    with col_met2:
                                        st.metric(
                                            "Readability Δ",
                                            f"{metrics.get('readability_improvement', 0):+.1f}",
                                        )

                                    with col_met3:
                                        st.metric(
                                            "Est. CTR Boost",
                                            f"{metrics.get('predicted_ctr_improvement', 0):+.4f}",
                                        )

                                # Show all variants
                                if "all_variants" in rewrite_result:
                                    st.write("**All Strategy Variants:**")
                                    for strategy, variant in rewrite_result[
                                        "all_variants"
                                    ].items():
                                        if variant != title:
                                            st.write(
                                                f"• **{strategy.title()}:** {variant}"
                                            )

                            else:
                                st.info(
                                    "No significant improvements suggested for this headline"
                                )

                        # Feature Analysis
                        st.subheader("📈 Feature Analysis")
                        features = result["features"]

                        col_feat1, col_feat2 = st.columns(2)

                        with col_feat1:
                            st.write("**Title Characteristics:**")
                            st.write(f"• Length: {features['title_length']} characters")
                            st.write(
                                f"• Word count: {features['title_word_count']} words"
                            )
                            st.write(
                                f"• Reading ease: {features['title_reading_ease']:.1f}"
                            )
                            st.write(
                                f"• Has question: {'Yes' if features['has_question'] else 'No'}"
                            )
                            st.write(
                                f"• Has numbers: {'Yes' if features['has_number'] else 'No'}"
                            )

                        with col_feat2:
                            st.write("**Engagement Factors:**")
                            st.write(
                                f"• Exclamation: {'Yes' if features['has_exclamation'] else 'No'}"
                            )
                            st.write(
                                f"• Colon: {'Yes' if features['has_colon'] else 'No'}"
                            )
                            st.write(
                                f"• Quotes: {'Yes' if features['has_quotes'] else 'No'}"
                            )
                            st.write(f"• Capital words: {features['title_caps_words']}")
                            st.write(
                                f"• Abstract: {'Yes' if features['has_abstract'] else 'No'}"
                            )

                    else:
                        st.error(
                            "Failed to predict engagement. Please check your inputs."
                        )
                else:
                    st.warning("Please enter an article title.")

        with col2:
            # Tips and guidelines
            st.subheader("💡 Editorial Guidelines")
            st.write("**High-engagement headlines:**")
            st.write("• 8-12 words optimal")
            st.write("• Include numbers/questions")
            st.write("• High readability (60+ score)")
            st.write("• Front-load key information")
            st.write("• Under 75 characters")

            st.subheader("📝 Examples")
            st.write("**High Engagement:**")
            st.code("5 Ways AI Will Transform Healthcare")
            st.code("Why Tesla Stock Dropped 15% Today")

            st.write("**Low Engagement:**")
            st.code("Technology Report Published")
            st.code("General Business Update Information")

    # Tab 2: Search Articles
    with tab2:
        st.header("Search Articles & Rewrite Variants")

        search_query = st.text_input(
            "Search Query",
            placeholder="Enter keywords or describe the topic...",
            help="Search through articles and their rewrite variants",
        )

        col_search1, col_search2 = st.columns(2)

        with col_search1:
            num_results = st.slider("Number of results", 5, 20, 10)

        with col_search2:
            include_rewrites = st.checkbox("Include rewrite variants", value=True)

        if st.button("🔍 Search", type="primary"):
            if search_query.strip():
                with st.spinner("Searching articles..."):
                    # Search function (simplified)
                    embedder = load_embedder()
                    query_embedding = embedder.encode([search_query])
                    query_embedding = query_embedding.astype(np.float32)
                    faiss.normalize_L2(query_embedding)

                    search_k = num_results * 3 if include_rewrites else num_results
                    distances, indices = search_system["index"].search(
                        query_embedding, search_k
                    )

                    results = []
                    for dist, idx in zip(distances[0], indices[0]):
                        if idx in search_system["mappings"]["idx_to_article_id"]:
                            article_id = search_system["mappings"]["idx_to_article_id"][
                                idx
                            ]
                            if article_id in search_system["article_lookup"]:
                                article_info = search_system["article_lookup"][
                                    article_id
                                ].copy()
                                article_info["similarity_score"] = float(dist)
                                article_info["newsID"] = article_id

                                if (
                                    not include_rewrites
                                    and article_info.get("dataset") == "rewrite_variant"
                                ):
                                    continue

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
                                    st.write(f"**Abstract:** {abstract_preview}")

                            with col_art2:
                                st.metric(
                                    "Similarity", f"{article['similarity_score']:.3f}"
                                )
                                st.write(f"**Category:** {article['category']}")
                                st.write(f"**Dataset:** {article['dataset']}")

                                if article.get("dataset") == "rewrite_variant":
                                    st.write(
                                        f"**Strategy:** {article.get('rewrite_strategy', 'N/A')}"
                                    )
                                    st.write(
                                        f"**Quality:** {article.get('quality_score', 'N/A')}"
                                    )

                                if not pd.isna(article.get("ctr")):
                                    st.write(f"**CTR:** {article['ctr']:.4f}")

                                if not pd.isna(article.get("high_engagement")):
                                    engagement_status = (
                                        "🔥 High"
                                        if article["high_engagement"]
                                        else "📉 Low"
                                    )
                                    st.write(f"**Engagement:** {engagement_status}")
                else:
                    st.info("No articles found. Try different keywords.")
            else:
                st.warning("Please enter a search query.")

    # Tab 3: Rewrite Analysis
    with tab3:
        st.header("Headline Rewrite Analysis")

        if search_system and search_system["metadata"].get("rewrite_analysis"):
            rewrite_stats = search_system["metadata"]["rewrite_analysis"]

            st.subheader("📊 Rewrite Performance Summary")

            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

            with col_stat1:
                st.metric(
                    "Headlines Analyzed", rewrite_stats.get("unique_originals", 0)
                )

            with col_stat2:
                st.metric("Variants Generated", rewrite_stats.get("total_rewrites", 0))

            with col_stat3:
                st.metric(
                    "Best Strategy",
                    rewrite_stats.get("best_performing_strategy", "N/A"),
                )

            with col_stat4:
                model_used = rewrite_stats.get("model_based_ctr_used", False)
                st.metric("Model CTR", "✅ Yes" if model_used else "❌ No")

            # Load detailed rewrite results if available
            rewrite_file = FAISS_DIR / "rewrite_analysis" / "headline_rewrites.parquet"
            if rewrite_file.exists():
                try:
                    rewrite_df = pd.read_parquet(rewrite_file)

                    st.subheader("📈 Strategy Performance")

                    # Strategy comparison
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

                    # Create performance chart
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
                        "original_ctr",
                    ]
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

    # Tab 4: Top Articles
    with tab4:
        st.header("Top Performing Articles")

        if search_system:
            # Get high engagement articles
            high_engagement_articles = []
            for newsid, info in search_system["article_lookup"].items():
                if (
                    pd.notna(info.get("high_engagement"))
                    and info["high_engagement"] == 1
                ):
                    high_engagement_articles.append(
                        {
                            "newsID": newsid,
                            "title": info["title"],
                            "category": info["category"],
                            "ctr": info.get("ctr", 0),
                            "dataset": info["dataset"],
                            "abstract": info.get("abstract", ""),
                        }
                    )

            high_engagement_articles.sort(key=lambda x: x["ctr"], reverse=True)

            if high_engagement_articles:
                st.subheader(
                    f"🏆 Top {min(20, len(high_engagement_articles))} High-Engagement Articles"
                )

                # Filter options
                col_filter1, col_filter2 = st.columns(2)

                with col_filter1:
                    available_categories = sorted(
                        list(set(art["category"] for art in high_engagement_articles))
                    )
                    selected_categories = st.multiselect(
                        "Filter by Category", options=available_categories, default=[]
                    )

                with col_filter2:
                    selected_datasets = st.multiselect(
                        "Filter by Dataset",
                        options=["train", "val", "test"],
                        default=["train", "val"],
                    )

                # Apply filters
                filtered_articles = high_engagement_articles
                if selected_categories:
                    filtered_articles = [
                        art
                        for art in filtered_articles
                        if art["category"] in selected_categories
                    ]
                if selected_datasets:
                    filtered_articles = [
                        art
                        for art in filtered_articles
                        if art["dataset"] in selected_datasets
                    ]

                # Display top articles
                st.write(f"Showing top {min(20, len(filtered_articles))} articles:")

                for i, article in enumerate(filtered_articles[:20], 1):
                    with st.expander(f"{i}. [{article['ctr']:.4f}] {article['title']}"):
                        col_top1, col_top2 = st.columns([3, 1])

                        with col_top1:
                            st.write(f"**Title:** {article['title']}")
                            if article["abstract"]:
                                abstract_preview = (
                                    article["abstract"][:300] + "..."
                                    if len(article["abstract"]) > 300
                                    else article["abstract"]
                                )
                                st.write(f"**Abstract:** {abstract_preview}")

                        with col_top2:
                            st.metric("CTR", f"{article['ctr']:.4f}")
                            st.write(f"**Category:** {article['category']}")
                            st.write(f"**Dataset:** {article['dataset']}")
                            st.write(f"**ID:** {article['newsID']}")

            else:
                st.info("No high-engagement articles found in the dataset.")

        else:
            st.error("Top articles not available.")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Article Engagement Predictor & AI Rewriter** | Built with Streamlit, XGBoost, and OpenAI"
    )

    # Enhanced sidebar info
    if model_pipeline:
        st.sidebar.markdown("---")
        st.sidebar.header("📋 Model Details")
        st.sidebar.write(f"**Type:** {model_pipeline['model_name']}")

        if "performance" in model_pipeline:
            perf = model_pipeline["performance"]
            if "auc" in perf:
                st.sidebar.write(f"**AUC:** {perf['auc']:.4f}")
            if "ctr_gain_achieved" in perf:
                st.sidebar.write(f"**CTR Gain:** {perf['ctr_gain_achieved']:.4f}")

    if search_system:
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Search Index")
        metadata = search_system["metadata"]
        st.sidebar.write(f"**Articles:** {metadata['total_articles']:,}")
        if metadata.get("rewrite_variants", 0) > 0:
            st.sidebar.write(f"**Rewrites:** {metadata['rewrite_variants']:,}")

        # Show model integration status
        model_integration = metadata.get("model_integration", {})
        model_available = model_integration.get("model_available", False)
        st.sidebar.write(
            f"**Model Integration:** {'✅ Yes' if model_available else '❌ No'}"
        )


if __name__ == "__main__":
    main()
