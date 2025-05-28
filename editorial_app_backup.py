import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import json
import openai
from sentence_transformers import SentenceTransformer
from pathlib import Path
from model_loader import get_model_paths, check_pipeline_status, load_ctr_model
from textstat import flesch_reading_ease
import datetime
import random

# App Configuration & Custom Styles
st.set_page_config(
    page_title="AI News Editor Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    html, body, [class*=\"css\"]  {
        font-family: 'Poppins', sans-serif;
        background-color: #f5f5f5;
    }
    .block-container {
        background-color: #ffffff;
        border: 1px solid #dcdcdc;
        padding: 2rem;
        border-radius: 0.5rem;
    }
    h1, h2, h3, .stHeader {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }
    .stTable table {
        width: 100% !important;
    }
    .stTable td, .stTable th {
        padding: 4px 8px !important;
        font-size: 12px !important;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Helper Functions
@st.cache_resource
def load_resources():
    """Load all required resources with error handling."""
    paths = get_model_paths()
    status = check_pipeline_status()

    # Check if essential components are available
    if not status["preprocessing_completed"]:
        st.error(
            "Preprocessing not completed. Please run preprocessing pipeline first."
        )
        st.stop()

    try:
        # Load processed data
        train_features = pd.read_parquet(paths["train_features"])
        train_targets = pd.read_parquet(paths["train_targets"])

        # Load metadata
        with open(paths["feature_metadata"], "r") as f:
            feature_metadata = json.load(f)

        with open(paths["label_encoder"], "r") as f:
            label_encoder_info = json.load(f)

        # Load embeddings and FAISS index
        if status.get("embeddings_completed", True):
            with open(paths["embeddings"], "rb") as f:
                emb_data = pickle.load(f)
            index = faiss.read_index(str(paths["faiss_index"]))
        else:
            st.warning("Embeddings not available. Some features will be limited.")
            emb_data = None
            index = None

        # Load CTR model (prioritize LightGBM for compatibility)
        if status["training_completed"]:
            model_data, err = load_ctr_model(paths["ctr_model"])
            if err:
                st.error(f"Error loading CTR model: {err}")
                st.stop()
            ctr_model = model_data["model"]
            feature_names = model_data.get("feature_names", [])
            model_type = model_data.get("model_type", "lightgbm")
        else:
            st.warning("CTR model not trained. Please run training script first.")
            ctr_model = None
            feature_names = None
            model_type = None

        # Load sentence transformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        return {
            "train_features": train_features,
            "train_targets": train_targets,
            "feature_metadata": feature_metadata,
            "label_encoder_info": label_encoder_info,
            "emb_data": emb_data,
            "index": index,
            "ctr_model": ctr_model,
            "feature_names": feature_names,
            "model_type": model_type,
            "embedder": embedder,
            "status": status,
        }

    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        st.stop()


def extract_enhanced_features(
    headline,
    category="news",
    publish_time=None,
    day_type="Weekday",
    embedder=None,
    feature_metadata=None,
):
    """Enhanced feature extraction that matches your preprocessing pipeline."""

    if publish_time is None:
        publish_time = datetime.datetime.now()

    # Generate embeddings
    if embedder is not None:
        embs = embedder.encode([headline])
        emb_dim = feature_metadata.get("embedding_dim", 50)
    else:
        embs = np.zeros((1, 50))
        emb_dim = 50

    # Basic text features
    title_words = headline.split() if headline else []

    features = {
        # Text analysis features
        "title_length": len(headline) if headline else 0,
        "abstract_length": 100,  # Default for abstract
        "title_word_count": len(title_words),
        "abstract_word_count": 15,  # Default
        "title_reading_ease": flesch_reading_ease(headline) if headline.strip() else 50,
        "abstract_reading_ease": 60,  # Default
        # Pattern features
        "has_question": int("?" in headline),
        "has_exclamation": int("!" in headline),
        "has_number": int(any(c.isdigit() for c in headline)),
        "has_colon": int(":" in headline),
        "has_quotes": int(any(q in headline for q in ['"', "'"])),
        "has_hyphen": int("-" in headline),
        "has_brackets": int(any(b in headline for b in "[]())")),
        # Capitalization features
        "title_upper_ratio": (
            sum(c.isupper() for c in headline) / len(headline) if headline else 0
        ),
        "starts_with_caps": int(headline[0].isupper() if headline else False),
        # Enhanced temporal features
        "hour": publish_time.hour,
        "day_of_week": publish_time.weekday(),
        "is_weekend": int(day_type == "Weekend" or publish_time.weekday() >= 5),
        # News-specific temporal patterns
        "is_morning": int(6 <= publish_time.hour < 12),
        "is_lunch": int(11 <= publish_time.hour < 14),
        "is_evening": int(17 <= publish_time.hour < 22),
        "is_night": int(publish_time.hour >= 22 or publish_time.hour < 6),
        "is_commute_time": int(
            (7 <= publish_time.hour <= 9) or (17 <= publish_time.hour <= 19)
        ),
        "is_prime_time": int(19 <= publish_time.hour <= 23),
        # Cyclical encoding
        "hour_sin": np.sin(2 * np.pi * publish_time.hour / 24),
        "hour_cos": np.cos(2 * np.pi * publish_time.hour / 24),
        "day_sin": np.sin(2 * np.pi * publish_time.weekday() / 7),
        "day_cos": np.cos(2 * np.pi * publish_time.weekday() / 7),
        # Content freshness (assume recent)
        "hours_since_published": 1.0,
        "is_breaking_news": int(
            any(word in headline.lower() for word in ["breaking", "urgent", "just in"])
        ),
        "is_stale_news": 0,  # Assume fresh
        # Category encoding
        "category_enc": {
            "news": 0,
            "sports": 1,
            "entertainment": 2,
            "finance": 3,
            "lifestyle": 4,
            "technology": 5,
        }.get(category.lower(), 0),
        # Context features
        "title_complexity": len(headline) / len(title_words) if title_words else 0,
        "title_sentiment_words": len(
            [
                w
                for w in title_words
                if w.lower()
                in ["amazing", "shocking", "exclusive", "breaking", "urgent"]
            ]
        ),
        "has_detailed_abstract": 1,  # Assume yes
        "abstract_title_ratio": 100 / (len(headline) + 1),  # Default ratio
    }

    # Add embeddings
    for j in range(min(emb_dim, embs.shape[1])):
        features[f"emb_{j}"] = embs[0, j]

    return pd.DataFrame([features])


def predict_ctr_enhanced(headline, category, publish_time, day_type, resources):
    """Enhanced CTR prediction with better feature matching and NaN handling."""
    try:
        # Extract features with enhanced pipeline
        features_df = extract_enhanced_features(
            headline,
            category,
            publish_time,
            day_type,
            resources["embedder"],
            resources["feature_metadata"],
        )

        # Get available features in the model
        model_features = resources["feature_names"]

        if not model_features:
            st.warning("No model features available. Using default prediction.")
            return 0.01

        # Match features (use available ones, fill missing with defaults)
        final_features = []
        for feat_name in model_features:
            if feat_name in features_df.columns:
                value = features_df[feat_name].iloc[0]
                # Handle NaN values
                if pd.isna(value):
                    if "emb_" in feat_name:
                        final_features.append(0.0)
                    elif "category" in feat_name:
                        final_features.append(0)
                    else:
                        final_features.append(0.0)
                else:
                    final_features.append(float(value))
            else:
                # Provide reasonable defaults for missing features
                if "emb_" in feat_name:
                    final_features.append(0.0)
                elif "category" in feat_name:
                    final_features.append(0)
                elif "hour" in feat_name:
                    final_features.append(12.0)  # Default to noon
                elif "day" in feat_name:
                    final_features.append(2.0)  # Default to Tuesday
                else:
                    final_features.append(0.0)

        # Ensure we have the right number of features
        if len(final_features) != len(model_features):
            st.warning(
                f"Feature mismatch: expected {len(model_features)}, got {len(final_features)}"
            )
            return 0.01

        # Make prediction
        if resources["model_type"] == "lightgbm":
            prediction = resources["ctr_model"].predict([final_features])[0]
        else:
            # Fallback for other model types
            prediction = 0.01  # Default prediction

        # Handle NaN predictions
        if pd.isna(prediction):
            prediction = 0.01

        return max(0, min(1, float(prediction)))  # Clamp between 0 and 1

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0.01


def get_ctr_insights(ctr_prediction, category, publish_time):
    """Provide insights about the CTR prediction."""
    insights = []

    # Performance assessment
    if ctr_prediction > 0.02:
        insights.append(
            "🚀 **High Performance** - This headline is predicted to perform well"
        )
    elif ctr_prediction > 0.01:
        insights.append("📈 **Moderate Performance** - Decent engagement expected")
    else:
        insights.append("📉 **Low Performance** - Consider optimizing this headline")

    # Time-based insights
    hour = publish_time.hour
    if 7 <= hour <= 9:
        insights.append("☕ **Morning Commute** - Good time for news consumption")
    elif 12 <= hour <= 14:
        insights.append("🍽️ **Lunch Time** - Popular reading period")
    elif 17 <= hour <= 19:
        insights.append("🚗 **Evening Commute** - Peak engagement time")
    elif 19 <= hour <= 23:
        insights.append("📺 **Prime Time** - High attention period")

    # Category insights
    category_tips = {
        "news": "📰 Breaking news and urgent updates perform best",
        "sports": "⚽ Game results and player news drive engagement",
        "entertainment": "🎬 Celebrity news and trending topics work well",
        "finance": "💰 Market updates and investment advice are popular",
        "lifestyle": "✨ How-to guides and tips generate interest",
        "technology": "🔧 Product launches and reviews perform well",
    }

    if category.lower() in category_tips:
        insights.append(category_tips[category.lower()])

    return insights


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Initialize Data
with st.spinner("Loading models and data..."):
    resources = load_resources()

# Load header image
img_dir = Path("images")
if img_dir.exists():
    all_images = list(img_dir.glob("*.jpg"))
    if all_images:
        selected = random.choice(all_images)
        st.image(str(selected), width=600)

st.title("📰 AI News Editor Assistant")
st.markdown("*Enhanced with temporal intelligence and contextual features*")

# Display pipeline status
if not all(resources["status"].values()):
    with st.expander("⚙️ Pipeline Status", expanded=True):
        for stage, completed in resources["status"].items():
            status_icon = "✅" if completed else "❌"
            st.write(f"{status_icon} {stage.replace('_', ' ').title()}")

# UI Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Retrieve & Rank", "⚡ Predict CTR", "✍️ Rewrite Headlines", "📊 Analytics"]
)

with tab1:
    st.header("🔍 Article Retrieval & Ranking")
    st.markdown("Find similar articles using semantic search")

    if not resources.get("status", {}).get("embeddings_completed", True):
        st.warning("⚠️ Embeddings not available. Please run embedding script first.")
    else:
        col1, col2 = st.columns([3, 1])

        with col1:
            query = st.text_input(
                "Search Query:",
                "big tech layoffs",
                help="Enter keywords to find similar articles",
            )
        with col2:
            k = st.slider("Top K results:", 1, 20, 5)

        if st.button("🔍 Search Articles", type="primary"):
            try:
                # Generate query embedding
                q_emb = resources["embedder"].encode([query]).astype("float32")

                # Handle dimension matching
                index_dim = resources["index"].d
                query_dim = q_emb.shape[1]

                if query_dim != index_dim:
                    if query_dim > index_dim:
                        q_emb = q_emb[:, :index_dim]
                        st.info(
                            f"Adjusted embedding dimensions: {query_dim} → {index_dim}"
                        )
                    else:
                        padding = np.zeros(
                            (q_emb.shape[0], index_dim - query_dim), dtype=np.float32
                        )
                        q_emb = np.concatenate([q_emb, padding], axis=1)

                # Search
                D, I = resources["index"].search(q_emb, k)

                if len(I[0]) > 0:
                    # Create results
                    results = []
                    for i, (similarity_score, idx) in enumerate(zip(D[0], I[0])):
                        if idx < len(resources["emb_data"]["article_indices"]):
                            article_idx = resources["emb_data"]["article_indices"][idx]
                            split_label = (
                                resources["emb_data"]["split_labels"][idx]
                                if idx < len(resources["emb_data"]["split_labels"])
                                else "unknown"
                            )

                            results.append(
                                {
                                    "Rank": i + 1,
                                    "Article_ID": f"Article_{article_idx}",
                                    "Similarity": f"{float(similarity_score):.4f}",
                                    "Dataset": split_label.title(),
                                    "Relevance": (
                                        "🔥 High"
                                        if similarity_score < 0.5
                                        else (
                                            "📈 Medium"
                                            if similarity_score < 1.0
                                            else "📊 Low"
                                        )
                                    ),
                                }
                            )

                    if results:
                        st.success(f"✅ Found {len(results)} similar articles")

                        # Display results in a nice format
                        results_df = pd.DataFrame(results)
                        st.dataframe(
                            results_df, use_container_width=True, hide_index=True
                        )

                        # Show statistics
                        avg_similarity = np.mean(
                            [float(r["Similarity"]) for r in results]
                        )
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🎯 Avg Similarity", f"{avg_similarity:.4f}")
                        with col2:
                            st.metric("📊 Results Found", len(results))
                        with col3:
                            st.metric("🔍 Query Terms", len(query.split()))
                    else:
                        st.error("❌ No valid results found")
                else:
                    st.error("❌ No results found for this query")

            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")

with tab2:
    st.header("⚡ Advanced CTR Prediction")
    st.markdown("Get precise CTR estimates with contextual intelligence")

    if not resources["status"]["training_completed"]:
        st.warning("⚠️ CTR model not available. Please run training script first.")
    else:
        # Enhanced input form
        with st.form("ctr_prediction_form"):
            st.subheader("📝 Headline Details")

            col1, col2 = st.columns(2)

            with col1:
                headline = st.text_input(
                    "Headline Text:",
                    placeholder="Enter your headline here...",
                    help="The headline you want to analyze",
                )

                category = st.selectbox(
                    "📂 Category:",
                    [
                        "news",
                        "sports",
                        "entertainment",
                        "finance",
                        "lifestyle",
                        "technology",
                    ],
                    help="Select the content category",
                )

            with col2:
                publish_time = st.time_input(
                    "🕐 Publish Time:",
                    value=datetime.time(12, 0),
                    help="When will this be published?",
                )

                day_type = st.radio(
                    "📅 Day Type:",
                    ["Weekday", "Weekend"],
                    help="Affects reader behavior patterns",
                )

            predict_btn = st.form_submit_button("🚀 Predict CTR", type="primary")

        if predict_btn and headline:
            # Create datetime object for prediction
            current_date = datetime.datetime.now().date()
            publish_datetime = datetime.datetime.combine(current_date, publish_time)

            # Make prediction
            with st.spinner("🔮 Analyzing headline..."):
                ctr_prediction = predict_ctr_enhanced(
                    headline, category, publish_datetime, day_type, resources
                )

            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")

            # Main metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "🎯 Predicted CTR",
                    f"{ctr_prediction:.3%}",
                    help="Expected click-through rate",
                )

            with col2:
                # Calculate relative performance
                baseline_ctr = resources["train_targets"]["ctr"].mean()
                improvement = ((ctr_prediction - baseline_ctr) / baseline_ctr) * 100
                st.metric(
                    "📈 vs Baseline",
                    f"{improvement:+.1f}%",
                    delta=f"{improvement:+.1f}%",
                    help="Performance vs average headline",
                )

            with col3:
                # Estimated clicks per 1000 impressions
                clicks_per_1k = ctr_prediction * 1000
                st.metric(
                    "👆 Clicks/1K Views",
                    f"{clicks_per_1k:.1f}",
                    help="Expected clicks per 1000 impressions",
                )

            # Insights
            st.subheader("💡 AI Insights")
            insights = get_ctr_insights(ctr_prediction, category, publish_datetime)
            for insight in insights:
                st.markdown(f"• {insight}")

            # Feature analysis
            with st.expander("🔍 Feature Analysis"):
                features_df = extract_enhanced_features(
                    headline,
                    category,
                    publish_datetime,
                    day_type,
                    resources["embedder"],
                    resources["feature_metadata"],
                )

                # Show key features
                key_features = {
                    "Title Length": features_df["title_length"].iloc[0],
                    "Word Count": features_df["title_word_count"].iloc[0],
                    "Readability Score": f"{features_df['title_reading_ease'].iloc[0]:.1f}",
                    "Hour": features_df["hour"].iloc[0],
                    "Has Question": (
                        "Yes" if features_df["has_question"].iloc[0] else "No"
                    ),
                    "Has Numbers": "Yes" if features_df["has_number"].iloc[0] else "No",
                    "Breaking News": (
                        "Yes" if features_df["is_breaking_news"].iloc[0] else "No"
                    ),
                }

                for feature, value in key_features.items():
                    st.markdown(f"**{feature}:** {value}")

        # Reference statistics
        st.markdown("---")
        st.subheader("📋 Reference Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🔎 Dataset Statistics**")
            train_ctr_mean = resources["train_targets"]["ctr"].mean()
            train_ctr_with_impressions = resources["train_targets"][
                resources["train_targets"]["ctr"] > 0
            ]["ctr"]

            st.metric("Overall Mean CTR", f"{train_ctr_mean:.3%}")
            if len(train_ctr_with_impressions) > 0:
                st.metric(
                    "Mean CTR (with clicks)", f"{train_ctr_with_impressions.mean():.3%}"
                )

        with col2:
            st.markdown("**⏰ Best Publishing Times**")
            st.markdown("• **Morning**: 7-9 AM (commute)")
            st.markdown("• **Lunch**: 12-2 PM (break time)")
            st.markdown("• **Evening**: 5-7 PM (commute)")
            st.markdown("• **Prime**: 7-11 PM (leisure)")

with tab3:
    st.header("✍️ AI-Powered Headline Rewriting")
    st.markdown("Generate optimized headlines using GPT-4")

    if not resources["status"]["training_completed"]:
        st.warning("⚠️ CTR model not available. Please run training script first.")
    else:
        # Input form
        with st.form("headline_rewriting_form"):
            st.subheader("📝 Original Headline")

            col1, col2 = st.columns([3, 1])

            with col1:
                original = st.text_input(
                    "Original Headline:",
                    placeholder="Enter the headline you want to improve...",
                    help="The headline you want to optimize",
                )

            with col2:
                n_variations = st.slider("Variations:", 1, 5, 3)

            st.subheader("🔑 OpenAI Configuration")
            openai_key = st.text_input(
                "OpenAI API Key:",
                type="password",
                help="Your OpenAI API key for GPT-4 access",
            )

            # Context settings
            col1, col2 = st.columns(2)
            with col1:
                rewrite_category = st.selectbox(
                    "Category:",
                    [
                        "news",
                        "sports",
                        "entertainment",
                        "finance",
                        "lifestyle",
                        "technology",
                    ],
                )

            with col2:
                optimization_focus = st.selectbox(
                    "Optimization Focus:",
                    [
                        "Click-through Rate",
                        "Readability",
                        "Emotional Appeal",
                        "Urgency",
                        "Curiosity",
                    ],
                )

            rewrite_btn = st.form_submit_button(
                "✨ Generate Variations", type="primary"
            )

        if rewrite_btn and original and openai_key:
            try:
                # Set up OpenAI
                client = openai.OpenAI(api_key=openai_key)

                # Create focused instructions based on optimization focus
                focus_instructions = {
                    "Click-through Rate": "Focus on creating headlines that maximize click-through rates with compelling hooks and curiosity gaps.",
                    "Readability": "Prioritize clarity and easy-to-understand language with high readability scores.",
                    "Emotional Appeal": "Use emotional triggers and power words that resonate with readers.",
                    "Urgency": "Create a sense of urgency and timeliness that compels immediate action.",
                    "Curiosity": "Build curiosity gaps that make readers want to learn more.",
                }

                detailed_instructions = f"""
You are an expert news editor optimizing headlines for {rewrite_category} content.

OPTIMIZATION FOCUS: {focus_instructions[optimization_focus]}

Guidelines:
1. Preserve factual accuracy and key information
2. Keep headlines under 70 characters for optimal display
3. Maintain semantic similarity to the original
4. Use power words and emotional triggers appropriately
5. Consider the target category: {rewrite_category}

Original headline: {original}

Generate {n_variations} improved variations, each on a separate line:
"""

                with st.spinner("🤖 Generating optimized headlines..."):
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert headline optimizer.",
                            },
                            {"role": "user", "content": detailed_instructions},
                        ],
                        temperature=0.7,
                        max_tokens=500,
                    )

                    # Extract candidates
                    text = response.choices[0].message.content.strip()
                    candidates = [
                        line.strip("- ").strip()
                        for line in text.splitlines()
                        if line.strip() and not line.startswith("Generate")
                    ][:n_variations]

                if candidates:
                    st.markdown("---")
                    st.subheader("📊 Headline Analysis")

                    # Analyze original headline
                    current_time = datetime.datetime.now()
                    orig_ctr = predict_ctr_enhanced(
                        original, rewrite_category, current_time, "Weekday", resources
                    )
                    orig_emb = resources["embedder"].encode([original])[0]

                    # Analyze all candidates
                    results = []
                    for i, candidate in enumerate(candidates, 1):
                        ctr_pred = predict_ctr_enhanced(
                            candidate,
                            rewrite_category,
                            current_time,
                            "Weekday",
                            resources,
                        )
                        readability = (
                            flesch_reading_ease(candidate) if candidate.strip() else 0
                        )
                        similarity = cosine_sim(
                            orig_emb, resources["embedder"].encode([candidate])[0]
                        )

                        results.append(
                            {
                                "Rank": i,
                                "Headline": candidate,
                                "CTR": ctr_pred,
                                "CTR_Display": f"{ctr_pred:.2%}",
                                "Readability": f"{readability:.1f}",
                                "Similarity": f"{similarity:.2f}",
                                "Length": len(candidate),
                                "Improvement": f"{((ctr_pred - orig_ctr) / orig_ctr * 100):+.1f}%",
                            }
                        )

                    # Sort by CTR
                    results.sort(key=lambda x: x["CTR"], reverse=True)

                    # Display original vs best
                    best_ctr = results[0]["CTR"]
                    improvement = ((best_ctr - orig_ctr) / orig_ctr) * 100

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📄 Original CTR", f"{orig_ctr:.2%}")
                    with col2:
                        st.metric("🚀 Best Variation CTR", f"{best_ctr:.2%}")
                    with col3:
                        st.metric(
                            "📈 Improvement",
                            f"{improvement:+.1f}%",
                            delta=f"{improvement:+.1f}%",
                        )

                    # Display results table
                    st.subheader("🏆 Ranked Headlines")

                    display_df = pd.DataFrame(results)[
                        [
                            "Rank",
                            "Headline",
                            "CTR_Display",
                            "Readability",
                            "Similarity",
                            "Length",
                            "Improvement",
                        ]
                    ]
                    display_df.columns = [
                        "Rank",
                        "Headline",
                        "CTR",
                        "Readability",
                        "Similarity",
                        "Length",
                        "vs Original",
                    ]

                    # Color code the best result
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    # Show usage info
                    st.info(f"💰 OpenAI tokens used: {response.usage.total_tokens}")

                    # Best headline recommendation
                    st.markdown("---")
                    st.subheader("🎯 Recommendation")
                    best_headline = results[0]
                    st.success(
                        f"**Best Performing Headline:** {best_headline['Headline']}"
                    )
                    st.markdown(
                        f"**Expected Performance:** {best_headline['CTR_Display']} CTR ({best_headline['Improvement']} improvement)"
                    )

            except Exception as e:
                st.error(f"❌ Error generating headlines: {str(e)}")

with tab4:
    st.header("📊 Model Analytics")
    st.markdown("Performance insights and model information")

    # Model information
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Model Information")
        st.markdown(
            f"**Model Type:** {resources['model_type'].title() if resources['model_type'] else 'Unknown'}"
        )
        st.markdown(
            f"**Features Used:** {len(resources['feature_names']) if resources['feature_names'] else 'Unknown'}"
        )
        st.markdown(f"**Training Samples:** {len(resources['train_targets']):,}")

        # Feature importance (if available)
        if hasattr(resources["ctr_model"], "feature_importances_"):
            st.subheader("🎯 Top Features")
            importances = resources["ctr_model"].feature_importances_
            feature_importance = (
                pd.DataFrame(
                    {
                        "Feature": resources["feature_names"][: len(importances)],
                        "Importance": importances,
                    }
                )
                .sort_values("Importance", ascending=False)
                .head(10)
            )

            st.bar_chart(feature_importance.set_index("Feature")["Importance"])

    with col2:
        st.subheader("📈 Dataset Statistics")

        # CTR distribution
        ctr_data = resources["train_targets"]["ctr"]

        st.markdown
