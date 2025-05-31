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
import llm_rewriter
from feature_utils import create_article_features_exact, load_preprocessing_components
import datetime
import io

warnings.filterwarnings("ignore")


# Add this logging function at the top, after your imports
def log_event(event_type, data):
    """Log events to track user behavior"""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "event": event_type,
        "data": data,
    }

    try:
        with open("usage_log.txt", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except:
        pass  # Fail silently if can't write


def get_usage_stats():
    """Read usage statistics"""
    try:
        with open("usage_log.txt", "r") as f:
            logs = [json.loads(line) for line in f]
        return logs
    except:
        return []


# Page configuration
st.set_page_config(
    page_title="Headline Hunter",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add this to track page visits
# AFTER: Only logs when YOU set development mode
if os.getenv("STREAMLIT_ENV") == "development":
    log_event("page_visit", {"user_agent": "streamlit_user"})

# Constants
MODEL_DIR = Path("model_output")
FAISS_DIR = Path("faiss_index")
PREP_DIR = Path("data/preprocessed")

# Enhanced CSS Styles with Mobile Optimization
st.markdown(
    """
    <style>
      .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
      }
      
      /* Mobile-responsive header */
      @media (max-width: 768px) {
        .main-header {
          padding: 1rem;
          margin-bottom: 1rem;
        }
        .header-content {
          flex-direction: column !important;
          gap: 15px !important;
        }
        .header-logo {
          width: 60px !important;
          height: 60px !important;
        }
        .header-title {
          font-size: 2rem !important;
          line-height: 1.2 !important;
        }
        .header-tagline {
          font-size: 0.9rem !important;
          margin-top: 0.5rem !important;
        }
      }
      
      .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
      }
      
      @media (max-width: 768px) {
        .metric-card {
          padding: 1rem;
          margin-bottom: 0.75rem;
        }
      }
      
      .improvement-positive {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
      }
      .improvement-negative {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
      }
      .improvement-neutral {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
      }
      
      @media (max-width: 768px) {
        .improvement-positive, .improvement-negative, .improvement-neutral {
          padding: 0.75rem;
          margin: 0.4rem 0;
          font-size: 14px;
        }
      }
      
      .comparison-container {
        display: flex;
        gap: 20px;
        margin: 20px 0;
      }
      
      @media (max-width: 768px) {
        .comparison-container {
          flex-direction: column;
          gap: 15px;
          margin: 15px 0;
        }
      }
      
      .comparison-box {
        flex: 1;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #ddd;
      }
      
      @media (max-width: 768px) {
        .comparison-box {
          padding: 15px;
        }
        .comparison-box h4 {
          font-size: 16px;
          margin-bottom: 10px;
        }
      }
      
      .original-box {
        background-color: #fff5f5;
        border-color: #fed7d7;
      }
      .optimized-box {
        background-color: #f0fff4;
        border-color: #9ae6b4;
      }
      
      /* Mobile-responsive mode selector */
      @media (max-width: 768px) {
        .stRadio > div {
          flex-direction: column !important;
          gap: 10px !important;
        }
        .stRadio > div > label {
          margin-right: 0 !important;
          margin-bottom: 8px !important;
        }
      }
      
      /* Mobile text areas and inputs */
      @media (max-width: 768px) {
        .stTextArea textarea {
          font-size: 16px !important; /* Prevents zoom on iOS */
        }
        .stTextInput input {
          font-size: 16px !important; /* Prevents zoom on iOS */
        }
        .stSelectbox select {
          font-size: 16px !important;
        }
      }
      
      /* Better mobile button spacing */
      @media (max-width: 768px) {
        .stButton button {
          width: 100%;
          margin: 10px 0;
          padding: 12px 20px;
          font-size: 16px;
        }
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
def load_llm_rewriter():
    """Load the efficient LLM headline rewriter with correct EDA insights path"""
    try:
        model_pipeline = load_model()
        components = load_preprocessing_components()

        if model_pipeline and components:
            # Try the preprocessed data directory first
            eda_insights_path = (
                "data/preprocessed/processed_data/headline_eda_insights.json"
            )
            if not Path(eda_insights_path).exists():
                # Fall back to project root
                eda_insights_path = "headline_eda_insights.json"

            return llm_rewriter.EnhancedLLMHeadlineRewriter(
                model_pipeline=model_pipeline,
                components=components,
                eda_insights_path=eda_insights_path,
            )
        else:
            st.warning("Could not load model pipeline or components for rewriter")
            return None

    except Exception as e:
        st.error(f"Error loading LLM rewriter: {e}")
        # Show the actual error for debugging
        with st.expander("🔍 Debug Info"):
            import traceback

            st.code(traceback.format_exc())
        return None


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


def get_personalized_tips(features, result, improvement):
    """Generate personalized optimization tips based on analysis"""
    tips = []

    # Only provide tips if there's room for improvement
    if improvement >= 2:  # If improvement is good, no tips needed
        return []

    # Feature-based tips for headlines that need improvement
    if not features.get("has_number", 0):
        tips.append("Add specific numbers or statistics for credibility")

    if not features.get("has_question", 0):
        tips.append("Turn into a question to spark curiosity")

    if features.get("title_length", 0) > 75:
        tips.append("Shorten headline - aim for under 75 characters")

    word_count = features.get("title_word_count", 0)
    if word_count < 6:
        tips.append("Add more context - headlines with 8-12 words perform better")
    elif word_count > 15:
        tips.append("Simplify language - long headlines lose engagement")

    # Readability tips
    if features.get("title_reading_ease", 50) < 50:
        tips.append("Use simpler words to improve readability")

    # Performance-based tips
    if result["estimated_ctr"] < 0.03:
        tips.append("Consider using emotional triggers or urgency words")
        tips.append("Try power words like 'exclusive', 'breaking', or 'revealed'")

    return tips[:3]  # Return top 3 tips


def process_batch_headlines(uploaded_file, llm_rewriter, model_pipeline, components):
    """Process batch uploaded headlines"""
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Validate required columns
        if "headline" not in df.columns:
            st.error("CSV must contain a 'headline' column")
            return None

        # Add category column if not present
        if "category" not in df.columns:
            df["category"] = "news"

        results = []
        progress_bar = st.progress(0)

        for idx, row in df.iterrows():
            progress_bar.progress((idx + 1) / len(df))

            headline = row["headline"]
            category = row.get("category", "news")

            # Get original prediction
            original_result = predict_engagement(
                headline, "", category, model_pipeline, components
            )

            # Get optimized headline
            article_data = {"category": category, "abstract": ""}
            rewrite_result = llm_rewriter.get_best_headline(headline, article_data)

            optimized_headline = rewrite_result.get("best_headline", headline)
            optimized_result = predict_engagement(
                optimized_headline, "", category, model_pipeline, components
            )

            # Calculate improvement
            original_ctr = original_result["estimated_ctr"] if original_result else 0
            optimized_ctr = optimized_result["estimated_ctr"] if optimized_result else 0
            improvement = (
                ((optimized_ctr - original_ctr) / original_ctr * 100)
                if original_ctr > 0
                else 0
            )

            results.append(
                {
                    "original_headline": headline,
                    "optimized_headline": optimized_headline,
                    "original_ctr": f"{original_ctr*100:.2f}%",
                    "optimized_ctr": f"{optimized_ctr*100:.2f}%",
                    "improvement": f"{improvement:+.1f}%",
                    "category": category,
                }
            )

        progress_bar.empty()
        return pd.DataFrame(results)

    except Exception as e:
        st.error(f"Error processing batch file: {e}")
        return None


def main():

    # Header with logo - try to load your logo first, fallback to placeholder
    logo_path = Path("NEXUS_MARK_cmyk_page-0001-remove-background.com.png")

    if logo_path.exists():
        # Display your actual logo with proper header
        st.markdown(
            """
        <div class="main-header">
            <div class="header-content" style="display: flex; align-items: center; justify-content: center; gap: 25px; margin-bottom: 0.5rem;">
        """,
            unsafe_allow_html=True,
        )

        # Display logo using Streamlit's image function
        col_logo, col_title = st.columns([1, 4])
        with col_logo:
            st.image(str(logo_path), width=100)

        with col_title:
            st.markdown(
                """
                <h1 class="header-title" style="margin: 0; color: white; font-size: 3rem; font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Inter', sans-serif; font-weight: 700; letter-spacing: -0.02em;">Headline Hunter</h1>
                <div class="header-tagline" style="font-size: 1rem; color: rgba(255,255,255,0.9); margin-top: 0.3rem; font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Inter', sans-serif; font-weight: 400;">AI-powered headline optimization that drives engagement</div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("</div></div>", unsafe_allow_html=True)
    else:
        # Fallback header with placeholder
        st.markdown(
            """
        <div class="main-header">
            <div class="header-content" style="display: flex; align-items: center; justify-content: center; gap: 25px; margin-bottom: 0.5rem;">
                <div class="header-logo" style="width: 80px; height: 80px; display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.2); border-radius: 12px; font-size: 2rem; color: white; font-weight: bold;">HL</div>
                <div style="text-align: center;">
                    <h1 class="header-title" style="margin: 0; color: white; font-size: 3rem; font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Inter', sans-serif; font-weight: 700; letter-spacing: -0.02em;">Headline Hunter</h1>
                    <div class="header-tagline" style="font-size: 1rem; color: rgba(255,255,255,0.9); margin-top: 0.3rem; font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Inter', sans-serif; font-weight: 400;">AI-powered headline optimization that drives engagement</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Load all systems once at startup
    with st.spinner("Loading AI systems..."):
        model_pipeline = load_model()
        preprocessing_components = load_preprocessing_components()
        llm_rewriter = load_llm_rewriter()

    # Admin panel - completely hidden until correct password
    admin_key = os.getenv("ADMIN_PASSWORD")
    show_admin = False

    if admin_key:
        # Check if correct password was entered (using session state)
        if "admin_authenticated" not in st.session_state:
            st.session_state.admin_authenticated = False

        # Only show password field in main area if not authenticated
        if not st.session_state.admin_authenticated:
            # Hide sidebar completely for public users
            st.markdown(
                """
            <style>
            section[data-testid="stSidebar"] {display: none !important;}
            </style>
            """,
                unsafe_allow_html=True,
            )

            # Show password input in main area (discrete)
            with st.expander("🔐 Admin Access", expanded=False):
                entered_password = st.text_input(
                    "Password:", type="password", key="admin_main"
                )
                if st.button("Login") and entered_password == admin_key:
                    st.session_state.admin_authenticated = True
                    st.rerun()
        else:
            # Authenticated - show admin in sidebar
            st.sidebar.markdown("👤 **Admin Panel**")

            # Logout button
            if st.sidebar.button("🚪 Logout"):
                st.session_state.admin_authenticated = False
                st.rerun()

            logs = get_usage_stats()
            st.sidebar.success(f"✅ {len(logs)} interactions logged")

            # Metrics
            st.sidebar.metric("Total Interactions", len(logs))
            st.sidebar.metric(
                "Headlines Optimized",
                len([l for l in logs if l["event"] == "headline_optimization"]),
            )
            st.sidebar.metric(
                "Batch Uploads",
                len([l for l in logs if l["event"] == "batch_optimization"]),
            )

            # Download button
            if logs:
                log_data = "\n".join([json.dumps(log) for log in logs])
                st.sidebar.download_button(
                    label="📥 Download Analytics",
                    data=log_data,
                    file_name=f"analytics_{datetime.date.today()}.txt",
                    mime="text/plain",
                )
    else:
        # No admin password set - hide sidebar completely
        st.markdown(
            """
        <style>
        section[data-testid="stSidebar"] {display: none !important;}
        </style>
        """,
            unsafe_allow_html=True,
        )

    # Mode selector
    mode = st.radio(
        "Choose optimization mode:",
        ["🎯 Single Headline", "📊 Batch Upload", "⚖️ Comparison Mode"],
        horizontal=True,
    )

    if mode == "🎯 Single Headline":
        # Single headline optimization
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Optimize Your Headline")

            title = st.text_area(
                "Enter your headline:",
                placeholder="Local Team Wins Championship Game",
                height=100,
                help="Enter the headline you want to optimize",
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
                # "kids",
                # "northamerica",
                # "middleeast",
                # "unknown",
            ]

            category = st.selectbox("Category:", categories, index=0)
            optimize_btn = st.button("🚀 Optimize Headline", type="primary")

        with col2:
            # Simplified How to Use guide
            st.markdown("### 🚀 How to Use")
            st.markdown(
                """
            1. **Enter your headline** in the text box
            2. **Select the category** from the dropdown  
            3. **Click "Optimize Headline"** to see AI suggestions
            4. **Compare** original vs optimized performance
            5. **Try batch mode** for multiple headlines at once
            """
            )

        if optimize_btn and title.strip():
            if model_pipeline and preprocessing_components and llm_rewriter:
                # Log the event
                log_event(
                    "headline_optimization",
                    {"headline": title, "category": category, "mode": "single"},
                )

                with st.spinner("🤖 Optimizing your headline..."):
                    # Get original prediction
                    original_result = predict_engagement(
                        title, "", category, model_pipeline, preprocessing_components
                    )

                    # Get optimized headline
                    article_data = {"category": category, "abstract": ""}
                    rewrite_result = llm_rewriter.get_best_headline(title, article_data)

                    optimized_headline = rewrite_result.get("best_headline", title)
                    optimized_result = predict_engagement(
                        optimized_headline,
                        "",
                        category,
                        model_pipeline,
                        preprocessing_components,
                    )

                    if original_result and optimized_result:
                        # Calculate improvement
                        original_ctr = original_result["estimated_ctr"]
                        optimized_ctr = optimized_result["estimated_ctr"]
                        improvement = (
                            ((optimized_ctr - original_ctr) / original_ctr * 100)
                            if original_ctr > 0
                            else 0
                        )

                        # Display results in comparison format
                        st.markdown("### 📊 Optimization Results")

                        col_orig, col_opt = st.columns(2)

                        with col_orig:
                            st.markdown("**📝 Original Headline**")
                            st.info(title)
                            st.metric("Estimated CTR", f"{original_ctr*100:.2f}%")

                        with col_opt:
                            st.markdown("**✨ Optimized Headline**")
                            if optimized_headline != title:
                                st.success(optimized_headline)
                            else:
                                st.info(optimized_headline)
                            st.metric(
                                "Estimated CTR",
                                f"{optimized_ctr*100:.2f}%",
                                f"{improvement:+.1f}%",
                            )

                        # Improvement summary
                        if improvement > 5:
                            st.success(
                                f"🎉 **Excellent!** {improvement:+.1f}% CTR improvement"
                            )
                        elif improvement > 0:
                            st.success(
                                f"📈 **Good improvement:** {improvement:+.1f}% CTR boost"
                            )
                        elif improvement > -2:
                            st.info(
                                f"📊 **Minimal change** - original was already well-optimized"
                            )
                        else:
                            st.warning(
                                f"⚠️ **Consider alternative approach:** {improvement:.1f}% change"
                            )

                        # Personalized tips - only show if there's meaningful room for improvement
                        if (
                            improvement < 2
                        ):  # Only show tips if improvement is less than 2%
                            tips = get_personalized_tips(
                                original_result["features"],
                                original_result,
                                improvement,
                            )
                            if tips:
                                st.markdown(
                                    "### 💡 Personalized Tips for Better Headlines"
                                )
                                for tip in tips:
                                    st.markdown(f"• {tip}")

    elif mode == "📊 Batch Upload":
        # Batch upload section
        st.subheader("📊 Batch Headline Optimization")
        st.write(
            "Upload a CSV file with headlines to optimize multiple headlines at once"
        )

        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="CSV should contain a 'headline' column. Optional 'category' column.",
        )

        # Download sample template
        sample_df = pd.DataFrame(
            {
                "headline": [
                    "Local Team Wins Game",
                    "Stock Market Changes Today",
                    "New Health Study Released",
                ],
                "category": ["sports", "finance", "health"],
            }
        )

        csv_sample = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=csv_sample,
            file_name="headline_template.csv",
            mime="text/csv",
        )

        if uploaded_file and llm_rewriter:
            if st.button("🚀 Optimize All Headlines", type="primary"):
                # Log batch optimization
                log_event("batch_optimization", {"filename": uploaded_file.name})

                # Process batch
                results_df = process_batch_headlines(
                    uploaded_file,
                    llm_rewriter,
                    model_pipeline,
                    preprocessing_components,
                )

                if results_df is not None:
                    st.success(f"✅ Optimized {len(results_df)} headlines!")

                    # Show summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        improved_count = sum(
                            1
                            for imp in results_df["improvement"]
                            if float(imp.replace("%", "").replace("+", "")) > 0
                        )
                        st.metric(
                            "Headlines Improved", f"{improved_count}/{len(results_df)}"
                        )

                    with col2:
                        avg_improvement = np.mean(
                            [
                                float(imp.replace("%", "").replace("+", ""))
                                for imp in results_df["improvement"]
                            ]
                        )
                        st.metric("Average Improvement", f"{avg_improvement:+.1f}%")

                    with col3:
                        best_improvement = max(
                            [
                                float(imp.replace("%", "").replace("+", ""))
                                for imp in results_df["improvement"]
                            ]
                        )
                        st.metric("Best Improvement", f"{best_improvement:+.1f}%")

                    # Display results
                    st.subheader("📋 Optimization Results")
                    st.dataframe(results_df, use_container_width=True)

                    # Download results
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Optimized Headlines",
                        data=csv_results,
                        file_name=f"optimized_headlines_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

    elif mode == "⚖️ Comparison Mode":
        # Comparison mode
        st.subheader("⚖️ Compare Headlines Side-by-Side")
        st.write("Test multiple headline variations against each other")

        # Better UX layout for both desktop and mobile
        col1, col2 = st.columns([2, 1])

        with col1:
            # Input multiple headlines
            num_headlines = st.slider("Number of headlines to compare:", 2, 5, 3)

        with col2:
            # Enhanced category selection with beautiful UI
            st.markdown(
                """
            <style>
            .category-container {
                margin: 5px 0 15px 0;
            }
            
            .category-container .stButton > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white !important;
                border: none;
                border-radius: 10px;
                padding: 6px 12px;
                font-weight: 500;
                font-size: 12px;
                transition: all 0.3s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                width: 100%;
                height: 38px;
                margin: 2px 0;
            }
            
            .category-container .stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 3px 6px rgba(0,0,0,0.15);
                background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
            }
            
            .category-container .stButton > button:focus:not(:active) {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                box-shadow: 0 3px 8px rgba(255,107,107,0.3);
                border: 2px solid #ff6b6b;
            }
            
            .category-selected {
                background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
                border-left: 3px solid #667eea;
                padding: 8px 12px;
                border-radius: 6px;
                margin: 8px 0;
                font-size: 13px;
                font-weight: 500;
                text-align: center;
            }
            
            @media (max-width: 768px) {
                .category-container .stButton > button {
                    font-size: 11px;
                    height: 35px;
                    padding: 5px 8px;
                }
                .category-selected {
                    font-size: 12px;
                    padding: 6px 10px;
                }
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("**Category:**")
            st.markdown('<div class="category-container">', unsafe_allow_html=True)

            # Initialize session state for category if not exists
            if "selected_category" not in st.session_state:
                st.session_state.selected_category = "news"

            # Category options with emojis
            category_options = {
                "news": "📰 News",
                "sports": "⚽ Sports",
                "finance": "💰 Finance",
                "travel": "✈️ Travel",
                "lifestyle": "🌟 Lifestyle",
                "health": "🏥 Health",
            }

            # Create 2 rows of 3 columns for mobile-friendly layout
            row1_cols = st.columns(3)
            row2_cols = st.columns(3)

            categories = ["news", "sports", "finance", "travel", "lifestyle", "health"]

            for i, cat in enumerate(categories):
                if i < 3:
                    col = row1_cols[i]
                else:
                    col = row2_cols[i - 3]

                with col:
                    if st.button(
                        category_options[cat],
                        key=f"cat_btn_{cat}",
                        help=f"Select {cat} category - will be applied to all headlines",
                        use_container_width=True,
                    ):
                        st.session_state.selected_category = cat

            st.markdown("</div>", unsafe_allow_html=True)

            # Set the category variable (this replaces your selectbox output)
            category = st.session_state.selected_category

            # Show current selection with nice styling
            st.markdown(
                f"""
            <div class="category-selected">
                ✅ Selected: <strong>{category_options[category]}</strong>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("### Enter Headlines to Compare")

        headlines = []
        # Use responsive columns that stack on mobile
        if num_headlines <= 2:
            cols = st.columns(2)
        elif num_headlines <= 3:
            cols = st.columns(3)
        else:
            cols = st.columns(2)  # For 4-5 headlines, use 2 columns to avoid cramping

        for i in range(num_headlines):
            col_idx = i % len(cols)
            with cols[col_idx]:
                headline = st.text_area(
                    f"Headline {i+1}:",
                    placeholder=f"Enter headline {i+1}...",
                    key=f"headline_{i}",
                    height=80,
                )
                if headline.strip():
                    headlines.append(headline)

        if st.button("📊 Compare Headlines", type="primary") and len(headlines) >= 2:
            # Log comparison
            log_event(
                "headline_comparison",
                {"headlines_count": len(headlines), "category": category},
            )

            with st.spinner("Analyzing headlines..."):
                comparison_results = []

                for i, headline in enumerate(headlines):
                    result = predict_engagement(
                        headline, "", category, model_pipeline, preprocessing_components
                    )

                    if result:
                        comparison_results.append(
                            {
                                "Headline": headline,
                                "CTR": f"{result['estimated_ctr']*100:.2f}%",
                                "Engagement": (
                                    "🔥 High" if result["high_engagement"] else "📉 Low"
                                ),
                                "Score": result["estimated_ctr"],
                            }
                        )

                # Sort by CTR score
                comparison_results.sort(key=lambda x: x["Score"], reverse=True)

                # Display results
                st.subheader("🏆 Comparison Results")

                for i, result in enumerate(comparison_results):
                    if i == 0:
                        st.success(
                            f"🥇 **Winner:** {result['Headline']} - {result['CTR']}"
                        )
                    elif i == 1:
                        st.info(
                            f"🥈 **Runner-up:** {result['Headline']} - {result['CTR']}"
                        )
                    else:
                        st.write(f"{i+1}. {result['Headline']} - {result['CTR']}")

                # Detailed comparison table
                comparison_df = pd.DataFrame(comparison_results)[
                    ["Headline", "CTR", "Engagement"]
                ]
                st.dataframe(comparison_df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "**Headline Hunter** | Built with Streamlit, XGBoost, and OpenAI | "
        "Boost your content engagement with data-driven headline optimization"
    )


if __name__ == "__main__":
    main()

# import os
# import torch

# # Fix for Streamlit's local_sources_watcher crashing on torch.classes
# try:
#     torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
# except Exception:
#     pass

# import streamlit as st
# import json
# import pandas as pd
# import numpy as np
# import pickle
# import faiss
# from pathlib import Path
# from sentence_transformers import SentenceTransformer
# import plotly.express as px
# import plotly.graph_objects as go
# from textstat import flesch_reading_ease
# import warnings
# import llm_rewriter
# from feature_utils import create_article_features_exact, load_preprocessing_components

# warnings.filterwarnings("ignore")

# import streamlit as st
# import datetime
# import json
# import os


# # Add this logging function at the top, after your imports
# def log_event(event_type, data):
#     """Log events to track user behavior"""
#     log_entry = {
#         "timestamp": datetime.datetime.now().isoformat(),
#         "event": event_type,
#         "data": data,
#     }

#     try:
#         with open("usage_log.txt", "a") as f:
#             f.write(json.dumps(log_entry) + "\n")
#     except:
#         pass  # Fail silently if can't write


# def get_usage_stats():
#     """Read usage statistics"""
#     try:
#         with open("usage_log.txt", "r") as f:
#             logs = [json.loads(line) for line in f]
#         return logs
#     except:
#         return []


# # Page configuration
# st.set_page_config(
#     page_title="Article Engagement Predictor & Rewriter",
#     page_icon="📰",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # Add this to track page visits
# log_event("page_visit", {"user_agent": "streamlit_user"})

# # Constants
# MODEL_DIR = Path("model_output")
# FAISS_DIR = Path("faiss_index")
# PREP_DIR = Path("data/preprocessed")

# # CSS Styles
# st.markdown(
#     """
#     <style>
#       .card {
#         background-color: white;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         box-shadow: 0 2px 6px rgba(0,0,0,0.1);
#         margin-bottom: 1rem;
#       }
#       .full-width { width: 100% !important; }
#       .guidelines-box {
#         background-color: #f0f2f6;
#         border: 2px solid #4f8bf9;
#         border-radius: 10px;
#         padding: 20px;
#         margin-top: 20px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#       }
#       .guidelines-title {
#         font-weight: bold;
#         font-size: 18px;
#         margin-bottom: 15px;
#         color: #1f4e79;
#         text-align: center;
#       }
#       .guidelines-content {
#         font-size: 15px;
#         line-height: 1.6;
#         color: #2c3e50;
#       }
#       .guidelines-content ul {
#         margin: 10px 0;
#         padding-left: 20px;
#       }
#       .guidelines-content li {
#         margin: 8px 0;
#         color: #34495e;
#       }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# @st.cache_resource
# def load_model():
#     """Load the trained engagement prediction model"""
#     try:
#         model_files = list(MODEL_DIR.glob("*_optimized_model.pkl"))
#         model_file = (
#             model_files[0] if model_files else MODEL_DIR / "xgboost_optimized_model.pkl"
#         )

#         with open(model_file, "rb") as f:
#             model = pickle.load(f)

#         metadata_file = MODEL_DIR / "model_metadata.json"
#         metadata = {}
#         if metadata_file.exists():
#             with open(metadata_file, "r") as f:
#                 metadata = json.load(f)

#         # Load baseline metrics from your actual EDA insights
#         baseline_metrics = {}
#         try:
#             # Try the preprocessed data directory first
#             eda_path = "data/preprocessed/processed_data/headline_eda_insights.json"
#             if not Path(eda_path).exists():
#                 eda_path = "headline_eda_insights.json"

#             with open(eda_path, "r") as f:
#                 eda_insights = json.load(f)
#                 baseline_metrics = eda_insights.get(
#                     "baseline_metrics",
#                     {
#                         "overall_avg_ctr": 0.041238101089957464,
#                         "training_median_ctr": 0.019230769230769232,
#                         "ctr_threshold": 0.05,
#                     },
#                 )
#         except:
#             # Fallback to metadata values
#             baseline_metrics = {
#                 "overall_avg_ctr": metadata.get("target_statistics", {}).get(
#                     "mean_ctr", 0.041
#                 ),
#                 "training_median_ctr": metadata.get("target_statistics", {}).get(
#                     "median_ctr", 0.019
#                 ),
#                 "ctr_threshold": metadata.get("target_statistics", {}).get(
#                     "ctr_threshold", 0.05
#                 ),
#             }

#         model_pipeline = {
#             "model": model,
#             "model_name": metadata.get("model_type", "XGBoost"),
#             "target": "high_engagement",
#             "performance": metadata.get("final_evaluation", {}),
#             "feature_names": (
#                 list(model.feature_names_in_)
#                 if hasattr(model, "feature_names_in_")
#                 else []
#             ),
#             "scaler": None,
#             "baseline_metrics": baseline_metrics,
#             "metadata": metadata,
#         }

#         return model_pipeline
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None


# @st.cache_resource
# def load_search_system():
#     """Load the enhanced FAISS search system with rewrite capabilities"""
#     try:
#         index = faiss.read_index(str(FAISS_DIR / "article_index.faiss"))

#         with open(FAISS_DIR / "article_lookup.pkl", "rb") as f:
#             article_lookup = pickle.load(f)

#         with open(FAISS_DIR / "article_id_mappings.pkl", "rb") as f:
#             mappings = pickle.load(f)

#         with open(FAISS_DIR / "index_metadata.json", "r") as f:
#             metadata = json.load(f)

#         return {
#             "index": index,
#             "article_lookup": article_lookup,
#             "mappings": mappings,
#             "metadata": metadata,
#         }
#     except Exception as e:
#         st.error(f"Error loading search system: {e}")
#         return None


# @st.cache_resource
# def load_llm_rewriter():
#     """Load the efficient LLM headline rewriter with correct EDA insights path"""
#     try:
#         model_pipeline = load_model()
#         components = load_preprocessing_components()

#         if model_pipeline and components:
#             # Try the preprocessed data directory first
#             eda_insights_path = (
#                 "data/preprocessed/processed_data/headline_eda_insights.json"
#             )
#             if not Path(eda_insights_path).exists():
#                 # Fall back to project root
#                 eda_insights_path = "headline_eda_insights.json"

#             return llm_rewriter.EnhancedLLMHeadlineRewriter(
#                 model_pipeline=model_pipeline,
#                 components=components,
#                 eda_insights_path=eda_insights_path,
#             )
#         else:
#             st.warning("Could not load model pipeline or components for rewriter")
#             return None

#     except Exception as e:
#         st.error(f"Error loading LLM rewriter: {e}")
#         # Show the actual error for debugging
#         with st.expander("🔍 Debug Info"):
#             import traceback

#             st.code(traceback.format_exc())
#         return None


# @st.cache_resource
# def load_embedder():
#     """Load the sentence transformer model"""
#     try:
#         from feature_utils import get_embedder

#         return get_embedder()
#     except Exception as e:
#         st.error(f"Error loading embedder: {e}")
#         return SentenceTransformer("all-MiniLM-L6-v2")


# def predict_engagement(
#     title, abstract="", category="news", model_pipeline=None, components=None
# ):
#     """Predict engagement for a single article using exact feature replication"""
#     if model_pipeline is None or components is None:
#         return None

#     try:
#         # Create features using exact replication of preprocessing pipeline
#         features = create_article_features_exact(title, abstract, category, components)

#         # Create feature vector in the exact order expected by the model
#         feature_order = components.get("feature_order", [])

#         if feature_order:
#             feature_vector = [
#                 features.get(feat_name, 0.0) for feat_name in feature_order
#             ]
#         elif hasattr(model_pipeline["model"], "feature_names_in_"):
#             expected_features = list(model_pipeline["model"].feature_names_in_)
#             feature_vector = [
#                 features.get(feat_name, 0.0) for feat_name in expected_features
#             ]
#         else:
#             feature_vector = [features[k] for k in sorted(features.keys())]

#         feature_vector = np.array(feature_vector).reshape(1, -1)

#         # Predict
#         prediction = model_pipeline["model"].predict(feature_vector)[0]
#         prediction_proba = model_pipeline["model"].predict_proba(feature_vector)[0]

#         # Convert engagement probability to estimated CTR
#         engagement_prob = float(prediction_proba[1])
#         estimated_ctr = max(0.01, engagement_prob * 0.1)

#         return {
#             "high_engagement": bool(prediction),
#             "engagement_probability": engagement_prob,
#             "estimated_ctr": estimated_ctr,
#             "confidence": float(max(prediction_proba)),
#             "features": features,
#         }

#     except Exception as e:
#         st.error(f"Error in prediction: {e}")
#         return None


# def main():
#     st.title("📰 AI-Assisted Headline Hunter")
#     st.write("**Predict engagement and optimize headlines with AI-powered rewriting**")

#     # Load all systems once at startup
#     with st.spinner("Loading AI systems..."):
#         model_pipeline = load_model()
#         preprocessing_components = load_preprocessing_components()
#         search_system = load_search_system()
#         llm_rewriter = load_llm_rewriter()

#     # ENHANCED ADMIN PANEL WITH DOWNLOAD BUTTON
#     st.sidebar.markdown("---")
#     if st.sidebar.text_input("📊 Analytics (Enter: admin)", type="password") == "admin":
#         logs = get_usage_stats()
#         st.sidebar.success("✅ Admin Mode Activated")

#         # Metrics
#         st.sidebar.metric("Total Interactions", len(logs))
#         st.sidebar.metric(
#             "CTR Predictions",
#             len([l for l in logs if l["event"] == "ctr_prediction"]),
#         )
#         st.sidebar.metric(
#             "Headlines Rewritten",
#             len([l for l in logs if l["event"] == "headline_rewrite"]),
#         )
#         st.sidebar.metric(
#             "Searches Made", len([l for l in logs if l["event"] == "search"])
#         )

#         # Download button
#         if logs:
#             log_data = "\n".join([json.dumps(log) for log in logs])
#             st.sidebar.download_button(
#                 label="📥 Download Analytics Data",
#                 data=log_data,
#                 file_name=f"headline_hunter_analytics_{datetime.date.today()}.txt",
#                 mime="text/plain",
#             )

#             # Show recent activity
#             st.sidebar.write("**📈 Recent Activity:**")
#             for log in logs[-5:]:  # Last 5 events
#                 st.sidebar.caption(f"• {log['event']} - {log['timestamp'][11:16]}")
#         else:
#             st.sidebar.info("No analytics data yet")

#     # Main tabs
#     (tab1,) = st.tabs(
#         ["🔮 Predict & Rewrite"]  # "🔍 Search Articles", "📊 Headline Rewrite Analysis"
#     )

#     # Tab 1: Predict & Rewrite
#     with tab1:
#         # Main layout with input on left and guidelines on right
#         col1, col2 = st.columns([2, 1])

#         with col1:
#             title = st.text_area(
#                 "Article Title",
#                 placeholder="Enter your article headline here...",
#                 height=100,
#                 help="Enter the headline you want to test and optimize",
#             )

#             categories = [
#                 "news",
#                 "sports",
#                 "finance",
#                 "travel",
#                 "lifestyle",
#                 "video",
#                 "foodanddrink",
#                 "weather",
#                 "autos",
#                 "health",
#                 "entertainment",
#                 "tv",
#                 "music",
#                 "movies",
#                 "kids",
#                 "northamerica",
#                 "middleeast",
#                 "unknown",
#             ]

#             category = st.selectbox("Article Category", categories, index=0)

#             predict_and_rewrite = st.button("🤖 Analyze & Optimize", type="primary")

#         with col2:
#             # Editorial Guidelines in right column
#             st.markdown(
#                 """
#             <div class="guidelines-box">
#                 <div class="guidelines-title">💡 Editorial Guidelines</div>
#                 <div class="guidelines-content">
#                     <strong>High-engagement headlines:</strong>
#                     <ul>
#                         <li>8-12 words optimal</li>
#                         <li>Include numbers/questions</li>
#                         <li>High readability (60+ score)</li>
#                         <li>Front-load key information</li>
#                         <li>Under 75 characters</li>
#                     </ul>
#                 </div>
#             </div>
#             """,
#                 unsafe_allow_html=True,
#             )

#         # Process prediction and rewriting - FIXED VERSION
#         if predict_and_rewrite:
#             if title.strip():
#                 # Log the event with actual data
#                 log_event(
#                     "ctr_prediction",
#                     {
#                         "headline": title,
#                         "category": category,
#                         "headline_length": len(title),
#                         "word_count": len(title.split()),
#                     },
#                 )

#                 if model_pipeline and preprocessing_components:
#                     # Step 1: Predict engagement
#                     with st.spinner("🔮 Analyzing engagement potential..."):
#                         result = predict_engagement(
#                             title,
#                             "",
#                             category,
#                             model_pipeline,
#                             preprocessing_components,
#                         )

#                     if result and isinstance(result, dict):
#                         # Get threshold from model pipeline
#                         threshold = model_pipeline["baseline_metrics"].get(
#                             "ctr_threshold", 0.05
#                         )

#                         # Display prediction results
#                         st.subheader("📊 Engagement Analysis")

#                         col_pred1, col_pred2, col_pred3 = st.columns(3)

#                         with col_pred1:
#                             # Show engagement level with threshold
#                             engagement_level = (
#                                 "High Engagement"
#                                 if result["high_engagement"]
#                                 else "Low Engagement"
#                             )
#                             engagement_emoji = (
#                                 "🔥" if result["high_engagement"] else "📉"
#                             )
#                             st.metric(
#                                 "Engagement Level",
#                                 f"{engagement_emoji} {engagement_level}",
#                             )

#                         with col_pred2:
#                             # Show CTR as percentage with threshold
#                             ctr_percentage = result["estimated_ctr"] * 100
#                             st.metric("Estimated CTR", f"{ctr_percentage:.2f}%")
#                             st.caption(f"Threshold: {threshold*100:.1f}%")

#                         # Step 2: Generate AI rewrites
#                         st.subheader("✨ AI-Optimized Headlines")

#                         if llm_rewriter:
#                             with st.spinner("🤖 Generating AI-optimized headlines..."):
#                                 try:
#                                     article_data = {
#                                         "category": category,
#                                         "abstract": "",
#                                         "current_ctr": result["estimated_ctr"],
#                                     }

#                                     rewrite_result = llm_rewriter.get_best_headline(
#                                         title, article_data
#                                     )

#                                     if (
#                                         rewrite_result
#                                         and "best_headline" in rewrite_result
#                                     ):
#                                         best_headline = rewrite_result["best_headline"]

#                                         # Log the rewrite event
#                                         log_event(
#                                             "headline_rewrite",
#                                             {
#                                                 "original": title,
#                                                 "rewritten": best_headline,
#                                                 "category": category,
#                                             },
#                                         )

#                                         if best_headline.strip() != title.strip():
#                                             # Predict engagement for the rewritten headline
#                                             rewritten_result = predict_engagement(
#                                                 best_headline,
#                                                 "",
#                                                 category,
#                                                 model_pipeline,
#                                                 preprocessing_components,
#                                             )

#                                             # Display comparison
#                                             col_orig, col_rewrite = st.columns(2)

#                                             with col_orig:
#                                                 st.markdown("**Original Headline:**")
#                                                 st.info(f"📝 {title}")
#                                                 st.write(
#                                                     f"CTR: {result['estimated_ctr']*100:.2f}%"
#                                                 )

#                                             with col_rewrite:
#                                                 st.markdown(
#                                                     "**AI-Optimized Headline:**"
#                                                 )
#                                                 st.success(f"✨ {best_headline}")

#                                                 if rewritten_result:
#                                                     st.write(
#                                                         f"CTR: {rewritten_result['estimated_ctr']*100:.2f}%"
#                                                     )

#                                                     # Show improvement
#                                                     ctr_improvement = (
#                                                         rewritten_result[
#                                                             "estimated_ctr"
#                                                         ]
#                                                         - result["estimated_ctr"]
#                                                     ) * 100
#                                                     if ctr_improvement > 0:
#                                                         st.success(
#                                                             f"📈 CTR Improvement: +{ctr_improvement:.2f}%"
#                                                         )
#                                                     elif ctr_improvement < 0:
#                                                         st.warning(
#                                                             f"📉 CTR Change: {ctr_improvement:.2f}%"
#                                                         )
#                                                     else:
#                                                         st.info(
#                                                             "📊 No significant change"
#                                                         )

#                                             # Show all candidates
#                                             if "all_candidates" in rewrite_result:
#                                                 with st.expander(
#                                                     "🔍 View All AI-Generated Candidates"
#                                                 ):
#                                                     for i, (
#                                                         candidate,
#                                                         ctr,
#                                                     ) in enumerate(
#                                                         rewrite_result[
#                                                             "all_candidates"
#                                                         ],
#                                                         1,
#                                                     ):
#                                                         st.write(
#                                                             f"{i}. **{candidate}** (CTR: {ctr*100:.2f}%)"
#                                                         )
#                                         else:
#                                             st.info(
#                                                 "🎯 Original headline is already well-optimized!"
#                                             )

#                                             # Show the analysis anyway
#                                             if "all_candidates" in rewrite_result:
#                                                 with st.expander(
#                                                     "🔍 View Alternative Suggestions"
#                                                 ):
#                                                     for i, (
#                                                         candidate,
#                                                         ctr,
#                                                     ) in enumerate(
#                                                         rewrite_result[
#                                                             "all_candidates"
#                                                         ],
#                                                         1,
#                                                     ):
#                                                         if candidate != title:
#                                                             st.write(
#                                                                 f"{i}. **{candidate}** (CTR: {ctr*100:.2f}%)"
#                                                             )
#                                     else:
#                                         st.warning(
#                                             "🤔 No rewrite suggestions generated"
#                                         )

#                                 except Exception as e:
#                                     st.error(f"❌ Rewriting failed: {e}")

#                                     # Show the exact error for debugging
#                                     with st.expander("🔍 Debug Info"):
#                                         import traceback

#                                         st.code(traceback.format_exc())
#                         else:
#                             st.error(
#                                 "❌ AI Rewriter not available. Please check your OpenAI API key."
#                             )

#                     else:
#                         st.error("❌ Prediction failed. Please check your inputs.")
#                 else:
#                     st.error("❌ Model or preprocessing components not loaded.")
#             else:
#                 st.warning("⚠️ Please enter an article title.")

#     # # Tab 2: Search Articles
#     # with tab2:
#     #     st.header("Search Articles")

#     #     search_query = st.text_input(
#     #         "Search Query",
#     #         placeholder="Enter keywords or describe the topic...",
#     #         help="Search through articles by keywords, title, or content.",
#     #     )

#     #     col_search1, col_search2 = st.columns(2)
#     #     with col_search1:
#     #         num_results = st.slider("Number of results", 5, 20, 10)

#     #     if st.button("🔍 Search", type="primary"):
#     #         if search_query.strip() and search_system:
#     #             # Log the search event
#     #             log_event(
#     #                 "search",
#     #                 {"query": search_query, "num_results_requested": num_results},
#     #             )

#     #             with st.spinner("Searching articles..."):
#     #                 try:
#     #                     embedder = load_embedder()
#     #                     query_embedding = embedder.encode([search_query])
#     #                     query_embedding = query_embedding.astype(np.float32)
#     #                     faiss.normalize_L2(query_embedding)

#     #                     search_k = num_results * 3
#     #                     distances, indices = search_system["index"].search(
#     #                         query_embedding, search_k
#     #                     )

#     #                     results = []
#     #                     for dist, idx in zip(distances[0], indices[0]):
#     #                         if idx in search_system["mappings"]["idx_to_article_id"]:
#     #                             article_id = search_system["mappings"][
#     #                                 "idx_to_article_id"
#     #                             ][idx]
#     #                             if article_id in search_system["article_lookup"]:
#     #                                 article_info = search_system["article_lookup"][
#     #                                     article_id
#     #                                 ].copy()
#     #                                 l2_distance = float(dist)
#     #                                 similarity_score = 1.0 / (1.0 + l2_distance)
#     #                                 article_info["similarity_score"] = similarity_score
#     #                                 results.append(article_info)

#     #                                 if len(results) >= num_results:
#     #                                     break

#     #                     if results:
#     #                         st.subheader(f"📰 Found {len(results)} articles")
#     #                         for i, article in enumerate(results, 1):
#     #                             with st.expander(f"{i}. {article['title'][:70]}..."):
#     #                                 col_art1, col_art2 = st.columns([3, 1])

#     #                                 with col_art1:
#     #                                     st.write(f"**Title:** {article['title']}")
#     #                                     if article.get("abstract"):
#     #                                         abstract_preview = (
#     #                                             article["abstract"][:200] + "..."
#     #                                             if len(article["abstract"]) > 200
#     #                                             else article["abstract"]
#     #                                         )
#     #                                         st.write(
#     #                                             f"**Abstract:** {abstract_preview}"
#     #                                         )

#     #                                 with col_art2:
#     #                                     st.metric(
#     #                                         "Similarity",
#     #                                         f"{article['similarity_score']:.3f}",
#     #                                     )
#     #                                     st.write(f"**Category:** {article['category']}")

#     #                                     if not pd.isna(article.get("ctr")):
#     #                                         st.write(
#     #                                             f"**CTR:** {article['ctr']*100:.2f}%"
#     #                                         )

#     #                                     if not pd.isna(article.get("high_engagement")):
#     #                                         engagement_status = (
#     #                                             "🔥 High"
#     #                                             if article["high_engagement"]
#     #                                             else "📉 Low"
#     #                                         )
#     #                                         st.write(
#     #                                             f"**Engagement:** {engagement_status}"
#     #                                         )
#     #                     else:
#     #                         st.info("No articles found. Try different keywords.")
#     #                 except Exception as e:
#     #                     st.error(f"Search failed: {e}")
#     #         else:
#     #             if not search_query.strip():
#     #                 st.warning("Please enter a search query.")
#     #             else:
#     #                 st.error("Search system not available.")

#     # # Tab 3: Rewrite Analysis
#     # with tab3:
#     #     st.header("Headline Rewrite Analysis")

#     #     if search_system and search_system["metadata"].get("rewrite_analysis"):
#     #         rewrite_stats = search_system["metadata"]["rewrite_analysis"]

#     #         # Load detailed rewrite results if available
#     #         rewrite_file = FAISS_DIR / "rewrite_analysis" / "headline_rewrites.parquet"
#     #         if rewrite_file.exists():
#     #             try:
#     #                 rewrite_df = pd.read_parquet(rewrite_file)

#     #                 st.subheader("📈 Strategy Performance")

#     #                 # Strategy comparison
#     #                 if "model_ctr_improvement" in rewrite_df.columns:
#     #                     strategy_performance = (
#     #                         rewrite_df.groupby("strategy")
#     #                         .agg(
#     #                             {
#     #                                 "quality_score": "mean",
#     #                                 "readability_improvement": "mean",
#     #                                 "model_ctr_improvement": "mean",
#     #                             }
#     #                         )
#     #                         .round(4)
#     #                     )

#     #                     fig = px.bar(
#     #                         strategy_performance.reset_index(),
#     #                         x="strategy",
#     #                         y="model_ctr_improvement",
#     #                         title="Average Model-Based CTR Improvement by Strategy",
#     #                     )
#     #                     st.plotly_chart(fig, use_container_width=True)
#     #                 else:
#     #                     strategy_performance = (
#     #                         rewrite_df.groupby("strategy")
#     #                         .agg(
#     #                             {
#     #                                 "quality_score": "mean",
#     #                                 "readability_improvement": "mean",
#     #                                 "predicted_ctr_improvement": "mean",
#     #                             }
#     #                         )
#     #                         .round(3)
#     #                     )

#     #                     fig = px.bar(
#     #                         strategy_performance.reset_index(),
#     #                         x="strategy",
#     #                         y="quality_score",
#     #                         title="Average Quality Score by Strategy",
#     #                     )
#     #                     st.plotly_chart(fig, use_container_width=True)

#     #                 # Show detailed results
#     #                 st.subheader("🔍 Detailed Results")

#     #                 display_columns = [
#     #                     "original_title",
#     #                     "strategy",
#     #                     "rewritten_title",
#     #                     "quality_score",
#     #                     "readability_improvement",
#     #                     "predicted_ctr_improvement",
#     #                 ]

#     #                 if "model_ctr_improvement" in rewrite_df.columns:
#     #                     display_columns.extend(
#     #                         ["model_ctr_improvement", "original_ctr", "rewritten_ctr"]
#     #                     )

#     #                 available_columns = [
#     #                     col for col in display_columns if col in rewrite_df.columns
#     #                 ]
#     #                 st.dataframe(
#     #                     rewrite_df[available_columns].head(20), use_container_width=True
#     #                 )

#     #             except Exception as e:
#     #                 st.error(f"Error loading rewrite analysis: {e}")
#     #         else:
#     #             st.info(
#     #                 "Detailed rewrite analysis not available. Run the FAISS index creation script to generate analysis."
#     #             )
#     #     else:
#     #         st.info(
#     #             "No rewrite analysis available. The system may be running in offline mode."
#     #         )

#     # Footer
#     st.markdown("---")
#     st.markdown(
#         "**Article Engagement Predictor & AI Rewriter** | Built with Streamlit, XGBoost, and OpenAI"
#     )


# if __name__ == "__main__":
#     main()
