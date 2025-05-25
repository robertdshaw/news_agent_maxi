import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import json
import openai
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from textstat import flesch_reading_ease
from dotenv import load_dotenv

load_dotenv()
import os

from model_loader import get_model_paths, check_pipeline_status, load_ctr_model
import logging

logger = logging.getLogger(__name__)

# Clear cache for debugging
st.cache_resource.clear()

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
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper Functions
# Add this code to your streamlit_app.py right after the load_resources function definition
# Replace the existing load_resources function with this version:


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

        # Look for original headlines from MIND dataset
        titles_mapping = {}
        print("🔍 Looking for MIND dataset headlines...")

        try:
            base_dir = Path(__file__).parent
            source_data_paths = [
                base_dir / "source_data" / "train_data",
                base_dir / "source_data" / "val_data",
                base_dir / "source_data" / "test_data",
            ]

            for data_path in source_data_paths:
                print(f"📂 Checking: {data_path}")
                if data_path.exists():
                    # Look for MIND dataset news.tsv files
                    news_file = data_path / "news.tsv"
                    if news_file.exists():
                        print(f"📄 Found news.tsv in {data_path.name}")
                        try:
                            # MIND dataset format: NewsID, Category, SubCategory, Title, Abstract, URL, TitleEntities, AbstractEntities
                            df = pd.read_csv(
                                news_file,
                                sep="\t",
                                header=None,
                                names=[
                                    "NewsID",
                                    "Category",
                                    "SubCategory",
                                    "Title",
                                    "Abstract",
                                    "URL",
                                    "TitleEntities",
                                    "AbstractEntities",
                                ],
                            )

                            print(f"📊 Read {len(df)} articles from {data_path.name}")

                            # Create mapping from index to title
                            for idx, title in enumerate(df["Title"]):
                                if pd.notna(title) and str(title).strip():
                                    titles_mapping[idx] = str(title).strip()

                            print(
                                f"✅ Added {len([t for t in df['Title'] if pd.notna(t)])} valid headlines"
                            )

                            # Show a sample
                            sample_titles = df["Title"].dropna().head(3).tolist()
                            print(f"📝 Sample headlines: {sample_titles}")

                            break  # Use first successful file (train data preferred)

                        except Exception as e:
                            print(f"❌ Error reading {news_file}: {e}")
                    else:
                        print(f"❌ No news.tsv found in {data_path}")
                else:
                    print(f"❌ Directory doesn't exist: {data_path}")

            if titles_mapping:
                print(f"🎉 Successfully loaded {len(titles_mapping)} headlines total")
            else:
                print("⚠️ No headlines loaded - will use placeholders")

        except Exception as e:
            print(f"💥 Error in headline loading: {e}")
            titles_mapping = {}

        # Load metadata
        with open(paths["feature_metadata"], "r") as f:
            feature_metadata = json.load(f)

        with open(paths["label_encoder"], "r") as f:
            label_encoder_info = json.load(f)

        # Load embeddings and FAISS index
        if status["embeddings_completed"]:
            with open(paths["embeddings"], "rb") as f:
                emb_data = pickle.load(f)
            index = faiss.read_index(str(paths["faiss_index"]))
        else:
            st.warning("Embeddings not available. Some features will be limited.")
            emb_data = None
            index = None

        # Load CTR model
        if status["training_completed"]:
            model_data, err = load_ctr_model(paths["ctr_model"])
            if err:
                st.error(f"Error loading CTR model: {err}")
                st.stop()
            ctr_model = model_data["model"]
            feature_names = model_data["feature_names"]
        else:
            st.warning("CTR model not trained. Please run training script first.")
            ctr_model = None
            feature_names = None

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
            "embedder": embedder,
            "titles_mapping": titles_mapping,
            "status": status,
        }

    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        st.stop()


import os
import requests
from pathlib import Path

# Create images directory
images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

print("Creating images folder for Streamlit app...")


def download_placeholder_images():
    """Download placeholder images for the news editor app."""

    # Placeholder image URLs (from picsum for demo purposes)
    image_urls = [
        "https://picsum.photos/800/400?random=1",
        "https://picsum.photos/800/400?random=2",
        "https://picsum.photos/800/400?random=3",
        "https://picsum.photos/800/400?random=4",
        "https://picsum.photos/800/400?random=5",
        "https://picsum.photos/800/400?random=6",
        "https://picsum.photos/800/400?random=7",
        "https://picsum.photos/800/400?random=8",
        "https://picsum.photos/800/400?random=9",
        "https://picsum.photos/800/400?random=10",
    ]

    for i, url in enumerate(image_urls, 1):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(images_dir / f"news_image_{i}.jpg", "wb") as f:
                    f.write(response.content)
                print(f"Downloaded image {i}/10")
            else:
                print(f"Failed to download image {i}")
        except Exception as e:
            print(f"Error downloading image {i}: {e}")


# Or manually add your own images:
print(f"\n📁 Add 10 JPG images to: {images_dir.absolute()}")
print("Suggested news-related images:")
print("- news_header_1.jpg")
print("- news_header_2.jpg")
print("- ... (any 10 JPG files)")
print("\nThe Streamlit app will randomly select one to display as a header image.")

# Check current images
current_images = list(images_dir.glob("*.jpg"))
print(f"\nCurrently found {len(current_images)} JPG images in images folder:")
for img in current_images:
    print(f"- {img.name}")

if len(current_images) == 0:
    print(
        "\n⚠️  No images found. Add some JPG files to the images/ folder or run download_placeholder_images()"
    )
else:
    print(
        f"\n✅ Ready! Streamlit app will randomly show one of these {len(current_images)} images."
    )


def extract_features_updated(titles, embedder, feature_metadata):
    """Extract features matching the preprocessing pipeline."""
    embs = embedder.encode(titles)
    rows = []

    for i, title in enumerate(titles):
        feat = {
            # Text features
            "title_length": len(title),
            "abstract_length": 50,  # Default
            "title_word_count": len(title.split()),
            "abstract_word_count": 10,  # Default
            "title_reading_ease": flesch_reading_ease(title) if title.strip() else 0,
            "abstract_reading_ease": 60,  # Default
            # Pattern features
            "has_question": int("?" in title),
            "has_exclamation": int("!" in title),
            "has_number": int(any(c.isdigit() for c in title)),
            "has_colon": int(":" in title),
            "has_quotes": int(any(q in title for q in ['"', "'"])),
            "has_hyphen": int("-" in title),
            "has_brackets": int(any(b in title for b in "[]())")),
            # Capitalization features
            "title_upper_ratio": (
                sum(c.isupper() for c in title) / len(title) if title else 0
            ),
            "starts_with_caps": int(title[0].isupper() if title else False),
            # Temporal features (defaults)
            "hour": 12,
            "day_of_week": 1,
            "is_weekend": 0,
            "time_of_day": 1,
            # Category (default)
            "category_enc": 0,
        }

        # Add embeddings
        emb_dim = feature_metadata.get("embedding_dim", 50)
        for j in range(min(emb_dim, embs.shape[1])):
            feat[f"emb_{j}"] = embs[i, j]

        rows.append(feat)

    return pd.DataFrame(rows)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Initialize Data
with st.spinner("Loading models and data..."):
    resources = load_resources()

# Load header image
import random

img_dir = Path(__file__).parent / "images"
if img_dir.exists():
    all_images = list(img_dir.glob("*.jpg"))
    selected = random.choice(all_images) if all_images else None
    if selected:
        st.image(str(selected), width=500)

st.title("📰 AI News Editor Assistant")

# Display pipeline status
if not all(resources["status"].values()):
    with st.expander("Pipeline Status", expanded=True):
        for stage, completed in resources["status"].items():
            status_icon = "✅" if completed else "❌"
            st.write(f"{status_icon} {stage.replace('_', ' ').title()}")

# UI Tabs
tab1, tab2, tab3 = st.tabs(["Retrieve & Rank", "Predict CTR", "Rewrite Headline"])

with tab1:
    st.header("🔍 Retrieval & Ranking")

    if not resources["status"]["embeddings_completed"]:
        st.warning("Embeddings not available. Please run embedding script first.")
    else:
        query = st.text_input("Query:", "big tech layoffs")
        k = st.slider("Top K articles:", 1, 20, 5)

        if st.button("Search & Rank Articles"):
            try:
                # Generate query embedding (truncated to 50D to match index)
                # Generate query embedding using same method as preprocessing
                query_features = extract_features_updated(
                    [query], resources["embedder"], resources["feature_metadata"]
                )
                emb_cols = [
                    col for col in query_features.columns if col.startswith("emb_")
                ]
                q_emb = query_features[emb_cols].values.astype("float32")

                # Search the index
                D, I = resources["index"].search(q_emb, k)

                # Check if we have valid results
                if len(I[0]) == 0:
                    st.error("No results found")
                else:
                    # Get article indices and create results
                    results = []
                    for i, (distance, idx) in enumerate(zip(D[0], I[0])):
                        if idx < len(resources["emb_data"]["article_indices"]):
                            article_idx = resources["emb_data"]["article_indices"][idx]

                            # Convert FAISS distance to relevance score with amplified differences
                            relevance_score = np.exp(-distance * 3)  # Exponential decay
                            relevance_percentage = 0.50 + (relevance_score * 0.45)
                            relevance_percentage = min(
                                0.95, max(0.10, relevance_percentage)
                            )

                            # Try to get actual headline from loaded titles
                            headline = None

                            # Check if we have titles_mapping
                            if (
                                "titles_mapping" in resources
                                and resources["titles_mapping"]
                            ):
                                titles_dict = resources["titles_mapping"]

                                # Method 1: Direct lookup by article_idx
                                if article_idx in titles_dict:
                                    headline = titles_dict[article_idx]

                                # Method 2: Use modulo mapping for large indices
                                elif len(titles_dict) > 0:
                                    mapped_idx = article_idx % len(titles_dict)
                                    headlines_list = list(titles_dict.values())
                                    if mapped_idx < len(headlines_list):
                                        headline = headlines_list[mapped_idx]

                            # Fallback: create meaningful placeholder
                            if not headline or not headline.strip():
                                query_words = query.lower().split()
                                if any(
                                    word in query_words
                                    for word in ["tech", "technology", "ai", "software"]
                                ):
                                    topic = "Tech"
                                elif any(
                                    word in query_words
                                    for word in ["business", "finance", "economy"]
                                ):
                                    topic = "Business"
                                elif any(
                                    word in query_words
                                    for word in ["health", "medical", "healthcare"]
                                ):
                                    topic = "Health"
                                elif any(
                                    word in query_words
                                    for word in ["politics", "election", "government"]
                                ):
                                    topic = "Politics"
                                elif any(
                                    word in query_words
                                    for word in ["climate", "environment", "green"]
                                ):
                                    topic = "Climate"
                                else:
                                    topic = "News"

                                headline = f"{topic} Article #{article_idx} (Related to: {query})"

                            results.append(
                                {
                                    "Rank": i + 1,
                                    "Related Article": headline,
                                    "Relevance": f"{relevance_percentage:.1%}",
                                    "ID": article_idx,
                                    "Raw_Distance": float(distance),
                                }
                            )

                    if results:
                        # Sort by raw distance (smaller = better)
                        results_df = pd.DataFrame(results)
                        results_df = results_df.sort_values(
                            "Raw_Distance", ascending=True
                        )
                        results_df["Rank"] = range(
                            1, len(results_df) + 1
                        )  # Rerank after sorting

                        # Display main results
                        display_df = results_df[
                            ["Rank", "Related Article", "Relevance"]
                        ].copy()

                        st.success(
                            f"Found {len(results)} relevant articles for: '{query}'"
                        )
                        st.dataframe(
                            display_df, use_container_width=True, hide_index=True
                        )

                        # Add this right after: st.dataframe(display_df, use_container_width=True, hide_index=True)

                        # TEMP DEBUG TEST
                        if st.session_state.get(
                            "Show Debug Info (distances and IDs)", False
                        ):
                            st.write("**🔧 DEBUG TABLE:**")
                            debug_cols = [
                                "Rank",
                                "Related Article",
                                "Relevance",
                                "Raw_Distance",
                                "ID",
                            ]
                            if all(col in results_df.columns for col in debug_cols):
                                debug_df = results_df[debug_cols].copy()
                                st.dataframe(
                                    debug_df, use_container_width=True, hide_index=True
                                )
                            else:
                                st.write(
                                    f"Missing columns. Available: {list(results_df.columns)}"
                                )
                                st.write("Sample data:")
                                st.write(results_df.head())

                        # Show best match relevance
                        best_relevance = results_df.iloc[0]["Relevance"]
                        st.metric("Best Match Relevance", best_relevance)

                        # Debug info checkbox
                        if st.checkbox("Show Debug Info (distances and IDs)"):
                            debug_cols = [
                                "Rank",
                                "Related Article",
                                "Relevance",
                                "Raw_Distance",
                                "ID",
                            ]
                            debug_df = results_df[debug_cols].copy()
                            st.subheader("Debug Information")
                            st.dataframe(
                                debug_df, use_container_width=True, hide_index=True
                            )
                    else:
                        st.error("No valid results to display")

            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                st.write("Debug info:")
                st.write(f"- Query: '{query}'")
                st.write(f"- Embeddings available: {resources['emb_data'] is not None}")
                st.write(f"- Index available: {resources['index'] is not None}")
                import traceback

                st.code(traceback.format_exc())

with tab2:
    st.header("⚡ Instant CTR Prediction")
    st.markdown("Type or paste a candidate headline and get an instant CTR estimate.")

    if not resources["status"]["training_completed"]:
        st.warning("CTR model not available. Please run training script first.")
    else:
        col_input, col_ref = st.columns([2, 1])

        with col_input:
            headline = st.text_input("Headline:", key="predict_input")
            if st.button("Predict CTR", key="predict_btn") and headline:
                feats = extract_features_updated(
                    [headline], resources["embedder"], resources["feature_metadata"]
                )
                pred = resources["ctr_model"].predict(
                    feats[resources["feature_names"]]
                )[0]
                st.metric("Predicted CTR", f"{pred:.1%}")

        with col_ref:
            st.markdown("**🔎 Reference Statistics**")

            # Calculate stats from training data
            train_ctr_mean = resources["train_targets"]["ctr"].mean()
            train_ctr_with_impressions = resources["train_targets"][
                resources["train_targets"]["ctr"] > 0
            ]["ctr"]

            st.metric("Overall Mean CTR", f"{train_ctr_mean:.2%}")
            if len(train_ctr_with_impressions) > 0:
                st.metric(
                    "Mean CTR (with impressions)",
                    f"{train_ctr_with_impressions.mean():.2%}",
                )

with tab3:
    st.header("✍️ Headline Rewriting")

    if not resources["status"]["training_completed"]:
        st.warning("CTR model not available. Please run training script first.")
    else:
        original = st.text_input(
            "Original Headline:",
            "Billy Joel cancels tour after brain condition diagnosis",
        )
        n = st.slider("Variations to Create:", 1, 5, 3)

        if st.button("Generate & Score") and original:
            # Get API key from environment
            openai_key = os.getenv("OPENAI_API_KEY")

            if not openai_key:
                st.error(
                    "OpenAI API key not found. Please add OPENAI_API_KEY to your .env file."
                )
                st.stop()

            # Set OpenAI API key
            openai.api_key = openai_key
            client = openai.OpenAI(api_key=openai_key)

            improved_instructions = """
You are an expert news headline editor. Your goal is to rewrite headlines to increase click-through rates while maintaining accuracy.

Guidelines for high-CTR headlines:
1. Use compelling action words and emotional triggers
2. Create curiosity or urgency without being clickbait
3. Include specific numbers, names, or details when possible
4. Keep under 70 characters for optimal display
5. Maintain factual accuracy - never invent details
6. Consider the reader's emotional response

For medical/health news:
- Emphasize impact on people's lives
- Use accessible language, not medical jargon
- Focus on what readers need to know

For the headline: "{original}"

Please create {n} improved versions that would likely get higher click-through rates. Each should be on its own line without numbering or bullets.
"""

            prompt = improved_instructions.format(original=original, n=n)

            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert news headline editor focused on increasing engagement while maintaining journalistic integrity.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    n=1,
                    temperature=0.8,  # Higher temperature for more creative variations
                    max_tokens=200,
                )

                # Extract and clean candidates
                text = response.choices[0].message.content.strip()
                candidates = []
                for line in text.splitlines():
                    clean_line = line.strip()
                    # Remove numbering, bullets, dashes
                    clean_line = re.sub(r"^\d+[\.\)]\s*", "", clean_line)
                    clean_line = re.sub(r"^[-•*]\s*", "", clean_line)
                    clean_line = clean_line.strip("\"'")
                    if (
                        clean_line and len(clean_line) > 10
                    ):  # Only keep substantial headlines
                        candidates.append(clean_line)

                candidates = candidates[:n]  # Limit to requested number

                if not candidates:
                    st.error(
                        "Failed to generate valid headline variations. Please try again."
                    )
                    st.stop()

                # Score original and candidates
                orig_feats = extract_features_updated(
                    [original], resources["embedder"], resources["feature_metadata"]
                )
                orig_ctr = resources["ctr_model"].predict(
                    orig_feats[resources["feature_names"]]
                )[0]

                # Get original embedding (truncated to 50D)
                orig_emb = resources["embedder"].encode([original])[:, :50][0]

                rows = []
                for cand in candidates:
                    feats = extract_features_updated(
                        [cand], resources["embedder"], resources["feature_metadata"]
                    )
                    ctr_pred = resources["ctr_model"].predict(
                        feats[resources["feature_names"]]
                    )[0]
                    read = flesch_reading_ease(cand) if cand.strip() else 0

                    # Get candidate embedding (truncated to 50D)
                    cand_emb = resources["embedder"].encode([cand])[:, :50][0]
                    sim = cosine_sim(orig_emb, cand_emb)

                    rows.append(
                        {
                            "Candidate": cand,
                            "CTR": ctr_pred,
                            "Readability": read,
                            "Similarity": sim,
                        }
                    )

                # Sort by CTR (highest first)
                df_res = pd.DataFrame(rows).sort_values("CTR", ascending=False)

                if len(df_res) > 0:
                    best_ctr = df_res["CTR"].iloc[0]
                    improvement = best_ctr - orig_ctr

                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original CTR", f"{orig_ctr:.1%}")
                    with col2:
                        st.metric(
                            "Best Variation CTR",
                            f"{best_ctr:.1%}",
                            delta=f"{improvement:.1%}",
                        )

                    # Format and display table
                    df_display = df_res.copy()
                    df_display["CTR"] = df_display["CTR"].map(lambda v: f"{v:.1%}")
                    df_display["Readability"] = df_display["Readability"].map(
                        lambda v: f"{v:.1f}"
                    )
                    df_display["Similarity"] = df_display["Similarity"].map(
                        lambda v: f"{v:.2f}"
                    )

                    st.subheader("Generated Variations (Ranked by Predicted CTR)")
                    st.dataframe(
                        df_display.reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Display usage info
                    st.info(f"OpenAI tokens used: {response.usage.total_tokens}")
                else:
                    st.error("No valid headline variations were generated.")

            except Exception as e:
                st.error(f"Error calling OpenAI API: {str(e)}")
                import traceback

                st.code(traceback.format_exc())

    # Sidebar with additional info
    with st.sidebar:
        st.header("📊 Pipeline Status")
        for stage, completed in resources["status"].items():
            status_icon = "✅" if completed else "❌"
            st.write(f"{status_icon} {stage.replace('_', ' ').title()}")

        # Add model info section
        if resources["status"]["training_completed"]:
            st.header("🤖 Model Info")
            st.write(f"**Model Type:** {type(resources['ctr_model']).__name__}")
            st.write(f"**Features:** {len(resources['feature_names'])}")
            st.write(f"**Version:** {os.getenv('MODEL_VERSION', 'Unknown')}")
            model_path = get_model_paths()["ctr_model"]
            st.write(f"**File:** {model_path.name}")

        if resources["status"]["training_completed"] and "performance" in resources:
            st.header("📈 Model Performance")
            perf = resources["performance"]
            st.metric("Test RMSE", f"{perf.get('test_rmse', 'N/A'):.4f}")
            st.metric("Test MAE", f"{perf.get('test_mae', 'N/A'):.4f}")
