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

# App Configuration & Custom Styles
st.set_page_config(
    page_title="Agentic AI News Editor", layout="wide", initial_sidebar_state="expanded"
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

# You can either:
# Option 1: Add your own news-related JPG images to the images/ folder
# Option 2: Use this script to download some placeholder images


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


# Uncomment to download placeholder images:
# download_placeholder_images()

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

st.title("📰 Agentic AI News Editor")

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
                # Generate query embedding
                q_emb = resources["embedder"].encode([query]).astype("float32")

                # Check dimensions match
                index_dim = resources["index"].d
                query_dim = q_emb.shape[1]

                st.write(
                    f"Debug: Index dimension: {index_dim}, Query dimension: {query_dim}"
                )

                if query_dim != index_dim:
                    # Adjust query embedding to match index dimension
                    if query_dim > index_dim:
                        # Truncate if query embedding is larger
                        q_emb = q_emb[:, :index_dim]
                        st.warning(
                            f"Truncated query embedding from {query_dim} to {index_dim} dimensions"
                        )
                    else:
                        # Pad if query embedding is smaller
                        padding = np.zeros(
                            (q_emb.shape[0], index_dim - query_dim), dtype=np.float32
                        )
                        q_emb = np.concatenate([q_emb, padding], axis=1)
                        st.warning(
                            f"Padded query embedding from {query_dim} to {index_dim} dimensions"
                        )

                # Search with corrected dimensions
                D, I = resources["index"].search(q_emb, k)

                # Check if we have valid results
                if len(I[0]) == 0:
                    st.error("No results found")
                else:
                    # Get article indices (handle potential index mismatches)
                    article_indices = []
                    for idx in I[0]:
                        if idx < len(resources["emb_data"]["article_indices"]):
                            article_indices.append(
                                resources["emb_data"]["article_indices"][idx]
                            )
                        else:
                            st.warning(f"Index {idx} out of range, skipping")

                    # Create results dataframe
                    results = []
                    for i, (similarity_score, article_idx) in enumerate(
                        zip(D[0], article_indices)
                    ):
                        if i < len(resources["emb_data"]["split_labels"]):
                            split_label = resources["emb_data"]["split_labels"][I[0][i]]
                        else:
                            split_label = "unknown"

                        results.append(
                            {
                                "Rank": i + 1,
                                "Article_ID": f"Article_{article_idx}",
                                "Similarity_Score": float(similarity_score),
                                "Split": split_label,
                            }
                        )

                    if results:
                        results_df = pd.DataFrame(results)
                        st.success(f"Found {len(results)} similar articles")
                        st.dataframe(results_df, use_container_width=True)

                        # Show search statistics
                        avg_similarity = np.mean(
                            [r["Similarity_Score"] for r in results]
                        )
                        st.metric("Average Similarity", f"{avg_similarity:.4f}")
                    else:
                        st.error("No valid results to display")

            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                st.write("Debug info:")
                st.write(f"- Query: '{query}'")
                st.write(f"- Embeddings available: {resources['emb_data'] is not None}")
                st.write(f"- Index available: {resources['index'] is not None}")
                if resources["emb_data"]:
                    st.write(
                        f"- Embedding data keys: {list(resources['emb_data'].keys())}"
                    )

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
        original = st.text_input("Original Headline:", "")
        n = st.slider("Variations to Create:", 1, 5, 3)

        # OpenAI API key input
        openai_key = st.text_input("OpenAI API Key:", type="password")

        if st.button("Generate & Score") and original and openai_key:
            # Set OpenAI API key
            openai.api_key = openai_key
            client = openai.OpenAI(api_key=openai_key)

            detailed_instructions = """
You are a data-driven news editor. When rewriting the headline, follow these guidelines:
1. Preserve factual accuracy and key entities.
2. Improve predicted click-through rate (CTR).
3. Enhance readability (aim for a Flesch Reading Ease score ≥ 60).
4. Maintain semantic similarity to the original.
5. Keep the headline under 70 characters.
"""

            prompt = (
                f"{detailed_instructions}\n"
                f"Original: {original}\n"
                f"Rewrites (provide {n} options, each on its own line):"
            )

            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": detailed_instructions.strip()},
                        {"role": "user", "content": prompt},
                    ],
                    n=1,
                    temperature=0.7,
                )

                # Extract and score candidates
                text = response.choices[0].message.content.strip()
                candidates = [
                    line.strip("- ").strip()
                    for line in text.splitlines()
                    if line.strip()
                ][:n]

                # Score original and candidates
                orig_feats = extract_features_updated(
                    [original], resources["embedder"], resources["feature_metadata"]
                )
                orig_ctr = resources["ctr_model"].predict(
                    orig_feats[resources["feature_names"]]
                )[0]
                orig_emb = resources["embedder"].encode([original])[0]

                rows = []
                for cand in candidates:
                    feats = extract_features_updated(
                        [cand], resources["embedder"], resources["feature_metadata"]
                    )
                    ctr_pred = resources["ctr_model"].predict(
                        feats[resources["feature_names"]]
                    )[0]
                    read = flesch_reading_ease(cand) if cand.strip() else 0
                    sim = cosine_sim(orig_emb, resources["embedder"].encode([cand])[0])
                    rows.append(
                        {
                            "Candidate": cand,
                            "CTR": ctr_pred,
                            "Readability": read,
                            "Similarity": sim,
                        }
                    )

                df_res = pd.DataFrame(rows).sort_values("CTR", ascending=False)
                best_ctr = df_res["CTR"].iloc[0]
                improvement = best_ctr - orig_ctr

                # Display results
                st.metric("Original CTR", f"{orig_ctr:.1%}")
                st.metric(
                    "Best Variation CTR", f"{best_ctr:.1%}", delta=f"{improvement:.1%}"
                )

                # Format and display table
                df_res["CTR"] = df_res["CTR"].map(lambda v: f"{v:.1%}")
                df_res["Readability"] = df_res["Readability"].map(lambda v: f"{v:.1f}")
                df_res["Similarity"] = df_res["Similarity"].map(lambda v: f"{v:.2f}")
                st.table(df_res.reset_index(drop=True))

                # Display usage info
                st.info(f"OpenAI tokens used: {response.usage.total_tokens}")

            except Exception as e:
                st.error(f"Error calling OpenAI API: {str(e)}")

# Sidebar with additional info
with st.sidebar:
    st.header("📊 Pipeline Status")
    for stage, completed in resources["status"].items():
        status_icon = "✅" if completed else "❌"
        st.write(f"{status_icon} {stage.replace('_', ' ').title()}")

    if resources["status"]["training_completed"] and "performance" in resources:
        st.header("📈 Model Performance")
        perf = resources["performance"]
        st.metric("Test RMSE", f"{perf.get('test_rmse', 'N/A'):.4f}")
        st.metric("Test MAE", f"{perf.get('test_mae', 'N/A'):.4f}")
