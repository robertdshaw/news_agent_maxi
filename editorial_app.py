import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import openai
from sentence_transformers import SentenceTransformer
from pathlib import Path
from headline_utils import get_model_paths
from model_loader import load_ctr_model
from textstat import flesch_reading_ease

# ---- App Configuration & Custom Styles ----
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- Helper Functions ----
@st.cache_resource
def load_resources():
    paths = get_model_paths()
    df = pd.read_csv(paths["articles_csv"])
    with open(paths["embeddings"], "rb") as f:
        emb_data = pickle.load(f)
    index = faiss.read_index(str(paths["faiss_index"]))
    model_data, err = load_ctr_model(paths["ctr_model"])
    if err:
        st.error(f"Error loading CTR model: {err}")
        st.stop()
    ctr_model = model_data["model"]
    feature_names = model_data["feature_names"]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return (
        df,
        emb_data["embeddings"],
        emb_data["newsIDs"],
        index,
        ctr_model,
        feature_names,
        embedder,
    )


def extract_features(titles, embedder):
    """Compute pattern and embedding features for a list of titles."""
    embs = embedder.encode(titles)
    rows = []
    for i, title in enumerate(titles):
        feat = {
            "title_length": len(title),
            "title_reading_ease": flesch_reading_ease(title),
            "has_question": int("?" in title),
            "has_exclamation": int("!" in title),
            "has_number": int(any(c.isdigit() for c in title)),
            "has_colon": int(":" in title),
            "has_quotes": int(any(q in title for q in ['"', "'"])),
            "abstract_length": 50,  # Default value since not available
            "category_enc": 0,  # Default value since not available
            "hour": 12,  # Default value since not available
            "day_of_week": 1,  # Default value since not available
        }
        for j in range(min(50, embs.shape[1])):
            feat[f"emb_{j}"] = embs[i, j]
        rows.append(feat)
    return pd.DataFrame(rows)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---- Initialize Data & Load Header Image ----
with st.spinner("Loading models and data..."):
    df, all_embs, news_ids, index, ctr_model, feature_names, embedder = load_resources()
    df = df.set_index(pd.Index(news_ids))

import random

img_dir = Path(__file__).parent / "images"
all_images = list(img_dir.glob("*.jpg"))
selected = random.choice(all_images) if all_images else None
if selected:
    st.image(str(selected), width=500)

st.title("📰 Agentic AI News Editor")

# ---- UI Tabs ----
tab1, tab2, tab3 = st.tabs(["Retrieve & Rank", "Predict CTR", "Rewrite Headline"])

with tab1:
    st.header("🔍 Retrieval & Ranking")
    query = st.text_input("Query:", "big tech layoffs")
    k = st.slider("Top K articles:", 1, 20, 5)
    if st.button("Search & Rank Articles"):
        # semantic retrieval
        q_emb = embedder.encode([query]).astype("float32")
        D, I = index.search(q_emb, k)
        ids = [news_ids[i] for i in I[0]]
        subset = df.loc[ids].copy()

        # CTR + similarity
        feats = extract_features(subset["title"].tolist(), embedder)
        preds = ctr_model.predict(feats[feature_names])
        sims = [cosine_sim(q_emb[0], all_embs[i]) for i in I[0]]
        subset["Predicted CTR"] = preds
        subset["Similarity"] = sims

        # show metrics
        st.metric("Avg CTR", f"{np.mean(preds):.1%}")
        st.metric("Avg Sim", f"{np.mean(sims):.2f}")

        # display table
        disp = subset[["title", "category", "Predicted CTR", "Similarity"]]
        disp["Predicted CTR"] = disp["Predicted CTR"].map("{:.1%}".format)
        disp["Similarity"] = disp["Similarity"].map("{:.2f}".format)
        st.dataframe(disp.reset_index(drop=True), use_container_width=True)


with tab2:
    st.header("⚡ Instant CTR Prediction")
    st.markdown("Type or paste a candidate headline and get an instant CTR estimate.")

    # ── Define two columns: input on left, reference on right ──
    col_input, col_ref = st.columns([2, 1])

    with col_input:
        # Headline input + Predict button
        headline = st.text_input("Headline:", key="predict_input")
        if st.button("Predict CTR", key="predict_btn"):
            feats = extract_features([headline], embedder)
            pred = ctr_model.predict(feats[feature_names])[0]
            st.metric("Predicted CTR", f"{pred:.1%}")

    with col_ref:
        # Boxed reference stats
        st.markdown("**🔎 Reference CTR by Category**")
        overall_mean = df["ctr"].mean()
        st.metric("Global mean CTR", f"{overall_mean:.2%}")

        # Prepare category‐mean CTR
        ctr_by_cat = (
            df.groupby("category")["ctr"]
            .mean()
            .sort_values(ascending=False)
            .to_frame("mean_ctr")
        )
        ctr_by_cat["mean_ctr"] = ctr_by_cat["mean_ctr"].map("{:.2%}".format)

        # Inject CSS to widen cols and shrink row padding/font
        st.markdown(
            """
            <style>
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

        # Render full table without scroll
        st.table(ctr_by_cat)


# --- Tab 3: Rewrite Headline ---
with tab3:
    st.header("✍️ Headline Rewriting")
    original = st.text_input("Original Headline:", "")
    n = st.slider("Variations to Create:", 1, 5, 3)
    if st.button("Generate & Score") and original:
        # Initialize the OpenAI client
        client = openai.OpenAI()

        # Detailed system instructions for the LLM
        detailed_instructions = """
You are a data-driven news editor. When rewriting the headline, follow these guidelines:
1. Preserve factual accuracy and key entities.
2. Improve predicted click-through rate (CTR).
3. Enhance readability (aim for a Flesch Reading Ease score ≥ 60).
4. Maintain semantic similarity to the original (no extraneous angle).
5. Keep the headline under 70 characters.
"""

        # Build the user prompt
        prompt = (
            f"{detailed_instructions}\n"
            f"Original: {original}\n"
            f"Rewrites (provide {n} options, each on its own line):"
        )

        # Send to the model
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": detailed_instructions.strip()},
                {"role": "user", "content": prompt},
            ],
            n=1,
            temperature=0.7,
        )
        import logging

        logging.info(
            f"OpenAI call successful, response ID: {response.id}, tokens used: {response.usage.total_tokens}"
        )
        # Or display in-app:
        st.write("🔑 OpenAI usage:", response.usage)
        st.write("🆔 Request ID:", response.id)

        # Extract lines and limit to n
        text = response.choices[0].message.content.strip()
        candidates = [
            line.strip("- ").strip() for line in text.splitlines() if line.strip()
        ][:n]

        # Score originals and candidates
        orig_feats = extract_features([original], embedder)
        orig_ctr = ctr_model.predict(orig_feats[feature_names])[0]
        orig_emb = embedder.encode([original])[0]

        rows = []
        for cand in candidates:
            feats = extract_features([cand], embedder)
            ctr_pred = ctr_model.predict(feats[feature_names])[0]
            read = flesch_reading_ease(cand)
            sim = cosine_sim(orig_emb, embedder.encode([cand])[0])
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

        # Display metrics and table
        st.metric("Original CTR", f"{orig_ctr:.1%}")
        st.metric("Best Variation CTR", f"{best_ctr:.1%}", delta=f"{improvement:.1%}")
        df_res["CTR"] = df_res["CTR"].map(lambda v: f"{v:.1%}")
        df_res["Readability"] = df_res["Readability"].map(lambda v: f"{v:.1f}")
        df_res["Similarity"] = df_res["Similarity"].map(lambda v: f"{v:.2f}")
        st.table(df_res.reset_index(drop=True))
