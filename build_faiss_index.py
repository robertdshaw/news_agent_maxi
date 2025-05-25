import pickle
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Updated paths to match preprocessing pipeline
PREP_DIR = Path("data/preprocessed")
EMBEDDINGS_OUT = PREP_DIR / "article_embeddings.pkl"
INDEX_OUT = PREP_DIR / "faiss_index.idx"
METADATA_OUT = PREP_DIR / "embedding_metadata.pkl"


def rebuild_faiss_with_consistent_embeddings():
    """Rebuild FAISS index using the exact same method as Streamlit search."""

    print("🔧 Rebuilding FAISS index with consistent embeddings...")

    # Load headlines from MIND dataset (same as Streamlit)
    base_dir = Path.cwd()
    news_file = base_dir / "source_data" / "train_data" / "news.tsv"

    if not news_file.exists():
        raise FileNotFoundError(f"Cannot find {news_file}")

    print(f"📂 Loading headlines from {news_file}")

    # Load MIND dataset format
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

    # Clean titles
    titles = df["Title"].fillna("").astype(str).tolist()
    valid_titles = [t.strip() for t in titles if t.strip()]

    print(f"📊 Loaded {len(valid_titles)} valid headlines")
    print(f"📝 Sample titles: {valid_titles[:3]}")

    # Use same embedder as Streamlit
    print("🤖 Loading sentence transformer...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate full embeddings then truncate to 50D (exactly like Streamlit search after fix)
    print("⚡ Generating embeddings...")
    full_embeddings = embedder.encode(valid_titles, show_progress_bar=True)

    # Truncate to 50D to match your preprocessing pipeline expectation
    embeddings = full_embeddings[:, :50]

    print(
        f"✂️ Truncated embeddings to {embeddings.shape[1]}D (from {full_embeddings.shape[1]}D)"
    )

    # Create article indices that match the headline positions
    article_indices = list(range(len(valid_titles)))
    split_labels = ["train"] * len(valid_titles)

    # Build FAISS index
    print("🏗️ Building FAISS index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype("float32"))

    # Save embeddings and metadata
    print("💾 Saving embeddings...")
    with open(EMBEDDINGS_OUT, "wb") as f:
        pickle.dump(
            {
                "embeddings": embeddings,
                "article_indices": article_indices,
                "split_labels": split_labels,
                "embedding_dim": d,
                "titles": valid_titles,  # Include titles for easier debugging
            },
            f,
        )

    with open(METADATA_OUT, "wb") as f:
        pickle.dump(
            {
                "total_articles": len(valid_titles),
                "embedding_dimension": d,
                "method": "sentence_transformer_truncated_50d",
                "model": "all-MiniLM-L6-v2",
                "splits": {
                    "train": len(valid_titles),
                    "val": 0,
                    "test": 0,
                },
            },
            f,
        )

    # Save FAISS index
    print("💾 Saving FAISS index...")
    faiss.write_index(index, str(INDEX_OUT))

    # Test the index with a sample query
    print("\n🧪 Testing the rebuilt index...")
    test_query = "trump tariffs trade war"
    q_emb = embedder.encode([test_query])[:, :50].astype("float32")

    D, I = index.search(q_emb, 3)

    print(f"🔍 Test search for '{test_query}':")
    for i, (dist, idx) in enumerate(zip(D[0], I[0])):
        if idx < len(valid_titles):
            print(f"  {i+1}. {valid_titles[idx]} (distance: {dist:.3f})")

    print(f"\n✅ Successfully rebuilt FAISS index!")
    print(f"📁 Saved to:")
    print(f"   - {EMBEDDINGS_OUT}")
    print(f"   - {INDEX_OUT}")
    print(f"   - {METADATA_OUT}")
    print(f"\n🎯 Index contains {len(valid_titles)} articles with {d}D embeddings")


if __name__ == "__main__":
    rebuild_faiss_with_consistent_embeddings()
