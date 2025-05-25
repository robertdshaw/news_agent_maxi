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


def load_processed_data():
    """Load all processed datasets and combine for embedding generation."""
    # Load the three splits
    try:
        # Try loading from parquet files first (preferred)
        train_data = pd.read_parquet(PREP_DIR / "processed_data" / "X_train.parquet")
        val_data = pd.read_parquet(PREP_DIR / "processed_data" / "X_val.parquet")
        test_data = pd.read_parquet(PREP_DIR / "processed_data" / "X_test.parquet")

        # Load target data to get the original dataframes with titles
        y_train = pd.read_parquet(PREP_DIR / "processed_data" / "y_train.parquet")
        y_val = pd.read_parquet(PREP_DIR / "processed_data" / "y_val.parquet")
        y_test = pd.read_parquet(PREP_DIR / "processed_data" / "y_test.parquet")

        print("Loaded from parquet files")
        return None  # Need original data with titles

    except FileNotFoundError:
        # Fallback to CSV if available
        csv_path = PREP_DIR / "news_with_engagement.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"Loaded from CSV: {len(df)} articles")
            return df
        else:
            raise FileNotFoundError("No processed data found. Run preprocessing first.")


def build_embeddings_from_preprocessing():
    """Build embeddings using the same approach as preprocessing pipeline."""
    # Load processed feature matrices to get article indices
    X_train = pd.read_parquet(PREP_DIR / "processed_data" / "X_train.parquet")
    X_val = pd.read_parquet(PREP_DIR / "processed_data" / "X_val.parquet")
    X_test = pd.read_parquet(PREP_DIR / "processed_data" / "X_test.parquet")

    # Extract existing embeddings from feature matrices
    emb_cols = [col for col in X_train.columns if col.startswith("emb_")]

    if emb_cols:
        print(f"Found {len(emb_cols)} embedding columns in processed data")

        # Combine embeddings from all splits
        train_embs = X_train[emb_cols].values
        val_embs = X_val[emb_cols].values
        test_embs = X_test[emb_cols].values

        all_embeddings = np.vstack([train_embs, val_embs, test_embs])

        # Create newsID mapping (use indices as IDs since we don't have original newsIDs easily)
        all_indices = list(X_train.index) + list(X_val.index) + list(X_test.index)

        # Add split information for each article
        split_labels = (
            ["train"] * len(X_train) + ["val"] * len(X_val) + ["test"] * len(X_test)
        )

        return all_embeddings, all_indices, split_labels

    else:
        raise ValueError("No embedding columns found in processed data")


# Main execution
try:
    # Try to use embeddings from preprocessing pipeline
    print("Extracting embeddings from processed feature matrices...")
    embeddings, article_indices, split_labels = build_embeddings_from_preprocessing()

except Exception as e:
    print(f"Could not extract from processed data: {e}")
    print("Falling back to computing fresh embeddings...")

    # Fallback: load CSV and compute fresh embeddings
    df = load_processed_data()
    if df is None:
        raise ValueError("No data source available")

    titles = df["title"].fillna("").tolist()
    article_indices = df.index.tolist()
    split_labels = ["unknown"] * len(df)

    print("Computing fresh embeddings...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(titles, show_progress_bar=True)

print(f"Total articles: {len(embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")

# Build FAISS index
print("Building FAISS index...")
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings.astype("float32"))

# Save everything
with open(EMBEDDINGS_OUT, "wb") as f:
    pickle.dump(
        {
            "embeddings": embeddings,
            "article_indices": article_indices,
            "split_labels": split_labels,
            "embedding_dim": d,
        },
        f,
    )

with open(METADATA_OUT, "wb") as f:
    pickle.dump(
        {
            "total_articles": len(embeddings),
            "embedding_dimension": d,
            "splits": {
                "train": sum(1 for x in split_labels if x == "train"),
                "val": sum(1 for x in split_labels if x == "val"),
                "test": sum(1 for x in split_labels if x == "test"),
            },
        },
        f,
    )

faiss.write_index(index, str(INDEX_OUT))

print(f"Saved embeddings to {EMBEDDINGS_OUT}")
print(f"Saved metadata to {METADATA_OUT}")
print(f"Saved FAISS index to {INDEX_OUT}")
print("Embedding index build completed successfully")
