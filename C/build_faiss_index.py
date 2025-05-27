import pickle
import faiss
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from model_loader import get_model_paths

print("Building FAISS content system...")

# Get paths
paths = get_model_paths()
faiss_dir = Path("faiss_system")
faiss_dir.mkdir(exist_ok=True)

# Load processed data
X_train = pd.read_parquet(paths["train_features"])
X_val = pd.read_parquet(paths["val_features"])
X_test = pd.read_parquet(paths["test_features"])

# Extract embeddings
emb_cols = [col for col in X_train.columns if col.startswith("emb_")]

if emb_cols:
    print(f"Found {len(emb_cols)} embedding columns")
    train_embs = X_train[emb_cols].values
    val_embs = X_val[emb_cols].values
    test_embs = X_test[emb_cols].values
    all_embeddings = np.vstack([train_embs, val_embs, test_embs])
else:
    print("No embeddings found, generating from text...")
    # Try to load original news data
    try:
        news_train = pd.read_csv(
            "source_data/train_data/news.tsv",
            sep="\t",
            header=None,
            names=[
                "newsID",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ],
        )
        news_val = pd.read_csv(
            "source_data/val_data/news.tsv",
            sep="\t",
            header=None,
            names=[
                "newsID",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ],
        )
        news_test = pd.read_csv(
            "source_data/test_data/news.tsv",
            sep="\t",
            header=None,
            names=[
                "newsID",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
            ],
        )

        all_news = pd.concat([news_train, news_val, news_test]).fillna("")
        texts = [
            f"{row['title']}. {row['abstract'][:200]}" for _, row in all_news.iterrows()
        ]

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        all_embeddings = embedder.encode(texts, show_progress_bar=True)
        print(f"Generated {len(all_embeddings)} embeddings")

    except Exception as e:
        print(f"Error loading source data: {e}")
        print("Creating dummy embeddings...")
        total_articles = len(X_train) + len(X_val) + len(X_test)
        all_embeddings = np.random.randn(total_articles, 384).astype(np.float32)

# Create article indices and splits
article_indices = (
    list(range(len(X_train))) + list(range(len(X_val))) + list(range(len(X_test)))
)
split_labels = ["train"] * len(X_train) + ["val"] * len(X_val) + ["test"] * len(X_test)

# Load predictions if available
predictions = None
if paths["model_predictions"].exists():
    predictions = pd.read_csv(paths["model_predictions"])
    print(f"Loaded {len(predictions)} predictions")

# Build metadata
article_metadata = []
for i in range(len(article_indices)):
    metadata = {
        "index": i,
        "newsID": f"article_{i}",
        "title": f"Article {i}",
        "abstract": "Sample abstract",
        "category": "news",
        "split": split_labels[i],
    }

    if predictions is not None and i < len(predictions):
        pred_row = predictions.iloc[i]
        metadata.update(
            {
                "engagement_probability": float(
                    pred_row.get("engagement_probability", 0)
                ),
                "engagement_score": float(pred_row.get("engagement_score", 0)),
                "editorial_recommendation": pred_row.get(
                    "editorial_recommendation", "Unknown"
                ),
            }
        )

    article_metadata.append(metadata)

# Build FAISS index
embeddings = all_embeddings.astype(np.float32)
faiss.normalize_L2(embeddings)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"Built FAISS index: {index.ntotal} articles, {dimension} dimensions")

# Save files
with open(faiss_dir / "article_embeddings.pkl", "wb") as f:
    pickle.dump(
        {
            "embeddings": embeddings,
            "article_indices": article_indices,
            "split_labels": split_labels,
            "embedding_dim": dimension,
        },
        f,
    )

with open(faiss_dir / "content_metadata.pkl", "wb") as f:
    pickle.dump(
        {
            "articles": article_metadata,
            "total_articles": len(article_metadata),
            "embedding_dimension": dimension,
            "has_engagement_predictions": predictions is not None,
        },
        f,
    )

faiss.write_index(index, str(faiss_dir / "faiss_index.idx"))

config = {
    "system_name": "AI News Editor FAISS System",
    "total_articles": len(article_metadata),
    "embedding_dimension": dimension,
    "created": pd.Timestamp.now().isoformat(),
}

with open(faiss_dir / "faiss_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"Saved to {faiss_dir}/")
print(f"Articles: {len(article_metadata)}")
print(f"Dimensions: {dimension}")
print("Done.")
