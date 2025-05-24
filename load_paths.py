import logging
from pathlib import Path


# === Model Paths for Agentic AI News Editor ===
def get_model_paths():
    """
    Returns a dict of file paths required by the Streamlit dashboard:
    - articles_csv: preprocessed articles with engagement metrics
    - embeddings: pickle file with title embeddings and newsIDs
    - faiss_index: FAISS index file for semantic retrieval
    - ctr_model: trained CTR regressor pickle file
    """
    base = Path(__file__).parent.resolve()
    data_dir = base / "preprocessed_data" / "processed_data"
    model_dir = base / "model_output"

    paths = {
        "articles_csv": data_dir / "news_with_engagement.csv",
        "embeddings": data_dir / "article_embeddings.pkl",
        "faiss_index": data_dir / "faiss_index.idx",
        "ctr_model": model_dir / "ctr_regressor.pkl",
    }

    for name, p in paths.items():
        if not p.exists():
            logging.error(f"Required file '{name}' not found at: {p}")
    return paths
