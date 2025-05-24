import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path

# --- Paths ---
BASE = Path(__file__).parent
PREPROC = BASE / "preprocessed_data" / "processed_data" / "news_with_engagement.csv"
EMBEDDINGS_OUT = (
    BASE / "preprocessed_data" / "processed_data" / "article_embeddings.pkl"
)
INDEX_OUT = BASE / "preprocessed_data" / "processed_data" / "faiss_index.idx"

# 1) Load preprocessed articles
df = pd.read_csv(PREPROC)
titles = df["title"].fillna("").tolist()
newsIDs = df["newsID"].tolist()

# 2) Compute embeddings
print("Computing embeddings...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embs = embedder.encode(titles, show_progress_bar=True)

# 3) Save embeddings + IDs
with open(EMBEDDINGS_OUT, "wb") as f:
    pickle.dump({"embeddings": embs, "newsIDs": newsIDs}, f)
print(f"Saved embeddings to {EMBEDDINGS_OUT}")

# 4) Build FAISS index
print("Building FAISS index...")
d = embs.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embs.astype("float32"))
faiss.write_index(index, str(INDEX_OUT))
print(f"Saved FAISS index to {INDEX_OUT}")
