print("Testing FAISS system...")

try:
    from model_loader import load_faiss_system

    data, err = load_faiss_system()

    if err:
        print("ERROR:", err)
    else:
        print("SUCCESS: Loaded", data.get("total_articles", 0), "articles")

        # Check the key
        if "embeddings_completed" in data:
            print("SUCCESS: embeddings_completed found:", data["embeddings_completed"])
        else:
            print("MISSING: embeddings_completed not found")

except Exception as e:
    print("ERROR:", e)
