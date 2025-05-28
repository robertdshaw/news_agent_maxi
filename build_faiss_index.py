import pandas as pd
import numpy as np
import faiss
import pickle
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import warnings
from llm_rewriter import LLMHeadlineRewriter

warnings.filterwarnings("ignore")

print("=" * 80)
print("FAISS INDEX CREATION WITH LLM REWRITE INTEGRATION")
print("=" * 80)

# Configuration
PREP_DIR = Path("data/preprocessed")
FAISS_DIR = Path("faiss_index")
MODEL_DIR = Path("models")
FAISS_DIR.mkdir(parents=True, exist_ok=True)
(FAISS_DIR / "rewrite_analysis").mkdir(exist_ok=True)

CONFIG = {
    "embedding_dim": 384,
    "index_type": "IVF",
    "nlist": 100,
    "nprobe": 10,
    "rewrite_sample_size": 2,  # Number of headlines to rewrite for analysis
    "similarity_threshold": 0.85,  # Minimum similarity for rewrite variants
}

# ============================================================================
# STEP 1: LOAD ARTICLE DATA
# ============================================================================
print("\nStep 1: Loading article data...")

try:
    train_metadata = pd.read_parquet(
        PREP_DIR / "processed_data" / "article_metadata_train_optimized.parquet"
    )
    val_metadata = pd.read_parquet(
        PREP_DIR / "processed_data" / "article_metadata_val_optimized.parquet"
    )
    test_metadata = pd.read_parquet(
        PREP_DIR / "processed_data" / "article_metadata_test_optimized.parquet"
    )

    print(f"Loaded article metadata:")
    print(f"  Train: {len(train_metadata):,} articles")
    print(f"  Val: {len(val_metadata):,} articles")
    print(f"  Test: {len(test_metadata):,} articles")

except Exception as e:
    print(f"❌ Error loading article metadata: {e}")
    print("Please run the EDA preprocessing script first!")
    exit(1)

# ============================================================================
# STEP 2: LOAD MODEL AND PREPROCESSING COMPONENTS FOR CTR PREDICTION
# ============================================================================
print(
    "\nStep 2: Loading trained model and preprocessing components for CTR prediction..."
)

try:
    with open(MODEL_DIR / "xgboost_optimized_model.pkl", "rb") as f:
        trained_model = pickle.load(f)

    # Load preprocessing metadata (contains training median CTR and feature order)
    with open(PREP_DIR / "processed_data" / "preprocessing_metadata.json", "r") as f:
        preprocessing_metadata = json.load(f)

    # Load preprocessing components
    with open(PREP_DIR / "processed_data" / "category_encoder.pkl", "rb") as f:
        category_encoder = pickle.load(f)

    # Load PCA transformer if available
    pca_transformer = None
    pca_file = PREP_DIR / "processed_data" / "pca_transformer.pkl"
    if pca_file.exists():
        with open(pca_file, "rb") as f:
            pca_transformer = pickle.load(f)

    # Get training median CTR and feature order
    training_median_ctr = preprocessing_metadata.get("training_median_ctr", 0.030)
    feature_order = preprocessing_metadata.get("available_features", [])

    print("✅ Model and preprocessing components loaded successfully")
    print(f"  Training median CTR: {training_median_ctr:.6f}")
    print(f"  Expected features: {len(feature_order)}")

except Exception as e:
    print(f"⚠️ Warning: Could not load trained model or preprocessing components: {e}")
    print("Proceeding without model-based CTR predictions")
    trained_model = None
    category_encoder = None
    pca_transformer = None
    training_median_ctr = 0.030
    feature_order = []

# ============================================================================
# STEP 3: LLM HEADLINE REWRITING ANALYSIS
# ============================================================================
print(
    f"\nStep 3: Analyzing headline rewrites with LLM ({CONFIG['rewrite_sample_size']} samples)..."
)

rewriter = LLMHeadlineRewriter()

# Select low-performing headlines for rewriting analysis
low_performing_headlines = train_metadata[
    train_metadata["ctr"] < train_metadata["ctr"].median()
].nsmallest(CONFIG["rewrite_sample_size"], "ctr")

print(
    f"Selected {len(low_performing_headlines)} low-performing headlines for rewrite analysis"
)

# Initialize embedder for feature creation
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def create_features_for_headline_exact(title, abstract="", category="news"):
    """
    Create feature vector for a headline that EXACTLY replicates the preprocessing
    pipeline from EDA_preprocess_features.py
    """

    # Editorial criteria (must match EDA_preprocess_features.py)
    EDITORIAL_CRITERIA = {
        "target_reading_ease": 60,
        "readability_weight": 0.3,
        "engagement_weight": 0.4,
        "headline_quality_weight": 0.2,
        "timeliness_weight": 0.1,
        "target_ctr_gain": 0.05,
        "optimal_word_count": (8, 12),
        "max_title_length": 75,
    }

    features = {}

    # ========== STEP 1: Basic text features (exact replication) ==========
    features["title_length"] = len(title)
    features["abstract_length"] = len(abstract)
    features["title_word_count"] = len(title.split())
    features["abstract_word_count"] = len(abstract.split())

    # ========== STEP 2: Flesch Reading Ease (exact replication) ==========
    from textstat import flesch_reading_ease

    features["title_reading_ease"] = flesch_reading_ease(title) if title else 0
    features["abstract_reading_ease"] = flesch_reading_ease(abstract) if abstract else 0

    # ========== STEP 3: Headline Quality Indicators (exact replication) ==========
    features["has_question"] = 1 if "?" in title else 0
    features["has_exclamation"] = 1 if "!" in title else 0
    features["has_number"] = 1 if any(c.isdigit() for c in title) else 0
    features["has_colon"] = 1 if ":" in title else 0
    features["has_quotes"] = 1 if any(q in title for q in ['"', "'", '"', '"']) else 0
    features["has_dash"] = 1 if any(d in title for d in ["-", "–", "—"]) else 0

    # ========== STEP 4: Advanced headline metrics (exact replication) ==========
    features["title_upper_ratio"] = (
        sum(c.isupper() for c in title) / len(title) if title else 0
    )
    features["title_caps_words"] = len(
        [w for w in title.split() if w.isupper() and len(w) > 1]
    )
    features["avg_word_length"] = (
        np.mean([len(word) for word in title.split()]) if title.split() else 0
    )

    # ========== STEP 5: Content depth indicators (exact replication) ==========
    features["has_abstract"] = 1 if len(abstract) > 0 else 0
    features["title_abstract_ratio"] = features["title_length"] / (
        features["abstract_length"] + 1
    )

    # ========== STEP 6: Editorial scores (exact replication) ==========
    features["editorial_readability_score"] = (
        np.clip(features["title_reading_ease"] / 100, 0, 1)
        * EDITORIAL_CRITERIA["readability_weight"]
    )
    features["editorial_headline_score"] = (
        (features["has_question"] + features["has_number"] + features["has_colon"])
        / 3
        * EDITORIAL_CRITERIA["headline_quality_weight"]
    )

    # ========== STEP 7: CTR gain potential - REMOVED (DATA LEAKAGE) ==========
    # NOTE: ctr_gain_potential and below_median_ctr are NOT included in model features
    # to prevent data leakage (they depend on historical performance not available at publication)

    # ========== STEP 7: Editorial quality flags (exact replication) ==========
    features["needs_readability_improvement"] = (
        1
        if features["title_reading_ease"] < EDITORIAL_CRITERIA["target_reading_ease"]
        else 0
    )
    features["suboptimal_word_count"] = (
        1
        if (
            features["title_word_count"] < EDITORIAL_CRITERIA["optimal_word_count"][0]
            or features["title_word_count"]
            > EDITORIAL_CRITERIA["optimal_word_count"][1]
        )
        else 0
    )
    features["too_long_title"] = (
        1 if features["title_length"] > EDITORIAL_CRITERIA["max_title_length"] else 0
    )

    # ========== STEP 8: Category encoding (exact replication) ==========
    if category_encoder is not None:
        try:
            # Clean category the same way as in EDA preprocessing
            category_clean = (
                str(category).replace("nan", "unknown")
                if pd.notna(category)
                else "unknown"
            )

            if category_clean in category_encoder.classes_:
                features["category_enc"] = category_encoder.transform([category_clean])[
                    0
                ]
            else:
                # Use "unknown" if available, otherwise use 0
                features["category_enc"] = (
                    category_encoder.transform(["unknown"])[0]
                    if "unknown" in category_encoder.classes_
                    else 0
                )
        except Exception as e:
            print(f"Warning: Category encoding failed for '{category}': {e}")
            features["category_enc"] = 0
    else:
        features["category_enc"] = 0

    # ========== STEP 9: Create title embeddings (exact replication) ==========
    try:
        title_embedding = embedder.encode([title])[0]

        # Add full embeddings first
        for i, emb_val in enumerate(title_embedding[:384]):
            features[f"title_emb_{i}"] = float(emb_val)

        # Apply PCA if transformer is available
        if pca_transformer is not None:
            # Create embedding matrix for PCA transformation
            embedding_matrix = np.array([title_embedding[:384]]).astype(np.float32)
            pca_embeddings = pca_transformer.transform(embedding_matrix)[0]

            # Add PCA features (these will be used if model was trained with PCA)
            for i, pca_val in enumerate(pca_embeddings):
                features[f"title_pca_{i}"] = float(pca_val)

    except Exception as e:
        print(f"Warning: Could not create embeddings for title: {e}")
        # Add zero embeddings as fallback
        for i in range(384):
            features[f"title_emb_{i}"] = 0.0
        # Add zero PCA features if PCA was expected
        if pca_transformer is not None:
            for i in range(pca_transformer.n_components_):
                features[f"title_pca_{i}"] = 0.0

    return features


def predict_ctr_for_headline_exact(title, abstract="", category="news"):
    """
    Predict CTR for a specific headline using the trained model with EXACT feature replication
    """
    if trained_model is None:
        return 0.035  # Default CTR

    try:
        # Step 1: Create features using exact replication
        features = create_features_for_headline_exact(title, abstract, category)

        # Step 2: Create feature vector in the exact order expected by the model
        if feature_order:
            # Use the exact feature order from training
            feature_vector = []
            for feat_name in feature_order:
                feature_vector.append(features.get(feat_name, 0.0))
        elif hasattr(trained_model, "feature_names_in_"):
            # Fallback to model's expected features
            expected_features = list(trained_model.feature_names_in_)
            feature_vector = []
            for feat_name in expected_features:
                feature_vector.append(features.get(feat_name, 0.0))
        else:
            # Last resort: use all features in sorted order
            feature_vector = [features[k] for k in sorted(features.keys())]

        # Step 3: Make prediction
        feature_array = np.array(feature_vector).reshape(1, -1)
        engagement_prob = trained_model.predict_proba(feature_array)[:, 1][0]

        # Step 4: Convert engagement probability to estimated CTR
        estimated_ctr = max(0.01, engagement_prob * 0.1)

        return estimated_ctr

    except Exception as e:
        print(f"Warning: CTR prediction failed for headline '{title[:50]}...': {e}")
        return 0.035


# Enhanced rewrite with model-based CTR prediction for BOTH original and rewritten headlines
rewrite_results = []

for idx, article in low_performing_headlines.iterrows():
    # Predict CTR for original article using exact feature replication
    if trained_model is not None:
        original_predicted_ctr = predict_ctr_for_headline_exact(
            article["title"],
            article.get("abstract", ""),
            article.get("category", "news"),
        )
    else:
        original_predicted_ctr = article.get("ctr", 0.035)

    article_data = {
        "category": article.get("category", "news"),
        "ctr": original_predicted_ctr,  # Use model-predicted CTR
        "readability": article.get("title_reading_ease", 60),
        "abstract": article.get("abstract", ""),
    }

    # Get rewrite variants
    variants = rewriter.create_rewrite_variants(article["title"], article_data)

    for strategy, rewritten_title in variants.items():
        if rewritten_title != article["title"]:  # Only include actual rewrites

            # **KEY CHANGE: Use model with exact feature replication for rewritten headline**
            if trained_model is not None:
                rewritten_predicted_ctr = predict_ctr_for_headline_exact(
                    rewritten_title,
                    article.get("abstract", ""),
                    article.get("category", "news"),
                )
                # Calculate ACTUAL model-based CTR improvement
                model_ctr_improvement = rewritten_predicted_ctr - original_predicted_ctr
            else:
                rewritten_predicted_ctr = original_predicted_ctr
                model_ctr_improvement = 0.0

            # Still get LLM quality metrics for additional insights
            quality_metrics = rewriter.evaluate_rewrite_quality(
                article["title"], rewritten_title
            )

            rewrite_results.append(
                {
                    "newsID": article.get("newsID", idx),
                    "original_title": article["title"],
                    "strategy": strategy,
                    "rewritten_title": rewritten_title,
                    "original_ctr": original_predicted_ctr,  # Model-predicted original CTR
                    "rewritten_ctr": rewritten_predicted_ctr,  # Model-predicted rewritten CTR
                    "model_ctr_improvement": model_ctr_improvement,  # ACTUAL model-based improvement
                    "category": article_data["category"],
                    "quality_score": quality_metrics.get("overall_quality_score", 0),
                    "readability_improvement": quality_metrics.get(
                        "readability_improvement", 0
                    ),
                    "predicted_ctr_improvement": quality_metrics.get(
                        "predicted_ctr_improvement", 0
                    ),  # Keep LLM heuristic for comparison
                    "semantic_similarity": quality_metrics.get(
                        "semantic_similarity", 0
                    ),
                    "overall_quality_score": quality_metrics.get(
                        "overall_quality_score", 0
                    ),
                }
            )

rewrite_results = pd.DataFrame(rewrite_results)

if not rewrite_results.empty:
    print(f"Generated {len(rewrite_results)} rewrite variants")

    # Save rewrite analysis
    rewrite_results.to_parquet(
        FAISS_DIR / "rewrite_analysis" / "headline_rewrites.parquet"
    )

    # Create summary statistics with model-based improvements
    rewrite_summary = {
        "total_rewrites": len(rewrite_results),
        "unique_originals": rewrite_results["newsID"].nunique(),
        "strategies_tested": rewrite_results["strategy"].unique().tolist(),
        "average_quality_improvement": rewrite_results["overall_quality_score"].mean(),
        "average_readability_improvement": rewrite_results[
            "readability_improvement"
        ].mean(),
        "average_predicted_ctr_improvement": rewrite_results[
            "predicted_ctr_improvement"
        ].mean(),  # LLM heuristic-based improvement
        "average_model_ctr_improvement": rewrite_results[
            "model_ctr_improvement"
        ].mean(),  # ACTUAL model-based improvement
        "best_performing_strategy_by_model": (
            rewrite_results.loc[
                rewrite_results["model_ctr_improvement"].idxmax(), "strategy"
            ]
            if len(rewrite_results) > 0
            else "N/A"
        ),
        "best_performing_strategy_by_quality": (
            rewrite_results.loc[
                rewrite_results["overall_quality_score"].idxmax(), "strategy"
            ]
            if len(rewrite_results) > 0
            else "N/A"
        ),
        "semantic_similarity_maintained": (
            rewrite_results["semantic_similarity"] >= CONFIG["similarity_threshold"]
        ).mean(),
        "model_based_ctr_used": trained_model is not None,
        "positive_model_improvements": (
            rewrite_results["model_ctr_improvement"] > 0
        ).sum(),
        "negative_model_improvements": (
            rewrite_results["model_ctr_improvement"] < 0
        ).sum(),
        "best_model_improvement": (
            rewrite_results["model_ctr_improvement"].max()
            if len(rewrite_results) > 0
            else 0
        ),
    }

    with open(FAISS_DIR / "rewrite_analysis" / "rewrite_summary.json", "w") as f:
        json.dump(rewrite_summary, f, indent=2)

    print(f"Rewrite analysis completed:")
    print(
        f"  Best strategy (by model): {rewrite_summary['best_performing_strategy_by_model']}"
    )
    print(
        f"  Best strategy (by quality): {rewrite_summary['best_performing_strategy_by_quality']}"
    )
    print(
        f"  Avg quality improvement: {rewrite_summary['average_quality_improvement']:.2f}"
    )
    print(
        f"  Avg CTR improvement (LLM heuristic): {rewrite_summary['average_predicted_ctr_improvement']:.4f}"
    )
    print(
        f"  Avg CTR improvement (XGBoost model): {rewrite_summary['average_model_ctr_improvement']:.4f}"
    )
    print(f"  Model-based CTR: {rewrite_summary['model_based_ctr_used']}")
    print(f"  Positive improvements: {rewrite_summary['positive_model_improvements']}")
    print(f"  Best improvement: {rewrite_summary['best_model_improvement']:.4f}")

else:
    print("No rewrite results generated (API may be unavailable)")
    rewrite_summary = {"model_based_ctr_used": trained_model is not None}

# ============================================================================
# STEP 4: CREATE COMPREHENSIVE DATASET WITH REWRITES
# ============================================================================
print(
    "\nStep 4: Creating comprehensive dataset with original and rewritten headlines..."
)

# Combine all metadata
train_meta = train_metadata.copy()
train_meta["dataset"] = "train"

val_meta = val_metadata.copy()
val_meta["dataset"] = "val"

test_meta = test_metadata.copy()
test_meta["dataset"] = "test"

# Add placeholder engagement metrics for test data
for col in ["ctr", "high_engagement"]:
    if col not in test_meta.columns:
        test_meta[col] = np.nan

all_metadata = pd.concat([train_meta, val_meta, test_meta], ignore_index=True)

# Add rewritten variants to the dataset for embedding
if not rewrite_results.empty:
    rewrite_variants = []

    for _, row in rewrite_results.iterrows():
        variant_row = {
            "newsID": f"{row['newsID']}_rewrite_{row['strategy']}",
            "title": row["rewritten_title"],
            "category": row["category"],
            "dataset": "rewrite_variant",
            "original_newsID": row["newsID"],
            "rewrite_strategy": row["strategy"],
            "overall_quality_score": row["overall_quality_score"],
            "predicted_ctr_improvement": row["predicted_ctr_improvement"],
            "model_ctr_improvement": row.get(
                "model_ctr_improvement", 0
            ),  # Add model-based improvement
        }

        # Copy other fields from original
        original_article = all_metadata[all_metadata["newsID"] == row["newsID"]]
        if not original_article.empty:
            for col in ["abstract", "ctr", "high_engagement"]:
                if col in original_article.columns:
                    variant_row[col] = original_article[col].iloc[0]

        rewrite_variants.append(variant_row)

    rewrite_variants_df = pd.DataFrame(rewrite_variants)
    all_metadata = pd.concat([all_metadata, rewrite_variants_df], ignore_index=True)

    print(f"Added {len(rewrite_variants)} rewrite variants to embedding dataset")

print(
    f"Combined dataset: {len(all_metadata):,} articles (including {len(rewrite_variants) if not rewrite_results.empty else 0} rewrite variants)"
)

# ============================================================================
# STEP 5: CREATE EMBEDDINGS FOR ALL CONTENT
# ============================================================================
print("\nStep 5: Creating embeddings for all articles and rewrite variants...")

titles = all_metadata["title"].fillna("").tolist()

# Process in batches
batch_size = 1000
all_embeddings = []

for i in range(0, len(titles), batch_size):
    batch_titles = titles[i : i + batch_size]
    print(
        f"  Processing batch {i//batch_size + 1}/{(len(titles) + batch_size - 1)//batch_size}"
    )

    batch_embeddings = embedder.encode(
        batch_titles, show_progress_bar=False, batch_size=32
    )
    all_embeddings.append(batch_embeddings)

embedding_matrix = np.vstack(all_embeddings)
embedding_matrix = embedding_matrix.astype(np.float32)
faiss.normalize_L2(embedding_matrix)

print(f"Created embeddings: {embedding_matrix.shape}")

# ============================================================================
# STEP 6: CREATE FAISS INDEX
# ============================================================================
print("\nStep 6: Creating FAISS index...")

d = embedding_matrix.shape[1]
n = embedding_matrix.shape[0]

if n < 1000:
    print("Using Flat index for exact search")
    index = faiss.IndexFlatIP(d)
    index_type = "Flat"
else:
    print("Using IVF index for approximate search")
    quantizer = faiss.IndexFlatIP(d)
    nlist = min(CONFIG["nlist"], n // 10)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index_type = "IVF"

    print(f"Training IVF index with {nlist} clusters...")
    index.train(embedding_matrix)
    index.nprobe = CONFIG["nprobe"]

print(f"Adding {n:,} embeddings to index...")
index.add(embedding_matrix)

print(f"FAISS index created: {index_type}, {index.ntotal:,} vectors, {d}D")

# ============================================================================
# STEP 7: CREATE ENHANCED LOOKUP SYSTEM
# ============================================================================
print("\nStep 7: Creating enhanced article lookup system...")

if all_metadata.duplicated(subset=["newsID"]).any():
    print(
        f"INFO: Duplicate newsIDs detected in `all_metadata`. This is expected if articles span train/val/test periods."
    )
    print(
        f"      For `article_lookup`, the metadata from the *first occurrence* of each newsID will be used."
    )
    # Create a version of all_metadata with unique newsIDs for the article_lookup dictionary
    all_metadata_for_lookup = all_metadata.drop_duplicates(
        subset=["newsID"], keep="first"
    )
    article_lookup = all_metadata_for_lookup.set_index("newsID").to_dict("index")
else:
    # If no duplicates, proceed as before
    article_lookup = all_metadata.set_index("newsID").to_dict("index")

# This maps the FAISS vector index (0 to N-1, where N is total rows in all_metadata)
# to the newsID present at that specific row in the original all_metadata.
# This is essential for interpreting FAISS search results correctly.
idx_to_article_id = dict(enumerate(all_metadata["newsID"]))

# This maps a newsID to its *first appearing* row index in all_metadata
# (which corresponds to its first FAISS vector index).
# Useful if you have a newsID and want to find its primary vector or the version stored in article_lookup.
article_id_to_idx = {}
for i, news_id_val in enumerate(all_metadata["newsID"]):
    if (
        news_id_val not in article_id_to_idx
    ):  # Store only the first encountered index for each news_id
        article_id_to_idx[news_id_val] = i

# --- The rest of your Step 7 print statements ---
print(f"Article lookup system created:")
print(
    f"   Total rows in all_metadata (and FAISS index): {len(all_metadata):,}"
)  # Should match FAISS index.ntotal
print(f"   Unique newsIDs in article_lookup: {len(article_lookup):,}")
print(
    f"   Original articles in all_metadata: {len(all_metadata[all_metadata['dataset'] != 'rewrite_variant']):,}"
)
print(
    f"   Rewrite variants in all_metadata: {len(all_metadata[all_metadata['dataset'] == 'rewrite_variant']):,}"
)


# ============================================================================
# STEP 8: ENHANCED SEARCH FUNCTIONS
# ============================================================================
print("\nStep 8: Creating enhanced search functions...")


def search_similar_articles(query_text, top_k=10, include_rewrites=True):
    """Search for articles with option to include/exclude rewrite variants"""

    query_embedding = embedder.encode([query_text])
    query_embedding = query_embedding.astype(np.float32)
    faiss.normalize_L2(query_embedding)

    # Search more results to filter
    search_k = top_k * 3 if include_rewrites else top_k
    distances, indices = index.search(query_embedding, search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        article_id = idx_to_article_id[idx]
        article_info = article_lookup[article_id].copy()
        article_info["similarity_score"] = float(dist)
        article_info["newsID"] = article_id

        # Filter rewrite variants if requested
        if not include_rewrites and article_info.get("dataset") == "rewrite_variant":
            continue

        results.append(article_info)

        if len(results) >= top_k:
            break

    return results


def compare_original_vs_rewrites(original_newsID):
    """Compare original article with its rewrite variants"""

    if original_newsID not in article_lookup:
        return None

    original_article = article_lookup[original_newsID]

    # Find rewrite variants
    rewrite_variants = []
    for newsid, info in article_lookup.items():
        if info.get("original_newsID") == original_newsID:
            rewrite_variants.append(info)

    # Get similarity scores between original and variants
    if rewrite_variants:
        original_idx = article_id_to_idx[original_newsID]
        original_embedding = embedding_matrix[original_idx : original_idx + 1]

        for variant in rewrite_variants:
            variant_idx = article_id_to_idx[variant["newsID"]]
            variant_embedding = embedding_matrix[variant_idx : variant_idx + 1]

            similarity = np.dot(original_embedding, variant_embedding.T)[0, 0]
            variant["similarity_to_original"] = float(similarity)

    return {
        "original": original_article,
        "variants": rewrite_variants,
        "comparison_available": len(rewrite_variants) > 0,
    }


def get_top_rewrite_improvements():
    """Get articles with the best rewrite improvements"""

    improvements = []
    for newsid, info in article_lookup.items():
        if info.get("dataset") == "rewrite_variant":
            improvements.append(
                {
                    "newsID": newsid,
                    "original_newsID": info.get("original_newsID"),
                    "strategy": info.get("rewrite_strategy"),
                    "overall_quality_score": info.get("overall_quality_score", 0),
                    "predicted_ctr_improvement": info.get(
                        "predicted_ctr_improvement", 0
                    ),
                    "model_ctr_improvement": info.get(
                        "model_ctr_improvement", 0
                    ),  # Add model-based improvement
                    "title": info["title"],
                }
            )

    # Sort by model-based CTR improvement (the real prediction)
    improvements.sort(key=lambda x: x["model_ctr_improvement"], reverse=True)
    return improvements[:2]


# ============================================================================
# STEP 9: SAVE ENHANCED SYSTEM
# ============================================================================
print("\nStep 9: Saving enhanced FAISS system...")

# Save FAISS index
faiss.write_index(index, str(FAISS_DIR / "article_index.faiss"))

# Save enhanced lookup data
with open(FAISS_DIR / "article_lookup.pkl", "wb") as f:
    pickle.dump(article_lookup, f)

with open(FAISS_DIR / "article_id_mappings.pkl", "wb") as f:
    pickle.dump(
        {
            "article_id_to_idx": article_id_to_idx,
            "idx_to_article_id": idx_to_article_id,
        },
        f,
    )

# Save embedding matrix
np.save(FAISS_DIR / "embedding_matrix.npy", embedding_matrix)

# Save enhanced search functions
search_functions = {
    "search_similar_articles": search_similar_articles,
    "compare_original_vs_rewrites": compare_original_vs_rewrites,
    "get_top_rewrite_improvements": get_top_rewrite_improvements,
}

with open(FAISS_DIR / "search_functions.pkl", "wb") as f:
    pickle.dump(search_functions, f)

# Save comprehensive metadata
index_metadata = {
    "index_type": index_type,
    "total_articles": len(all_metadata),
    "original_articles": len(
        all_metadata[all_metadata["dataset"] != "rewrite_variant"]
    ),
    "rewrite_variants": len(all_metadata[all_metadata["dataset"] == "rewrite_variant"]),
    "embedding_dim": d,
    "datasets_included": all_metadata["dataset"].unique().tolist(),
    "rewrite_analysis": rewrite_summary,
    "config": CONFIG,
    "model_integration": {
        "model_available": trained_model is not None,
        "ctr_prediction_method": (
            "trained_model" if trained_model is not None else "actual_ctr"
        ),
    },
    "files_created": [
        "article_index.faiss",
        "article_lookup.pkl",
        "article_id_mappings.pkl",
        "embedding_matrix.npy",
        "search_functions.pkl",
        "index_metadata.json",
        "rewrite_analysis/headline_rewrites.parquet",
        "rewrite_analysis/rewrite_summary.json",
    ],
}

with open(FAISS_DIR / "index_metadata.json", "w") as f:
    json.dump(index_metadata, f, indent=2)

print("Enhanced FAISS system saved successfully!")

# ============================================================================
# STEP 10: TEST ENHANCED SYSTEM
# ============================================================================
print("\nStep 10: Testing enhanced search capabilities...")

# Test original vs rewrite comparison
if not rewrite_results.empty:
    sample_original = rewrite_results["newsID"].iloc[0]
    comparison = compare_original_vs_rewrites(sample_original)

    if comparison and comparison["comparison_available"]:
        print(f"\nSample comparison for {sample_original}:")
        print(f"Original: {comparison['original']['title']}")

        for variant in comparison["variants"][:2]:
            print(f"Rewrite ({variant['rewrite_strategy']}): {variant['title']}")
            print(f"  Quality score: {variant.get('overall_quality_score', 'N/A')}")
            print(f"  Similarity: {variant.get('similarity_to_original', 'N/A'):.3f}")

# Test rewrite improvements
top_improvements = get_top_rewrite_improvements()
if top_improvements:
    print(f"\nTop 3 rewrite improvements:")
    for i, improvement in enumerate(top_improvements[:3], 1):
        print(f"{i}. Strategy: {improvement['strategy']}")
        print(f"   Title: {improvement['title'][:60]}...")
        print(f"   Model CTR improvement: +{improvement['model_ctr_improvement']:.4f}")
        print(
            f"   LLM CTR improvement: +{improvement['predicted_ctr_improvement']:.4f}"
        )

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ENHANCED FAISS INDEX WITH LLM REWRITES COMPLETED")
print("=" * 80)

print(f"\n📊 INDEX STATISTICS:")
print(f"  Total articles indexed: {index_metadata['total_articles']:,}")
print(f"  Original articles: {index_metadata['original_articles']:,}")
print(f"  Rewrite variants: {index_metadata['rewrite_variants']:,}")
print(f"  Embedding dimension: {d}")
print(f"  Index type: {index_type}")

if rewrite_summary:
    print(f"\n🔄 REWRITE ANALYSIS:")
    print(f"  Headlines analyzed: {rewrite_summary['unique_originals']}")
    print(f"  Variants generated: {rewrite_summary['total_rewrites']}")
    print(
        f"  Best strategy (by model): {rewrite_summary['best_performing_strategy_by_model']}"
    )
    print(
        f"  Best strategy (by quality): {rewrite_summary['best_performing_strategy_by_quality']}"
    )
    print(
        f"  Avg quality improvement: {rewrite_summary['average_quality_improvement']:.2f}"
    )
    print(
        f"  Avg CTR improvement (LLM): {rewrite_summary['average_predicted_ctr_improvement']:.4f}"
    )
    print(
        f"  Avg CTR improvement (Model): {rewrite_summary['average_model_ctr_improvement']:.4f}"
    )
    print(f"  Model-based CTR: {rewrite_summary['model_based_ctr_used']}")
    print(f"  Positive improvements: {rewrite_summary['positive_model_improvements']}")
    print(f"  Best model improvement: +{rewrite_summary['best_model_improvement']:.4f}")

print(f"\n🔍 ENHANCED CAPABILITIES:")
print(f"  ✅ Original article search")
print(f"  ✅ Rewrite variant comparison")
print(f"  ✅ A/B testing preparation")
print(f"  ✅ Quality improvement tracking")
print(f"  ✅ Editorial optimization insights")
print(f"  ✅ Model-integrated CTR prediction")

print(f"\n📁 FILES CREATED:")
for file in index_metadata["files_created"]:
    print(f"  📁 {file}")

print("\n" + "=" * 80)
print("READY FOR EDITORIAL DASHBOARD INTEGRATION")
print("=" * 80)
