import pandas as pd
import numpy as np
import faiss
import pickle
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import warnings
from llm_rewriter import EnhancedLLMHeadlineRewriter

warnings.filterwarnings("ignore")

print("=" * 80)
print("FAISS INDEX CREATION WITH LLM REWRITE INTEGRATION")
print("=" * 80)

# Configuration
PREP_DIR = Path("data/preprocessed")
FAISS_DIR = Path("faiss_index")
MODEL_DIR = Path("model_output")
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
        try:
            with open(pca_file, "rb") as f:
                pca_transformer = pickle.load(f)
            print(f"✅ PCA transformer loaded: {type(pca_transformer)}")
            print(
                f"   PCA components: {getattr(pca_transformer, 'n_components_', 'Unknown')}"
            )
        except Exception as e:
            print(f"❌ PCA transformer loading failed: {e}")
            pca_transformer = None
    else:
        print("ℹ️ No PCA transformer file found - proceeding without PCA")

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

# Create model pipeline and components for the rewriter
model_pipeline = {
    "model": trained_model,
    "baseline_metrics": {"overall_avg_ctr": training_median_ctr},
}

components = {
    "feature_order": feature_order,
    "category_encoder": category_encoder,
    "pca_transformer": pca_transformer,  # Add this line!
}

rewriter = EnhancedLLMHeadlineRewriter(
    model_pipeline=model_pipeline, components=components
)

# Select low-performing headlines for rewriting analysis
low_performing_headlines = train_metadata[
    train_metadata["ctr"] < train_metadata["ctr"].median()
].nsmallest(CONFIG["rewrite_sample_size"], "ctr")

print(
    f"Selected {len(low_performing_headlines)} low-performing headlines for rewrite analysis"
)

# Initialize embedder for feature creation
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Enhanced rewrite with model-based CTR prediction for BOTH original and rewritten headlines
rewrite_results = []

for idx, article in low_performing_headlines.iterrows():
    # Predict CTR for original article using exact feature replication
    if trained_model is not None:
        original_predicted_ctr = rewriter.predict_ctr_with_model(
            article["title"],
            {
                "abstract": article.get("abstract", ""),
                "category": article.get("category", "news"),
            },
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
                rewritten_predicted_ctr = rewriter.predict_ctr_with_model(
                    rewritten_title,
                    {
                        "abstract": article.get("abstract", ""),
                        "category": article.get("category", "news"),
                    },
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

    # Create summary statistics with explicit type conversions for JSON compatibility
    avg_quality_imp = rewrite_results["overall_quality_score"].mean()
    avg_readability_imp = rewrite_results["readability_improvement"].mean()
    avg_heuristic_ctr_imp = rewrite_results["predicted_ctr_improvement"].mean()

    # Handle model_ctr_improvement potentially not existing if model didn't load
    if (
        "model_ctr_improvement" in rewrite_results.columns
        and not rewrite_results["model_ctr_improvement"].isnull().all()
    ):
        avg_model_ctr_imp_val = rewrite_results["model_ctr_improvement"].mean()
        best_strat_model_val = rewrite_results.loc[
            rewrite_results["model_ctr_improvement"].idxmax(), "strategy"
        ]
        positive_model_imp_val = (rewrite_results["model_ctr_improvement"] > 0).sum()
        negative_model_imp_val = (rewrite_results["model_ctr_improvement"] < 0).sum()
        best_model_imp_val = rewrite_results["model_ctr_improvement"].max()
    else:  # Fallback if model-based metrics aren't available
        avg_model_ctr_imp_val = 0.0
        best_strat_model_val = "N/A (model not used)"
        positive_model_imp_val = 0
        negative_model_imp_val = 0
        best_model_imp_val = 0.0

    rewrite_summary = {
        "total_rewrites": int(len(rewrite_results)),
        "unique_originals": int(rewrite_results["newsID"].nunique()),
        "strategies_tested": rewrite_results["strategy"]
        .unique()
        .tolist(),  # List of strings is fine
        "average_quality_improvement": (
            float(avg_quality_imp) if pd.notna(avg_quality_imp) else 0.0
        ),
        "average_readability_improvement": (
            float(avg_readability_imp) if pd.notna(avg_readability_imp) else 0.0
        ),
        "average_predicted_ctr_improvement": (
            float(avg_heuristic_ctr_imp) if pd.notna(avg_heuristic_ctr_imp) else 0.0
        ),  # LLM heuristic
        "average_model_ctr_improvement": (
            float(avg_model_ctr_imp_val) if pd.notna(avg_model_ctr_imp_val) else 0.0
        ),  # Actual model-based
        "best_performing_strategy_by_model": str(best_strat_model_val),
        "best_performing_strategy_by_quality": str(
            rewrite_results.loc[
                rewrite_results["overall_quality_score"].idxmax(), "strategy"
            ]
            if len(rewrite_results) > 0
            and not rewrite_results["overall_quality_score"].isnull().all()
            else "N/A"
        ),
        "semantic_similarity_maintained": float(
            (
                rewrite_results["semantic_similarity"] >= CONFIG["similarity_threshold"]
            ).mean()
            if "semantic_similarity" in rewrite_results.columns
            else 0.0
        ),
        "model_based_ctr_used": bool(trained_model is not None),  # Python bool
        "positive_model_improvements": int(positive_model_imp_val),
        "negative_model_improvements": int(negative_model_imp_val),
        "best_model_improvement": (
            float(best_model_imp_val) if pd.notna(best_model_imp_val) else 0.0
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

print(f"Combined dataset before deduplication: {len(all_metadata):,} articles")
# Deduplicate all_metadata to ensure each newsID (original or rewrite variant) is unique
# This means if an original newsID was in train, val, and test, only its first occurrence is kept.
# Rewrite variants already have unique newsIDs.
all_metadata = all_metadata.drop_duplicates(
    subset=["newsID"], keep="first"
).reset_index(drop=True)
print(
    f"Combined dataset AFTER deduplication by newsID: {len(all_metadata):,} unique articles/variants"
)

all_metadata = all_metadata.drop_duplicates(subset=["title"], keep="first").reset_index(
    drop=True
)
print(
    f"Combined dataset AFTER title deduplication: {len(all_metadata):,} unique titles to be indexed."
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
print(
    f"Embedding norms (should be ~1.0): {np.linalg.norm(embedding_matrix[:5], axis=1)}"
)

# ============================================================================
# STEP 6: CREATE FAISS INDEX
# ============================================================================
print("\nStep 6: Creating FAISS index...")

d = embedding_matrix.shape[1]
n = embedding_matrix.shape[0]

if n < 1000:
    print("Using Flat index for exact search")
    index = index = faiss.IndexFlatL2(d)
    index_type = "Flat"
else:
    print("Using IVF index for approximate search")
    quantizer = faiss.IndexFlatL2(d)
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
article_lookup = all_metadata.set_index("newsID").to_dict("index")

idx_to_article_id = dict(enumerate(all_metadata["newsID"]))
article_id_to_idx = {}
for i, news_id_val in enumerate(all_metadata["newsID"]):
    if news_id_val not in article_id_to_idx:
        article_id_to_idx[news_id_val] = i

# print("\nStep 7: Creating enhanced article lookup system...")

# if all_metadata.duplicated(subset=["newsID"]).any():
#     print(
#         f"INFO: Duplicate newsIDs detected in `all_metadata`. This is expected if articles span train/val/test periods."
#     )
#     print(
#         f"      For `article_lookup`, the metadata from the *first occurrence* of each newsID will be used."
#     )
#     # Create a version of all_metadata with unique newsIDs for the article_lookup dictionary
#     all_metadata_for_lookup = all_metadata.drop_duplicates(
#         subset=["newsID"], keep="first"
#     )
#     article_lookup = all_metadata_for_lookup.set_index("newsID").to_dict("index")
# else:
#     # If no duplicates, proceed as before
#     article_lookup = all_metadata.set_index("newsID").to_dict("index")

idx_to_article_id = dict(enumerate(all_metadata["newsID"]))

article_id_to_idx = {}
for i, news_id_val in enumerate(all_metadata["newsID"]):
    if news_id_val not in article_id_to_idx:
        article_id_to_idx[news_id_val] = i

print(f"Article lookup system created:")
print(f"   Total rows in all_metadata (and FAISS index): {len(all_metadata):,}")
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

    search_k = top_k * 3 if include_rewrites else top_k
    distances, indices = index.search(query_embedding, search_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        article_id = idx_to_article_id[idx]
        article_info = article_lookup[article_id].copy()
        l2_distance = float(dist)
        similarity_score = 1.0 / (1.0 + l2_distance)
        article_info["similarity_score"] = similarity_score
        article_info["newsID"] = article_id

        if not include_rewrites and article_info.get("dataset") == "rewrite_variant":
            continue

        results.append(article_info)

        if len(results) >= top_k:
            break

    return results


def compare_original_vs_rewrites(original_newsID):
    """Compare original article with its rewrite variants"""

    if original_newsID not in article_lookup:
        print(
            f"Warning: Original newsID {original_newsID} not found in article_lookup."
        )
        return None

    original_article_data = article_lookup[original_newsID].copy()
    original_article_data["newsID"] = original_newsID

    # Find rewrite variants
    rewrite_variants_list = []
    for (
        current_newsid,
        info_dict,
    ) in article_lookup.items():  # newsid_key is the actual newsID from article_lookup
        if info_dict.get("original_newsID") == original_newsID:
            # info_dict does not contain 'newsID' as a key because it was the index.
            # Add it back for consistent structure.
            variant_to_append = info_dict.copy()
            variant_to_append["newsID"] = (
                current_newsid  # Add the actual newsID of the variant
            )
            rewrite_variants_list.append(variant_to_append)

    # Get similarity scores between original and variants
    if rewrite_variants_list:
        if (
            original_newsID in article_id_to_idx
        ):  # Check if original_newsID has a direct index mapping
            original_idx = article_id_to_idx[original_newsID]
            # Ensure embedding_matrix is accessible (it should be global in the script or passed appropriately)
            original_embedding = embedding_matrix[original_idx : original_idx + 1]

            for variant in rewrite_variants_list:
                if (
                    variant["newsID"] in article_id_to_idx
                ):  # Check if variant newsID has a mapping
                    variant_idx = article_id_to_idx[variant["newsID"]]
                    variant_embedding = embedding_matrix[variant_idx : variant_idx + 1]

                    similarity = np.dot(original_embedding, variant_embedding.T)[0, 0]
                    variant["similarity_to_original"] = float(similarity)
                else:
                    variant["similarity_to_original"] = (
                        np.nan
                    )  # Or some other placeholder
                    print(
                        f"Warning: newsID {variant['newsID']} for variant not found in article_id_to_idx."
                    )
        else:
            print(
                f"Warning: original_newsID {original_newsID} not found in article_id_to_idx for similarity calculation."
            )
            for variant in rewrite_variants_list:
                variant["similarity_to_original"] = np.nan

    return {
        "original": original_article_data,  # Use the fetched and potentially augmented original_article_data
        "variants": rewrite_variants_list,
        "comparison_available": len(rewrite_variants_list) > 0,
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
