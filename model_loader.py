import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
from load_paths import get_model_paths, get_critical_paths


def load_xgboost_model(model_path=None):
    """Load the XGBoost engagement prediction model from disk"""
    if model_path is None:
        paths = get_model_paths()
        model_path = paths["xgboost_model"]

    model_path = Path(model_path)
    if not model_path.exists():
        err = f"XGBoost model file not found: {model_path}"
        logging.error(err)
        return None, err

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load model metadata
        paths = get_model_paths()
        metadata_path = paths["model_metadata"]
        metadata = None

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        logging.info(f"Loaded XGBoost model from: {model_path}")
        if metadata:
            auc = metadata.get("final_evaluation", {}).get("auc", 0)
            logging.info(f"Model AUC: {auc:.4f}")

        return {"model": model, "metadata": metadata}, None

    except Exception as e:
        err = f"Failed to load XGBoost model: {str(e)}"
        logging.error(err)
        return None, err


def load_faiss_search_system():
    """Load the complete FAISS article search system"""
    paths = get_model_paths()

    try:
        import faiss

        # Load FAISS index
        index = faiss.read_index(str(paths["faiss_index"]))

        # Load article lookup
        with open(paths["article_lookup"], "rb") as f:
            article_lookup = pickle.load(f)

        # Load ID mappings
        with open(paths["article_id_mappings"], "rb") as f:
            mappings = pickle.load(f)

        # Load metadata
        with open(paths["index_metadata"], "r") as f:
            metadata = json.load(f)

        search_data = {
            "index": index,
            "article_lookup": article_lookup,
            "mappings": mappings,
            "metadata": metadata,
            "total_articles": metadata.get("total_articles", 0),
            "high_engagement_articles": metadata.get("high_engagement_articles", 0),
            "rewrite_variants": metadata.get("rewrite_variants", 0),
            "system_ready": True,
            "embedding_dim": metadata.get("embedding_dim", 384),
            "index_type": metadata.get("index_type", "Unknown"),
        }

        logging.info(
            f"Loaded FAISS search system: {search_data['total_articles']:,} articles indexed"
        )
        logging.info(
            f"High engagement articles: {search_data['high_engagement_articles']:,}"
        )
        logging.info(f"Rewrite variants: {search_data['rewrite_variants']:,}")
        logging.info(f"Index type: {search_data['index_type']}")

        return search_data, None

    except Exception as e:
        err = f"Failed to load FAISS search system: {str(e)}"
        logging.error(err)
        return None, err


def load_preprocessing_components():
    """Load preprocessing components (encoders, transformers, etc.)"""
    paths = get_model_paths()

    components = {}
    errors = []

    # Load category encoder
    try:
        with open(paths["category_encoder"], "rb") as f:
            components["category_encoder"] = pickle.load(f)
        logging.info("Loaded category encoder")
    except Exception as e:
        errors.append(f"Category encoder: {e}")

    # Load PCA transformer if available
    try:
        if paths["pca_transformer"].exists():
            with open(paths["pca_transformer"], "rb") as f:
                components["pca_transformer"] = pickle.load(f)
            logging.info("Loaded PCA transformer")
    except Exception as e:
        errors.append(f"PCA transformer: {e}")

    # Load preprocessing metadata
    try:
        with open(paths["preprocessing_metadata"], "r") as f:
            components["preprocessing_metadata"] = json.load(f)
        logging.info("Loaded preprocessing metadata")
    except Exception as e:
        errors.append(f"Preprocessing metadata: {e}")

    return components, errors


def load_rewrite_analysis():
    """Load LLM headline rewrite analysis data"""
    paths = get_model_paths()

    rewrite_data = {}
    errors = []

    # Load rewrite analysis data
    try:
        if paths["rewrite_analysis_data"].exists():
            rewrite_data["rewrite_results"] = pd.read_parquet(
                paths["rewrite_analysis_data"]
            )
            logging.info(
                f"Loaded rewrite analysis: {len(rewrite_data['rewrite_results'])} variants"
            )
    except Exception as e:
        errors.append(f"Rewrite analysis data: {e}")

    # Load rewrite summary
    try:
        if paths["rewrite_summary"].exists():
            with open(paths["rewrite_summary"], "r") as f:
                rewrite_data["rewrite_summary"] = json.load(f)
            logging.info("Loaded rewrite summary")
    except Exception as e:
        errors.append(f"Rewrite summary: {e}")

    return rewrite_data, errors


def predict_article_engagement(model_data, features_df):
    """Predict article engagement using the loaded XGBoost model"""
    if model_data is None or features_df is None or len(features_df) == 0:
        logging.error("Model data or features dataframe is empty")
        return {"error": "Invalid input data"}

    try:
        model = model_data["model"]

        # Validate features_df has the required structure
        if not isinstance(features_df, pd.DataFrame):
            logging.error("features_df must be a pandas DataFrame")
            return {"error": "Invalid features format"}

        # Handle feature alignment
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)

            # Reorder columns to match model expectations
            missing_features = set(expected_features) - set(features_df.columns)
            if missing_features:
                logging.warning(f"Missing features: {missing_features}")
                # Add missing features with default values
                for feature in missing_features:
                    features_df[feature] = 0

            # Reorder columns to match model training order
            features_df = features_df[expected_features]

        # Make predictions
        predictions = model.predict(features_df)
        probabilities = model.predict_proba(features_df)[:, 1]

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "high_engagement": predictions.astype(bool),
            "engagement_scores": probabilities,
            "model_type": "XGBoost",
            "feature_count": len(features_df.columns),
        }

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}


def search_similar_articles(search_data, query_text, top_k=10, include_rewrites=False):
    """Search for articles similar to query text"""
    if not query_text or not query_text.strip():
        logging.error("Query text is empty")
        return []

    try:
        from sentence_transformers import SentenceTransformer
        import faiss

        # Create embedding for query
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = embedder.encode([query_text])
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search more results to allow filtering
        search_k = top_k * 3 if include_rewrites else top_k
        distances, indices = search_data["index"].search(query_embedding, search_k)

        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(search_data["mappings"]["idx_to_article_id"]):
                article_id = search_data["mappings"]["idx_to_article_id"][idx]
                if article_id in search_data["article_lookup"]:
                    article_info = search_data["article_lookup"][article_id].copy()
                    article_info["similarity_score"] = float(dist)
                    article_info["newsID"] = article_id

                    # Filter rewrite variants if not requested
                    if (
                        not include_rewrites
                        and article_info.get("dataset") == "rewrite_variant"
                    ):
                        continue

                    results.append(article_info)

                    if len(results) >= top_k:
                        break

        return results

    except Exception as e:
        logging.error(f"Search error: {str(e)}")
        return []


def get_high_engagement_articles(search_data, top_k=50):
    """Get top high-engagement articles for recommendations"""
    try:
        engaged_articles = []
        for newsid, info in search_data["article_lookup"].items():
            if pd.notna(info.get("high_engagement")) and info["high_engagement"] == 1:
                engaged_articles.append(
                    {
                        "newsID": newsid,
                        "title": info["title"],
                        "category": info["category"],
                        "ctr": info.get("ctr", 0),
                        "dataset": info["dataset"],
                        "abstract": info.get("abstract", ""),
                    }
                )

        # Sort by CTR
        engaged_articles.sort(key=lambda x: x["ctr"], reverse=True)
        return engaged_articles[:top_k]

    except Exception as e:
        logging.error(f"Error getting high engagement articles: {str(e)}")
        return []


def get_rewrite_improvements(search_data, top_k=20):
    """Get articles with the best rewrite improvements"""
    try:
        improvements = []
        for newsid, info in search_data["article_lookup"].items():
            if info.get("dataset") == "rewrite_variant":
                improvements.append(
                    {
                        "newsID": newsid,
                        "original_newsID": info.get("original_newsID"),
                        "strategy": info.get("rewrite_strategy"),
                        "quality_score": info.get("quality_score", 0),
                        "predicted_ctr_improvement": info.get(
                            "predicted_ctr_improvement", 0
                        ),
                        "title": info["title"],
                        "original_title": search_data["article_lookup"]
                        .get(info.get("original_newsID"), {})
                        .get("title", "Unknown"),
                    }
                )

        # Sort by predicted CTR improvement
        improvements.sort(key=lambda x: x["predicted_ctr_improvement"], reverse=True)
        return improvements[:top_k]

    except Exception as e:
        logging.error(f"Error getting rewrite improvements: {str(e)}")
        return []


def compare_original_vs_rewrites(search_data, original_newsID):
    """Compare original article with its rewrite variants"""
    try:
        if original_newsID not in search_data["article_lookup"]:
            return None

        original_article = search_data["article_lookup"][original_newsID]

        # Find rewrite variants
        rewrite_variants = []
        for newsid, info in search_data["article_lookup"].items():
            if info.get("original_newsID") == original_newsID:
                rewrite_variants.append(info)

        # Calculate similarities if embeddings available
        if rewrite_variants and "embedding_matrix" in search_data:
            try:
                original_idx = search_data["mappings"]["article_id_to_idx"][
                    original_newsID
                ]
                original_embedding = search_data["embedding_matrix"][original_idx]

                for variant in rewrite_variants:
                    variant_idx = search_data["mappings"]["article_id_to_idx"][
                        variant["newsID"]
                    ]
                    variant_embedding = search_data["embedding_matrix"][variant_idx]

                    similarity = np.dot(original_embedding, variant_embedding)
                    variant["similarity_to_original"] = float(similarity)
            except Exception as e:
                logging.warning(f"Could not calculate similarities: {e}")

        return {
            "original": original_article,
            "variants": rewrite_variants,
            "comparison_available": len(rewrite_variants) > 0,
        }

    except Exception as e:
        logging.error(f"Error comparing original vs rewrites: {str(e)}")
        return None


def interpret_engagement_prediction(probability, threshold=0.15):
    """Interpret article engagement probability for editorial decisions"""
    interpretation = {
        "raw_probability": float(probability),
        "editorial_score": min(100, probability * 100),
        "confidence": (
            "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        ),
        "action": (
            "Prioritize"
            if probability > 0.6
            else "Consider" if probability > 0.3 else "Review"
        ),
        "engagement_level": ("High" if probability >= threshold else "Low"),
        "threshold_used": threshold,
        "recommendation": (
            "Strong headline - likely to engage readers well"
            if probability > 0.6
            else (
                "Good headline - solid engagement potential"
                if probability > 0.3
                else "Headline may need improvement for better engagement"
            )
        ),
    }
    return interpretation


def load_complete_system():
    """Load all system components"""
    logging.info("Loading complete News Editor system...")

    system = {
        "model": None,
        "search": None,
        "preprocessing": None,
        "rewrite_analysis": None,
        "errors": [],
        "warnings": [],
        "system_ready": False,
    }

    # Load model
    model_data, model_error = load_xgboost_model()
    if model_error:
        system["errors"].append(f"Model: {model_error}")
    else:
        system["model"] = model_data
        logging.info("✅ Model loaded successfully")

    # Load search system
    search_data, search_error = load_faiss_search_system()
    if search_error:
        system["errors"].append(f"Search: {search_error}")
    else:
        system["search"] = search_data
        logging.info("✅ Search system loaded successfully")

    # Load preprocessing components
    prep_data, prep_errors = load_preprocessing_components()
    if prep_errors:
        system["warnings"].extend([f"Preprocessing: {err}" for err in prep_errors])
    system["preprocessing"] = prep_data

    # Load rewrite analysis (optional)
    rewrite_data, rewrite_errors = load_rewrite_analysis()
    if rewrite_errors:
        system["warnings"].extend([f"Rewrite: {err}" for err in rewrite_errors])
    system["rewrite_analysis"] = rewrite_data

    # Determine system readiness
    critical_components = ["model", "search", "preprocessing"]
    system["system_ready"] = all(
        system[comp] is not None for comp in critical_components
    )

    if system["system_ready"]:
        logging.info("🎉 Complete system loaded successfully!")

        # Log system capabilities
        if system["model"]:
            model_perf = system["model"]["metadata"].get("final_evaluation", {})
            auc = model_perf.get("auc", 0)
            logging.info(f"📊 Model AUC: {auc:.4f}")

        if system["search"]:
            total_articles = system["search"]["total_articles"]
            rewrite_variants = system["search"]["rewrite_variants"]
            logging.info(
                f"🔍 Search: {total_articles:,} articles, {rewrite_variants:,} variants"
            )

        if (
            system["rewrite_analysis"]
            and "rewrite_results" in system["rewrite_analysis"]
        ):
            rewrite_count = len(system["rewrite_analysis"]["rewrite_results"])
            logging.info(f"🤖 Rewrite analysis: {rewrite_count} variants analyzed")

    else:
        logging.error("❌ System not ready - missing critical components")
        for error in system["errors"]:
            logging.error(f"  {error}")

    # Log warnings
    for warning in system["warnings"]:
        logging.warning(warning)

    return system


def validate_system_integrity():
    """Validate that all critical system components are present and compatible"""
    issues = []
    paths = get_critical_paths()

    # Check critical files
    for file_key, path in paths.items():
        if not path.exists():
            issues.append(f"Critical file missing: {file_key} at {path}")

    # Test model loading
    if paths["xgboost_model"].exists():
        try:
            model_data, error = load_xgboost_model(paths["xgboost_model"])
            if error:
                issues.append(f"XGBoost model loading error: {error}")
        except Exception as e:
            issues.append(f"XGBoost model validation error: {str(e)}")

    # Test FAISS system
    faiss_files = ["faiss_index", "article_lookup"]
    if all(paths.get(f, Path("")).exists() for f in faiss_files):
        try:
            search_data, error = load_faiss_search_system()
            if error:
                issues.append(f"FAISS search system error: {error}")
            elif not search_data.get("system_ready", False):
                issues.append("FAISS search system not ready")
        except Exception as e:
            issues.append(f"FAISS search validation error: {str(e)}")

    # Test preprocessing components
    try:
        components, errors = load_preprocessing_components()
        if errors:
            issues.extend([f"Preprocessing component error: {err}" for err in errors])
    except Exception as e:
        issues.append(f"Preprocessing validation error: {str(e)}")

    is_valid = len(issues) == 0
    return is_valid, issues


def test_complete_system():
    """Test the complete system functionality"""
    print("Testing Complete News Editor System...")

    try:
        # Load complete system
        system = load_complete_system()

        if not system["system_ready"]:
            print("❌ System not ready:")
            for error in system["errors"]:
                print(f"  {error}")
            return False

        # Test model predictions
        print("Testing model predictions...")
        if system["model"]:
            # Load sample data for testing
            paths = get_model_paths()
            if paths["train_features"].exists():
                train_data = pd.read_parquet(paths["train_features"])
                sample_features = train_data.head(1)

                result = predict_article_engagement(system["model"], sample_features)
                if "error" in result:
                    print(f"❌ Prediction failed: {result['error']}")
                    return False
                else:
                    prob = result["probabilities"][0]
                    pred = result["predictions"][0]
                    print(
                        f"✅ Model prediction: probability={prob:.3f}, high_engagement={pred}"
                    )

        # Test search functionality
        print("Testing search functionality...")
        if system["search"]:
            search_results = search_similar_articles(
                system["search"], "technology news", top_k=3
            )
            if search_results:
                print(f"✅ Search works: found {len(search_results)} similar articles")
            else:
                print("⚠️  Search returned no results (may be expected)")

        # Test rewrite analysis (optional)
        if (
            system["rewrite_analysis"]
            and "rewrite_results" in system["rewrite_analysis"]
        ):
            rewrite_count = len(system["rewrite_analysis"]["rewrite_results"])
            print(f"✅ Rewrite analysis: {rewrite_count} variants available")

        print("\n🎉 All system tests passed!")
        print("System is ready for deployment!")
        return True

    except Exception as e:
        print(f"❌ System test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test the complete system
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("NEWS EDITOR SYSTEM LOADER TEST")
    print("=" * 60)

    # Check system integrity
    is_valid, issues = validate_system_integrity()
    if not is_valid:
        print("❌ System integrity check failed:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease run the preprocessing and training scripts first.")
    else:
        print("✅ System integrity check passed")

        # Run comprehensive test
        success = test_complete_system()

        if success:
            print("\n🚀 News Editor System is ready for deployment!")
            print("Focus: Article engagement prediction with LLM optimization")
            print("Target: high_engagement classification with headline rewriting")
        else:
            print("\n⚠️  System tests failed. Please check the logs above.")

    print("=" * 60)
