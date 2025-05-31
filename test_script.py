# test_rewriter.py
# Simple test to verify the EnhancedLLMHeadlineRewriter class works

import sys
import os

# Add the current directory to path to find modules
sys.path.append(".")

try:
    print("Testing EnhancedLLMHeadlineRewriter import...")
    from llm_rewriter import EnhancedLLMHeadlineRewriter

    print("✅ Import successful")

    # Create a minimal mock model pipeline for testing
    class MockModel:
        def predict_proba(self, X):
            # Return a simple prediction
            return [[0.6, 0.4]]  # [low_engagement_prob, high_engagement_prob]

    # Create mock components
    mock_model_pipeline = {
        "model": MockModel(),
        "baseline_metrics": {"overall_avg_ctr": 0.041},
    }

    mock_components = {
        "feature_order": ["title_length", "title_word_count", "has_number"],
        "category_encoder": None,
    }

    print("Testing class instantiation...")
    try:
        rewriter = EnhancedLLMHeadlineRewriter(
            model_pipeline=mock_model_pipeline, components=mock_components
        )
        print("✅ Class instantiation successful")

        # Test the compatibility methods
        print("Testing create_rewrite_variants method...")
        variants = rewriter.create_rewrite_variants(
            "Test headline", {"category": "news", "abstract": ""}
        )
        print(f"✅ create_rewrite_variants works: {list(variants.keys())}")

        print("Testing evaluate_rewrite_quality method...")
        quality = rewriter.evaluate_rewrite_quality(
            "Original headline", "Rewritten headline"
        )
        print(f"✅ evaluate_rewrite_quality works: {list(quality.keys())}")

        print("Testing predict_ctr_with_model method...")
        ctr = rewriter.predict_ctr_with_model(
            "Test headline", {"category": "news", "abstract": ""}
        )
        print(f"✅ predict_ctr_with_model works: {ctr:.4f}")

        print("\n🎉 All tests passed! The class is working correctly.")

    except Exception as e:
        print(f"❌ Class instantiation failed: {e}")
        import traceback

        traceback.print_exc()

except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure llm_rewriter.py is in the current directory")
    import traceback

    traceback.print_exc()

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback

    traceback.print_exc()
