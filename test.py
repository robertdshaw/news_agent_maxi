import json
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from openai import OpenAI
import os

# Import your enhanced rewriter
from llm_rewriter import EnhancedLLMHeadlineRewriter
from streamlit_app import load_model, load_preprocessing_components


def load_components():
    """Load your preprocessing components"""
    try:
        return load_preprocessing_components()
    except:
        # Fallback if streamlit_app function doesn't work
        components_path = "data/preprocessed/preprocessing_components.pkl"
        if Path(components_path).exists():
            with open(components_path, "rb") as f:
                return pickle.load(f)
        else:
            print("⚠️ Components file not found")
            return None


def load_model_pipeline():
    """Load your model pipeline"""
    try:
        return load_model()
    except:
        # Fallback
        model_path = "model_output/xgboost_optimized_model.pkl"
        if Path(model_path).exists():
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return {"model": model}
        else:
            print("⚠️ Model file not found")
            return None


def setup_enhanced_rewriter():
    """Setup the enhanced rewriter with your components"""

    # Load your model and components
    model_pipeline = load_model_pipeline()
    components = load_components()

    if not model_pipeline:
        print("❌ Failed to load model pipeline")
        return None

    if not components:
        print("❌ Failed to load preprocessing components")
        return None

    # Setup OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️ No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        print("Or enter it manually:")
        api_key = input("Enter OpenAI API Key (or press Enter to skip): ").strip()
        if api_key:
            openai_api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key

    llm_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

    # Find EDA insights file
    eda_paths = [
        "headline_eda_insights.json",
        "data/preprocessed/processed_data/headline_eda_insights.json",
        "eda_insights.json",
    ]

    eda_insights_path = None
    for path in eda_paths:
        if Path(path).exists():
            eda_insights_path = path
            print(f"✅ Found EDA insights: {path}")
            break

    if not eda_insights_path:
        print("⚠️ No EDA insights file found - will use defaults")

    # Initialize enhanced rewriter
    enhanced_rewriter = EnhancedLLMHeadlineRewriter(
        model_pipeline=model_pipeline,
        components=components,
        llm_client=llm_client,
        eda_insights_path=eda_insights_path,
    )

    print("✅ Enhanced rewriter initialized successfully!")
    return enhanced_rewriter


def test_enhanced_features():
    """Test the enhanced prompting features"""

    print("🚀 Setting up Enhanced LLM Headline Rewriter...")
    enhanced_rewriter = setup_enhanced_rewriter()

    if not enhanced_rewriter:
        print("❌ Setup failed. Please check your files and API key.")
        return

    # Test headlines
    test_headlines = [
        {"title": "Local Team Wins Game", "category": "sports"},
        {"title": "Stock Market Changes Today", "category": "business"},
        {"title": "New Health Study Released", "category": "health"},
        {"title": "Breaking News Update", "category": "news"},
    ]

    print("\n" + "=" * 60)
    print("🧪 TESTING ENHANCED PROMPTING FEATURES")
    print("=" * 60)

    for i, test_case in enumerate(test_headlines, 1):
        print(f"\n--- Test {i}: {test_case['title']} ---")

        article_data = {
            "category": test_case["category"],
            "abstract": f"Sample abstract about {test_case['category']} topic",
        }

        try:
            # Test the enhanced method
            print("🔄 Running enhanced headline optimization...")
            result = enhanced_rewriter.get_best_headline_enhanced(
                test_case["title"], article_data
            )

            # Display results
            print(f"📝 Original: {test_case['title']}")
            print(f"✨ Enhanced: {result['best_headline']}")
            print(f"👤 Persona: {result['persona']}")
            print(f"🎯 Best Layer: {result['best_layer']}")
            print(f"📈 CTR Improvement: +{result['ctr_improvement']:.4f}")

            if result.get("layer_results"):
                print("\n🔍 Layer-by-layer results:")
                for layer, layer_result in result["layer_results"].items():
                    print(
                        f"  {layer}: {layer_result['best_headline'][:50]}... "
                        f"(CTR: {layer_result['best_score']:.4f})"
                    )

        except Exception as e:
            print(f"❌ Error testing {test_case['title']}: {e}")
            import traceback

            print(traceback.format_exc())

    # Test comparison
    print(f"\n" + "=" * 60)
    print("🏆 RUNNING COMPARISON TEST")
    print("=" * 60)

    try:
        comparison_headlines = [
            "Local Team Wins Game",
            "Stock Market Changes Today",
            "New Health Study Released",
        ]

        print("🔄 Running comparison test...")
        df, summary = enhanced_rewriter.run_comparison_test(comparison_headlines)

        print(
            f"✅ Enhanced method won {summary['enhanced_wins']}/{summary['total_tests']} tests"
        )
        print(f"📊 Average advantage: +{summary['avg_enhancement_advantage']:.4f} CTR")

        if not df.empty:
            print("\n📋 Detailed Results:")
            for _, row in df.iterrows():
                print(f"  Original: {row['original_headline'][:40]}...")
                print(f"  Enhanced: {row['enhanced_headline'][:40]}...")
                print(f"  Advantage: +{row['enhancement_advantage']:.4f}")
                print()

    except Exception as e:
        print(f"❌ Comparison test failed: {e}")

    print("\n🎉 Testing complete!")


if __name__ == "__main__":
    test_enhanced_features()
