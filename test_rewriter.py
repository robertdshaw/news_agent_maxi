import sys
sys.path.append('.')

import llm_rewriter
from feature_utils import load_preprocessing_components
import os

# Check API key
print(f"OpenAI API Key set: {bool(os.getenv('OPENAI_API_KEY'))}")

# Try to load rewriter
try:
    from pathlib import Path
    import pickle

    # Load model pipeline (simplified)
    MODEL_DIR = Path("model_output")
    model_files = list(MODEL_DIR.glob("*_optimized_model.pkl"))
    
    if model_files:
        with open(model_files[0], "rb") as f:
            model = pickle.load(f)
        
        model_pipeline = {
            "model": model,
            "baseline_metrics": {"ctr_threshold": 0.05}
        }
        
        components = load_preprocessing_components()
        
        rewriter = llm_rewriter.EnhancedLLMHeadlineRewriter(
            model_pipeline=model_pipeline,
            components=components,
            eda_insights_path="headline_eda_insights.json"
        )
        
        # Test rewriting
        test_headline = "Local Team Wins Game"
        print(f"Testing headline: '{test_headline}'")
        
        result = rewriter.get_best_headline(test_headline, {"category": "sports"})
        print(f"Rewrite result: {result}")
        
        if result and 'best_headline' in result:
            print(f"Original: '{test_headline}'")
            print(f"Optimized: '{result['best_headline']}'")
            print(f"Different: {result['best_headline'] != test_headline}")
        else:
            print("❌ No valid rewrite result")
            
    else:
        print("❌ No model files found")
        
except Exception as e:
    print(f"❌ Rewriter test failed: {e}")
    import traceback
    traceback.print_exc()
