import sys
sys.path.append('.')
import numpy as np
from feature_utils import create_article_features_exact, load_preprocessing_components
import pickle
from pathlib import Path

# Load model and components
MODEL_DIR = Path("model_output")
model_files = list(MODEL_DIR.glob("*_optimized_model.pkl"))

if model_files:
    with open(model_files[0], "rb") as f:
        model = pickle.load(f)
    
    components = load_preprocessing_components()
    
    # Test different headlines
    headlines = [
        "Breaking News: Major Event",
        "Simple Local Story",
        "5 Amazing Tips for Success",
        "Why This Changes Everything"
    ]
    
    for headline in headlines:
        features = create_article_features_exact(headline, "", "news", components)
        
        # Get feature vector
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
            feature_vector = [features.get(name, 0.0) for name in feature_names]
        else:
            feature_vector = [features[k] for k in sorted(features.keys())]
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        prediction_proba = model.predict_proba(feature_vector)[0]
        ctr = max(0.01, prediction_proba[1] * 0.1)
        
        print(f"'{headline}' -> CTR: {ctr*100:.3f}%")
else:
    print("No model found")
