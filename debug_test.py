import sys
sys.path.append('.')

from feature_utils import load_preprocessing_components, create_article_features_exact
import pickle
from pathlib import Path

# Test model loading
MODEL_DIR = Path("model_output")
model_files = list(MODEL_DIR.glob("*_optimized_model.pkl"))
print(f"Found model files: {model_files}")

# Test preprocessing components
try:
    components = load_preprocessing_components()
    print(f"Components loaded: {list(components.keys()) if components else 'Failed'}")
except Exception as e:
    print(f"Component loading error: {e}")

# Test feature extraction with different headlines
headlines = [
    "Breaking: Major News Event Shakes Nation",
    "Local Team Wins Championship Game",
    "Stock Market Drops 5% Today"
]

for headline in headlines:
    try:
        features = create_article_features_exact(headline, "", "news", components)
        print(f"Headline: '{headline[:30]}...'")
        print(f"  Word count: {features.get('title_word_count', 'N/A')}")
        print(f"  Length: {features.get('title_length', 'N/A')}")
        print(f"  Has number: {features.get('has_number', 'N/A')}")
        print()
    except Exception as e:
        print(f"Feature extraction error for '{headline}': {e}")
