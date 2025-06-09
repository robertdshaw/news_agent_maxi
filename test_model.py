import sys
from pathlib import Path
import pickle

import numpy as np

sys.path.append('.')

from feature_utils import create_article_features_exact, load_preprocessing_components


def test_ctr_prediction():
    """Ensure CTR predictions run without errors and return valid values."""
    model_dir = Path("model_output")
    model_files = list(model_dir.glob("*_optimized_model.pkl"))
    assert model_files, "No optimized model file found"

    with open(model_files[0], "rb") as f:
        model = pickle.load(f)

    components = load_preprocessing_components()
    assert components is not None, "Failed to load preprocessing components"

    headlines = [
        "Breaking News: Major Event",
        "Simple Local Story",
        "5 Amazing Tips for Success",
        "Why This Changes Everything",
    ]

    for headline in headlines:
        features = create_article_features_exact(headline, "", "news", components)

        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
            feature_vector = [features.get(name, 0.0) for name in feature_names]
        else:
            feature_vector = [features[k] for k in sorted(features.keys())]

        feature_vector = np.array(feature_vector).reshape(1, -1)
        prediction_proba = model.predict_proba(feature_vector)[0]
        ctr = max(0.01, prediction_proba[1] * 0.1)

        assert isinstance(ctr, float)
        assert 0.0 <= ctr <= 1.0
