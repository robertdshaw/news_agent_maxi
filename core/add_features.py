import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

# Configuration
PREP_DIR = Path("data/preprocessed")
OUTPUT_DIR = Path("data/preprocessed/processed_data")
PLOT_OUTPUT_DIR = Path("data/preprocessed/processed_data/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("INTERACTION FEATURE ENGINEERING & ANALYSIS")
print("=" * 80)


def add_interaction_features(df_X_input: pd.DataFrame):
    """
    Adds 5 specific interaction features to the input DataFrame.
    Assumes that base features like 'has_question', 'has_number', 'title_word_count',
    'has_abstract', 'title_reading_ease', 'title_upper_ratio', 'category_enc',
    'title_pca_0', 'avg_word_length' are already present in df_X_input.
    """
    df = df_X_input.copy()

    # 1. Binary * Binary
    df["interaction_question_x_number"] = df["has_question"] * df["has_number"]

    # 2. Numerical * Binary
    df["interaction_word_count_x_abstract"] = (
        df["title_word_count"] * df["has_abstract"]
    )

    # 3. Numerical * Numerical
    df["interaction_read_ease_x_upper_ratio"] = (
        df["title_reading_ease"] * df["title_upper_ratio"]
    )

    # 4. Label Encoded Categorical * Numerical
    df["interaction_cat_enc_x_pca_0"] = df["category_enc"] * df["title_pca_0"]

    # 5. Numerical * Binary
    df["interaction_avg_word_len_x_number"] = df["avg_word_length"] * df["has_number"]

    new_interaction_feature_names = [
        "interaction_question_x_number",
        "interaction_word_count_x_abstract",
        "interaction_read_ease_x_upper_ratio",
        "interaction_cat_enc_x_pca_0",
        "interaction_avg_word_len_x_number",
    ]

    print(f"  Added {len(new_interaction_feature_names)} interaction features.")
    return df, new_interaction_feature_names


def main():
    print("\nStep 1: Loading preprocessed data (features and targets)...")
    try:
        X_train_orig = pd.read_parquet(OUTPUT_DIR / "X_train_optimized.parquet")
        y_train_orig = pd.read_parquet(OUTPUT_DIR / "y_train_optimized.parquet")

        X_val_orig = pd.read_parquet(OUTPUT_DIR / "X_val_optimized.parquet")

        X_test_orig = pd.read_parquet(OUTPUT_DIR / "X_test_optimized.parquet")

        print("  Original data loaded successfully.")
        print(f"    X_train_orig shape: {X_train_orig.shape}")

    except FileNotFoundError as e:
        print(f"Error: Required preprocessed file not found: {e}")
        print("Please ensure 'EDA_preprocess_features.py' has been run successfully.")
        return

    print("\nStep 2: Adding interaction features...")
    X_train_with_interactions, new_feature_names = add_interaction_features(
        X_train_orig
    )
    X_val_with_interactions, _ = add_interaction_features(X_val_orig)
    X_test_with_interactions, _ = add_interaction_features(X_test_orig)

    print(
        "\nStep 3: Analyzing importance of NEW interaction features using RandomForestRegressor on training data..."
    )

    df_analysis = X_train_with_interactions[new_feature_names].copy()
    df_analysis["ctr"] = y_train_orig["ctr"]
    df_analysis.dropna(subset=new_feature_names + ["ctr"], inplace=True)

    if df_analysis.empty:
        print(
            "DataFrame for analysis is empty after dropping NaNs. Skipping feature importance analysis."
        )
    else:
        X_for_analysis = df_analysis[new_feature_names]
        y_for_analysis = df_analysis["ctr"]

        print(
            f"Training RandomForestRegressor on {len(X_for_analysis)} samples to assess interaction feature importance..."
        )
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_for_analysis, y_for_analysis)

        print("  Calculating permutation importance...")
        results = permutation_importance(
            model,
            X_for_analysis,
            y_for_analysis,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        importances = pd.Series(
            results.importances_mean, index=new_feature_names
        ).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.values, y=importances.index, palette="viridis")
        plt.title(
            "Permutation Importance of New Interaction Features on CTR (Training Data)"
        )
        plt.xlabel("Mean Importance Decrease (Permutation)")
        plt.tight_layout()
        plot_path = PLOT_OUTPUT_DIR / "interaction_feature_importance.png"
        plt.savefig(plot_path)
        print(f"  Interaction feature importance plot saved to: {plot_path}")

    print("\nStep 4: Saving datasets with added interaction features...")
    X_train_with_interactions.to_parquet(
        OUTPUT_DIR / "X_train_with_interactions.parquet", index=False
    )
    X_val_with_interactions.to_parquet(
        OUTPUT_DIR / "X_val_with_interactions.parquet", index=False
    )
    X_test_with_interactions.to_parquet(
        OUTPUT_DIR / "X_test_with_interactions.parquet", index=False
    )
    print(
        f"  Saved X_train_with_interactions.parquet (shape: {X_train_with_interactions.shape})"
    )
    print(
        f"  Saved X_val_with_interactions.parquet (shape: {X_val_with_interactions.shape})"
    )
    print(
        f"  Saved X_test_with_interactions.parquet (shape: {X_test_with_interactions.shape})"
    )

    print("\nStep 5: Updating preprocessing metadata...")
    metadata_path = OUTPUT_DIR / "preprocessing_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            preprocessing_metadata = json.load(f)

        original_available_features = preprocessing_metadata.get(
            "available_features", []
        )
        updated_available_features = original_available_features + [
            f_name
            for f_name in new_feature_names
            if f_name not in original_available_features
        ]

        preprocessing_metadata["available_features"] = updated_available_features
        preprocessing_metadata["features_created"] = len(updated_available_features)

        if "feature_categories" not in preprocessing_metadata:
            preprocessing_metadata["feature_categories"] = {}

        preprocessing_metadata["feature_categories"]["interaction_features"] = len(
            new_feature_names
        )
        preprocessing_metadata["feature_categories"]["total_features"] = len(
            updated_available_features
        )

        if (
            "X_train_with_interactions.parquet"
            not in preprocessing_metadata["files_created"]
        ):
            preprocessing_metadata["files_created"].extend(
                [
                    "X_train_with_interactions.parquet",
                    "X_val_with_interactions.parquet",
                    "X_test_with_interactions.parquet",
                ]
            )

        with open(metadata_path, "w") as f:
            json.dump(preprocessing_metadata, f, indent=2)
        print(f"  Preprocessing metadata updated at: {metadata_path}")
    else:
        print(
            f"Warning: Preprocessing metadata file not found at {metadata_path}. Cannot update."
        )

    print("\n" + "=" * 80)
    print("INTERACTION FEATURE CREATION AND ANALYSIS COMPLETE")
    print(
        f"Next step: Run 'model_class.py' using the '*_with_interactions.parquet' files."
    )
    print("=" * 80)


if __name__ == "__main__":

    main()
