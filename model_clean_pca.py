import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

print("=" * 60)
print("CTR PREDICTION MODEL TRAINING WITH PCA")
print("=" * 60)

# Load preprocessed data
PREP_DIR = Path("data/preprocessed")
print("Loading clean data...")

X_train = pd.read_parquet(PREP_DIR / "processed_data" / "X_train_clean.parquet")
y_train = pd.read_parquet(PREP_DIR / "processed_data" / "y_train_clean.parquet")["ctr"]

X_val = pd.read_parquet(PREP_DIR / "processed_data" / "X_val_clean.parquet")
y_val = pd.read_parquet(PREP_DIR / "processed_data" / "y_val_clean.parquet")["ctr"]

X_test = pd.read_parquet(PREP_DIR / "processed_data" / "X_test_clean.parquet")

print(f"Data loaded:")
print(f"- Training: {len(X_train)} samples, {len(X_train.columns)} features")
print(f"- Validation: {len(X_val)} samples")
print(f"- Test: {len(X_test)} samples")

# Separate embedding and traditional features
emb_features = [col for col in X_train.columns if col.startswith("emb_")]
traditional_features = [col for col in X_train.columns if not col.startswith("emb_")]

print(f"Features breakdown:")
print(f"- Embedding features: {len(emb_features)} (will apply PCA)")
print(f"- Traditional features: {len(traditional_features)}")

# ================================
# PCA ANALYSIS AND DIMENSIONALITY REDUCTION
# ================================
print("\n" + "=" * 60)
print("PCA ANALYSIS ON EMBEDDING FEATURES")
print("=" * 60)

# Scale embedding features for PCA
scaler_emb = StandardScaler()
X_train_emb_scaled = scaler_emb.fit_transform(X_train[emb_features])
X_val_emb_scaled = scaler_emb.transform(X_val[emb_features])
X_test_emb_scaled = scaler_emb.transform(X_test[emb_features])

# Fit PCA to see explained variance
pca_full = PCA()
pca_full.fit(X_train_emb_scaled)

# Plot explained variance to choose optimal number of components
cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(12, 5))

# Plot 1: Explained variance by component
plt.subplot(1, 2, 1)
plt.plot(range(1, 51), pca_full.explained_variance_ratio_[:50], "b-", alpha=0.7)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Individual Explained Variance (First 50 Components)")
plt.grid(True, alpha=0.3)

# Plot 2: Cumulative explained variance
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, "r-", linewidth=2)
plt.axhline(y=0.95, color="g", linestyle="--", label="95% Variance")
plt.axhline(y=0.90, color="orange", linestyle="--", label="90% Variance")
plt.axhline(y=0.85, color="purple", linestyle="--", label="85% Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    PREP_DIR / "plots" / "pca_explained_variance.png", dpi=300, bbox_inches="tight"
)
plt.show()


# Find optimal number of components for different variance thresholds
def find_components_for_variance(variance_threshold, cumsum_var):
    return np.argmax(cumsum_var >= variance_threshold) + 1


pca_configs = {
    "95%_variance": find_components_for_variance(0.95, cumsum_variance),
    "90%_variance": find_components_for_variance(0.90, cumsum_variance),
    "85%_variance": find_components_for_variance(0.85, cumsum_variance),
    "fixed_50": 50,
    "fixed_100": 100,
}

print("Optimal PCA components:")
for config_name, n_components in pca_configs.items():
    variance_explained = cumsum_variance[n_components - 1]
    reduction = (1 - n_components / len(emb_features)) * 100
    print(
        f"  {config_name:15s}: {n_components:3d} components ({variance_explained:.3f} variance, {reduction:.1f}% reduction)"
    )

# ================================
# CREATE FEATURE SETS WITH PCA
# ================================
print("\n" + "=" * 60)
print("CREATING FEATURE SETS WITH PCA")
print("=" * 60)

# Scale traditional features
scaler_trad = StandardScaler()
X_train_trad_scaled = scaler_trad.fit_transform(X_train[traditional_features])
X_val_trad_scaled = scaler_trad.transform(X_val[traditional_features])
X_test_trad_scaled = scaler_trad.transform(X_test[traditional_features])

# Create PCA transformers and feature sets
pca_transformers = {}
feature_sets = {}

# Traditional features only
feature_sets["traditional_only"] = {
    "X_train": X_train_trad_scaled,
    "X_val": X_val_trad_scaled,
    "X_test": X_test_trad_scaled,
    "feature_names": traditional_features,
}

# Original embeddings + traditional
X_train_all_scaled = np.hstack([X_train_trad_scaled, X_train_emb_scaled])
X_val_all_scaled = np.hstack([X_val_trad_scaled, X_val_emb_scaled])
X_test_all_scaled = np.hstack([X_test_trad_scaled, X_test_emb_scaled])

feature_sets["all_features_original"] = {
    "X_train": X_train_all_scaled,
    "X_val": X_val_all_scaled,
    "X_test": X_test_all_scaled,
    "feature_names": traditional_features + emb_features,
}

# PCA-reduced embeddings + traditional
for config_name, n_components in pca_configs.items():
    print(f"Creating PCA transformer for {config_name} ({n_components} components)...")

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_emb_scaled)
    X_val_pca = pca.transform(X_val_emb_scaled)
    X_test_pca = pca.transform(X_test_emb_scaled)

    # Combine traditional + PCA features
    X_train_combined = np.hstack([X_train_trad_scaled, X_train_pca])
    X_val_combined = np.hstack([X_val_trad_scaled, X_val_pca])
    X_test_combined = np.hstack([X_test_trad_scaled, X_test_pca])

    pca_feature_names = [f"pca_{i}" for i in range(n_components)]
    combined_feature_names = traditional_features + pca_feature_names

    feature_sets[f"traditional_plus_pca_{config_name}"] = {
        "X_train": X_train_combined,
        "X_val": X_val_combined,
        "X_test": X_test_combined,
        "feature_names": combined_feature_names,
    }

    pca_transformers[config_name] = pca

print(f"Created {len(feature_sets)} feature sets for comparison")

# ================================
# MODEL TRAINING WITH PCA FEATURES
# ================================
print("\n" + "=" * 60)
print("TRAINING MODELS WITH PCA FEATURES")
print("=" * 60)

# Define models to test
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Neural Network": MLPRegressor(
        hidden_layer_sizes=(128, 64), max_iter=500, random_state=42
    ),
}

# Results storage
results = {}

for feature_set_name, feature_data in feature_sets.items():
    print(f"\n🔧 Testing feature set: {feature_set_name.upper()}")
    print(f"   Features: {feature_data['X_train'].shape[1]}")

    results[feature_set_name] = {}

    for model_name, model in models.items():
        print(f"  Training {model_name}...")

        try:
            # Train model
            model.fit(feature_data["X_train"], y_train)
            y_val_pred = model.predict(feature_data["X_val"])

            # Calculate metrics
            mse = mean_squared_error(y_val, y_val_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val, y_val_pred)
            r2 = r2_score(y_val, y_val_pred)

            results[feature_set_name][model_name] = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "model": model,
                "n_features": feature_data["X_train"].shape[1],
            }

            print(
                f"    RMSE: {rmse:.4f}, R²: {r2:.4f}, Features: {feature_data['X_train'].shape[1]}"
            )

        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            results[feature_set_name][model_name] = None

# ================================
# RESULTS ANALYSIS AND VISUALIZATION
# ================================
print("\n" + "=" * 60)
print("PCA PERFORMANCE ANALYSIS")
print("=" * 60)

# Find best model
best_model = None
best_rmse = float("inf")
best_config = None

comparison_data = []

for fs_name, fs_results in results.items():
    print(f"\n📊 {fs_name.upper()}:")
    for model_name, metrics in fs_results.items():
        if metrics is not None:
            print(
                f"  {model_name:20s}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}, Features={metrics['n_features']}"
            )

            comparison_data.append(
                {
                    "Feature Set": fs_name,
                    "Model": model_name,
                    "RMSE": metrics["rmse"],
                    "R²": metrics["r2"],
                    "MAE": metrics["mae"],
                    "N_Features": metrics["n_features"],
                }
            )

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model = metrics["model"]
                best_config = (fs_name, model_name)

# Create comprehensive comparison visualization
df_comparison = pd.DataFrame(comparison_data)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# RMSE vs Number of Features
axes[0, 0].scatter(
    df_comparison["N_Features"],
    df_comparison["RMSE"],
    c=df_comparison["Model"].astype("category").cat.codes,
    alpha=0.7,
)
axes[0, 0].set_xlabel("Number of Features")
axes[0, 0].set_ylabel("RMSE")
axes[0, 0].set_title("Model Performance vs Feature Count")
axes[0, 0].grid(True, alpha=0.3)

# R² vs Number of Features
axes[0, 1].scatter(
    df_comparison["N_Features"],
    df_comparison["R²"],
    c=df_comparison["Model"].astype("category").cat.codes,
    alpha=0.7,
)
axes[0, 1].set_xlabel("Number of Features")
axes[0, 1].set_ylabel("R²")
axes[0, 1].set_title("Model R² vs Feature Count")
axes[0, 1].grid(True, alpha=0.3)

# Best performance by feature set
fs_best = df_comparison.groupby("Feature Set")["RMSE"].min().reset_index()
fs_best = fs_best.sort_values("RMSE")

axes[1, 0].barh(range(len(fs_best)), fs_best["RMSE"])
axes[1, 0].set_yticks(range(len(fs_best)))
axes[1, 0].set_yticklabels(fs_best["Feature Set"], fontsize=8)
axes[1, 0].set_xlabel("Best RMSE")
axes[1, 0].set_title("Best Performance by Feature Set")

# Performance improvement with PCA
pca_results = df_comparison[df_comparison["Feature Set"].str.contains("pca")]
if len(pca_results) > 0:
    avg_by_model = pca_results.groupby("Model")[["RMSE", "N_Features"]].mean()
    axes[1, 1].scatter(avg_by_model["N_Features"], avg_by_model["RMSE"])
    for i, (model, row) in enumerate(avg_by_model.iterrows()):
        axes[1, 1].annotate(
            model,
            (row["N_Features"], row["RMSE"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1, 1].set_xlabel("Average Features (PCA methods)")
    axes[1, 1].set_ylabel("Average RMSE (PCA methods)")
    axes[1, 1].set_title("PCA Methods: Efficiency vs Performance")
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    PREP_DIR / "plots" / "pca_model_comparison.png", dpi=300, bbox_inches="tight"
)
plt.show()

print(f"\n🏆 BEST MODEL WITH PCA:")
print(f"   Configuration: {best_config[1]} with {best_config[0]}")
print(f"   RMSE: {best_rmse:.4f}")
print(f"   Features used: {results[best_config[0]][best_config[1]]['n_features']}")

# PCA insights
print(f"\n🔍 PCA INSIGHTS:")
original_features = len(emb_features) + len(traditional_features)
best_features = results[best_config[0]][best_config[1]]["n_features"]
reduction = ((original_features - best_features) / original_features) * 100

print(f"   Original features: {original_features}")
print(f"   Best model features: {best_features}")
print(f"   Dimensionality reduction: {reduction:.1f}%")

# Save best model and transformers
model_dir = PREP_DIR / "models"
model_dir.mkdir(exist_ok=True)

joblib.dump(best_model, model_dir / "best_ctr_model_pca.pkl")
joblib.dump(scaler_emb, model_dir / "embedding_scaler.pkl")
joblib.dump(scaler_trad, model_dir / "traditional_scaler.pkl")

# Save PCA transformers
for config_name, pca_transformer in pca_transformers.items():
    joblib.dump(pca_transformer, model_dir / f"pca_transformer_{config_name}.pkl")

print(f"\n💾 Models and transformers saved to: {model_dir}/")

print("\n" + "=" * 60)
print("🎯 PCA TRAINING COMPLETE!")
print("=" * 60)
print("Key findings:")
print(f"1. Best performance: {best_config[1]} with {best_config[0]}")
print(f"2. Dimensionality reduction: {reduction:.1f}%")
print(f"3. Performance vs efficiency tradeoff visualized")
print("4. All PCA transformers saved for production use")
print("=" * 60)
