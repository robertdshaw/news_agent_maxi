import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Paths
PREP_DIR = Path("data/preprocessed/processed_data")


def quick_leakage_check():
    """
    Quick checks for data leakage
    """
    print("🔍 QUICK DATA LEAKAGE CHECKER")
    print("=" * 50)

    # Load data
    print("📂 Loading data...")
    X_train = pd.read_parquet(PREP_DIR / "X_train.parquet")
    y_train = pd.read_parquet(PREP_DIR / "y_train.parquet")["ctr"]
    X_val = pd.read_parquet(PREP_DIR / "X_val.parquet")
    y_val = pd.read_parquet(PREP_DIR / "y_val.parquet")["ctr"]

    # Load flags
    try:
        train_flags = pd.read_parquet(PREP_DIR / "train_flags.parquet")["ctr_is_real"]
        val_flags = pd.read_parquet(PREP_DIR / "val_flags.parquet")["ctr_is_real"]

        X_train = X_train[train_flags]
        y_train = y_train[train_flags]
        X_val = X_val[val_flags]
        y_val = y_val[val_flags]
        print("✅ Applied CTR validity filtering")
    except:
        print("⚠️ No CTR flags found")

    print(f"📊 Data shapes: Train {X_train.shape}, Val {X_val.shape}")
    print()

    # CHECK 1: Index overlap
    print("🔍 CHECK 1: Index Overlap")
    print("-" * 30)
    train_indices = set(X_train.index)
    val_indices = set(X_val.index)
    overlap = train_indices.intersection(val_indices)

    print(f"Train indices: {len(train_indices):,}")
    print(f"Val indices: {len(val_indices):,}")
    print(f"Overlapping indices: {len(overlap):,}")

    if len(overlap) > 0:
        print("🚨 LEAKAGE DETECTED: Same articles in train and val!")
        print(f"First few overlapping indices: {list(overlap)[:10]}")
    else:
        print("✅ No index overlap")
    print()

    # CHECK 2: Identical rows
    print("🔍 CHECK 2: Identical Feature Vectors")
    print("-" * 30)

    # Sample check (full check would be too slow)
    sample_size = min(1000, len(X_train), len(X_val))
    train_sample = X_train.sample(sample_size, random_state=42)
    val_sample = X_val.sample(sample_size, random_state=42)

    # Check for identical rows
    identical_count = 0
    for i, (_, train_row) in enumerate(train_sample.iterrows()):
        if i % 100 == 0:
            print(f"  Checking row {i}/{sample_size}...", end="\r")

        # Check if this exact row exists in validation
        matches = (val_sample == train_row).all(axis=1).sum()
        if matches > 0:
            identical_count += 1

    print(f"\nIdentical feature vectors found: {identical_count}/{sample_size}")
    if identical_count > 10:
        print("🚨 POTENTIAL LEAKAGE: Many identical feature vectors!")
    else:
        print("✅ No significant feature duplication")
    print()

    # CHECK 3: Target distribution analysis
    print("🔍 CHECK 3: Target Distribution Differences")
    print("-" * 30)

    y_train_binary = (y_train > 0).astype(int)
    y_val_binary = (y_val > 0).astype(int)

    train_stats = {
        "mean_ctr": y_train.mean(),
        "click_rate": y_train_binary.mean(),
        "std_ctr": y_train.std(),
        "median_ctr": y_train.median(),
        "max_ctr": y_train.max(),
    }

    val_stats = {
        "mean_ctr": y_val.mean(),
        "click_rate": y_val_binary.mean(),
        "std_ctr": y_val.std(),
        "median_ctr": y_val.median(),
        "max_ctr": y_val.max(),
    }

    print("Train set stats:")
    for key, value in train_stats.items():
        print(f"  {key}: {value:.4f}")

    print("\nVal set stats:")
    for key, value in val_stats.items():
        print(f"  {key}: {value:.4f}")

    # Calculate differences
    click_rate_diff = abs(train_stats["click_rate"] - val_stats["click_rate"])
    print(f"\nClick rate difference: {click_rate_diff:.3f}")

    if click_rate_diff > 0.05:  # More than 5% difference
        print(
            "🚨 CONCERNING: Large click rate difference suggests different data distributions"
        )
    else:
        print("✅ Click rates reasonably similar")
    print()

    # CHECK 4: Feature value ranges
    print("🔍 CHECK 4: Feature Value Range Comparison")
    print("-" * 30)

    # Check if train and val have similar feature ranges
    suspicious_features = []

    for col in X_train.columns[:20]:  # Check first 20 features
        train_min, train_max = X_train[col].min(), X_train[col].max()
        val_min, val_max = X_val[col].min(), X_val[col].max()

        # Check if validation has values completely outside training range
        if val_min < train_min or val_max > train_max:
            range_diff = max(abs(val_min - train_min), abs(val_max - train_max))
            if range_diff > 0.1:  # Significant difference
                suspicious_features.append((col, range_diff))

    if suspicious_features:
        print(f"Features with suspicious value ranges: {len(suspicious_features)}")
        for feat, diff in suspicious_features[:5]:
            print(f"  {feat}: range difference {diff:.3f}")
        if len(suspicious_features) > 10:
            print("🚨 Many features have different ranges - possible data shift")
    else:
        print("✅ Feature ranges look consistent")
    print()

    # CHECK 5: Temporal features analysis
    print("🔍 CHECK 5: Temporal Features Check")
    print("-" * 30)

    temporal_cols = [
        col
        for col in X_train.columns
        if any(x in col for x in ["hour", "weekend", "time_period", "temporal"])
    ]

    if temporal_cols:
        print(f"Found {len(temporal_cols)} temporal features: {temporal_cols}")

        for col in temporal_cols:
            train_unique = X_train[col].nunique()
            val_unique = X_val[col].nunique()

            # Check if temporal features are all zeros (suspicious)
            train_zero_rate = (X_train[col] == 0).mean()
            val_zero_rate = (X_val[col] == 0).mean()

            print(f"  {col}:")
            print(
                f"    Train: {train_unique} unique values, {train_zero_rate:.1%} zeros"
            )
            print(f"    Val: {val_unique} unique values, {val_zero_rate:.1%} zeros")

            if train_zero_rate > 0.95 or val_zero_rate > 0.95:
                print(f"    🚨 {col} is mostly zeros - may be corrupted")
    else:
        print("No temporal features found")
    print()

    # CHECK 6: Quick model performance sanity check
    print("🔍 CHECK 6: Performance Sanity Check")
    print("-" * 30)

    from sklearn.metrics import roc_auc_score
    from sklearn.dummy import DummyClassifier

    # Baseline models
    dummy_most_frequent = DummyClassifier(strategy="most_frequent")
    dummy_stratified = DummyClassifier(strategy="stratified", random_state=42)

    dummy_most_frequent.fit(X_train, y_train_binary)
    dummy_stratified.fit(X_train, y_train_binary)

    # Predict on validation
    dummy_mf_pred = dummy_most_frequent.predict_proba(X_val)[:, 1]
    dummy_strat_pred = dummy_stratified.predict_proba(X_val)[:, 1]

    try:
        dummy_mf_auc = roc_auc_score(y_val_binary, dummy_mf_pred)
    except:
        dummy_mf_auc = 0.5

    try:
        dummy_strat_auc = roc_auc_score(y_val_binary, dummy_strat_pred)
    except:
        dummy_strat_auc = 0.5

    print(f"Dummy (most frequent) AUC: {dummy_mf_auc:.4f}")
    print(f"Dummy (stratified) AUC: {dummy_strat_auc:.4f}")
    print(f"Your model AUC: 0.7928")

    improvement = 0.7928 - max(dummy_mf_auc, dummy_strat_auc)
    print(f"Improvement over baseline: {improvement:.4f}")

    if improvement > 0.4:
        print("🚨 SUSPICIOUS: Improvement seems too large")
    elif improvement > 0.2:
        print("✅ Good improvement, plausible")
    else:
        print("⚠️ Small improvement")
    print()

    # SUMMARY
    print("📋 LEAKAGE CHECK SUMMARY")
    print("=" * 30)

    leakage_score = 0
    warnings_list = []

    if len(overlap) > 0:
        leakage_score += 3
        warnings_list.append("Index overlap detected")

    if identical_count > 10:
        leakage_score += 2
        warnings_list.append("Many identical feature vectors")

    if click_rate_diff > 0.05:
        leakage_score += 2
        warnings_list.append("Large click rate difference")

    if improvement > 0.4:
        leakage_score += 2
        warnings_list.append("Suspiciously high performance")

    if len(suspicious_features) > 10:
        leakage_score += 1
        warnings_list.append("Feature range inconsistencies")

    print(f"Leakage Risk Score: {leakage_score}/10")

    if leakage_score >= 5:
        print("🚨 HIGH RISK: Multiple leakage indicators detected!")
    elif leakage_score >= 3:
        print("⚠️ MEDIUM RISK: Some concerning signs")
    elif leakage_score >= 1:
        print("⚠️ LOW RISK: Minor issues detected")
    else:
        print("✅ LOW RISK: No major leakage indicators")

    if warnings_list:
        print("\nWarnings:")
        for warning in warnings_list:
            print(f"  - {warning}")

    print(f"\n💡 RECOMMENDATION:")
    if leakage_score >= 5:
        print("   Investigate data splits and feature engineering process")
        print("   Consider re-doing train/val split")
        print("   True performance likely ~0.60-0.65 AUC")
    elif leakage_score >= 3:
        print("   Monitor model performance on new data carefully")
        print("   Consider more conservative performance estimates")
    else:
        print("   Model performance appears legitimate")
        print("   Proceed with deployment but monitor results")


if __name__ == "__main__":
    quick_leakage_check()
