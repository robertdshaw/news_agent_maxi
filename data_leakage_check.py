# Run this code to check for data leakage in your features

import pandas as pd
import numpy as np

# 1. EXAMINE YOUR RAW DATA
print("=== CHECKING FOR DATA LEAKAGE ===\n")

# Load your original data before preprocessing
# Replace with your actual data loading path
try:
    # Check your raw data files
    train_raw = pd.read_parquet("data/preprocessed/processed_data/X_train.parquet")

    print("1. SUSPICIOUS FEATURES ANALYSIS:")
    print("-" * 40)

    # Check the suspicious features
    suspicious_features = ["total_users", "total_impressions"]

    for feature in suspicious_features:
        if feature in train_raw.columns:
            print(f"\n{feature.upper()}:")
            print(f"  Min value: {train_raw[feature].min()}")
            print(f"  Max value: {train_raw[feature].max()}")
            print(f"  Mean: {train_raw[feature].mean():.2f}")
            print(f"  Unique values: {train_raw[feature].nunique()}")
            print(f"  Sample values: {train_raw[feature].head(10).tolist()}")

            # Check if values correlate suspiciously with target
            if "ctr" in train_raw.columns:
                correlation = train_raw[feature].corr(train_raw["ctr"])
                print(f"  Correlation with CTR: {correlation:.4f}")
                if abs(correlation) > 0.5:
                    print(f"  ⚠️  HIGH CORRELATION - POTENTIAL LEAKAGE!")
        else:
            print(f"{feature} not found in data")

except Exception as e:
    print(f"Error loading data: {e}")
    print("Please check your data file paths")

print("\n" + "=" * 50)

# 2. TEMPORAL ANALYSIS
print("2. TEMPORAL LEAKAGE CHECK:")
print("-" * 30)

# This helps identify if features contain future information
try:
    # Check if your data has timestamp information
    if "timestamp" in train_raw.columns or "date" in train_raw.columns:
        timestamp_col = "timestamp" if "timestamp" in train_raw.columns else "date"
        print(f"Found timestamp column: {timestamp_col}")

        # Check date range
        print(
            f"Date range: {train_raw[timestamp_col].min()} to {train_raw[timestamp_col].max()}"
        )

        # Check if features change over time (they shouldn't if calculated pre-publication)
        for feature in suspicious_features:
            if feature in train_raw.columns:
                # Group by date and see if feature values vary significantly
                daily_stats = train_raw.groupby(train_raw[timestamp_col].dt.date)[
                    feature
                ].agg(["mean", "std", "count"])
                print(f"\n{feature} daily variation:")
                print(f"  Mean std deviation: {daily_stats['std'].mean():.4f}")
                if daily_stats["std"].mean() > daily_stats["mean"].mean() * 0.1:
                    print(f"  ⚠️  HIGH DAILY VARIATION - CHECK IF PRE-PUBLICATION!")

except Exception as e:
    print(f"Timestamp analysis failed: {e}")

print("\n" + "=" * 50)

# 3. FEATURE LOGIC CHECK
print("3. FEATURE DEFINITION QUESTIONS:")
print("-" * 35)

questions = [
    "total_users - Is this:",
    "  a) Total users who saw THIS article (LEAKAGE)",
    "  b) Historical total users on the site (OK)",
    "  c) Expected users for this time slot (OK)",
    "",
    "total_impressions - Is this:",
    "  a) Actual impressions THIS article received (LEAKAGE)",
    "  b) Historical impressions for similar articles (OK)",
    "  c) Expected impressions based on publish time (OK)",
    "",
    "Key Questions:",
    "- When are these features calculated?",
    "- What time period do they represent?",
    "- Would an editor know this BEFORE publishing?",
]

for q in questions:
    print(q)

print("\n" + "=" * 50)

# 4. CROSS-VALIDATION LEAKAGE CHECK
print("4. CV LEAKAGE INDICATORS:")
print("-" * 28)

print("Signs of leakage in your results:")
print(f"✓ Training AUC: 0.9647 (very high)")
print(f"✓ Validation AUC: 0.8103 (still high)")
print(f"✓ Difference: {0.9647 - 0.8103:.4f}")

if 0.9647 - 0.8103 > 0.1:
    print("⚠️  Large train/val gap suggests possible overfitting or leakage")
else:
    print("✓ Reasonable train/val gap")

print(f"\nCategory performance ranges from 0.809 to 0.896 AUC")
print("⚠️  These are suspiciously high - typical CTR prediction is 0.65-0.75")

print("\n" + "=" * 50)

# 5. MANUAL INSPECTION
print("5. MANUAL DATA INSPECTION:")
print("-" * 30)

try:
    # Show some actual examples
    sample_rows = train_raw[["total_users", "total_impressions"]].head(20)
    print("Sample of suspicious features:")
    print(sample_rows)

    print("\nLook for patterns like:")
    print("- Are values always the same? (Might be OK)")
    print("- Do they vary dramatically? (Check timing)")
    print("- Do they correlate perfectly with target? (LEAKAGE)")

except Exception as e:
    print(f"Could not show sample data: {e}")
