# Quick script to update your model.py to use clean data

import fileinput
import sys

print("Updating model.py to use clean data...")

# Read the current model.py with UTF-8 encoding
with open("model.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace the data loading lines
old_lines = [
    'X_train = pd.read_parquet(PREP_DIR / "processed_data" / "X_train.parquet")',
    'y_train = pd.read_parquet(PREP_DIR / "processed_data" / "y_train.parquet")["ctr"]',
    'X_val = pd.read_parquet(PREP_DIR / "processed_data" / "X_val.parquet")',
    'y_val = pd.read_parquet(PREP_DIR / "processed_data" / "y_val.parquet")["ctr"]',
    'X_test = pd.read_parquet(PREP_DIR / "processed_data" / "X_test.parquet")',
    'y_test = pd.read_parquet(PREP_DIR / "processed_data" / "y_test.parquet")["ctr"]',
]

new_lines = [
    'X_train = pd.read_parquet(PREP_DIR / "processed_data" / "X_train_clean.parquet")',
    'y_train = pd.read_parquet(PREP_DIR / "processed_data" / "y_train_clean.parquet")["ctr"]',
    'X_val = pd.read_parquet(PREP_DIR / "processed_data" / "X_val_clean.parquet")',
    'y_val = pd.read_parquet(PREP_DIR / "processed_data" / "y_val_clean.parquet")["ctr"]',
    'X_test = pd.read_parquet(PREP_DIR / "processed_data" / "X_test_clean.parquet")',
    'y_test = pd.read_parquet(PREP_DIR / "processed_data" / "y_test_clean.parquet")["ctr"]',
]

# Replace each line
for old_line, new_line in zip(old_lines, new_lines):
    content = content.replace(old_line, new_line)

# Update the header comment
content = content.replace(
    "AI NEWS EDITOR ASSISTANT - TIME-AWARE CTR PREDICTION MODEL",
    "AI NEWS EDITOR ASSISTANT - CLEAN TIME-AWARE CTR PREDICTION MODEL (NO LEAKAGE)",
)

# Write back to model.py with UTF-8 encoding
with open("model_clean.py", "w", encoding="utf-8") as f:
    f.write(content)

print("✅ Created model_clean.py with leak-free data loading")
print("✅ Now run: python model_clean.py")
