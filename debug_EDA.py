# Quick debug script to see where EDA is stuck
from pathlib import Path
import os

PREP_DIR = Path("data/preprocessed")
cache_dir = PREP_DIR / "cache"

print("Checking EDA progress...")
print("=" * 40)

# Check if directories exist
print(f"PREP_DIR exists: {PREP_DIR.exists()}")
print(f"Cache dir exists: {cache_dir.exists()}")

if cache_dir.exists():
    cache_files = list(cache_dir.glob("*.pkl"))
    print(f"Cache files found: {len(cache_files)}")
    for f in cache_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")

# Check processed data
processed_dir = PREP_DIR / "processed_data"
if processed_dir.exists():
    parquet_files = list(processed_dir.glob("*.parquet"))
    print(f"Parquet files found: {len(parquet_files)}")
    for f in parquet_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")

# Check plots
plots_dir = PREP_DIR / "plots"
if plots_dir.exists():
    plot_files = list(plots_dir.glob("*.png"))
    print(f"Plot files found: {len(plot_files)}")
    for f in plot_files:
        print(f"  {f.name}")

print(
    "\nIf embedding files exist in cache, the script might be working on plots or saving data."
)
print("If no cache files exist, it's likely stuck on embedding generation.")
