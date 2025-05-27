import os
from pathlib import Path

# Check where plots should be
PREP_DIR = Path("data/preprocessed")
plots_dir = PREP_DIR / "plots"

print(f"Looking for plots in: {plots_dir.absolute()}")
print(f"Directory exists: {plots_dir.exists()}")

if plots_dir.exists():
    plot_files = list(plots_dir.glob("*.png"))
    print(f"Found {len(plot_files)} PNG files:")
    for plot_file in plot_files:
        size_kb = plot_file.stat().st_size / 1024
        print(f"  {plot_file.name} ({size_kb:.1f} KB)")
else:
    print("Plots directory doesn't exist - creating it...")
    plots_dir.mkdir(parents=True, exist_ok=True)

# Check current working directory
print(f"\nCurrent working directory: {Path.cwd()}")

# Check if plots might be in current directory
current_plots = list(Path.cwd().glob("*.png"))
if current_plots:
    print(f"Found {len(current_plots)} PNG files in current directory:")
    for plot_file in current_plots:
        print(f"  {plot_file.name}")

# Check common plot locations
common_locations = [
    Path("plots"),
    Path("data/plots"),
    Path("figures"),
    Path("output"),
    Path("."),
]

print("\nChecking common plot locations:")
for loc in common_locations:
    if loc.exists():
        png_files = list(loc.glob("*.png"))
        if png_files:
            print(f"  {loc.absolute()}: {len(png_files)} PNG files")
            for f in png_files:
                print(f"    {f.name}")
