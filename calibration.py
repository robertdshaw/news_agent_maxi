from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Load model and validation set predictions
with open("model_output/ai_news_editor_model.pkl", "rb") as f:
    model_package = pickle.load(f)

# Load the original validation set
X_val = pd.read_parquet("data/preprocessed/processed_data/X_val_clean.parquet")
y_val = pd.read_parquet("data/preprocessed/processed_data/y_val_clean.parquet")["ctr"]

# Align features with model (drop any extra columns to avoid mismatch)
model_features = model_package["feature_names"]
X_val = X_val[model_features].fillna(0)

# Create binary label for high engagement
threshold = model_package["ctr_threshold"]
y_val_binary = (y_val > threshold).astype(int)


# Predict probabilities (if not already saved)
probs = model_package["model"].predict_proba(X_val.fillna(0))[:, 1]

# Reliability curve
prob_true, prob_pred = calibration_curve(y_val_binary, probs, n_bins=10)

# Plot
plt.plot(prob_pred, prob_true, marker="o", label="Model")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("Predicted Probability")
plt.ylabel("True Frequency")
plt.title("Reliability Diagram")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("model_output/calibration_plot.png")

# Print Brier score
brier = brier_score_loss(y_val_binary, probs)
print(f"Brier Score: {brier:.4f}")
