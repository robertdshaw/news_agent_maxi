from model_loader import get_model_paths, check_pipeline_status, load_ctr_model

# Check what the pipeline status function is seeing
print("=== DEBUGGING PIPELINE STATUS ===")

paths = get_model_paths()
print(f"\nCTR model path: {paths['ctr_model']}")
print(f"CTR model exists: {paths['ctr_model'].exists()}")

status = check_pipeline_status()
print(f"\nPipeline status: {status}")
print(f"Training completed: {status['training_completed']}")

# Test the model loading directly
print("\n=== TESTING MODEL LOADING ===")
model_data, error = load_ctr_model(paths["ctr_model"])
if error:
    print(f"ERROR: {error}")
else:
    print("SUCCESS: Model loaded without errors")
    print(f"Model type: {type(model_data['model'])}")
    print(f"Number of features: {len(model_data['feature_names'])}")
