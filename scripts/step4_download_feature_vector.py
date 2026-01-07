"""
Download and validate Feature Vector Model
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import pickle

print("="*80)
print("DOWNLOADING FEATURE VECTOR MODEL")
print("="*80)

# Download from HuggingFace
print("\n1. Downloading from HuggingFace...")
print("   Repository: swan07/final-models")
print("   File: featuremodel.p")

try:
    model_path = hf_hub_download(
        repo_id="swan07/final-models",
        filename="featuremodel.p",
        repo_type="dataset"
    )

    print(f"   ✓ Downloaded to: {model_path}")

    # Check file size
    size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")

except Exception as e:
    print(f"   ✗ Error downloading: {e}")
    exit(1)

# Try to load and validate
print("\n2. Validating model file...")

try:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # Check what's in the pickle file
    if isinstance(model_data, tuple):
        print(f"   ✓ Model loaded (tuple with {len(model_data)} components)")

        # Expected: (clf, transformer, scaler, secondary_scaler)
        if len(model_data) == 4:
            clf, transformer, scaler, secondary_scaler = model_data
            print(f"   Components:")
            print(f"     - Classifier: {type(clf).__name__}")
            print(f"     - Transformer: {type(transformer).__name__}")
            print(f"     - Scaler: {type(scaler).__name__}")
            print(f"     - Secondary Scaler: {type(secondary_scaler).__name__}")

            # Check if classifier has expected attributes
            if hasattr(clf, 'coef_'):
                n_features = clf.coef_.shape[1]
                print(f"     - Number of features: {n_features:,}")

            if hasattr(clf, 'classes_'):
                print(f"     - Classes: {clf.classes_}")
        else:
            print(f"   ⚠ Unexpected number of components: {len(model_data)}")
            print(f"   Expected 4 (clf, transformer, scaler, secondary_scaler)")
    else:
        print(f"   ⚠ Unexpected model type: {type(model_data)}")

    print("   ✓ Model validation complete!")

except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    print(f"   The file may be corrupted or in an incompatible format")
    exit(1)

# Copy to local directory for easier access
print("\n3. Copying to local directory...")

local_path = Path("/Users/swan/Documents/GitHub/authorship-verification/featurevector/featuremodel.p")

try:
    import shutil
    shutil.copy(model_path, local_path)
    print(f"   ✓ Copied to: {local_path}")

except Exception as e:
    print(f"   ⚠ Could not copy: {e}")
    print(f"   Model is still accessible at: {model_path}")

print("\n" + "="*80)
print("✓ FEATURE VECTOR MODEL READY")
print("="*80)

print(f"\nModel location:")
print(f"  - Cache: {model_path}")
if local_path.exists():
    print(f"  - Local: {local_path}")

print(f"\nNext steps:")
print(f"  1. Test feature vector model on sample data")
print(f"  2. Compare performance with BERT model")
print(f"  3. Create ensemble combining both models")
