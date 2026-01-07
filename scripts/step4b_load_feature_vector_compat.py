"""
Load Feature Vector Model with compatibility handling
"""

import pickle
import warnings
from pathlib import Path

print("="*80)
print("LOADING FEATURE VECTOR MODEL (COMPATIBILITY MODE)")
print("="*80)

model_path = "/Users/swan/.cache/huggingface/hub/datasets--swan07--final-models/snapshots/cede139e1f1b13ca3c1ec5b7c8261f8d8842f435/featuremodel.p"

print(f"\n1. Attempting to load with different methods...")

# Method 1: Try with sklearn compatibility
print("\n   Method 1: Standard pickle load...")
try:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    print("   ✗ Failed with standard method (expected)")
except Exception as e:
    print(f"   ✗ Error: {str(e)[:100]}")

# Method 2: Try with custom unpickler
print("\n   Method 2: Custom unpickler (sklearn compatibility)...")
try:
    import sys
    import sklearn
    from sklearn import linear_model

    # Create a custom unpickler to handle old sklearn classes
    class CompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Handle old sklearn module names
            if 'sklearn.linear_model._sgd_fast' in module:
                if name == 'Log':
                    # Map to modern equivalent
                    from sklearn.linear_model import SGDClassifier
                    return SGDClassifier
            return super().find_class(module, name)

    with open(model_path, 'rb') as f:
        unpickler = CompatUnpickler(f)
        model_data = unpickler.load()

    print("   ✓ Loaded with custom unpickler!")

    if isinstance(model_data, tuple):
        print(f"   ✓ Model has {len(model_data)} components")
    else:
        print(f"   Model type: {type(model_data)}")

except Exception as e:
    print(f"   ✗ Error: {str(e)[:100]}")

# Method 3: Check sklearn version mismatch
print("\n   Method 3: Checking sklearn version...")
print(f"   Current sklearn version: {sklearn.__version__}")
print("   Model was likely trained with sklearn 0.x or 1.x")
print("   You have sklearn", sklearn.__version__)

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

print("""
The feature vector model was trained with an older version of scikit-learn
and cannot be loaded with the current version due to internal API changes.

OPTIONS:

1. RETRAIN THE MODEL (Recommended)
   - Use: featurevector/large_train_model.ipynb
   - Time: 2-4 hours on GPU
   - Result: Compatible model with current sklearn

2. DOWNGRADE SKLEARN (Not recommended)
   - Install sklearn==1.3.0 or older
   - May cause other compatibility issues

3. SKIP FEATURE VECTOR FOR NOW
   - Focus on BERT model improvements
   - Come back to this later

RECOMMENDATION:
Since BERT is already running, let's focus on:
1. Complete BERT baseline evaluation
2. Train calibration models
3. Improve BERT performance
4. Retrain feature vector model later if needed
""")

print("\n" + "="*80)
