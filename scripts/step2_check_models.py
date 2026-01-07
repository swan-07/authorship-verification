"""
Phase 1, Step 3: Check for Trained Models
Attempts to locate or download the trained models.
"""

import os
from pathlib import Path

print("="*80)
print("CHECKING FOR TRAINED MODELS")
print("="*80)

repo_root = Path("/Users/swan/Documents/GitHub/authorship-verification")

models_found = {}

# 1. Check for Feature Vector Model
print("\n1. Feature Vector Model (large_model.p)")
fv_paths = [
    repo_root / "featurevector" / "large_model.p",
    repo_root / "large_model.p",
    repo_root / "featurevector" / "temp_data" / "large_model.p",
]

found = False
for path in fv_paths:
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"   ✓ Found at: {path}")
        print(f"     Size: {size_mb:.2f} MB")
        models_found['feature_vector'] = str(path)
        found = True
        break

if not found:
    print("   ✗ Not found locally")
    print("   → You'll need to retrain this model using large_train_model.ipynb")
    print("      OR recover it from your RunPod instance if it still exists")

# 2. Check for Calibration Models
print("\n2. Calibration Models (*.pkl)")
calib_paths = [
    repo_root / "siamesebert" / "methods" / "calibration_model.pkl",
    repo_root / "siamesebert" / "methods" / "calibration_model1.pkl",
    repo_root / "calibration_model.pkl",
    repo_root / "calibration_model1.pkl",
]

for path in calib_paths:
    if path.exists():
        size_kb = path.stat().st_size / 1024
        print(f"   ✓ Found: {path.name} ({size_kb:.2f} KB)")
        models_found[path.name] = str(path)

if not any(p.exists() for p in calib_paths):
    print("   ✗ Not found locally")
    print("   → These are quick to retrain using logreg.ipynb")

# 3. Check for BERT Model (local)
print("\n3. BERT Model (local directory)")
bert_paths = [
    repo_root / "siamesebert" / "methods" / "bertmodel",
    repo_root / "bertmodel",
]

found = False
for path in bert_paths:
    if path.exists() and path.is_dir():
        files = list(path.glob("*"))
        print(f"   ✓ Found at: {path}")
        print(f"     Files: {len(files)}")
        print(f"     Contents: {[f.name for f in files[:5]]}")
        models_found['bert_local'] = str(path)
        found = True
        break

if not found:
    print("   ✗ Not found locally")

# 4. Try to access BERT model from HuggingFace
print("\n4. BERT Model (HuggingFace: swan07/final-models)")

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    api = HfApi()

    print("   Checking HuggingFace repository...")

    # Try to list files in the repo
    try:
        files = api.list_repo_files(
            repo_id="swan07/final-models",
            repo_type="dataset"
        )
        print(f"   ✓ Repository exists!")
        print(f"     Total files: {len(files)}")

        # Check for bertmodel directory
        bert_files = [f for f in files if 'bertmodel' in f]
        if bert_files:
            print(f"     BERT model files found: {len(bert_files)}")
            print(f"     Sample: {bert_files[:3]}")
            models_found['bert_hf'] = "swan07/final-models"
        else:
            print("     ⚠ No bertmodel directory found in repo")

    except Exception as e:
        print(f"   ✗ Cannot access repository: {e}")
        print("   → Repository might be private or deleted")
        print("   → You may need to retrain the BERT model")

except ImportError:
    print("   ✗ huggingface_hub not installed")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if models_found:
    print("\n✓ Found models:")
    for name, path in models_found.items():
        print(f"   - {name}: {path}")
else:
    print("\n✗ No trained models found")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

if 'bert_hf' in models_found or 'bert_local' in models_found:
    print("\n✓ Can proceed with BERT evaluation")
    print("  Next: Run step3_test_bert_model.py")
else:
    print("\n⚠ No BERT model available")
    print("  Options:")
    print("    1. Download from HuggingFace (if available)")
    print("    2. Use base BERT model (bert-base-cased) without fine-tuning")
    print("    3. Retrain using siamesebert/methods/bert.ipynb")

if 'feature_vector' in models_found:
    print("\n✓ Can proceed with Feature Vector evaluation")
else:
    print("\n⚠ No Feature Vector model available")
    print("  Must retrain using featurevector/large_train_model.ipynb")
    print("  (This takes several hours on GPU)")

if not any('calibration' in k for k in models_found.keys()):
    print("\n⚠ No calibration models available")
    print("  These are needed to fix BERT threshold issue")
    print("  Can quickly train using logreg.ipynb (~10 minutes)")
