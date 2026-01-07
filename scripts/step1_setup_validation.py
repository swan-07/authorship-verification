"""
Phase 1, Step 1-2: Environment Setup & Dataset Validation
Run this to verify everything is set up correctly.
"""

import sys
print("Checking dependencies...")

# Check imports
try:
    import torch
    import transformers
    import sentence_transformers
    import datasets
    import sklearn
    import pandas as pd
    import numpy as np
    print("✓ All required packages installed")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - Transformers: {transformers.__version__}")
    print(f"  - Sentence-Transformers: {sentence_transformers.__version__}")
    print(f"  - Datasets: {datasets.__version__}")
except ImportError as e:
    print(f"✗ Missing package: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

print("\n" + "="*80)
print("DOWNLOADING SAMPLE DATASET")
print("="*80)

from datasets import load_dataset

# Start with just one small test set
print("\nDownloading arxiv test set (smallest dataset)...")
try:
    arxiv_test = load_dataset(
        'swan07/authorship-verification',
        data_files={'test': 'arxiv_test.csv'},
        split='test'
    )
    print(f"✓ Downloaded {len(arxiv_test)} samples")

    # Show sample
    print("\nSample data:")
    sample = arxiv_test[0]
    print(f"  Columns: {list(sample.keys())}")
    print(f"  Text 1 length: {len(sample['text1'])} chars")
    print(f"  Text 2 length: {len(sample['text2'])} chars")

    # Determine label column
    label_col = 'same' if 'same' in sample else 'score'
    print(f"  Label ({label_col}): {sample[label_col]}")

    print(f"\n  Text 1 preview: {sample['text1'][:200]}...")
    print(f"\n  Text 2 preview: {sample['text2'][:200]}...")

    # Class distribution
    from collections import Counter
    labels = arxiv_test[label_col]
    counts = Counter(labels)
    print(f"\nClass distribution:")
    print(f"  Same author (1): {counts.get(1, 0)} ({counts.get(1, 0)/len(labels)*100:.1f}%)")
    print(f"  Different author (0): {counts.get(0, 0)} ({counts.get(0, 0)/len(labels)*100:.1f}%)")

except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✓ SETUP VALIDATION COMPLETE!")
print("="*80)

print("\nNext: Check for trained models")
print("  - BERT model should be at: swan07/final-models on HuggingFace")
print("  - Feature vector model: large_model.p (need to locate)")
print("  - Calibration models: calibration_model.pkl, calibration_model1.pkl")
