"""
Step 1: Validate Dataset and Environment
This script downloads and validates the authorship verification dataset.
"""

import os
from datasets import load_dataset
import pandas as pd
from collections import Counter

print("="*80)
print("STEP 1: VALIDATING DATASET AND ENVIRONMENT")
print("="*80)

# Define test data files (from your README)
test_data_files = {
    'arxiv': 'arxiv_test.csv',
    'blogs': 'blogs_test.csv',
    'british': 'british_test.csv',
    'darkreddit': 'darkreddit_test.csv',
    'imdb': 'imdb_test.csv',
    'pan11': 'pan11_test.csv',
    'pan13': 'pan13_test.csv',
    'pan14': 'pan14_test.csv',
    'pan15': 'pan15_test.csv',
    'pan20': 'pan20_test.csv',
    'reuters': 'reuters_test.csv',
    'victorian': 'victorian_test.csv'
}

print("\n1. Downloading datasets from HuggingFace...")
print("   Repository: swan07/authorship-verification")

try:
    # Download train and validation sets
    print("\n   Loading training data...")
    train_dataset = load_dataset(
        "swan07/authorship-verification",
        data_files="*_train.csv",
        split='train'
    )

    print("   Loading validation data...")
    val_dataset = load_dataset(
        "swan07/authorship-verification",
        data_files="*_val.csv",
        split='train'
    )

    print("\n   Loading test datasets...")
    test_datasets = {}
    for name, file in test_data_files.items():
        print(f"   - {name}: {file}")
        test_datasets[name] = load_dataset(
            'swan07/authorship-verification',
            data_files={"test": file},
            split='test'
        )

    print("\n✓ All datasets downloaded successfully!")

except Exception as e:
    print(f"\n✗ Error downloading datasets: {e}")
    exit(1)

print("\n" + "="*80)
print("2. DATASET STATISTICS")
print("="*80)

# Training set stats
print(f"\nTraining Set:")
print(f"  Total samples: {len(train_dataset):,}")
print(f"  Columns: {train_dataset.column_names}")

if 'same' in train_dataset.column_names or 'score' in train_dataset.column_names:
    label_col = 'same' if 'same' in train_dataset.column_names else 'score'
    labels = train_dataset[label_col]
    label_counts = Counter(labels)
    print(f"  Class distribution:")
    print(f"    Same author (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(labels)*100:.1f}%)")
    print(f"    Different author (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(labels)*100:.1f}%)")

# Validation set stats
print(f"\nValidation Set:")
print(f"  Total samples: {len(val_dataset):,}")

if 'same' in val_dataset.column_names or 'score' in val_dataset.column_names:
    label_col = 'same' if 'same' in val_dataset.column_names else 'score'
    labels = val_dataset[label_col]
    label_counts = Counter(labels)
    print(f"  Class distribution:")
    print(f"    Same author (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(labels)*100:.1f}%)")
    print(f"    Different author (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(labels)*100:.1f}%)")

# Test sets stats
print(f"\nTest Sets:")
for name, dataset in test_datasets.items():
    print(f"\n  {name}:")
    print(f"    Samples: {len(dataset):,}")

    if 'same' in dataset.column_names or 'score' in dataset.column_names:
        label_col = 'same' if 'same' in dataset.column_names else 'score'
        labels = dataset[label_col]
        label_counts = Counter(labels)
        print(f"    Same author: {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(labels)*100:.1f}%)")
        print(f"    Different author: {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(labels)*100:.1f}%)")

print("\n" + "="*80)
print("3. DATA QUALITY CHECKS")
print("="*80)

# Sample a few examples
print("\nSample from training set:")
sample = train_dataset[0]
print(f"  Text 1 (first 100 chars): {sample['text1'][:100]}...")
print(f"  Text 2 (first 100 chars): {sample['text2'][:100]}...")
label_col = 'same' if 'same' in sample else 'score'
print(f"  Label ({label_col}): {sample[label_col]}")

# Check for missing values
print("\nChecking for missing values...")
missing_found = False
for col in train_dataset.column_names:
    missing = sum(1 for x in train_dataset[col] if x is None or x == '')
    if missing > 0:
        print(f"  ✗ {col}: {missing} missing values")
        missing_found = True

if not missing_found:
    print("  ✓ No missing values found")

# Text length statistics
print("\nText length statistics (training set):")
text1_lengths = [len(str(text)) for text in train_dataset['text1'][:1000]]  # Sample 1000
text2_lengths = [len(str(text)) for text in train_dataset['text2'][:1000]]

import numpy as np
print(f"  Text 1 - Mean: {np.mean(text1_lengths):.0f}, Median: {np.median(text1_lengths):.0f}, Max: {np.max(text1_lengths)}")
print(f"  Text 2 - Mean: {np.mean(text2_lengths):.0f}, Median: {np.median(text2_lengths):.0f}, Max: {np.max(text2_lengths)}")

print("\n" + "="*80)
print("✓ VALIDATION COMPLETE!")
print("="*80)
print("\nNext steps:")
print("  1. Check if you have the trained BERT model")
print("  2. Check if you have the feature vector model (large_model.p)")
print("  3. Run baseline evaluation")
