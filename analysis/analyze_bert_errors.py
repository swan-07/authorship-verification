import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import numpy as np
import pickle
import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

from sentence_transformers import SentenceTransformer

print("=" * 80)
print("BERT ERROR ANALYSIS")
print("=" * 80)

# Load test results
print("\nLoading test results...")
results = np.load('finetuned_test_results_full.npz')
scores = results['scores']
probs = results['probs']
predictions = results['predictions']
labels = results['labels']
threshold = float(results['threshold'])

print(f"Threshold: {threshold:.4f}")
print(f"Total test pairs: {len(labels):,}")

# Find errors
errors = predictions != labels
false_positives = (predictions == 1) & (labels == 0)
false_negatives = (predictions == 0) & (labels == 1)

print(f"\nTotal errors: {errors.sum():,} ({errors.mean()*100:.1f}%)")
print(f"False positives: {false_positives.sum():,} (predicted same author, but different)")
print(f"False negatives: {false_negatives.sum():,} (predicted different author, but same)")

# Analyze confidence of errors
print("\n" + "=" * 80)
print("ERROR CONFIDENCE ANALYSIS")
print("=" * 80)

fp_probs = probs[false_positives]
fn_probs = probs[false_negatives]

print(f"\nFalse Positives (said same, but different):")
print(f"  Mean confidence: {fp_probs.mean():.3f}")
print(f"  Median confidence: {np.median(fp_probs):.3f}")
print(f"  Min confidence: {fp_probs.min():.3f}")
print(f"  Max confidence: {fp_probs.max():.3f}")

print(f"\nFalse Negatives (said different, but same):")
print(f"  Mean confidence: {fn_probs.mean():.3f}")
print(f"  Median confidence: {np.median(fn_probs):.3f}")
print(f"  Min confidence: {fn_probs.min():.3f}")
print(f"  Max confidence: {fn_probs.max():.3f}")

# High confidence errors (most interesting)
high_conf_fp = (false_positives) & (probs > 0.7)
high_conf_fn = (false_negatives) & (probs < 0.3)

print(f"\nHigh-confidence errors:")
print(f"  High-conf FP (>0.7 prob, but wrong): {high_conf_fp.sum():,}")
print(f"  High-conf FN (<0.3 prob, but wrong): {high_conf_fn.sum():,}")

# Load dataset to examine specific errors
print("\n" + "=" * 80)
print("EXAMINING SPECIFIC ERROR EXAMPLES")
print("=" * 80)

with open('dataset_cached.pkl', 'rb') as f:
    dataset = pickle.load(f)

test_data = dataset['test']

# Show a few high-confidence false positives
print("\nHigh-confidence FALSE POSITIVES (model very sure same author, but wrong):")
print("-" * 80)

fp_indices = np.where(high_conf_fp)[0][:3]
for i, idx in enumerate(fp_indices):
    ex = test_data[int(idx)]
    print(f"\nExample {i+1}:")
    print(f"  Probability: {probs[idx]:.3f} (threshold: {threshold:.3f})")
    print(f"  Cosine similarity: {scores[idx]:.3f}")
    print(f"  Text 1 (first 200 chars): {ex['text1'][:200]}...")
    print(f"  Text 2 (first 200 chars): {ex['text2'][:200]}...")
    print(f"  True label: Different authors")
    print(f"  Model said: Same author (WRONG)")

# Show a few high-confidence false negatives
print("\n" + "=" * 80)
print("\nHigh-confidence FALSE NEGATIVES (model very sure different author, but wrong):")
print("-" * 80)

fn_indices = np.where(high_conf_fn)[0][:3]
for i, idx in enumerate(fn_indices):
    ex = test_data[int(idx)]
    print(f"\nExample {i+1}:")
    print(f"  Probability: {probs[idx]:.3f} (threshold: {threshold:.3f})")
    print(f"  Cosine similarity: {scores[idx]:.3f}")
    print(f"  Text 1 (first 200 chars): {ex['text1'][:200]}...")
    print(f"  Text 2 (first 200 chars): {ex['text2'][:200]}...")
    print(f"  True label: Same author")
    print(f"  Model said: Different author (WRONG)")

# Correct predictions for comparison
correct = predictions == labels
correct_same = (predictions == 1) & (labels == 1)
correct_diff = (predictions == 0) & (labels == 0)

print("\n" + "=" * 80)
print("COMPARISON WITH CORRECT PREDICTIONS")
print("=" * 80)

print(f"\nCorrect 'same author' predictions:")
print(f"  Count: {correct_same.sum():,}")
print(f"  Mean probability: {probs[correct_same].mean():.3f}")
print(f"  Mean cosine similarity: {scores[correct_same].mean():.3f}")

print(f"\nCorrect 'different author' predictions:")
print(f"  Count: {correct_diff.sum():,}")
print(f"  Mean probability: {probs[correct_diff].mean():.3f}")
print(f"  Mean cosine similarity: {scores[correct_diff].mean():.3f}")

print(f"\nFalse positives (errors):")
print(f"  Count: {false_positives.sum():,}")
print(f"  Mean probability: {probs[false_positives].mean():.3f}")
print(f"  Mean cosine similarity: {scores[false_positives].mean():.3f}")

print(f"\nFalse negatives (errors):")
print(f"  Count: {false_negatives.sum():,}")
print(f"  Mean probability: {probs[false_negatives].mean():.3f}")
print(f"  Mean cosine similarity: {scores[false_negatives].mean():.3f}")

# Score distributions
print("\n" + "=" * 80)
print("SCORE DISTRIBUTION ANALYSIS")
print("=" * 80)

print("\nCosine Similarity Distribution:")
print(f"  Same author pairs (correct):     {scores[labels == 1].mean():.3f} ± {scores[labels == 1].std():.3f}")
print(f"  Different author pairs (correct): {scores[labels == 0].mean():.3f} ± {scores[labels == 0].std():.3f}")
print(f"  Overlap: {'High' if abs(scores[labels == 1].mean() - scores[labels == 0].mean()) < 0.1 else 'Moderate' if abs(scores[labels == 1].mean() - scores[labels == 0].mean()) < 0.2 else 'Low'}")

print("\nDone!")
