#!/usr/bin/env python3
"""
Test the calibrated BERT model on all test datasets.
"""

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

print("=" * 80)
print("TESTING CALIBRATED BASE BERT MODEL")
print("=" * 80)

# Load calibration model
print("\n1. Loading calibration model...")
with open('calibration_model_base_bert.pkl', 'rb') as f:
    calibration_model = pickle.load(f)

with open('calibration_config.pkl', 'rb') as f:
    config = pickle.load(f)

print(f"   ✓ Optimal threshold: {config['best_threshold']:.4f}")

# Load BERT model
print("\n2. Loading base BERT model...")
model = SentenceTransformer('bert-base-cased')
print("   ✓ Model loaded")

# Load all test datasets
print("\n3. Loading test datasets...")
dataset = load_dataset('swan07/authorship-verification')

test_datasets = [
    'arxiv_test', 'blogs_test', 'british_test', 'darkreddit_test',
    'imdb_test', 'pan11_test', 'pan13_test', 'pan14_test',
    'pan15_test', 'pan20_test', 'reuters_test', 'victorian_test'
]

print(f"   ✓ {len(test_datasets)} test datasets loaded")

# Test on each dataset
print("\n4. Evaluating on all test datasets...")
print("=" * 80)

all_results = []

for test_name in test_datasets:
    print(f"\n{test_name}:")

    test_data = dataset[test_name]

    # Compute cosine similarities
    cosine_scores = []
    true_labels = []

    for i, example in enumerate(tqdm(test_data, desc=f"  Computing scores")):
        emb1 = model.encode(example['text1'], convert_to_tensor=False)
        emb2 = model.encode(example['text2'], convert_to_tensor=False)

        # Cosine similarity
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        cosine_scores.append(cosine_sim)
        true_labels.append(example['same'])

    cosine_scores = np.array(cosine_scores).reshape(-1, 1)
    true_labels = np.array(true_labels)

    # Get calibrated probabilities
    calibrated_probs = calibration_model.predict_proba(cosine_scores)[:, 1]

    # Make predictions using optimal threshold
    predictions = (calibrated_probs >= config['best_threshold']).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.1%}")

    all_results.append({
        'dataset': test_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_samples': len(test_data)
    })

# Print summary
print("\n" + "=" * 80)
print("SUMMARY - ALL DATASETS")
print("=" * 80)

print(f"\n{'Dataset':<20} {'Samples':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 80)

for result in all_results:
    print(f"{result['dataset']:<20} {result['n_samples']:>8} "
          f"{result['accuracy']:>9.1%} {result['precision']:>9.1%} "
          f"{result['recall']:>9.1%} {result['f1']:>9.1%}")

# Overall average
avg_accuracy = np.mean([r['accuracy'] for r in all_results])
avg_precision = np.mean([r['precision'] for r in all_results])
avg_recall = np.mean([r['recall'] for r in all_results])
avg_f1 = np.mean([r['f1'] for r in all_results])

print("-" * 80)
print(f"{'AVERAGE':<20} {'':>8} {avg_accuracy:>9.1%} {avg_precision:>9.1%} "
      f"{avg_recall:>9.1%} {avg_f1:>9.1%}")

print("\n" + "=" * 80)
print("COMPARISON TO BASELINE")
print("=" * 80)
print("\nOriginal models:")
print("  - Fine-tuned BERT (broken): 56.6%")
print("  - Feature Vector: 64.6% AUC, 62.6% overall")
print("\nCurrent (Base BERT + Calibration):")
print(f"  - Average Accuracy: {avg_accuracy:.1%}")
print(f"  - Average F1: {avg_f1:.1%}")

if avg_accuracy > 0.646:
    print(f"\n✓ BETTER than original feature vector by {(avg_accuracy - 0.646)*100:.1f} percentage points!")
else:
    print(f"\n✗ Worse than original feature vector by {(0.646 - avg_accuracy)*100:.1f} percentage points")

print("\n" + "=" * 80)
